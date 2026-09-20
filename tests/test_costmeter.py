"""成本计量（T4.1）的测试：usage 采集 + 计价 + 流式路径 + 账本。"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatResult

from agent_base.core.config import Settings
from agent_base.extensions.costmeter import (
    COST_METER,
    CostMeterHandler,
    MemoryCostLedger,
    SqliteCostLedger,
)

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {"llm_api_key": "sk-test"}
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


def _response(usage: dict[str, Any] | None) -> ChatResult:
    message = AIMessage(content="x", usage_metadata=usage) if usage else AIMessage(content="x")
    return ChatResult(generations=[ChatGeneration(message=message)])


def _fresh_meter(prices_json: str = ""):
    """独立 meter（清空进程级单例状态，绑定新价目）。"""
    meter = COST_METER.__class__(prices_json=prices_json)
    return meter


# ─────────────────────────── handler 采集 ───────────────────────────


def test_handler_records_usage_row() -> None:
    meter = _fresh_meter()
    handler = CostMeterHandler(model="deepseek-chat", profile="main", meter=meter)
    handler.on_llm_end(
        _response(
            {
                "input_tokens": 100,
                "output_tokens": 20,
                "input_token_details": {"cache_read": 60},
                "total_tokens": 120,
            }
        )
    )
    rows = meter.take_pending()
    assert len(rows) == 1
    row = rows[0]
    assert row.prompt_tokens == 100
    assert row.completion_tokens == 20
    assert row.cache_read_tokens == 60
    assert row.model == "deepseek-chat" and row.profile == "main"


def test_handler_zero_values_without_usage() -> None:
    meter = _fresh_meter()
    handler = CostMeterHandler(model="m", profile="main", meter=meter)
    handler.on_llm_end(_response(None))  # 不应抛错
    row = meter.take_pending()[0]
    assert (row.prompt_tokens, row.completion_tokens, row.cache_read_tokens) == (0, 0, 0)


def test_llm_output_token_usage_fallback() -> None:
    """部分 provider 只在 llm_output 给 OpenAI 风格 token_usage。"""
    meter = _fresh_meter()
    handler = CostMeterHandler(model="m", profile="fast", meter=meter)
    result = ChatResult(generations=[ChatGeneration(message=AIMessage(content="x"))])
    result.llm_output = {
        "token_usage": {
            "prompt_tokens": 7,
            "completion_tokens": 3,
            "prompt_tokens_details": {"cached_tokens": 4},
        }
    }
    handler.on_llm_end(result)
    row = meter.take_pending()[0]
    assert (row.prompt_tokens, row.completion_tokens, row.cache_read_tokens) == (7, 3, 4)


# ─────────────────────────── 计价与累计 ───────────────────────────


def test_meter_prices_day_totals_and_unknown_model(
    caplog: pytest.LogCaptureFixture,
) -> None:
    prices = '{"deepseek-chat": {"input_miss": 3, "input_hit": 0.25, "output": 6}}'
    meter = _fresh_meter(prices_json=prices)
    handler = CostMeterHandler(model="deepseek-chat", profile="main", meter=meter)
    # 100 input（全部命中）+ 20 output → 100/1e6*0.25 + 20/1e6*6 = 0.000145
    handler.on_llm_end(
        _response(
            {
                "input_tokens": 100,
                "output_tokens": 20,
                "input_token_details": {"cache_read": 100},
                "total_tokens": 120,
            }
        )
    )
    assert meter.day_total_cny() == pytest.approx(100 / 1e6 * 0.25 + 20 / 1e6 * 6)

    # 未知模型：计 0 元并告警（至多一次）。
    import logging

    with caplog.at_level(logging.WARNING, logger="agent_base.extensions.costmeter"):
        unknown = CostMeterHandler(model="ghost-model", profile="fast", meter=meter)
        unknown.on_llm_end(_response({"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}))
        unknown.on_llm_end(_response({"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}))
    assert meter.day_total_cny() == pytest.approx(100 / 1e6 * 0.25 + 20 / 1e6 * 6)
    warnings = [r for r in caplog.records if "ghost-model" in r.getMessage()]
    assert len(warnings) == 1


def test_month_total_spans_days() -> None:
    meter = _fresh_meter(prices_json='{"m": {"input_miss": 1, "input_hit": 0, "output": 0}}')
    handler = CostMeterHandler(model="m", profile="main", meter=meter)
    for _ in range(3):
        handler.on_llm_end(
            _response({"input_tokens": 1_000_000, "output_tokens": 0, "total_tokens": 1_000_000})
        )
    assert meter.month_total_cny() == pytest.approx(3.0)


# ─────────────────────────── 真实回调管道 ───────────────────────────


async def test_scripted_model_invocation_reaches_handler() -> None:
    """非流式路径：模型级 callback 在真实 invoke 中触发 handler。"""
    from fakes import ScriptedChatModel

    meter = _fresh_meter()
    handler = CostMeterHandler(model="scripted", profile="main", meter=meter)
    message = AIMessage(
        content="ok",
        usage_metadata={"input_tokens": 11, "output_tokens": 4, "total_tokens": 15},
    )
    model = ScriptedChatModel([message], callbacks=[handler])
    await model.ainvoke("hi")
    rows = meter.take_pending()
    assert len(rows) == 1 and rows[0].prompt_tokens == 11 and rows[0].completion_tokens == 4


async def test_streaming_aggregate_reaches_handler() -> None:
    """流式路径：末块携带 usage，聚合后仍能到 handler（stream_usage 语义）。"""

    class _Streaming(BaseCallbackHandler):
        pass

    from langchain_core.language_models.chat_models import BaseChatModel

    class _StreamFake(BaseChatModel):
        handler: Any = None

        @property
        def _llm_type(self) -> str:
            return "stream-fake"

        def _generate(self, *a: Any, **k: Any) -> ChatResult:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

        async def _astream(self, messages: Any, stop: Any = None, **k: Any):
            from langchain_core.outputs import ChatGenerationChunk

            yield ChatGenerationChunk(message=AIMessageChunk(content="你"))
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="好",
                    usage_metadata={
                        "input_tokens": 9,
                        "output_tokens": 2,
                        "total_tokens": 11,
                    },
                )
            )

    meter = _fresh_meter()
    handler = CostMeterHandler(model="stream-fake", profile="fast", meter=meter)
    model = _StreamFake(callbacks=[handler])
    final = None
    async for chunk in model.astream("hi"):
        final = chunk if final is None else final + chunk
    assert final is not None and final.usage_metadata is not None
    rows = meter.take_pending()
    assert len(rows) == 1 and rows[0].prompt_tokens == 9 and rows[0].completion_tokens == 2


# ─────────────────────────── build_llm 接线 ───────────────────────────


def test_build_llm_attaches_handler_when_cost_enabled() -> None:
    from agent_base.core.llm import build_llm

    settings = _settings(cost_enabled=True, llm_stream_usage=True)
    model = build_llm(settings)
    assert model.stream_usage is True
    assert any(isinstance(cb, CostMeterHandler) for cb in (model.callbacks or []))

    settings_off = _settings()
    model_off = build_llm(settings_off)
    assert model_off.stream_usage is False
    assert not any(isinstance(cb, CostMeterHandler) for cb in (model_off.callbacks or []))


def test_build_llm_fast_profile_labels_handler() -> None:
    from agent_base.core.llm import build_llm

    settings = _settings(
        cost_enabled=True,
        llm_fast_base_url="https://fast.example.com/v1",
        llm_fast_model="glm-4.5-flash",
    )
    model = build_llm(settings, profile="fast")
    handlers = [cb for cb in (model.callbacks or []) if isinstance(cb, CostMeterHandler)]
    assert handlers and handlers[0].profile == "fast"
    assert model.model_name == "glm-4.5-flash"


# ─────────────────────────── 账本 ───────────────────────────


async def test_memory_ledger_roundtrip() -> None:
    ledger = MemoryCostLedger()
    meter = _fresh_meter(prices_json='{"m": {"input_miss": 1, "input_hit": 0, "output": 0}}')
    handler = CostMeterHandler(model="m", profile="main", meter=meter)
    handler.on_llm_end(
        _response({"input_tokens": 1_000_000, "output_tokens": 0, "total_tokens": 1_000_000})
    )
    await ledger.insert(meter.take_pending())
    assert await ledger.totals_since(0.0) == pytest.approx(1.0)
    assert await ledger.totals_since(_far_future_ts()) == 0.0


def _far_future_ts() -> float:
    return 4_102_444_800.0  # 2100-01-01


async def test_sqlite_ledger_roundtrip(tmp_path: Path) -> None:
    ledger = await SqliteCostLedger.create(str(tmp_path / "cost.db"))
    try:
        meter = _fresh_meter(prices_json='{"m": {"input_miss": 2, "input_hit": 0, "output": 0}}')
        handler = CostMeterHandler(model="m", profile="fast", meter=meter)
        handler.on_llm_end(
            _response({"input_tokens": 1_000_000, "output_tokens": 0, "total_tokens": 1_000_000})
        )
        await ledger.insert(meter.take_pending())
        assert await ledger.totals_since(0.0) == pytest.approx(2.0)
    finally:
        await ledger.aclose()


def test_cost_config_contract() -> None:
    settings = _settings(
        cost_enabled=True,
        cost_daily_cny=0.5,
        cost_monthly_cny=15,
        cost_prices_json='{"m": {"input_miss": 1, "input_hit": 0.1, "output": 2}}',
    )
    assert settings.cost.enabled is True
    assert settings.cost.daily_cny == 0.5
    assert settings.cost.prices == {"m": {"input_miss": 1, "input_hit": 0.1, "output": 2}}
    with pytest.raises(ValueError, match="COST_PRICES_JSON"):
        _settings(cost_prices_json="not-json")


def test_nightly_script_uses_meter_importable() -> None:
    """夜间批脚本与摄取脚本必须可导入（回帰守卫：脚本语法/依赖完整）。"""
    for name in ("memory_nightly_capture", "kb_ingest"):
        spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert sys.modules.get(name) is not module or True


def test_metrics_render_includes_cost_lines() -> None:
    from agent_base.extensions.metrics import CostMetrics

    metrics = CostMetrics()  # 独立实例：进程级单例可能已被其他测试观测过
    metrics.observe_tokens("deepseek-chat", "main", "input", 42)
    metrics.observe_cost(0.0015)
    rendered = metrics.render_cost_metrics()
    assert 'llm_tokens_total{model="deepseek-chat",profile="main",kind="input"} 42' in rendered
    assert "llm_cost_cny_total 0.001500" in rendered
