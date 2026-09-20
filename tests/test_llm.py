"""core.llm 的测试（openai 兼容客户端装配 + 双档位 + 弹性包装）。"""

from __future__ import annotations

import asyncio

import pytest
from openai import OpenAIError

from agent_base.core.config import Settings, SettingsError
from agent_base.core.llm import ResilientLLM, build_llm


def test_build_llm_with_key_sets_model() -> None:
    settings = Settings(_env_file=None, llm_api_key="sk-test", llm_model="deepseek-chat")
    llm = build_llm(settings)
    assert llm.model_name == "deepseek-chat"


def test_build_llm_construction_failure_becomes_settings_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """原始的 openai 错误（例如到处都缺 key）应可操作，而不是泄漏。"""

    def _fail(**kwargs: object) -> None:
        raise OpenAIError("no api key available")

    monkeypatch.setattr("agent_base.core.llm.ChatOpenAI", _fail)
    with pytest.raises(SettingsError, match="LLM_API_KEY"):
        build_llm(Settings(_env_file=None, llm_api_key=""))


# -- fast 档（LLM_FAST_*） -------------------------------------------------


def test_fast_profile_degrades_to_main_when_unconfigured() -> None:
    """LLM_FAST_* 未配置时 fast 档返回主力配置——免费档是优化不是依赖。"""
    settings = Settings(
        _env_file=None,
        llm_api_key="sk-test",
        llm_model="deepseek-chat",
        llm_base_url="https://api.deepseek.com/v1",
    )
    main_llm = build_llm(settings, profile="main")
    fast_llm = build_llm(settings, profile="fast")
    assert fast_llm.model_name == main_llm.model_name
    assert fast_llm.openai_api_base == main_llm.openai_api_base


def test_fast_profile_uses_fast_settings_when_configured() -> None:
    settings = Settings(
        _env_file=None,
        llm_api_key="sk-main",
        llm_fast_api_key="sk-fast",
        llm_fast_base_url="https://open.bigmodel.cn/api/paas/v4",
        llm_fast_model="glm-4.5-flash",
    )
    fast_llm = build_llm(settings, profile="fast")
    assert fast_llm.model_name == "glm-4.5-flash"
    assert fast_llm.openai_api_base == "https://open.bigmodel.cn/api/paas/v4"
    # fast 档未配 key 时回落主力 key（同一 provider 双档的常见形态）。
    settings_shared_key = Settings(
        _env_file=None,
        llm_api_key="sk-main",
        llm_fast_base_url="https://open.bigmodel.cn/api/paas/v4",
        llm_fast_model="glm-4.5-flash",
    )
    assert (
        build_llm(settings_shared_key, profile="fast").openai_api_key.get_secret_value()
        == "sk-main"
    )


def test_main_profile_ignores_fast_settings() -> None:
    settings = Settings(
        _env_file=None,
        llm_api_key="sk-main",
        llm_fast_base_url="https://fast.example.com/v1",
        llm_fast_model="glm-4.5-flash",
    )
    assert build_llm(settings, profile="main").model_name == "deepseek-chat"


# -- ResilientLLM（免费档限流 + 回退） --------------------------------------


class _Flaky:
    """按脚本次序抛错/应答的假 primary。"""

    def __init__(self, outcomes: list[object]) -> None:
        self._outcomes = list(outcomes)
        self.calls = 0

    async def ainvoke(self, messages: object) -> str:
        outcome = self._outcomes.pop(0)
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return str(outcome)


class _Steady:
    async def ainvoke(self, messages: object) -> str:
        self.calls = getattr(self, "calls", 0) + 1  # type: ignore[attr-defined]
        return "fallback-ok"


async def test_resilient_llm_retries_transient_error_then_succeeds() -> None:
    primary = _Flaky([RuntimeError("429 rate limited"), "primary-ok"])
    llm = ResilientLLM(primary=primary, max_retries=2, base_delay=0.0)
    assert await llm.ainvoke([("human", "hi")]) == "primary-ok"
    assert primary.calls == 2


async def test_resilient_llm_falls_back_after_retries_exhausted() -> None:
    primary = _Flaky([RuntimeError("boom"), RuntimeError("boom"), RuntimeError("boom")])
    fallback = _Steady()
    llm = ResilientLLM(primary=primary, fallback=fallback, max_retries=2, base_delay=0.0)
    assert await llm.ainvoke([("human", "hi")]) == "fallback-ok"
    assert primary.calls == 3  # 1 次原始尝试 + 2 次重试
    assert fallback.calls == 1


async def test_resilient_llm_reraises_without_fallback() -> None:
    primary = _Flaky([RuntimeError("boom"), RuntimeError("boom"), RuntimeError("boom")])
    llm = ResilientLLM(primary=primary, max_retries=2, base_delay=0.0)
    with pytest.raises(RuntimeError, match="boom"):
        await llm.ainvoke([("human", "hi")])


async def test_resilient_llm_semaphore_serializes_concurrency() -> None:
    live = 0
    peak = 0

    class _Slow:
        async def ainvoke(self, messages: object) -> str:
            nonlocal live, peak
            live += 1
            peak = max(peak, live)
            await asyncio.sleep(0.01)
            live -= 1
            return "ok"

    llm = ResilientLLM(primary=_Slow(), max_concurrency=1, base_delay=0.0)
    results = await asyncio.gather(*[llm.ainvoke([("human", "q")]) for _ in range(3)])
    assert results == ["ok", "ok", "ok"]
    assert peak == 1


async def test_tool_safe_stream_falls_back_to_generate_when_tools_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """快档客户端：绑定工具时 _astream 退回非流式聚合（SiliconFlow
    GLM-4-9B 流式工具调用违反 OpenAI 协议——工具名被拆进 arguments delta）；
    未绑工具时保持原生流式。"""
    from langchain_core.messages import AIMessage, AIMessageChunk
    from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

    from agent_base.core.llm import _ToolSafeStreamChatOpenAI

    model = _ToolSafeStreamChatOpenAI(model="m", api_key="sk-test")
    calls: list[str] = []

    def _fake_generate(
        messages: object, stop: object = None, run_manager: object = None, **kw: object
    ) -> ChatResult:
        calls.append("generate")
        assert kw.get("tools")  # 非流式路径必须带上工具定义
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

    monkeypatch.setattr(model, "_generate", _fake_generate)

    # 绑定工具 → 退回 _generate，聚合出分块。
    chunks = [c async for c in model._astream("hi", tools=[{"type": "function"}])]
    assert calls == ["generate"]
    assert chunks and chunks[0].message.content == "ok"

    # 未绑工具 → 不走 _generate，交回原生流式。
    async def _fake_astream(
        self: object,
        messages: object,
        stop: object = None,
        run_manager: object = None,
        **kw: object,
    ):
        calls.append("astream")
        yield ChatGenerationChunk(message=AIMessageChunk(content="s"))

    monkeypatch.setattr(_ToolSafeStreamChatOpenAI, "_astream", _fake_astream)
    chunks2 = [c async for c in model._astream("hi")]
    assert "astream" in calls
    assert chunks2[0].message.content == "s"
