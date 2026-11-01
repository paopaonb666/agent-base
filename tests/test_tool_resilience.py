"""工具韧性层（ADR-009）的测试：分类、健康注册表、结构化反馈、池集成。"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, ValidationError

from agent_base.core.config import Settings
from agent_base.core.graphs import build_single_agent_graph
from agent_base.core.tool_resilience import (
    SCHEMA_ERROR_MAX_CHARS,
    ToolErrorReporter,
    ToolHealthRegistry,
    ToolResilience,
    classify_tool_error,
)
from agent_base.core.tools import ToolTimeoutError, build_tool_pool
from tests.fakes import ScriptedChatModel


# ── 异常分类 ────────────────────────────────────────────────────────────


class _StrictArgs(BaseModel):
    limit: int
    query: str


def _validation_error() -> ValidationError:
    try:
        _StrictArgs(limit="十", query="ok")  # type: ignore[arg-type]
    except ValidationError as exc:
        return exc
    raise AssertionError("unreachable")


def test_classify_schema_for_validation_error() -> None:
    assert classify_tool_error(_validation_error()) == "schema"


def test_classify_schema_for_langgraph_invocation_error() -> None:
    from langgraph.prebuilt.tool_node import ToolInvocationError

    exc = ToolInvocationError("web_search", _validation_error(), {"limit": "十"})
    assert classify_tool_error(exc) == "schema"


def test_classify_transient_by_type_and_mro() -> None:
    assert classify_tool_error(ConnectionError("refused")) == "transient"
    assert classify_tool_error(TimeoutError("upstream timed out")) == "transient"

    class FakeHttpError(Exception):
        """httpx 错误族的替身：不 import httpx，靠 MRO 名字匹配。"""

    FakeHttpError.__name__ = "HTTPError"
    assert classify_tool_error(FakeHttpError("boom")) == "transient"


def test_classify_transient_by_message_pattern() -> None:
    assert classify_tool_error(RuntimeError("API rate limit exceeded")) == "transient"
    assert classify_tool_error(RuntimeError("service unavailable, retry later")) == "transient"


def test_classify_permanent_for_logic_errors() -> None:
    assert classify_tool_error(ValueError("路径在键 'a' 处失败")) == "permanent"
    assert classify_tool_error(KeyError("missing")) == "permanent"


# ── 健康注册表 ──────────────────────────────────────────────────────────


def _registry(**kwargs: Any) -> ToolHealthRegistry:
    return ToolHealthRegistry(**kwargs)


def test_consecutive_failures_block_and_recover() -> None:
    # 冷却时长给足（5s）保证断言时仍在冷却中；到期恢复在下一个测试覆盖。
    reg = _registry(blacklist_threshold=3, blacklist_cooldown_seconds=5.0)
    for _ in range(2):
        reg.record("t", "error")
    assert reg.is_blocked("t")[0] is False  # 阈值未到
    reg.record("t", "error")
    blocked, reason = reg.is_blocked("t")
    assert blocked
    assert "temporarily blocked" in reason and "3 consecutive" in reason
    assert reg.health() == "degraded"


def test_cooldown_expires_and_recovers() -> None:
    reg = _registry(blacklist_threshold=3, blacklist_cooldown_seconds=0.05)
    for _ in range(3):
        reg.record("t", "error")
    assert reg.is_blocked("t")[0] is True
    time.sleep(0.3)  # 6 倍余量：冷却到期自动恢复，无需显式复位
    assert reg.is_blocked("t")[0] is False
    assert reg.health() == "ok"


def test_success_resets_consecutive_counter() -> None:
    reg = _registry(blacklist_threshold=3, blacklist_cooldown_seconds=60.0)
    reg.record("t", "error")
    reg.record("t", "error")
    reg.record("t", "ok")  # 间歇性抖动不被历史惩罚
    reg.record("t", "error")
    reg.record("t", "error")
    assert reg.is_blocked("t")[0] is False


def test_cooldown_escalates_after_recovery() -> None:
    reg = _registry(blacklist_threshold=2, blacklist_cooldown_seconds=0.05)
    reg.record("t", "error")
    reg.record("t", "error")  # 第 1 档冷却
    time.sleep(0.3)
    assert reg.is_blocked("t")[0] is False
    reg.record("t", "error")  # 恢复后立刻又失败 → 第 2 档（时长翻倍）
    assert reg.is_blocked("t")[0] is True
    assert reg._tools["t"].cooldown_count == 2
    assert reg.health() == "degraded"


def test_blacklist_disabled_with_threshold_zero() -> None:
    reg = _registry(blacklist_threshold=0)
    for _ in range(10):
        reg.record("t", "error")
    assert reg.is_blocked("t")[0] is False


def test_global_budget_triggers_shed_and_recovers() -> None:
    reg = _registry(
        blacklist_threshold=0,
        global_failure_window_seconds=0.05,
        global_failure_budget=2,
    )
    assert reg.retries_allowed() is True
    reg.record("a", "error")
    reg.record("b", "error")  # 跨工具失败同样计入全局预算
    assert reg.global_shed_active is True
    assert reg.retries_allowed() is False
    assert reg.health() == "degraded"
    time.sleep(0.3)
    assert reg.global_shed_active is False  # 窗口滑过自动恢复
    assert reg.retries_allowed() is True


# ── 结构化参数反馈 + 修复预算 ───────────────────────────────────────────


def test_schema_feedback_lists_fields_with_values() -> None:
    reporter = ToolErrorReporter(max_repair_attempts=3)
    text = reporter.handle(_validation_error(), tool="web_search")
    assert text.startswith("argument validation failed for tool 'web_search'")
    assert 'field "limit"' in text  # per-field：模型能定位到具体字段
    assert "attempt 1/3" in text


def test_schema_feedback_survives_many_fields() -> None:
    """多字段不被 300 截断（放宽到 1000）——压平旧路径会丢后半字段。"""

    class _Wide(BaseModel):
        f0: int
        f1: int
        f2: int
        f3: int
        f4: int
        f5: int

    try:
        _Wide(f0="x", f1="x", f2="x", f3="x", f4="x", f5="x")  # type: ignore[arg-type]
    except ValidationError as exc:
        reporter = ToolErrorReporter(max_repair_attempts=99)
        text = reporter.handle(exc, tool="t")
        assert 'field "f5"' in text  # 最后一个字段仍在
        assert len(text) <= SCHEMA_ERROR_MAX_CHARS
    else:  # pragma: no cover
        raise AssertionError("unreachable")


def test_repair_budget_gives_terminal_feedback() -> None:
    reporter = ToolErrorReporter(max_repair_attempts=3)
    first = reporter.handle(_validation_error(), tool="t")
    assert "STOP" not in first
    reporter.handle(_validation_error(), tool="t")
    third = reporter.handle(_validation_error(), tool="t")
    assert "STOP calling this tool" in third  # 超限：终止性反馈


def test_repair_budget_resets_on_success() -> None:
    reporter = ToolErrorReporter(max_repair_attempts=2)
    reporter.handle(_validation_error(), tool="t")
    reporter.note_success("t")  # 成功调用清预算
    again = reporter.handle(_validation_error(), tool="t")
    assert "attempt 1/2" in again


def test_reporter_preserves_legacy_sanitization_for_non_schema() -> None:
    reporter = ToolErrorReporter()
    text = reporter.handle(ValueError("connect to http://10.0.0.1:8123/x failed"))
    assert "http://" not in text
    assert "<redacted-url>" in text
    assert len(text) <= 300


# ── ToolNode 路径：ToolInvocationError 自带 tool_name ──────────────────


async def test_tool_node_schema_error_gets_structured_feedback() -> None:
    """ToolNode 只传异常给回调——reporter 从 ToolInvocationError 自取工具名。"""

    @tool
    def strict(limit: int) -> str:
        """需要整数参数的工具。"""
        return str(limit)

    settings = Settings(_env_file=None)
    resilience = ToolResilience.from_settings(settings.toolkit)
    pool = build_tool_pool({}, extra_tools=[strict], resilience=resilience)
    model = ScriptedChatModel(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "strict", "args": {"limit": "十"}, "id": "c1"}],
            ),
            AIMessage(content="好的，我改用别的方式回答。"),
        ]
    )
    ctx = _make_context(settings, model, pool, resilience)
    graph = build_single_agent_graph(ctx, name="strict-demo")
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="调一下 strict")], },
        config={"recursion_limit": 8},
    )
    tool_messages = [m for m in result["messages"] if m.type == "tool"]
    assert tool_messages, "工具消息应存在"
    content = tool_messages[0].content
    assert "argument validation failed for tool 'strict'" in content
    assert 'field "limit"' in content  # per-field 结构化反馈到达模型


def _make_context(
    settings: Settings, model: ScriptedChatModel, pool: list[BaseTool], resilience: ToolResilience
) -> Any:
    from agent_base.core.contracts import ModuleContext

    return ModuleContext(
        settings=settings,
        llm=model,
        tools=pool,
        tool_resilience=resilience,
    )


# ── 池包装器集成：短路 / 重试 / 落账 ────────────────────────────────────


class _FlakyTool:
    """按脚本顺序失败/成功的工具内核：计数器记录真实执行次数。"""

    def __init__(self, outcomes: list[BaseException | str]) -> None:
        self.outcomes = outcomes
        self.calls = 0

    def make(self) -> BaseTool:
        outcomes = self.outcomes
        calls = self

        @tool
        def flaky() -> str:
            """脚本化结局的工具。"""
            idx = min(calls.calls, len(outcomes) - 1)
            calls.calls += 1
            outcome = outcomes[idx]
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        return flaky


def _fast_resilience(**overrides: Any) -> ToolResilience:
    settings = Settings(_env_file=None)
    bundle = ToolResilience.from_settings(settings.toolkit)
    bundle.retry_backoff_base_seconds = 0.01
    bundle.retry_backoff_cap_seconds = 0.02
    for key, value in overrides.items():
        setattr(bundle, key, value)
    return bundle


async def test_transient_error_retried_until_success() -> None:
    inner = _FlakyTool([ConnectionError("refused"), "recovered"])
    pool = build_tool_pool(
        {}, extra_tools=[inner.make()], resilience=_fast_resilience(retry_max_attempts=2)
    )
    result = await pool[0].ainvoke({})
    assert result == "recovered"
    assert inner.calls == 2  # 真的执行了两次


async def test_permanent_error_not_retried() -> None:
    inner = _FlakyTool([ValueError("logic error")])
    pool = build_tool_pool(
        {}, extra_tools=[inner.make()], resilience=_fast_resilience(retry_max_attempts=3)
    )
    with pytest.raises(ValueError):
        await pool[0].ainvoke({})
    assert inner.calls == 1  # permanent：重试也不会好


async def test_timeout_not_retried_by_default() -> None:
    sleeps = {"count": 0}

    @tool
    async def slow() -> str:
        """每次都睡过预算。"""
        sleeps["count"] += 1
        await asyncio.sleep(0.2)
        return "late"

    pool = build_tool_pool(
        {}, timeout=0.05, extra_tools=[slow], resilience=_fast_resilience(retry_max_attempts=2)
    )
    with pytest.raises(ToolTimeoutError):
        await pool[0].ainvoke({})
    assert sleeps["count"] == 1  # 默认不重试超时


async def test_timeout_retried_when_enabled() -> None:
    sleeps = {"count": 0}

    @tool
    async def slow_then_fast() -> str:
        """第一次睡过预算，第二次立即返回。"""
        sleeps["count"] += 1
        if sleeps["count"] == 1:
            await asyncio.sleep(0.2)
        return "ok"

    pool = build_tool_pool(
        {},
        timeout=0.06,
        extra_tools=[slow_then_fast],
        resilience=_fast_resilience(retry_max_attempts=2, retry_on_timeout=True),
    )
    assert await pool[0].ainvoke({}) == "ok"
    assert sleeps["count"] == 2


async def test_global_shed_disables_retries() -> None:
    bundle = _fast_resilience(retry_max_attempts=2)
    # 直接灌满全局预算（不同工具各失败一次）
    bundle.registry = ToolHealthRegistry(
        blacklist_threshold=0, global_failure_window_seconds=60.0, global_failure_budget=1
    )
    bundle.registry.record("other", "error")
    assert bundle.attempts_for() == 1  # 熔断态：重试关闭


async def test_blocked_tool_short_circuits_without_execution() -> None:
    inner = _FlakyTool([ValueError("boom"), "should-not-run"])
    bundle = _fast_resilience()
    bundle.registry = ToolHealthRegistry(
        blacklist_threshold=1, blacklist_cooldown_seconds=60.0, global_failure_budget=0
    )
    pool = build_tool_pool({}, extra_tools=[inner.make()], resilience=bundle)
    with pytest.raises(ValueError):
        await pool[0].ainvoke({})  # 第一次：真失败 → 进入冷却
    assert bundle.registry.is_blocked(pool[0].name)[0] is True
    result = await pool[0].ainvoke({})  # 第二次：短路返回原因，不再执行
    assert "temporarily blocked" in result
    assert inner.calls == 1
    # 短路在 metrics 里记为 blocked（不是 error——工具没被执行）
    from agent_base.extensions.metrics import TOOL_METRICS

    rendered = TOOL_METRICS.render_tool_metrics()
    assert 'outcome="blocked"' in rendered


async def test_success_recorded_to_registry() -> None:
    inner = _FlakyTool(["fine"])
    bundle = _fast_resilience()
    pool = build_tool_pool({}, extra_tools=[inner.make()], resilience=bundle)
    await pool[0].ainvoke({})
    assert bundle.registry.is_blocked(pool[0].name)[0] is False
    assert bundle.registry.retries_allowed() is True


# ── 配置解析 ────────────────────────────────────────────────────────────


def test_resilience_env_vars_reach_toolkit_section(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOOL_RETRY_MAX_ATTEMPTS", "4")
    monkeypatch.setenv("TOOL_BLACKLIST_THRESHOLD", "0")  # 禁用黑名单
    monkeypatch.setenv("TOOL_ARG_REPAIR_MAX_ATTEMPTS", "5")
    settings = Settings(_env_file=None)
    assert settings.toolkit.retry_max_attempts == 4
    assert settings.toolkit.blacklist_threshold == 0
    assert settings.toolkit.arg_repair_max_attempts == 5


def test_resilience_validator_rejects_zero_attempts() -> None:
    with pytest.raises(Exception, match="retry_max_attempts"):
        Settings(_env_file=None, tool_retry_max_attempts=0)
