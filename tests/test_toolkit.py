"""工具库注册表与池集成的测试（M1）：启用语义、快速失败、per-tool 超时、指标。"""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.tools import BaseTool, tool

from agent_base.core.config import Settings
from agent_base.core.tools import ToolPoolError, ToolTimeoutError, build_tool_pool
from agent_base.extensions.metrics import TOOL_METRICS, ToolMetrics
from agent_base.tools.registry import (
    ToolkitError,
    build_toolkit_tools,
    known_toolkit_names,
    toolkit_timeouts,
)


class _FakeModule:
    """最小化的 AgentModule 替身：对池而言只有 get_tools 重要。"""

    def __init__(self, name: str, tools: list[BaseTool]) -> None:
        self.name = name
        self.description = "stand-in"
        self._tools = tools

    def get_tools(self) -> list[BaseTool]:
        return self._tools


@tool
def _noop() -> str:
    """什么都不做的替身工具。"""
    return "ok"


@tool
def _boom() -> str:
    """总是失败的替身工具。"""
    raise RuntimeError("kaput")


@tool
async def _slow() -> str:
    """睡过任何预算的替身工具。"""
    await asyncio.sleep(10)
    return "late"


# ── 注册表 / 装配 ────────────────────────────────────────────────────


def test_default_settings_build_zero_dep_tools() -> None:
    settings = Settings(_env_file=None)
    names = {t.name for t in build_toolkit_tools(settings)}
    assert names == {"current_time", "calculator", "json_query"}


def test_unknown_enabled_name_fails_fast() -> None:
    settings = Settings(_env_file=None, toolkit_enabled=["nope"])
    with pytest.raises(ToolkitError, match="unknown TOOLKIT_ENABLED entry 'nope'"):
        build_toolkit_tools(settings)
    assert "nope" not in known_toolkit_names()


def test_enabled_but_unavailable_fails_with_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("agent_base.tools.search.engines.ddgs_available", lambda: False)
    settings = Settings(_env_file=None, toolkit_enabled=["web_search"])
    with pytest.raises(ToolkitError, match=r"agent-base\[search\]"):
        build_toolkit_tools(settings)


def test_disabled_tool_never_builds(monkeypatch: pytest.MonkeyPatch) -> None:
    # web_search 不可用（无 ddgs、无 key），但它未启用——装配必须无感成功，
    # 工厂根本不会被调用，"启用即承诺"只约束显式开启的工具。
    monkeypatch.setattr("agent_base.tools.search.engines.ddgs_available", lambda: False)
    settings = Settings(_env_file=None, toolkit_enabled=["current_time"])
    tools = build_toolkit_tools(settings)
    assert [t.name for t in tools] == ["current_time"]


def test_timeouts_collected_only_for_enabled() -> None:
    settings = Settings(_env_file=None, toolkit_enabled=["current_time", "web_search"])
    timeouts = toolkit_timeouts(settings)
    assert timeouts == {"web_search": 25.0}


def test_search_spec_declares_shorter_timeout() -> None:
    # 搜索工具的池超时应显著短于全局默认（30s）——它是交互对话的一步。
    assert (
        toolkit_timeouts(Settings(_env_file=None, toolkit_enabled=["web_search"]))["web_search"]
        < 30.0
    )


# ── 池集成 ───────────────────────────────────────────────────────────


def test_pool_merges_module_and_toolkit_tools_in_order() -> None:
    toolkit = build_toolkit_tools(Settings(_env_file=None))
    pool = build_tool_pool({"m": _FakeModule("m", [_noop])}, extra_tools=toolkit)
    assert [t.name for t in pool] == ["_noop", "current_time", "calculator", "json_query"]


def test_pool_rejects_toolkit_name_conflict() -> None:
    @tool
    def calculator(expression: str) -> str:
        """冒名的 calculator——与工具库重名。"""
        return expression

    toolkit = build_toolkit_tools(Settings(_env_file=None))
    with pytest.raises(ToolPoolError, match="duplicate tool name 'calculator'"):
        build_tool_pool({"m": _FakeModule("m", [calculator])}, extra_tools=toolkit)


def test_pool_conflict_message_names_toolkit_source() -> None:
    toolkit = build_toolkit_tools(Settings(_env_file=None))
    other = build_toolkit_tools(Settings(_env_file=None))
    with pytest.raises(ToolPoolError, match="toolkit"):
        build_tool_pool({"m": _FakeModule("m", [])}, extra_tools=[*toolkit, *other])


def test_pool_applies_per_tool_timeouts() -> None:
    pool = build_tool_pool(
        {"m": _FakeModule("m", [_noop])},
        timeout=30.0,
        timeouts={"_noop": 0.05},
    )
    wrapped = pool[0]
    assert wrapped.name == "_noop"
    assert getattr(wrapped, "timeout", None) == 0.05


# ── 工具指标挂钩 ─────────────────────────────────────────────────────


def _count(name: str, outcome: str) -> int:
    return TOOL_METRICS._executions.get((name, outcome), 0)


def test_metrics_record_success_and_error() -> None:
    pool = build_tool_pool({"m": _FakeModule("m", [_noop, _boom])})
    ok_before, err_before = _count("_noop", "ok"), _count("_boom", "error")
    assert pool[0].invoke({}) == "ok"
    with pytest.raises(RuntimeError, match="kaput"):
        pool[1].invoke({})
    assert _count("_noop", "ok") == ok_before + 1
    assert _count("_boom", "error") == err_before + 1


async def test_metrics_record_timeout() -> None:
    pool = build_tool_pool({"m": _FakeModule("m", [_slow])}, timeout=0.05)
    before = _count("_slow", "timeout")
    with pytest.raises(ToolTimeoutError, match="exceeded"):
        await pool[0].ainvoke({})
    assert _count("_slow", "timeout") == before + 1


def test_tool_metrics_render_and_validation() -> None:
    metrics = ToolMetrics()
    metrics.observe_tool("web_search", "ok", 0.5)
    metrics.observe_tool("web_search", "timeout", 20.0)
    rendered = metrics.render_tool_metrics()
    assert 'tool_executions_total{tool="web_search",outcome="ok"} 1' in rendered
    assert 'tool_duration_seconds_bucket{tool="web_search",le="1.0"} 1' in rendered
    assert 'tool_duration_seconds_count{tool="web_search"} 2' in rendered
    with pytest.raises(ValueError, match="unknown tool outcome"):
        metrics.observe_tool("web_search", "bogus", 1.0)
