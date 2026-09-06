"""Tests for the shared tool pool (Stage 3): assembly, timeout, fail-soft."""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.tools import BaseTool, tool

from agent_base.core.tools import (
    DEFAULT_TOOL_TIMEOUT_SECONDS,
    ToolPoolError,
    ToolTimeoutError,
    build_tool_pool,
)
from agent_base.modules.chat.module import ChatModule


class _FakeModule:
    """Minimal AgentModule stand-in: only get_tools matters to the pool."""

    def __init__(self, name: str, tools: list[BaseTool]) -> None:
        self.name = name
        self.description = "stand-in"
        self._tools = tools

    def get_tools(self) -> list[BaseTool]:
        return self._tools


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def test_pool_collects_in_module_order() -> None:
    modules = {
        "chat": _FakeModule("chat", ChatModule().get_tools()),
        "math": _FakeModule("math", [add]),
    }
    pool = build_tool_pool(modules)
    assert [t.name for t in pool] == ["echo", "add"]


def test_pool_duplicate_name_fails_fast() -> None:
    @tool
    def echo(text: str) -> str:
        """Another echo — collides with the chat module's tool."""
        return text

    modules = {
        "chat": _FakeModule("chat", ChatModule().get_tools()),
        "evil": _FakeModule("evil", [echo]),
    }
    with pytest.raises(ToolPoolError, match="duplicate tool name 'echo'"):
        build_tool_pool(modules)


def test_pool_wraps_with_default_timeout() -> None:
    pool = build_tool_pool({"chat": _FakeModule("chat", ChatModule().get_tools())})
    wrapped = pool[0]
    assert wrapped.name == "echo"
    assert getattr(wrapped, "timeout", None) == DEFAULT_TOOL_TIMEOUT_SECONDS


async def test_timeout_tool_async_cuts_off() -> None:
    @tool
    async def slow() -> str:
        """Sleeps past any budget."""
        await asyncio.sleep(10)
        return "late"

    pool = build_tool_pool({"m": _FakeModule("m", [slow])}, timeout=0.05)
    with pytest.raises(ToolTimeoutError, match="exceeded"):
        await pool[0].ainvoke({})


def test_timeout_tool_sync_cuts_off() -> None:
    import time

    @tool
    def slow_sync() -> str:
        """Sleeps past any budget (sync path)."""
        time.sleep(10)
        return "late"

    pool = build_tool_pool({"m": _FakeModule("m", [slow_sync])}, timeout=0.05)
    with pytest.raises(ToolTimeoutError, match="exceeded"):
        pool[0].invoke({})


def test_wrapped_tool_preserves_schema_and_result() -> None:
    pool = build_tool_pool({"m": _FakeModule("m", [add])}, timeout=5)
    assert pool[0].invoke({"a": 2, "b": 3}) == 5
    assert "a" in pool[0].args_schema.model_fields
