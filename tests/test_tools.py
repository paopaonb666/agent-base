"""共享工具池的测试（阶段 3）：装配、超时、软失败。"""

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
    """最小化的 AgentModule 替身：对池而言只有 get_tools 重要。"""

    def __init__(self, name: str, tools: list[BaseTool]) -> None:
        self.name = name
        self.description = "stand-in"
        self._tools = tools

    def get_tools(self) -> list[BaseTool]:
        return self._tools


@tool
def add(a: int, b: int) -> int:
    """两个整数相加。"""
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
        """另一个 echo——与 chat 模块的工具冲突。"""
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
        """睡过任何预算。"""
        await asyncio.sleep(10)
        return "late"

    pool = build_tool_pool({"m": _FakeModule("m", [slow])}, timeout=0.05)
    with pytest.raises(ToolTimeoutError, match="exceeded"):
        await pool[0].ainvoke({})


def test_timeout_tool_sync_cuts_off() -> None:
    import time

    @tool
    def slow_sync() -> str:
        """睡过任何预算（同步路径）。"""
        time.sleep(10)
        return "late"

    pool = build_tool_pool({"m": _FakeModule("m", [slow_sync])}, timeout=0.05)
    started = time.perf_counter()
    with pytest.raises(ToolTimeoutError, match="exceeded"):
        pool[0].invoke({})
    # 回归断言：超时必须在预算附近返回，而不是等失控工具跑满全程。
    assert time.perf_counter() - started < 5


def test_wrapped_tool_preserves_schema_and_result() -> None:
    pool = build_tool_pool({"m": _FakeModule("m", [add])}, timeout=5)
    assert pool[0].invoke({"a": 2, "b": 3}) == 5
    assert "a" in pool[0].args_schema.model_fields


def test_handle_tool_error_sanitized() -> None:
    """错误文本进 ToolMessage（模型 + 历史回放可见）：URL 脱敏、长度封顶。"""
    from agent_base.core.tools import handle_tool_error

    leaked = handle_tool_error(ValueError("connect to http://10.0.0.1:8123/internal failed"))
    assert "http://" not in leaked
    assert "<redacted-url>" in leaked

    huge = handle_tool_error(ValueError("x" * 2000))
    assert len(huge) <= 300
