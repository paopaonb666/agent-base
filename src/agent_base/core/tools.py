"""The shared tool pool (Stage 3).

Every module contributes tools via ``AgentModule.get_tools()``; the pool is
assembled once per runtime and handed back to modules through
``ModuleContext.tools``. Modules bind the pool to their LLM and execute it
through LangGraph's ``ToolNode``, which normalizes tool failures into
``ToolMessage`` feedback (the model sees the error and can recover — a tool
exception never crashes the graph).

Each tool is wrapped with a wall-clock timeout (``TOOL_TIMEOUT_SECONDS``).
In the async path (the one the server and CLI use) the timeout cancels the
await cleanly; in the sync path the underlying call keeps running in its
worker thread but the result is discarded past the deadline (bounded leak,
documented trade-off — Python threads cannot be killed).
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any

from langchain_core.tools import BaseTool

from agent_base.core.contracts import AgentModule

DEFAULT_TOOL_TIMEOUT_SECONDS = 30.0


class ToolTimeoutError(TimeoutError):
    """A tool exceeded its wall-clock budget and was cut off."""


class ToolPoolError(ValueError):
    """Raised when the pool cannot be assembled (duplicate tool names)."""


def handle_tool_error(exc: Exception) -> str:
    """Normalize any tool failure into model-readable feedback.

    LangGraph 1.2's default ``handle_tool_errors`` only converts argument
    errors and re-raises everything else; the base's contract is stronger —
    a broken tool must never crash the conversation, so every exception is
    turned into a ``ToolMessage`` the model can react to (Stage 3).
    """
    return f"tool execution failed: {exc!r}"


class _TimeoutTool(BaseTool):
    """Wrap ``inner`` so both run modes enforce the same wall-clock budget."""

    inner: BaseTool
    timeout: float

    def _run(self, **kwargs: Any) -> Any:
        # Executor + future.result: the deadline is honored on our side even
        # though the worker thread itself cannot be interrupted.
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.inner.invoke, kwargs)
            try:
                return future.result(timeout=self.timeout)
            except FutureTimeoutError as exc:
                raise ToolTimeoutError(
                    f"tool {self.name!r} exceeded {self.timeout}s timeout"
                ) from exc

    async def _arun(self, **kwargs: Any) -> Any:
        try:
            return await asyncio.wait_for(self.inner.ainvoke(kwargs), timeout=self.timeout)
        except TimeoutError as exc:  # asyncio.TimeoutError is builtin TimeoutError (3.11+)
            raise ToolTimeoutError(f"tool {self.name!r} exceeded {self.timeout}s timeout") from exc


def build_tool_pool(
    modules: dict[str, AgentModule],
    *,
    timeout: float = DEFAULT_TOOL_TIMEOUT_SECONDS,
) -> list[BaseTool]:
    """Collect every module's tools into one timeout-wrapped, conflict-free pool.

    Order follows module assembly order (``AGENT_MODULES``). A duplicate tool
    name aborts startup — two tools with the same name would make tool-call
    routing ambiguous, so this is a fail-fast configuration error.
    """
    pool: list[BaseTool] = []
    seen: set[str] = set()
    for module in modules.values():
        for tool in module.get_tools():
            if tool.name in seen:
                raise ToolPoolError(
                    f"duplicate tool name {tool.name!r} contributed by module "
                    f"{module.name!r}; tool names must be unique across modules"
                )
            seen.add(tool.name)
            pool.append(
                _TimeoutTool(
                    name=tool.name,
                    description=tool.description,
                    args_schema=tool.args_schema,
                    inner=tool,
                    timeout=timeout,
                )
            )
    return pool
