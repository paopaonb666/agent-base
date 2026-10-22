"""记忆子系统的装配入口（``agent_base.memory`` entry point 的实现）。

基座通过 entry point 发现这里，不反向 import；返回的 ``MemoryPlugin``
定义在基座 ``agent_base.core.memory_plugin``（契约在基座，实现在本包）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.tools import BaseTool

from agent_base.core.memory_plugin import MemoryPlugin
from agent_base_memory.service import MemoryService, build_memory_service
from agent_base_memory.store import build_memory_store
from agent_base_memory.tools import build_memory_tools

if TYPE_CHECKING:  # pragma: no cover - import avoided at runtime
    from agent_base.core.config import Settings


async def build(
    *,
    settings: Settings,
    llm: Any | None = None,
    fast_llm: Any | None = None,
) -> MemoryPlugin | None:
    """按 settings 装配记忆服务与工具；未启用返回 None。"""
    if not settings.memory.enabled:
        return None
    store = await build_memory_store(settings)
    if store is None:
        raise RuntimeError(
            "memory store unavailable: unknown checkpointer backend "
            f"{settings.checkpointer.backend!r}"
        )
    service = await build_memory_service(settings, llm, fast_llm=fast_llm, store=store)
    if service is None:  # pragma: no cover - enabled 已在上面判定
        return None
    tools: list[BaseTool] = list(build_memory_tools(service))
    return MemoryPlugin(service=service, tools=tools)


__all__ = ["MemoryService", "build"]
