"""记忆子系统的装配入口（``agent_base.memory`` entry point 的实现）。

基座通过 entry point 发现这里，不反向 import；返回的 ``MemoryPlugin``
定义在基座 ``agent_base.core.memory_plugin``（契约在基座，实现在本包）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.tools import BaseTool

from agent_base.core.memory_plugin import MemoryPlugin
from agent_base_memory.embeddings import EmbeddingClient, build_embedding_client
from agent_base_memory.orchestrator import RedisCaptureLocks, RedisSnapshotCache
from agent_base_memory.service import MemoryService, build_memory_service
from agent_base_memory.store import build_memory_store
from agent_base_memory.tools import build_memory_tools

if TYPE_CHECKING:  # pragma: no cover - import avoided at runtime
    from agent_base.core.config import Settings
    from agent_base.extensions.redissvc import RedisService


async def build(
    *,
    settings: Settings,
    llm: Any | None = None,
    fast_llm: Any | None = None,
    redis: RedisService | None = None,
) -> MemoryPlugin | None:
    """按 settings 装配记忆服务与工具；未启用返回 None。

    ``redis``（ADR-012，0.2 契约可选形参）：在场时捕获锁与注入快照缓存
    外置到 Redis（多 worker 口径——同一用户的捕获跨进程串行、冻结注入段
    跨进程逐字节一致）；缺席时全部进程内实现，行为与 0.1 一致。
    """
    if not settings.memory.enabled:
        return None
    store = await build_memory_store(settings)
    if store is None:
        raise RuntimeError(
            "memory store unavailable: unknown checkpointer backend "
            f"{settings.checkpointer.backend!r}"
        )
    locks = None
    snapshots = None
    if redis is not None:
        locks = RedisCaptureLocks(
            redis,
            ttl_seconds=settings.redis.lock_ttl_seconds,
            acquire_timeout_seconds=settings.redis.lock_acquire_timeout_seconds,
        )
        snapshots = RedisSnapshotCache(redis, ttl_seconds=settings.redis.snapshot_ttl_seconds)
    service = await build_memory_service(
        settings, llm, fast_llm=fast_llm, store=store, locks=locks, snapshots=snapshots
    )
    if service is None:  # pragma: no cover - enabled 已在上面判定
        return None
    tools: list[BaseTool] = list(build_memory_tools(service))
    return MemoryPlugin(service=service, tools=tools)


def build_embedder(settings: Settings) -> EmbeddingClient:
    """嵌入客户端入口（``agent_base.memory`` 组的 ``embedder`` 名）。

    与 ``MEMORY_ENABLED`` 无关——工具向量路由（ADR-010）与工具 few-shot
    （ADR-011）经它共享嵌入提供方；内部自带降级（未配 key → Null）。
    """
    return build_embedding_client(settings)


__all__ = ["EmbeddingClient", "MemoryService", "build", "build_embedder"]
