"""记忆服务门面（M6b 起）：入口层与模块只跟它说话。

职责是把"存储 + 向量 + 检索"组装成一个语义清晰的对象，让 server 与
模块代码不直接摆弄 store 的 SQL 细节：

- **写路径**（``add_memory`` / ``update_memory``）——内容落库前尽力
  向量化（embedding 不可用就存纯文本，检索自动走关键词路径）；
- **读路径**（``search`` / ``recall``）——混合检索（retrieval.py）；
- **生命周期**（``aclose``）——释放 sqlite 连接与 httpx 客户端。

M6c 将在此基础上追加形成管线（``capture_turn``：抽取/整合/画像/摘要）；
M6d 追加上下文组装（``compose_context``）。
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import replace
from typing import TYPE_CHECKING

from agent_base.extensions.metrics import MEMORY_METRICS
from agent_base.memory.embeddings import EmbeddingClient, NullEmbedding, build_embedding_client
from agent_base.memory.retrieval import ScoredMemory, recall_memories
from agent_base.memory.store import (
    KNOWN_MEMORY_KINDS,
    KNOWN_MEMORY_STATUSES,
    MemoryRecord,
    MemoryStore,
    MemoryStoreError,
    build_memory_store,
    encode_embedding,
)

if TYPE_CHECKING:  # pragma: no cover - import avoided at runtime
    from agent_base.core.config import Settings

logger = logging.getLogger(__name__)


def new_memory_id() -> str:
    """新生成的记忆 id（uuid hex 截断，与 file_id 同风格）。"""
    return uuid.uuid4().hex[:32]


class MemoryService:
    """记忆系统门面：一个实例服务整个运行时（三种后端一致）。"""

    def __init__(self, store: MemoryStore, embedder: EmbeddingClient, settings: Settings) -> None:
        self.store = store
        self.embedder = embedder
        self._settings = settings

    # -- 写路径 -----------------------------------------------------------------
    async def _embed_one(self, text: str) -> list[float] | None:
        vectors = await self.embedder.embed([text])
        if not vectors:
            return None
        return vectors[0]

    async def add_memory(
        self,
        *,
        user_id: str,
        agent_id: str,
        content: str,
        kind: str = "semantic",
        tags: list[str] | None = None,
        salience: float = 0.5,
        source_thread_id: str = "",
        source_refs: list[str] | None = None,
    ) -> MemoryRecord:
        """写入一条记忆；向量尽力而为（不可用就存纯文本）。"""
        if kind not in KNOWN_MEMORY_KINDS:
            raise MemoryStoreError(
                f"unknown memory kind {kind!r}; expected one of {KNOWN_MEMORY_KINDS}"
            )
        started = time.perf_counter()
        now = time.time()
        embedding = await self._embed_one(content)
        record = MemoryRecord(
            memory_id=new_memory_id(),
            user_id=user_id,
            agent_id=agent_id,
            kind=kind,
            content=content,
            tags=list(tags or []),
            embedding=encode_embedding(embedding) if embedding is not None else None,
            embedding_dim=self.embedder.dims if embedding is not None else None,
            salience=min(max(salience, 0.0), 1.0),
            source_thread_id=source_thread_id,
            source_refs=list(source_refs or []),
            created_at=now,
            updated_at=now,
        )
        await self.store.upsert_memory(record)
        MEMORY_METRICS.observe("add", "ok", time.perf_counter() - started)
        return record

    async def update_memory(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        tags: list[str] | None = None,
        salience: float | None = None,
        status: str | None = None,
    ) -> MemoryRecord | None:
        """更新一条记忆；内容变化时重新向量化。"""
        existing = await self.store.get_memory(memory_id)
        if existing is None:
            return None
        if status is not None and status not in KNOWN_MEMORY_STATUSES:
            raise MemoryStoreError(
                f"unknown memory status {status!r}; expected one of {KNOWN_MEMORY_STATUSES}"
            )
        started = time.perf_counter()
        new_content = content if content is not None else existing.content
        reembed = content is not None and content != existing.content
        embedding: list[float] | None = await self._embed_one(new_content) if reembed else None
        updated = replace(
            existing,
            content=new_content,
            tags=list(tags) if tags is not None else existing.tags,
            salience=min(max(salience, 0.0), 1.0) if salience is not None else existing.salience,
            status=status if status is not None else existing.status,
            embedding=encode_embedding(embedding) if embedding is not None else existing.embedding,
            embedding_dim=self.embedder.dims if embedding is not None else existing.embedding_dim,
            updated_at=time.time(),
        )
        await self.store.upsert_memory(updated)
        MEMORY_METRICS.observe("update", "ok", time.perf_counter() - started)
        return updated

    async def delete_memory(self, memory_id: str) -> bool:
        deleted = await self.store.delete_memory(memory_id)
        MEMORY_METRICS.observe("delete", "ok", 0.0)
        return deleted

    # -- 读路径 -----------------------------------------------------------------
    async def get_memory(self, memory_id: str) -> MemoryRecord | None:
        return await self.store.get_memory(memory_id)

    async def list_memories(
        self,
        user_id: str,
        *,
        agent_id: str | None = None,
        include_global: bool = True,
        kinds: list[str] | None = None,
        limit: int = 50,
    ) -> list[MemoryRecord]:
        return await self.store.list_memories(
            user_id,
            agent_id=agent_id,
            include_global=include_global,
            kinds=kinds,
            limit=limit,
        )

    async def search(
        self,
        *,
        user_id: str,
        agent_id: str | None,
        query: str,
        top_k: int | None = None,
    ) -> list[ScoredMemory]:
        """混合检索（注入路径与搜索端点共用）。"""
        started = time.perf_counter()
        results = await recall_memories(
            self.store,
            user_id=user_id,
            agent_id=agent_id,
            query=query,
            embedder=self.embedder if not isinstance(self.embedder, NullEmbedding) else None,
            top_k=top_k or self._settings.memory_recall_top_k,
            episodic_ttl_days=self._settings.memory_episodic_ttl_days,
            half_life_days=self._settings.memory_time_decay_half_life_days,
        )
        MEMORY_METRICS.observe("search", "ok", time.perf_counter() - started)
        return results

    # -- 生命周期 -----------------------------------------------------------------
    async def health_probe(self) -> str:
        """对 /health 的轻量探针：能读审计表即视为存储可用。"""
        try:
            await self.store.list_ops(limit=1)
        except Exception:
            logger.warning("memory: health probe failed", exc_info=True)
            return "error"
        return "ok"

    async def aclose(self) -> None:
        closer = getattr(self.store, "aclose", None)
        if closer is not None:
            await closer()
        embedder_close = getattr(self.embedder, "aclose", None)
        if embedder_close is not None:
            await embedder_close()


async def build_memory_service(settings: Settings) -> MemoryService | None:
    """按 settings 装配记忆服务；未启用或后端不可用时返回 None。"""
    if not settings.memory_enabled:
        logger.info("memory: 记忆系统已禁用（MEMORY_ENABLED=false）")
        return None
    store = await build_memory_store(settings)
    if store is None:
        return None
    embedder = build_embedding_client(settings)
    return MemoryService(store=store, embedder=embedder, settings=settings)


__all__ = [
    "MemoryService",
    "build_memory_service",
    "new_memory_id",
]
