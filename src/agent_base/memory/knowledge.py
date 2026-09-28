"""文档知识库子域（M6e，P0-2 拆分自 MemoryService）：摄取 / 检索 / 撤销。

知识分块是**用户级知识资产**：上传时按段落感知切块（空行对齐，表格/
列表不从中间劈开）+ 向量化入 ``doc_chunks``，独立于会话存活（删除线程
不回收分块）；检索是向量 + BM25 + 时间衰减的混合打分（retrieval.py）。

失败安全：ingest 是上传路径上的旁路任务，任何异常只记审计与日志。
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from agent_base.extensions.metrics import MEMORY_METRICS
from agent_base.memory.audit import MemoryAudit
from agent_base.memory.embeddings import EmbeddingClient, NullEmbedding
from agent_base.memory.retrieval import ScoredChunk, score_chunks, weights_from_settings
from agent_base.memory.store import DocChunk, MemoryStore, encode_embedding
from agent_base.memory.store.models import new_memory_id
from agent_base.memory.textkit import chunk_text

if TYPE_CHECKING:  # pragma: no cover - import avoided at runtime
    from agent_base.core.config import MemorySettings

logger = logging.getLogger(__name__)


class KnowledgeService:
    """文档知识库门面：依赖收敛到 store + embedder + MemorySettings 节 + 审计。"""

    def __init__(
        self,
        store: MemoryStore,
        embedder: EmbeddingClient,
        settings: MemorySettings,
        audit: MemoryAudit,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._settings = settings
        self._audit = audit

    async def ingest_document(
        self,
        *,
        file_id: str,
        user_id: str,
        agent_id: str,
        text: str,
        thread_id: str = "",
    ) -> int:
        """把已解析的文档文本切块 + 向量化后入知识库；返回分块数。"""
        if not text.strip():
            return 0
        started = time.perf_counter()
        try:
            parts = chunk_text(
                text,
                chunk_chars=self._settings.doc_chunk_chars,
                overlap=self._settings.doc_chunk_overlap,
            )
            # 幂等：同一 file_id 重复摄取（重试/重解析场景）先清旧分块，
            # 防止 chunk_id 随机生成导致的重复堆积。
            await self._store.delete_chunks_for_file(file_id)
            vectors = await self._embedder.embed([span.text for span in parts])
            chunks = [
                DocChunk(
                    chunk_id=new_memory_id(),
                    file_id=file_id,
                    thread_id=thread_id,
                    user_id=user_id,
                    agent_id=agent_id,
                    ordinal=ordinal,
                    text=span.text,
                    offsets=span.segments,
                    embedding=encode_embedding(vectors[ordinal]) if vectors is not None else None,
                    embedding_dim=self._embedder.dims if vectors is not None else None,
                )
                for ordinal, span in enumerate(parts)
            ]
            await self._store.put_chunks(chunks)
            duration_ms = int((time.perf_counter() - started) * 1000)
            MEMORY_METRICS.observe("ingest", "ok", time.perf_counter() - started)
            await self._audit.record(
                op="ingest",
                user_id=user_id,
                agent_id=agent_id,
                thread_id=thread_id,
                duration_ms=duration_ms,
                detail={"file_id": file_id, "chunks": len(chunks)},
            )
            return len(chunks)
        except Exception as exc:
            logger.warning("memory: 文档摄取失败（不影响上传）：%s", exc)
            MEMORY_METRICS.observe("ingest", "error", time.perf_counter() - started)
            await self._audit.record(
                op="ingest",
                user_id=user_id,
                agent_id=agent_id,
                thread_id=thread_id,
                status="error",
                error_text=f"{type(exc).__name__}: {exc}"[:300],
                duration_ms=int((time.perf_counter() - started) * 1000),
                detail={"file_id": file_id},
            )
            return 0

    async def search_knowledge(
        self, *, user_id: str, agent_id: str | None, query: str, top_k: int | None = None
    ) -> list[ScoredChunk]:
        """知识库混合检索（向量 + BM25 + 时间衰减）。"""
        chunks = await self._store.list_chunks(user_id, agent_id=agent_id)
        if len(chunks) >= 2000:
            # 容量契约（H2）：同召回候选——静默截断会让"检索质量莫名
            # 下降"，这里显式告警。
            logger.warning(
                "memory: 知识库候选分块达到上限 2000（user=%r）——更早的文件不参与本轮检索",
                user_id,
            )
        if not chunks:
            return []
        vectors = await self._embedder.embed([query])
        query_embedding = vectors[0] if vectors else None
        scored = score_chunks(
            chunks,
            query,
            query_embedding,
            now=time.time(),
            half_life_days=self._settings.time_decay_half_life_days,
            weights=weights_from_settings(self._settings),
        )
        scored.sort(key=lambda item: item.score, reverse=True)
        min_score = self._settings.recall_min_score
        if isinstance(self._embedder, NullEmbedding) and min_score > 0:
            # 降级关键词路径与 search 同规则：混合分缺向量主力分量，硬门槛
            # 会把关键词命中整体清零（天花板 ~0.45 < 0.5，Tier 3.1 验收
            # 发现）——放宽一半，宁可多给弱序结果也不静默丢命中。
            min_score = min_score / 2
        if min_score > 0:

            def _has_evidence(item: ScoredChunk) -> bool:
                return (
                    item.components.get("keyword", 0.0) > 0
                    or item.components.get("vector", 0.0) > 0
                )

            scored = [item for item in scored if item.score >= min_score and _has_evidence(item)]
        return scored[: max(1, top_k or self._settings.recall_top_k)]

    async def revoke_document(self, *, file_id: str, user_id: str) -> int:
        """撤销一份文档的知识库分块（用户级知识资产的显式收回路径，
        修复"传错文件无法撤回"的边界：DELETE /v1/agents/{module}/files/{id}）。
        """
        started = time.perf_counter()
        deleted = await self._store.delete_chunks_for_file(file_id)
        MEMORY_METRICS.observe("revoke", "ok", time.perf_counter() - started)
        await self._audit.record(
            op="revoke",
            user_id=user_id,
            detail={"file_id": file_id, "chunks": deleted},
        )
        return deleted
