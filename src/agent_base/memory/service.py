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

import asyncio
import json
import logging
import time
import uuid
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from agent_base.extensions.metrics import MEMORY_METRICS
from agent_base.memory.context import compose_context
from agent_base.memory.embeddings import EmbeddingClient, NullEmbedding, build_embedding_client
from agent_base.memory.pipeline import MemoryPipeline, render_transcript
from agent_base.memory.retrieval import ScoredChunk, ScoredMemory, recall_memories, score_chunks
from agent_base.memory.store import (
    KNOWN_MEMORY_KINDS,
    KNOWN_MEMORY_STATUSES,
    DocChunk,
    MemoryOp,
    MemoryRecord,
    MemoryStore,
    MemoryStoreError,
    build_memory_store,
    encode_embedding,
    profile_memory_id,
)

if TYPE_CHECKING:  # pragma: no cover - import avoided at runtime
    from agent_base.core.config import Settings

logger = logging.getLogger(__name__)


def new_memory_id() -> str:
    """新生成的记忆 id（uuid hex 截断，与 file_id 同风格）。"""
    return uuid.uuid4().hex[:32]


def _trim_profile_to_chars(profile: dict[str, Any], max_chars: int) -> dict[str, Any]:
    """画像结构级硬截断：从最后一个分节倒序逐条丢弃条目直到序列化
    长度达标；仍超长（存在巨型单条）则截断字符串值。输出永远是合法
    JSON——绝不让画像把注入预算缓慢撑爆。"""
    trimmed: dict[str, Any] = {}
    for key, value in profile.items():
        if isinstance(value, list):
            trimmed[key] = list(value)
        elif isinstance(value, dict):
            trimmed[key] = dict(value)
        else:
            trimmed[key] = value

    def _length() -> int:
        return len(json.dumps(trimmed, ensure_ascii=False))

    while _length() > max_chars:
        last_key = next(
            (
                key
                for key in reversed(list(trimmed))
                if isinstance(trimmed[key], list) and trimmed[key]
            ),
            None,
        )
        if last_key is None:
            break
        trimmed[last_key].pop()
        if not trimmed[last_key]:
            del trimmed[last_key]
    while _length() > max_chars:
        str_key: str | None = next(
            (k for k, v in trimmed.items() if isinstance(v, str) and v),
            None,
        )
        if str_key is None:
            # 只剩数字/空结构的极端兜底：几乎不可达，但绝不抛错。
            return {"truncated": True}
        trimmed[str_key] = trimmed[str_key][: max(1, max_chars // 2)]
    return trimmed


def chunk_text(text: str, *, chunk_chars: int, overlap: int) -> list[str]:
    """按字符窗口把文档文本切块（相邻块带重叠，保持跨块语义连续）。

    刻意用固定窗口而不是段落感知切分：行为可预测、中英文一视同仁，
    边界毛刺由重叠带兜住。空文本返回空列表。
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_chars:
        return [text]
    step = max(1, chunk_chars - overlap)
    return [text[start : start + chunk_chars] for start in range(0, len(text), step)]


class MemoryService:
    """记忆系统门面：一个实例服务整个运行时（三种后端一致）。"""

    def __init__(
        self,
        store: MemoryStore,
        embedder: EmbeddingClient,
        settings: Settings,
        llm: Any | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self._settings = settings
        self._llm = llm
        # 形成管线（M6c）：仅在拿到对话模型时可用；测试与禁用场景下为 None。
        self.pipeline: MemoryPipeline | None = (
            MemoryPipeline(llm, settings, self) if llm is not None else None
        )
        # 每用户的整合锁（锐评 #10）：同一用户并发轮次的后台捕获会互相
        # 看不到对方未写入的记忆，导致重复 ADD——串行化同一用户的捕获。
        self._capture_locks: dict[str, asyncio.Lock] = {}

    async def record_op(
        self,
        *,
        op: str,
        user_id: str = "",
        agent_id: str = "",
        thread_id: str = "",
        detail: dict[str, Any] | None = None,
        status: str = "ok",
        error_text: str = "",
        duration_ms: int = 0,
    ) -> None:
        """写一条操作审计（成功与失败都记录）；审计失败只记日志。"""
        try:
            await self.store.record_op(
                MemoryOp(
                    op_id=uuid.uuid4().hex[:32],
                    op=op,
                    user_id=user_id,
                    agent_id=agent_id,
                    thread_id=thread_id,
                    # default=str：审计 detail 是诊断信息而非契约数据，
                    # 序列化不了的对象降级为字符串表示，绝不让审计失败。
                    detail=detail or {},
                    status=status,
                    error_text=error_text,
                    duration_ms=duration_ms,
                )
            )
        except Exception:
            logger.warning("memory: 审计写入失败", exc_info=True)

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
            min_score=self._settings.memory_recall_min_score,
            weights=self._hybrid_weights(),
        )
        MEMORY_METRICS.observe("search", "ok", time.perf_counter() - started)
        return results

    # -- 知识库（M6e）：文档分块摄取与检索 ------------------------------------
    async def ingest_document(
        self,
        *,
        file_id: str,
        user_id: str,
        agent_id: str,
        text: str,
        thread_id: str = "",
    ) -> int:
        """把已解析的文档文本切块 + 向量化后入知识库；返回分块数。

        失败安全：ingest 是上传路径上的旁路任务，任何异常只记审计与
        日志。分块是用户级知识资产（``thread_id`` 仅作出处标注）——
        删除线程不回收分块，知识独立于会话存活。
        """
        if not text.strip():
            return 0
        started = time.perf_counter()
        try:
            parts = chunk_text(
                text,
                chunk_chars=self._settings.memory_doc_chunk_chars,
                overlap=self._settings.memory_doc_chunk_overlap,
            )
            vectors = await self.embedder.embed(parts)
            chunks = [
                DocChunk(
                    chunk_id=new_memory_id(),
                    file_id=file_id,
                    thread_id=thread_id,
                    user_id=user_id,
                    agent_id=agent_id,
                    ordinal=ordinal,
                    text=part,
                    embedding=encode_embedding(vectors[ordinal]) if vectors is not None else None,
                    embedding_dim=self.embedder.dims if vectors is not None else None,
                )
                for ordinal, part in enumerate(parts)
            ]
            await self.store.put_chunks(chunks)
            duration_ms = int((time.perf_counter() - started) * 1000)
            MEMORY_METRICS.observe("ingest", "ok", time.perf_counter() - started)
            await self.record_op(
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
            await self.record_op(
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
        chunks = await self.store.list_chunks(user_id, agent_id=agent_id)
        if not chunks:
            return []
        vectors = await self.embedder.embed([query])
        query_embedding = vectors[0] if vectors else None
        scored = score_chunks(
            chunks,
            query,
            query_embedding,
            now=time.time(),
            half_life_days=self._settings.memory_time_decay_half_life_days,
            weights=self._hybrid_weights(),
        )
        scored.sort(key=lambda item: item.score, reverse=True)
        min_score = self._settings.memory_recall_min_score
        if min_score > 0:

            def _has_evidence(item: ScoredChunk) -> bool:
                return item.components.get("keyword", 0.0) > 0 or item.components.get(
                    "vector", 0.0
                ) > 0

            scored = [item for item in scored if item.score >= min_score and _has_evidence(item)]
        return scored[: max(1, top_k or self._settings.memory_recall_top_k)]

    def _hybrid_weights(self) -> dict[str, float]:
        """从 settings 读混合权重（锐评 #4：可配置化，默认与 M6b 一致）。"""
        return {
            "vector": self._settings.memory_weight_vector,
            "keyword": self._settings.memory_weight_keyword,
            "recency": self._settings.memory_weight_recency,
            "salience": self._settings.memory_weight_salience,
        }

    # -- 生命周期 -----------------------------------------------------------------
    async def health_probe(self) -> str:
        """对 /health 的探针（锐评 #19）：存储可读 + embedding 可达。

        存储不可读 = error；embedding 端点不可达 = degraded（检索自动
        降级 BM25，服务仍在但能力受损）。embedding 探测带 TTL 缓存，
        不会被 LB 轮询打爆。
        """
        try:
            await self.store.list_ops(limit=1)
        except Exception:
            logger.warning("memory: health probe failed", exc_info=True)
            return "error"
        probe = getattr(self.embedder, "probe", None)
        if probe is not None and await probe() == "error":
            return "degraded"
        return "ok"

    async def backfill_embeddings(self, *, batch: int = 32) -> int:
        """为向量缺失/维度不符的 active 记忆回填 embedding（锐评 #3）。

        更换 embedding 模型/维度后运行（scripts/memory_backfill_embeddings.py）。
        回填保持原 updated_at——这是维护操作，不该刷高召回的时间衰减分。
        embedding 未配置或中途不可用都会安全终止并返回已回填数。
        """
        if self.embedder.dims is None:
            logger.info("memory: 未配置 embedding，回填无事可做")
            return 0
        total = 0
        while True:
            records = await self.store.list_memories_needing_embedding(
                self.embedder.dims, batch
            )
            if not records:
                break
            vectors = await self.embedder.embed([record.content for record in records])
            if vectors is None:
                logger.warning(
                    "memory: embedding 服务不可用，回填中断（已回填 %s 条）", total
                )
                break
            for record, vector in zip(records, vectors, strict=True):
                await self.store.upsert_memory(
                    replace(
                        record,
                        embedding=encode_embedding(vector),
                        embedding_dim=self.embedder.dims,
                    )
                )
                total += 1
            if len(records) < batch:
                break
        logger.info("memory: 向量回填完成，共 %s 条", total)
        return total

    # -- 用户画像（M6c；memories 表里的确定性记录） -------------------------------
    async def get_profile(self, user_id: str) -> dict[str, Any] | None:
        """读取结构化用户画像（JSON dict）；不存在或畸形返回 None。"""
        record = await self.store.get_memory(profile_memory_id(user_id))
        if record is None:
            return None
        try:
            parsed = json.loads(record.content)
        except ValueError:
            return None
        return parsed if isinstance(parsed, dict) else None

    async def save_profile(self, user_id: str, profile: dict[str, Any]) -> None:
        """整包保存用户画像（管线合并后的完整 JSON）。

        画像不需要向量——它由上下文组装整体注入而不是按相似度召回，
        跳过 embedding 省一次 API 调用；检索侧已按 id 前缀排除画像。
        保存前做结构级硬截断（锐评漏项）：prompt 里的"600 字以内"只是
        软约束，这里保证序列化长度不超过 MEMORY_PROFILE_MAX_CHARS，
        防止画像随对话缓慢膨胀注入预算。
        """
        memory_id = profile_memory_id(user_id)
        existing = await self.store.get_memory(memory_id)
        now = time.time()
        profile = _trim_profile_to_chars(profile, self._settings.memory_profile_max_chars)
        await self.store.upsert_memory(
            MemoryRecord(
                memory_id=memory_id,
                user_id=user_id,
                agent_id="*",
                kind="semantic",
                content=json.dumps(profile, ensure_ascii=False),
                tags=["profile"],
                salience=1.0,
                source_refs=["profile"],
                created_at=existing.created_at if existing else now,
                updated_at=now,
                last_accessed_at=existing.last_accessed_at if existing else None,
                access_count=existing.access_count if existing else 0,
            )
        )

    # -- 形成管线编排（M6c） ------------------------------------------------------
    async def capture_turn(
        self, *, user_id: str, agent_id: str, thread_id: str, messages: list[Any]
    ) -> dict[str, Any] | None:
        """一轮对话结束后的记忆形成（后台调用；绝不抛异常）。

        ``messages`` 是该线程的全部持久化消息（来自 checkpointer 快照）：
        转写渲染、轮次计数与触发阈值都在这里统一处理。
        """
        if self.pipeline is None:
            return None
        human_count = sum(1 for m in messages if getattr(m, "type", "") == "human")
        transcript = render_transcript(
            messages[-24:], self._settings.memory_extraction_max_input_chars
        )
        if not transcript.strip():
            return None
        lock = self._capture_locks.setdefault(user_id, asyncio.Lock())
        async with lock:
            try:
                return await self.pipeline.capture_turn(
                    user_id=user_id,
                    agent_id=agent_id,
                    thread_id=thread_id,
                    transcript=transcript,
                    human_count=human_count,
                )
            except Exception:
                # 管线内部已逐步失败安全；这里兜底防任何漏网异常干扰调用方。
                logger.exception("memory: capture_turn 意外失败")
                return None

    async def compose_context(
        self, *, user_id: str, agent_id: str, thread_id: str, query: str
    ) -> str | None:
        """组装本轮的注入块（M6d）：画像 + 记忆块 + 会话摘要 + 相关记忆。

        任何一步失败都降级为"缺那一节"；整体无内容时返回 None。
        """
        try:
            global_blocks = await self.store.list_blocks(user_id, "*")
            agent_blocks = await self.store.list_blocks(user_id, agent_id)
            blocks = [*global_blocks, *agent_blocks]
            profile = (
                await self.get_profile(user_id) if self._settings.memory_profile_enabled else None
            )
            summary_record = await self.store.get_summary(user_id, thread_id)
            recalled = await self.search(user_id=user_id, agent_id=agent_id, query=query)
            return compose_context(
                profile=profile,
                blocks=blocks,
                summary=summary_record.summary if summary_record else None,
                recalled=recalled,
                max_chars=self._settings.memory_context_max_chars,
            )
        except Exception:
            logger.warning("memory: 注入块组装失败（本轮不注入）", exc_info=True)
            return None

    async def purge_thread(self, thread_id: str) -> None:
        """线程删除的级联清理（滚动摘要 + 知识库分块；跨会话记忆保留）。"""
        await self.store.delete_for_thread(thread_id)

    async def aclose(self) -> None:
        closer = getattr(self.store, "aclose", None)
        if closer is not None:
            await closer()
        embedder_close = getattr(self.embedder, "aclose", None)
        if embedder_close is not None:
            await embedder_close()


async def build_memory_service(settings: Settings, llm: Any | None = None) -> MemoryService | None:
    """按 settings 装配记忆服务；未启用或后端不可用时返回 None。

    ``llm`` 是运行时的对话模型（形成管线复用它）；测试可以不传——
    检索与手工管理照常工作，仅形成管线缺席。
    """
    if not settings.memory_enabled:
        logger.info("memory: 记忆系统已禁用（MEMORY_ENABLED=false）")
        return None
    store = await build_memory_store(settings)
    if store is None:
        return None
    embedder = build_embedding_client(settings)
    return MemoryService(store=store, embedder=embedder, settings=settings, llm=llm)


__all__ = [
    "MemoryService",
    "build_memory_service",
    "chunk_text",
    "new_memory_id",
]
