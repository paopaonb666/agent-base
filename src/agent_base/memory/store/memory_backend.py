"""进程内字典实现的记忆后端：memory 后端（重启即丢）与测试用。"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import replace

from agent_base.memory.store.models import (
    KNOWN_MEMORY_KINDS,
    KNOWN_MEMORY_STATUSES,
    PROFILE_ID_PREFIX,
    DocChunk,
    MemoryBlock,
    MemoryOp,
    MemoryRecord,
    MemoryStoreError,
    MemoryVersion,
    SessionSummary,
    ThreadIndex,
)


class MemoryMemoryStore:
    """进程内字典实现：memory 后端（重启即丢）与测试用。"""

    def __init__(self) -> None:
        self._memories: dict[str, MemoryRecord] = {}
        self._threads: dict[str, ThreadIndex] = {}
        self._blocks: dict[tuple[str, str, str], MemoryBlock] = {}
        self._summaries: dict[tuple[str, str], SessionSummary] = {}
        self._chunks: dict[str, DocChunk] = {}
        # (op, 插入序号) 二元组：同秒并列时按插入顺序稳定排序。
        self._ops: list[tuple[MemoryOp, int]] = []
        self._versions: list[tuple[MemoryVersion, int]] = []
        self._op_counter = 0

    async def upsert_memory(self, record: MemoryRecord) -> None:
        if record.kind not in KNOWN_MEMORY_KINDS:
            raise MemoryStoreError(
                f"unknown memory kind {record.kind!r}; expected one of {KNOWN_MEMORY_KINDS}"
            )
        if record.status not in KNOWN_MEMORY_STATUSES:
            raise MemoryStoreError(
                f"unknown memory status {record.status!r}; expected one of {KNOWN_MEMORY_STATUSES}"
            )
        self._memories[record.memory_id] = record

    async def get_memory(self, memory_id: str) -> MemoryRecord | None:
        return self._memories.get(memory_id)

    async def delete_memory(self, memory_id: str) -> bool:
        return self._memories.pop(memory_id, None) is not None

    async def list_memories(
        self,
        user_id: str,
        *,
        agent_id: str | None = None,
        include_global: bool = True,
        kinds: Sequence[str] | None = None,
        statuses: Sequence[str] = ("active",),
        limit: int = 2000,
        exclude_profile: bool = False,
    ) -> list[MemoryRecord]:
        matched = [
            record
            for record in self._memories.values()
            if record.user_id == user_id
            and (not statuses or record.status in statuses)
            and (kinds is None or record.kind in kinds)
            and (not exclude_profile or not record.memory_id.startswith(PROFILE_ID_PREFIX))
            and (
                agent_id is None
                or record.agent_id == agent_id
                or (include_global and record.agent_id == "*")
            )
        ]
        matched.sort(key=lambda r: r.updated_at, reverse=True)
        return matched[: max(1, limit)]

    async def touch_memories(self, memory_ids: Sequence[str]) -> None:
        now = time.time()
        for mid in memory_ids:
            record = self._memories.get(mid)
            if record is None:
                continue
            self._memories[mid] = replace(
                record,
                last_accessed_at=now,
                access_count=record.access_count + 1,
            )

    async def list_memories_needing_embedding(
        self, dims: int, limit: int = 100
    ) -> list[MemoryRecord]:
        matched = [
            record
            for record in self._memories.values()
            if record.status == "active"
            and not record.memory_id.startswith(PROFILE_ID_PREFIX)
            and (
                record.embedding is None
                or record.embedding_dim is None
                or record.embedding_dim != dims
            )
        ]
        matched.sort(key=lambda r: r.updated_at, reverse=True)
        return matched[: max(1, limit)]

    async def upsert_block(self, block: MemoryBlock) -> None:
        self._blocks[(block.user_id, block.agent_id, block.label)] = block

    async def get_block(self, user_id: str, agent_id: str, label: str) -> MemoryBlock | None:
        return self._blocks.get((user_id, agent_id, label))

    async def list_blocks(self, user_id: str, agent_id: str) -> list[MemoryBlock]:
        blocks = [
            block
            for (uid, aid, _), block in self._blocks.items()
            if uid == user_id and aid == agent_id
        ]
        blocks.sort(key=lambda b: b.label)
        return blocks

    async def delete_block(self, user_id: str, agent_id: str, label: str) -> bool:
        return self._blocks.pop((user_id, agent_id, label), None) is not None

    async def get_summary(self, user_id: str, thread_id: str) -> SessionSummary | None:
        return self._summaries.get((user_id, thread_id))

    async def upsert_summary(self, summary: SessionSummary) -> None:
        self._summaries[(summary.user_id, summary.thread_id)] = summary

    async def put_chunks(self, chunks: Sequence[DocChunk]) -> None:
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk

    async def list_chunks(
        self,
        user_id: str,
        *,
        agent_id: str | None = None,
        file_id: str | None = None,
        limit: int = 2000,
    ) -> list[DocChunk]:
        matched = [
            chunk
            for chunk in self._chunks.values()
            if chunk.user_id == user_id
            and (agent_id is None or chunk.agent_id == agent_id or chunk.agent_id == "*")
            and (file_id is None or chunk.file_id == file_id)
        ]
        matched.sort(key=lambda c: (c.file_id, c.ordinal))
        return matched[: max(1, limit)]

    async def count_chunks_for_file(self, file_id: str) -> int:
        return sum(1 for chunk in self._chunks.values() if chunk.file_id == file_id)

    async def chunk_counts_by_file(self, user_id: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for chunk in self._chunks.values():
            if chunk.user_id == user_id:
                counts[chunk.file_id] = counts.get(chunk.file_id, 0) + 1
        return counts

    async def delete_chunks_for_file(self, file_id: str) -> int:
        stale = [cid for cid, chunk in self._chunks.items() if chunk.file_id == file_id]
        for cid in stale:
            del self._chunks[cid]
        return len(stale)

    async def delete_for_thread(self, thread_id: str) -> None:
        self._chunks = {
            cid: chunk for cid, chunk in self._chunks.items() if chunk.thread_id != thread_id
        }
        self._summaries = {
            key: summary
            for key, summary in self._summaries.items()
            if summary.thread_id != thread_id
        }
        self._threads.pop(thread_id, None)

    async def record_op(self, op: MemoryOp) -> None:
        self._op_counter += 1
        self._ops.append((op, self._op_counter))

    async def list_ops(self, user_id: str | None = None, limit: int = 100) -> list[MemoryOp]:
        pairs = [(op, seq) for op, seq in self._ops if user_id is None or op.user_id == user_id]
        pairs.sort(key=lambda pair: (pair[0].created_at, pair[1]), reverse=True)
        return [op for op, _ in pairs[: max(1, limit)]]

    async def record_version(self, version: MemoryVersion) -> None:
        self._op_counter += 1
        self._versions.append((version, self._op_counter))

    async def list_versions(self, memory_id: str, limit: int = 50) -> list[MemoryVersion]:
        pairs = [pair for pair in self._versions if pair[0].memory_id == memory_id]
        pairs.sort(key=lambda pair: (pair[0].created_at, pair[1]), reverse=True)
        return [version for version, _ in pairs[: max(1, limit)]]

    async def get_version(self, version_id: str) -> MemoryVersion | None:
        for version, _ in self._versions:
            if version.version_id == version_id:
                return version
        return None

    async def list_chunks_needing_embedding(self, dims: int, limit: int = 100) -> list[DocChunk]:
        matched = [
            chunk
            for chunk in self._chunks.values()
            if chunk.embedding is None or chunk.embedding_dim is None or chunk.embedding_dim != dims
        ]
        matched.sort(key=lambda chunk: chunk.created_at, reverse=True)
        return matched[: max(1, limit)]

    # -- 会话索引（thread_index；S1 线程作用域） --------------------------------

    async def upsert_thread_index(self, entry: ThreadIndex) -> None:
        self._threads[entry.thread_id] = entry

    async def get_thread_index(self, thread_id: str) -> ThreadIndex | None:
        return self._threads.get(thread_id)

    async def list_thread_index(
        self, user_id: str, *, module: str | None = None, limit: int = 50
    ) -> list[ThreadIndex]:
        matched = [
            entry
            for entry in self._threads.values()
            if entry.user_id == user_id and (module is None or entry.module == module)
        ]
        matched.sort(key=lambda entry: entry.updated_at, reverse=True)
        return matched[: max(1, limit)]

    async def delete_thread_index(self, thread_id: str) -> None:
        self._threads.pop(thread_id, None)
