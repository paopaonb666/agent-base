"""记忆存储协议（P0-3/P2-3 拆分）：按聚合根拆分的小接口 + 组合接口。

拆分动机：过去的 ``MemoryStore`` 是一个 ~30 方法的大接口——只想用知识库
检索的调用方被迫看到记忆版本史，新后端要一次性实现全部方法，测试桩随
之膨胀。现在每个聚合根一个 ``*Repo`` 协议；``MemoryStore`` 是它们的组合
接口（与拆分前的方法集完全一致），``MemoryService`` 与入口层依赖的仍是
``MemoryStore``。

实现必须保证并发安全。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from agent_base.memory.store.models import (
    DocChunk,
    MemoryBlock,
    MemoryOp,
    MemoryRecord,
    MemoryVersion,
    SessionSummary,
    ThreadIndex,
)


@runtime_checkable
class MemoryRepo(Protocol):
    """跨会话长期记忆（memories 表）。"""

    async def upsert_memory(self, record: MemoryRecord) -> None: ...

    async def get_memory(self, memory_id: str) -> MemoryRecord | None: ...

    async def delete_memory(self, memory_id: str) -> bool: ...

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
    ) -> list[MemoryRecord]: ...

    async def touch_memories(self, memory_ids: Sequence[str]) -> None: ...

    async def list_memories_needing_embedding(
        self, dims: int, limit: int = 100
    ) -> list[MemoryRecord]: ...


@runtime_checkable
class BlockRepo(Protocol):
    """常驻上下文的记忆块（memory_blocks 表，Letta 式）。"""

    async def upsert_block(self, block: MemoryBlock) -> None: ...

    async def get_block(self, user_id: str, agent_id: str, label: str) -> MemoryBlock | None: ...

    async def list_blocks(self, user_id: str, agent_id: str) -> list[MemoryBlock]: ...

    async def delete_block(self, user_id: str, agent_id: str, label: str) -> bool: ...


@runtime_checkable
class SummaryRepo(Protocol):
    """线程滚动摘要（session_summaries 表）。"""

    async def get_summary(self, user_id: str, thread_id: str) -> SessionSummary | None: ...

    async def upsert_summary(self, summary: SessionSummary) -> None: ...


@runtime_checkable
class KnowledgeRepo(Protocol):
    """文档知识库分块（doc_chunks 表）。"""

    async def put_chunks(self, chunks: Sequence[DocChunk]) -> None: ...

    async def list_chunks(
        self,
        user_id: str,
        *,
        agent_id: str | None = None,
        file_id: str | None = None,
        limit: int = 2000,
    ) -> list[DocChunk]: ...

    async def count_chunks_for_file(self, file_id: str) -> int: ...

    async def chunk_counts_by_file(self, user_id: str) -> dict[str, int]: ...

    async def delete_chunks_for_file(self, file_id: str) -> int: ...

    async def list_chunks_needing_embedding(
        self, dims: int, limit: int = 100
    ) -> list[DocChunk]: ...


@runtime_checkable
class AuditRepo(Protocol):
    """记忆系统操作审计（memory_ops 表）。"""

    async def record_op(self, op: MemoryOp) -> None: ...

    async def list_ops(self, user_id: str | None = None, limit: int = 100) -> list[MemoryOp]: ...


@runtime_checkable
class VersionRepo(Protocol):
    """记忆内容版本史（memory_versions 表，锐评 #7）。"""

    async def record_version(self, version: MemoryVersion) -> None: ...

    async def list_versions(self, memory_id: str, limit: int = 50) -> list[MemoryVersion]: ...

    async def get_version(self, version_id: str) -> MemoryVersion | None: ...


@runtime_checkable
class ThreadIndexRepo(Protocol):
    """会话索引（thread_index 表，S1 线程作用域的数据面）。

    线程元数据自管：invoke 时 upsert（属主/模块/标题/更新时间），线程
    端点的列表与属主校验改查该表——不再全表扫描 checkpointer，也不再
    向任意用户返回全库线程列表。
    """

    async def upsert_thread_index(self, entry: ThreadIndex) -> None: ...

    async def get_thread_index(self, thread_id: str) -> ThreadIndex | None: ...

    async def list_thread_index(
        self, user_id: str, *, module: str | None = None, limit: int = 50
    ) -> list[ThreadIndex]: ...

    async def delete_thread_index(self, thread_id: str) -> None: ...


class MemoryStore(
    MemoryRepo,
    BlockRepo,
    SummaryRepo,
    KnowledgeRepo,
    AuditRepo,
    VersionRepo,
    ThreadIndexRepo,
    Protocol,
):
    """完整存储接口：各聚合根 Repo 的组合（方法集与拆分前一致）。

    另含线程级联清理（``delete_for_thread``）。
    """

    async def delete_for_thread(self, thread_id: str) -> None: ...
