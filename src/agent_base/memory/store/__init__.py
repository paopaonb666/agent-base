"""记忆存储层包（P0-3 拆分自单文件 store.py）。

按关注点分模块：数据模型（``models``）、存储协议（``protocol``，按聚合
根拆小接口）、DDL（``ddl``）、SQL 共享层（``sql_base``）与三个后端实现
（memory/sqlite/mysql）。对外 API 与拆分前完全一致——全部名字从这里
re-export。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from agent_base.memory.store.memory_backend import MemoryMemoryStore
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
    decode_embedding,
    encode_embedding,
    profile_memory_id,
)
from agent_base.memory.store.mysql_backend import MysqlMemoryStore
from agent_base.memory.store.protocol import (
    AuditRepo,
    BlockRepo,
    KnowledgeRepo,
    MemoryRepo,
    MemoryStore,
    SummaryRepo,
    ThreadIndexRepo,
    VersionRepo,
)
from agent_base.memory.store.sqlite_backend import SqliteMemoryStore

if TYPE_CHECKING:  # pragma: no cover - import avoided at runtime
    from agent_base.core.config import Settings

logger = logging.getLogger(__name__)


async def build_memory_store(settings: Settings) -> MemoryStore | None:
    """按 settings 装配记忆存储（跟随 checkpointer 后端）。"""
    backend = settings.checkpointer.backend
    if backend == "memory":
        return MemoryMemoryStore()
    if backend == "sqlite":
        return await SqliteMemoryStore.create(settings.checkpointer.sqlite_path)
    if backend == "mysql":
        return MysqlMemoryStore(settings)
    logger.warning("memory: unknown checkpointer backend %r; memory store disabled", backend)
    return None


__all__ = [
    "KNOWN_MEMORY_KINDS",
    "KNOWN_MEMORY_STATUSES",
    "PROFILE_ID_PREFIX",
    "AuditRepo",
    "BlockRepo",
    "DocChunk",
    "KnowledgeRepo",
    "MemoryBlock",
    "MemoryMemoryStore",
    "MemoryOp",
    "MemoryRecord",
    "MemoryRepo",
    "MemoryStore",
    "MemoryStoreError",
    "MemoryVersion",
    "MysqlMemoryStore",
    "SessionSummary",
    "SqliteMemoryStore",
    "SummaryRepo",
    "ThreadIndex",
    "ThreadIndexRepo",
    "VersionRepo",
    "build_memory_store",
    "decode_embedding",
    "encode_embedding",
    "profile_memory_id",
]
