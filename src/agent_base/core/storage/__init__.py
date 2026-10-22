"""基座自有存储（线程索引、计划快照）。

这两张表在 ``MEMORY_ENABLED=false`` 时仍然要工作：线程列表与属主校验、
计划检查端点都依赖它们。0.3.0 之前它们寄居在记忆子系统的 ``MemoryStore``
里，导致基座无法在缺少记忆包时独立运行；现在由本模块拥有，记忆包只负责
自己的六张表。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from agent_base.core.storage.memory_backend import CoreMemoryStore
from agent_base.core.storage.models import ThreadIndex
from agent_base.core.storage.mysql_backend import CoreMysqlStore
from agent_base.core.storage.protocol import CoreStore, PlanRepo, ThreadIndexRepo
from agent_base.core.storage.sqlite_backend import CoreSqliteStore

if TYPE_CHECKING:  # pragma: no cover - import avoided at runtime
    from agent_base.core.config import Settings

logger = logging.getLogger(__name__)


async def build_core_store(settings: Settings) -> CoreStore | None:
    """按 settings 装配基座存储（跟随 checkpointer 后端）。"""
    backend = settings.checkpointer.backend
    if backend == "memory":
        return CoreMemoryStore()
    if backend == "sqlite":
        return await CoreSqliteStore.create(settings.checkpointer.sqlite_path)
    if backend == "mysql":
        return CoreMysqlStore(settings)
    logger.warning("core store: unknown checkpointer backend %r; storage disabled", backend)
    return None


__all__ = [
    "CoreMemoryStore",
    "CoreMysqlStore",
    "CoreSqliteStore",
    "CoreStore",
    "PlanRepo",
    "ThreadIndex",
    "ThreadIndexRepo",
    "build_core_store",
]
