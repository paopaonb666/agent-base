"""记忆子系统的失败安全写入助手（审计与版本史）。

两条约定（M6b 起成形）：

- 审计与版本史是**旁路数据面**——写失败只记日志，绝不影响主写入
  路径（用户的记忆操作不能因为审计表写不进去而失败）；
- ``detail`` 是诊断信息而非契约数据，序列化不了的对象降级为字符串
  表示（``default=str`` 的语义在调用侧的 json 序列化里）。
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from agent_base.memory.store import MemoryOp, MemoryStore, MemoryVersion

logger = logging.getLogger(__name__)


class MemoryAudit:
    """操作审计写入器（成功与失败都记录；审计失败只记日志）。"""

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    async def record(
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
        try:
            await self._store.record_op(
                MemoryOp(
                    op_id=uuid.uuid4().hex[:32],
                    op=op,
                    user_id=user_id,
                    agent_id=agent_id,
                    thread_id=thread_id,
                    detail=detail or {},
                    status=status,
                    error_text=error_text,
                    duration_ms=duration_ms,
                )
            )
        except Exception:
            logger.warning("memory: 审计写入失败", exc_info=True)


async def record_version_safe(store: MemoryStore, version: MemoryVersion) -> None:
    """版本史写入失败绝不影响主写入路径（与审计同语义）。"""
    try:
        await store.record_version(version)
    except Exception:
        logger.warning("memory: 版本史写入失败", exc_info=True)
