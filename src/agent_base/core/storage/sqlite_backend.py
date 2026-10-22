"""sqlite 基座表后端：与对话状态同一文件，aiosqlite 持久连接。"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from agent_base.core.storage.ddl import _SQLITE_DDLS, _SQLITE_INDEXES
from agent_base.core.storage.sql_base import _CoreSqlStoreBase

logger = logging.getLogger(__name__)


class CoreSqliteStore(_CoreSqlStoreBase):
    """sqlite 基座表：与对话状态同一文件，aiosqlite 持久连接。"""

    def __init__(self, connection: Any) -> None:
        self._conn = connection

    @classmethod
    async def create(cls, path: str) -> CoreSqliteStore:
        import aiosqlite

        conn = await aiosqlite.connect(path)
        for ddl in _SQLITE_DDLS:
            await conn.execute(ddl)
        for index in _SQLITE_INDEXES:
            await conn.execute(index)
        await conn.commit()
        return cls(conn)

    async def _execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        cursor = await self._conn.execute(sql, tuple(params))
        await self._conn.commit()
        return cursor.rowcount if cursor.rowcount and cursor.rowcount > 0 else 0

    async def _fetch_all(self, sql: str, params: Sequence[Any] = ()) -> list[tuple[Any, ...]]:
        rows: list[tuple[Any, ...]] = await self._conn.execute_fetchall(sql, tuple(params))
        return rows

    def _to_sql_ts(self, seconds: float) -> Any:
        # 微秒精度：线程列表的同秒并列依赖它稳定排序。
        return datetime.fromtimestamp(seconds).isoformat(sep=" ", timespec="microseconds")

    def _from_sql_ts(self, value: Any) -> float:
        try:
            return datetime.fromisoformat(str(value)).timestamp()
        except (TypeError, ValueError):
            logger.warning("sqlite core store: 无法解析时间戳 %r，按当前时间兜底", value)
            return time.time()

    async def aclose(self) -> None:
        await self._conn.close()
