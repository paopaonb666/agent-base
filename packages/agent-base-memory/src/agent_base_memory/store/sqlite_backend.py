"""sqlite 记忆表后端：与对话状态同一文件，aiosqlite 持久连接。"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from agent_base.memory.store.ddl import _SQLITE_DDLS, _SQLITE_INDEXES
from agent_base.memory.store.sql_base import _SqlMemoryStoreBase

logger = logging.getLogger(__name__)


class SqliteMemoryStore(_SqlMemoryStoreBase):
    """sqlite 记忆表：与对话状态同一文件，aiosqlite 持久连接。"""

    _OPS_TIEBREAK = "rowid DESC"

    def __init__(self, connection: Any) -> None:
        self._conn = connection

    @classmethod
    async def create(cls, path: str) -> SqliteMemoryStore:
        import aiosqlite

        conn = await aiosqlite.connect(path)
        for ddl in _SQLITE_DDLS:
            await conn.execute(ddl)
        # 旧库补列（sqlite 无 alembic，运行时自迁移）：offsets_json 是
        # 切片可视化（M7）后加的。
        cursor = await conn.execute("PRAGMA table_info(doc_chunks)")
        columns = {row[1] for row in await cursor.fetchall()}
        if "offsets_json" not in columns:
            await conn.execute("ALTER TABLE doc_chunks ADD COLUMN offsets_json TEXT")
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
        # 微秒精度：memory_ops 的同秒并列依赖它稳定排序。
        return datetime.fromtimestamp(seconds).isoformat(sep=" ", timespec="microseconds")

    def _from_sql_ts(self, value: Any) -> float:
        try:
            return datetime.fromisoformat(str(value)).timestamp()
        except (TypeError, ValueError):
            # 坏时间戳静默伪装成"刚刚更新"会污染时间衰减排序：显式告警
            # 而不是无声兜底（L3）。
            logger.warning("sqlite memory store: 无法解析时间戳 %r，按当前时间兜底", value)
            return time.time()

    async def aclose(self) -> None:
        await self._conn.close()
