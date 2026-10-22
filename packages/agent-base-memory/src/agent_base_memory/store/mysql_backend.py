"""MySQL 记忆表后端：aiomysql 连接池（锐评 #2——每操作建连在高负载下
开销显著；记忆读写是每轮一到两次的稳定流量，小池即可，池在首次操作时
惰性创建）。"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any

from agent_base.memory.store.ddl import _MYSQL_DDLS
from agent_base.memory.store.sql_base import _SqlMemoryStoreBase

if TYPE_CHECKING:  # pragma: no cover - import avoided at runtime
    from agent_base.core.config import Settings

logger = logging.getLogger(__name__)


class MysqlMemoryStore(_SqlMemoryStoreBase):
    """MySQL 记忆表：aiomysql 连接池（锐评 #2——每操作建连在高负载下

    开销显著；记忆读写是每轮一到两次的稳定流量，小池即可，池在首次
    操作时惰性创建。
    """

    _PLACEHOLDER = "%s"
    _OPS_TIEBREAK = "seq DESC"
    _VERSIONS_TIEBREAK = "seq DESC"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pool: Any = None
        self._table_ready = False

    async def _get_pool(self) -> Any:
        if self._pool is None:
            import aiomysql  # type: ignore[import-untyped]

            s = self._settings
            self._pool = await aiomysql.create_pool(
                host=s.checkpointer_mysql_host,
                port=s.checkpointer_mysql_port,
                user=s.checkpointer_mysql_user,
                password=s.checkpointer_mysql_password.get_secret_value(),
                db=s.checkpointer_mysql_database,
                minsize=1,
                maxsize=max(1, s.memory_mysql_pool_size),
                autocommit=True,
            )
        return self._pool

    async def _ensure_tables(self, cursor: Any) -> None:
        if self._table_ready:
            return
        for ddl in _MYSQL_DDLS:
            await cursor.execute(ddl)
        self._table_ready = True

    async def _execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        pool = await self._get_pool()
        async with pool.acquire() as conn, conn.cursor() as cur:
            await self._ensure_tables(cur)
            await cur.execute(self._sql(sql), tuple(params))
            return cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0

    async def _fetch_all(self, sql: str, params: Sequence[Any] = ()) -> list[tuple[Any, ...]]:
        pool = await self._get_pool()
        async with pool.acquire() as conn, conn.cursor() as cur:
            await self._ensure_tables(cur)
            await cur.execute(self._sql(sql), tuple(params))
            return list(await cur.fetchall())

    async def aclose(self) -> None:
        """归还并关闭连接池（MemoryService.aclose 经 getattr 调用）。"""
        if self._pool is not None:
            self._pool.close()
            await self._pool.wait_closed()
            self._pool = None

    def _to_sql_ts(self, seconds: float) -> Any:
        return datetime.fromtimestamp(seconds)

    def _from_sql_ts(self, value: Any) -> float:
        if isinstance(value, datetime):
            return value.timestamp()
        try:
            return datetime.fromisoformat(str(value)).timestamp()
        except (TypeError, ValueError):
            logger.warning("mysql memory store: 无法解析时间戳 %r，按当前时间兜底", value)
            return time.time()
