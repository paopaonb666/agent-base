"""MySQL 基座表后端：aiomysql 小连接池（与记忆池相互独立）。"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any

from agent_base.core.storage.ddl import _MYSQL_DDLS
from agent_base.core.storage.sql_base import _CoreSqlStoreBase

if TYPE_CHECKING:  # pragma: no cover - import avoided at runtime
    from agent_base.core.config import Settings

logger = logging.getLogger(__name__)

# 线程索引/计划快照是低频读写（每轮一到两次），小池足够；与记忆池分开
# 是为了让"关掉记忆"的部署不必为记忆流量付费。
_CORE_POOL_SIZE = 2


class CoreMysqlStore(_CoreSqlStoreBase):
    """MySQL 基座表：aiomysql 连接池，首次操作时惰性建池建表。"""

    _PLACEHOLDER = "%s"

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
                maxsize=_CORE_POOL_SIZE,
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
            logger.warning("mysql core store: 无法解析时间戳 %r，按当前时间兜底", value)
            return time.time()
