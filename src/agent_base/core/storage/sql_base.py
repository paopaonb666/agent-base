"""sqlite / MySQL 基座存储共享的 DML 与行编解码。

SQL 一律用 ``?`` 占位符书写（本项目语句不含字面问号），MySQL 执行前
整体替换为 ``%s``；整行覆盖式写入用 sqlite 的 ``INSERT OR REPLACE`` /
MySQL 的 ``REPLACE INTO``（同为"删旧插新"的原子语义）。
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from typing import Any

from agent_base.core.storage.models import ThreadIndex


class _CoreSqlStoreBase:
    """sqlite / MySQL 两个实现共享的 DML。

    子类只需提供四个连接原语（``_execute`` / ``_fetch_all`` / 时间戳互转）
    与建表自举；所有 SQL 用 ``?`` 占位符书写，MySQL 侧在执行前整体替换为
    ``%s``——本项目语句不含字面问号，整体替换是安全的。
    """

    _PLACEHOLDER = "?"

    def _sql(self, sql: str) -> str:
        if self._PLACEHOLDER == "?":
            return sql
        return sql.replace("?", "%s")

    def _replace_into(self, table: str, columns: tuple[str, ...]) -> str:
        """整行覆盖式 upsert：sqlite 是 INSERT OR REPLACE，MySQL 是 REPLACE
        INTO——两者同为"遇冲突删旧插新"的原子语义，要求提供全列值。"""
        if self._PLACEHOLDER == "?":
            return (
                f"INSERT OR REPLACE INTO {table} ({', '.join(columns)}) "
                f"VALUES ({', '.join('?' for _ in columns)})"
            )
        return (
            f"REPLACE INTO {table} ({', '.join(columns)}) "
            f"VALUES ({', '.join('%s' for _ in columns)})"
        )

    # -- 连接原语（子类实现） -------------------------------------------------
    async def _execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        raise NotImplementedError

    async def _fetch_all(self, sql: str, params: Sequence[Any] = ()) -> list[tuple[Any, ...]]:
        raise NotImplementedError

    def _to_sql_ts(self, seconds: float) -> Any:
        raise NotImplementedError

    def _from_sql_ts(self, value: Any) -> float:
        raise NotImplementedError

    # -- 会话索引（thread_index；S1 线程作用域） --------------------------------
    _THREAD_COLUMNS = "thread_id, user_id, module, title, created_at, updated_at"

    def _thread_from_row(self, row: tuple[Any, ...]) -> ThreadIndex:
        return ThreadIndex(
            thread_id=row[0],
            user_id=row[1] or "",
            module=row[2] or "",
            title=row[3] or "",
            created_at=self._from_sql_ts(row[4]),
            updated_at=self._from_sql_ts(row[5]),
        )

    async def upsert_thread_index(self, entry: ThreadIndex) -> None:
        await self._execute(
            self._replace_into(
                "thread_index",
                ("thread_id", "user_id", "module", "title", "created_at", "updated_at"),
            ),
            (
                entry.thread_id,
                entry.user_id,
                entry.module,
                entry.title,
                self._to_sql_ts(entry.created_at),
                self._to_sql_ts(entry.updated_at),
            ),
        )

    async def get_thread_index(self, thread_id: str) -> ThreadIndex | None:
        rows = await self._fetch_all(
            f"SELECT {self._THREAD_COLUMNS} FROM thread_index WHERE thread_id = ?",
            (thread_id,),
        )
        return self._thread_from_row(rows[0]) if rows else None

    async def list_thread_index(
        self, user_id: str, *, module: str | None = None, limit: int = 50
    ) -> list[ThreadIndex]:
        sql = f"SELECT {self._THREAD_COLUMNS} FROM thread_index WHERE user_id = ?"
        params: list[Any] = [user_id]
        if module is not None:
            sql += " AND module = ?"
            params.append(module)
        sql += " ORDER BY updated_at DESC LIMIT ?"
        params.append(max(1, limit))
        rows = await self._fetch_all(sql, params)
        return [self._thread_from_row(row) for row in rows]

    async def delete_thread_index(self, thread_id: str) -> None:
        await self._execute("DELETE FROM thread_index WHERE thread_id = ?", (thread_id,))

    # -- 计划快照（plan_snapshots；M10 跨轮延续） -------------------------------
    async def save_plan(
        self,
        *,
        thread_id: str,
        module: str,
        tasks: list[dict[str, Any]],
        cursor: int,
        replans: int,
    ) -> None:
        await self._execute(
            self._replace_into(
                "plan_snapshots",
                ("thread_id", "module", "tasks_json", "cursor", "replans", "updated_at"),
            ),
            (
                thread_id,
                module,
                json.dumps(tasks, ensure_ascii=False),
                int(cursor),
                int(replans),
                self._to_sql_ts(time.time()),
            ),
        )

    async def get_plan(self, thread_id: str) -> dict[str, Any] | None:
        rows = await self._fetch_all(
            "SELECT module, tasks_json, cursor, replans FROM plan_snapshots WHERE thread_id = ?",
            (thread_id,),
        )
        if not rows:
            return None
        module, tasks_json, cursor, replans = rows[0]
        try:
            tasks = json.loads(tasks_json or "[]")
        except ValueError:
            tasks = []
        return {
            "thread_id": thread_id,
            "module": str(module or ""),
            "tasks": tasks if isinstance(tasks, list) else [],
            "cursor": int(cursor or 0),
            "replans": int(replans or 0),
        }

    async def delete_plan(self, thread_id: str) -> None:
        await self._execute("DELETE FROM plan_snapshots WHERE thread_id = ?", (thread_id,))
