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

from agent_base.core.storage.models import MessageNode, ThreadIndex


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

    # -- 消息树（message_nodes；ADR-013） ----------------------------------------
    # append-only + 软删除；树形导航/子树计算在 tree.py 的纯函数里做，
    # 存储层只提供按 seq 排序的行集与逐行写。
    _NODE_COLUMNS = (
        "node_id, seq, thread_id, parent_id, msg_id, role, content, content_json, "
        "tool_calls_json, tool_call_id, name, status, attachments_json, "
        "label, starred, archived, created_at, deleted_at"
    )

    def _node_from_row(self, row: tuple[Any, ...]) -> MessageNode:
        return MessageNode(
            node_id=row[0],
            seq=int(row[1] or 0),
            thread_id=row[2],
            parent_id=row[3],
            msg_id=row[4] or "",
            role=row[5],
            content=row[6] or "",
            content_json=int(row[7] or 0),
            tool_calls_json=row[8] or "[]",
            tool_call_id=row[9] or "",
            name=row[10] or "",
            status=row[11] or "",
            attachments_json=row[12] or "[]",
            label=row[13] or "",
            starred=bool(row[14]),
            archived=bool(row[15]),
            created_at=self._from_sql_ts(row[16]),
            deleted_at=self._from_sql_ts(row[17]) if row[17] is not None else None,
        )

    async def next_node_seq(self, thread_id: str) -> int:
        """线程内的下一个节点序号（树锁内调用，天然无竞态）。"""
        rows = await self._fetch_all(
            "SELECT COALESCE(MAX(seq), 0) FROM message_nodes WHERE thread_id = ?",
            (thread_id,),
        )
        return int(rows[0][0] or 0) + 1

    async def append_node(self, node: MessageNode) -> None:
        seq = node.seq or await self.next_node_seq(node.thread_id)
        await self._execute(
            f"INSERT INTO message_nodes ({self._NODE_COLUMNS})"
            f" VALUES ({', '.join('?' for _ in range(18))})",
            (
                node.node_id,
                seq,
                node.thread_id,
                node.parent_id,
                node.msg_id,
                node.role,
                node.content,
                node.content_json,
                node.tool_calls_json,
                node.tool_call_id,
                node.name,
                node.status,
                node.attachments_json,
                node.label,
                int(node.starred),
                int(node.archived),
                self._to_sql_ts(node.created_at),
                self._to_sql_ts(node.deleted_at) if node.deleted_at is not None else None,
            ),
        )

    async def get_node(self, thread_id: str, node_id: str) -> MessageNode | None:
        rows = await self._fetch_all(
            f"SELECT {self._NODE_COLUMNS} FROM message_nodes"
            " WHERE thread_id = ? AND node_id = ?",
            (thread_id, node_id),
        )
        return self._node_from_row(rows[0]) if rows else None

    async def list_nodes(
        self, thread_id: str, *, include_deleted: bool = False
    ) -> list[MessageNode]:
        sql = f"SELECT {self._NODE_COLUMNS} FROM message_nodes WHERE thread_id = ?"
        if not include_deleted:
            sql += " AND deleted_at IS NULL"
        sql += " ORDER BY seq"
        rows = await self._fetch_all(sql, (thread_id,))
        return [self._node_from_row(row) for row in rows]

    async def update_node_meta(
        self,
        thread_id: str,
        node_id: str,
        *,
        label: str | None = None,
        starred: bool | None = None,
        archived: bool | None = None,
    ) -> bool:
        """按需更新维护元数据；None 的字段不动。返回是否存在该节点。"""
        sets: list[str] = []
        params: list[Any] = []
        if label is not None:
            sets.append("label = ?")
            params.append(label[:120])
        if starred is not None:
            sets.append("starred = ?")
            params.append(int(starred))
        if archived is not None:
            sets.append("archived = ?")
            params.append(int(archived))
        if not sets:
            return await self.get_node(thread_id, node_id) is not None
        params.extend((thread_id, node_id))
        rowcount = await self._execute(
            f"UPDATE message_nodes SET {', '.join(sets)}"
            " WHERE thread_id = ? AND node_id = ?",
            tuple(params),
        )
        return rowcount > 0

    async def set_nodes_deleted(
        self, thread_id: str, node_ids: Sequence[str], *, deleted: bool
    ) -> int:
        """子树软删 / 恢复：deleted=True 打标，False 清标（恢复）。"""
        if not node_ids:
            return 0
        marks = ", ".join("?" for _ in node_ids)
        value = self._to_sql_ts(time.time()) if deleted else None
        return await self._execute(
            f"UPDATE message_nodes SET deleted_at = ?"
            f" WHERE thread_id = ? AND node_id IN ({marks})",
            (value, thread_id, *node_ids),
        )

    async def set_current_node(self, thread_id: str, node_id: str | None) -> bool:
        """更新会话的活跃叶子指针（tree_pointers 表）；树写操作在锁内
        调用。返回是否写入了指针行。"""
        if node_id is None:
            await self._execute("DELETE FROM tree_pointers WHERE thread_id = ?", (thread_id,))
            return True
        await self._execute(
            self._replace_into("tree_pointers", ("thread_id", "current_node_id", "updated_at")),
            (thread_id, node_id, self._to_sql_ts(time.time())),
        )
        return True

    async def get_current_node(self, thread_id: str) -> str | None:
        rows = await self._fetch_all(
            "SELECT current_node_id FROM tree_pointers WHERE thread_id = ?",
            (thread_id,),
        )
        return str(rows[0][0]) if rows else None

    async def delete_nodes_for_thread(self, thread_id: str) -> None:
        await self._execute("DELETE FROM message_nodes WHERE thread_id = ?", (thread_id,))
        await self._execute("DELETE FROM tree_pointers WHERE thread_id = ?", (thread_id,))
