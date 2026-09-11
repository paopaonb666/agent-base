"""MySQL 5.7 兼容的 checkpointer。

官方 ``langgraph-checkpoint-mysql`` 把 MySQL >= 8.0.19 作为硬性要求：
它的迁移 SQL 使用表达式默认值（``metadata JSON DEFAULT ('{}')``），
读取 SQL（``SELECT_SQL``）使用 ``JSON_TABLE`` / ``JSON_ARRAYAGG``
把 checkpoint 里的 ``channel_versions`` 展开成行再聚合 blob——这些
都是 8.0 专属特性，5.7 直接报语法/函数不存在错误。

``MySQL57Saver`` 以子类覆写的方式补齐 5.7 支持：

- ``setup()``：直接按官方迁移链的**最终表结构**执行 5.7 等价 DDL
  （唯一差异：JSON 列去掉 DEFAULT——5.7 的 JSON 列不允许任何 DEFAULT，
  官方写入路径总是显式提供值，因此等价），并把官方迁移版本号一次性
  补齐，使官方包的未来升级仍能按增量迁移走。中间迁移的数据变换对
  空库无意义，跳过是安全的。
- ``aget_tuple`` / ``alist``：行查询用 5.7 兼容的基础 SQL，
  ``channel_values`` / ``pending_writes`` 在 Python 侧组装——官方
  SQL 里的 JSON_TABLE 展开本质只是"按 checkpoint.channel_versions
  匹配 blob 行"，放进 Python 后语义不变且更直白。
- 写入路径（``aput`` / ``aput_writes`` / ``adelete_thread``）与官方
  完全一致（``UNHEX(MD5())`` / ``ON DUPLICATE KEY`` / ``VALUE()`` 都
  是 5.7 特性），直接继承。

覆盖范围说明：不支持"在 8.0 上建库后再搬到 5.7"的数据迁移场景。
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint, CheckpointTuple, get_checkpoint_id
from langgraph.checkpoint.mysql.aio import AIOMySQLSaver
from langgraph.checkpoint.serde.types import TASKS

# 官方迁移链的最新版本号：setup() 把它一次性写入 checkpoint_migrations，
# 之后官方包升级（MIGRATIONS 变长）时增量迁移照常生效。
_OFFICIAL_LATEST_VERSION = len(AIOMySQLSaver.MIGRATIONS) - 1

# 最终表结构的 5.7 等价 DDL（列名/主键/索引与官方迁移链跑完后的形态
# 一一对应；唯一差异是 JSON 列无 DEFAULT，见模块 docstring）。
_SCHEMA_DDL: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS checkpoints (
        thread_id VARCHAR(150) NOT NULL,
        checkpoint_ns VARCHAR(2000) NOT NULL,
        checkpoint_ns_hash BINARY(16) NOT NULL,
        checkpoint_id VARCHAR(150) NOT NULL,
        parent_checkpoint_id VARCHAR(150),
        type VARCHAR(150),
        checkpoint JSON NOT NULL,
        metadata JSON NOT NULL,
        PRIMARY KEY (thread_id, checkpoint_ns_hash, checkpoint_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS checkpoint_blobs (
        thread_id VARCHAR(150) NOT NULL,
        checkpoint_ns VARCHAR(2000) NOT NULL,
        checkpoint_ns_hash BINARY(16) NOT NULL,
        channel VARCHAR(150) NOT NULL,
        version VARCHAR(150) NOT NULL,
        type VARCHAR(150) NOT NULL,
        `blob` LONGBLOB,
        PRIMARY KEY (thread_id, checkpoint_ns_hash, channel, version)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS checkpoint_writes (
        thread_id VARCHAR(150) NOT NULL,
        checkpoint_ns VARCHAR(2000) NOT NULL,
        checkpoint_ns_hash BINARY(16) NOT NULL,
        checkpoint_id VARCHAR(150) NOT NULL,
        task_id VARCHAR(150) NOT NULL,
        idx INTEGER NOT NULL,
        task_path VARCHAR(2000) NOT NULL DEFAULT '',
        channel VARCHAR(150) NOT NULL,
        type VARCHAR(150),
        `blob` LONGBLOB NOT NULL,
        PRIMARY KEY (thread_id, checkpoint_ns_hash, checkpoint_id, task_id, idx)
    )
    """,
    "CREATE INDEX checkpoints_thread_id_idx ON checkpoints (thread_id)",
    "CREATE INDEX checkpoint_blobs_thread_id_idx ON checkpoint_blobs (thread_id)",
    "CREATE INDEX checkpoint_writes_thread_id_idx ON checkpoint_writes (thread_id)",
    "CREATE INDEX checkpoints_checkpoint_id_idx ON checkpoints (checkpoint_id)",
)

# 与官方 SELECT_SQL 相同行结构的 5.7 兼容查询；channel_values /
# pending_writes 由 _attach_channel_values_and_writes 在 Python 侧组装。
_SELECT_CHECKPOINTS = """
SELECT thread_id, checkpoint, checkpoint_ns, checkpoint_id, parent_checkpoint_id, metadata
FROM checkpoints {WHERE}"""

_SELECT_BLOBS = """
SELECT channel, version, type, `blob` FROM checkpoint_blobs
WHERE thread_id = %s AND checkpoint_ns_hash = UNHEX(MD5(%s))"""

_SELECT_PENDING_WRITES = """
SELECT task_id, channel, type, `blob`, idx FROM checkpoint_writes
WHERE thread_id = %s AND checkpoint_ns_hash = UNHEX(MD5(%s)) AND checkpoint_id = %s"""

# alist 未显式给 limit 时的默认 SQL 上限：无 LIMIT 会把整个 checkpoints
# 表拉进内存（调用方如线程列表端点的迭代上限根本来不及生效）。
_ALIST_DEFAULT_LIMIT = 2000


class MySQL57Saver(AIOMySQLSaver):
    """在 MySQL 5.7 上工作的 ``AIOMySQLSaver``（见模块 docstring）。

    官方写入 SQL 用 ``/*!50700 ... *//*M! ...*/`` 条件注释区分 MySQL /
    MariaDB 分支，但 MySQL 分支选择的是 8.0.19 的 ``INSERT ... AS new``
    行别名语法——条件注释的阈值 50700 意味着 5.7 也会启用它，而 5.7
    不认识该语法。这里以 ``VALUES()`` 旧语法覆写两个 UPSERT 语句。
    """

    UPSERT_CHECKPOINTS_SQL = """
    INSERT INTO checkpoints
        (thread_id, checkpoint_ns, checkpoint_ns_hash, checkpoint_id,
         parent_checkpoint_id, checkpoint, metadata)
    VALUES (%s, %s, UNHEX(MD5(%s)), %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
        checkpoint = VALUES(checkpoint),
        metadata = VALUES(metadata)
"""

    UPSERT_CHECKPOINT_WRITES_SQL = """
    INSERT INTO checkpoint_writes
        (thread_id, checkpoint_ns, checkpoint_ns_hash, checkpoint_id,
         task_id, task_path, idx, channel, type, `blob`)
    VALUES (%s, %s, UNHEX(MD5(%s)), %s, %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
        channel = VALUES(channel),
        type = VALUES(type),
        `blob` = VALUES(`blob`)
"""

    async def setup(self) -> None:
        """以最终表结构的 5.7 等价 DDL 建库，并补齐官方迁移版本号。"""
        async with self._cursor() as cur:
            await cur.execute(
                "CREATE TABLE IF NOT EXISTS checkpoint_migrations (v INTEGER PRIMARY KEY)"
            )
            await cur.execute("SELECT v FROM checkpoint_migrations ORDER BY v DESC LIMIT 1")
            row = await cur.fetchone()
            version = row["v"] if row else -1
            if version >= _OFFICIAL_LATEST_VERSION:
                return
            for ddl in _SCHEMA_DDL:
                try:
                    await cur.execute(ddl)
                except Exception as exc:
                    # CREATE INDEX 没有幂等写法；重复建索引（1061）无害。
                    # args 可能为空元组：直接下标会抛 IndexError 掩盖真实错误。
                    if not exc.args or exc.args[0] != 1061:
                        raise
            for v in range(version + 1, _OFFICIAL_LATEST_VERSION + 1):
                await cur.execute("INSERT INTO checkpoint_migrations (v) VALUES (%s)", (v,))

    async def _load_checkpoint_tuple(self, value: dict[str, Any]) -> CheckpointTuple:
        """官方版本的等价实现，但直接消费已解析的 channel_values /
        pending_writes——_assemble_rows 已在 Python 侧组装好，无需再走
        官方的 JSON 字符串反序列化（那正是 5.7 不支持的 SQL 产物）。"""
        checkpoint = cast(
            Checkpoint,
            {
                **value["checkpoint"],
                "channel_values": {
                    **value["checkpoint"].get("channel_values"),
                    **self._load_blobs(value["channel_values"]),
                },
            },
        )
        return CheckpointTuple(
            {
                "configurable": {
                    "thread_id": value["thread_id"],
                    "checkpoint_ns": value["checkpoint_ns"],
                    "checkpoint_id": value["checkpoint_id"],
                }
            },
            checkpoint,
            json.loads(value["metadata"]),
            (
                {
                    "configurable": {
                        "thread_id": value["thread_id"],
                        "checkpoint_ns": value["checkpoint_ns"],
                        "checkpoint_id": value["parent_checkpoint_id"],
                    }
                }
                if value["parent_checkpoint_id"]
                else None
            ),
            await asyncio.to_thread(self._load_writes, value["pending_writes"]),
        )

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """按官方语义取最新（或指定）checkpoint，读取 SQL 为 5.7 兼容形态。"""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = get_checkpoint_id(config)
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        if checkpoint_id:
            where = (
                "WHERE thread_id = %(thread_id)s "
                "AND checkpoint_ns_hash = UNHEX(MD5(%(checkpoint_ns)s)) "
                "AND checkpoint_id = %(checkpoint_id)s"
            )
            args: dict[str, Any] = {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        else:
            where = (
                "WHERE thread_id = %(thread_id)s "
                "AND checkpoint_ns_hash = UNHEX(MD5(%(checkpoint_ns)s))"
            )
            args = {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}

        query = _SELECT_CHECKPOINTS.replace("{WHERE}", where)
        if not checkpoint_id:
            query += " ORDER BY checkpoint_id DESC LIMIT 1"
        async with self._cursor() as cur:
            await cur.execute(query, args)
            value = await cur.fetchone()
            if value is None:
                return None
            values = [value]
            await self._assemble_rows(cur, values)
            await self._migrate_pending_sends_if_needed(cur, values)
            return await self._load_checkpoint_tuple(value)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Any:
        """按官方语义列出 checkpoint（5.7 兼容读取路径）。"""
        where, args = self._search_where(config, filter, before)
        query = _SELECT_CHECKPOINTS.replace("{WHERE}", where) + " ORDER BY checkpoint_id DESC"
        if limit is not None:
            query += " LIMIT %(limit)s"
            args = {**args, "limit": int(limit)}
        else:
            query += f" LIMIT {_ALIST_DEFAULT_LIMIT}"
        async with self._cursor() as cur:
            await cur.execute(query, args)
            values = await cur.fetchall()
            if not values:
                return
            await self._assemble_rows(cur, values)
            await self._migrate_pending_sends_if_needed(cur, values)
            for value in values:
                yield await self._load_checkpoint_tuple(value)

    async def _assemble_rows(self, cur: Any, values: list[dict[str, Any]]) -> None:
        """把每行的 checkpoint JSON、channel_values、pending_writes 装配成
        官方 ``_load_checkpoint_tuple`` 所期望的行结构。"""
        for value in values:
            value["checkpoint"] = json.loads(value["checkpoint"])
            thread_id = value["thread_id"]
            checkpoint_ns = value["checkpoint_ns"]
            # 官方 _get_cursor_from_connection 用 DictCursor：行是 dict。
            await cur.execute(_SELECT_BLOBS, (thread_id, checkpoint_ns))
            blobs = {
                (r["channel"], r["version"]): (r["type"], r["blob"]) for r in await cur.fetchall()
            }
            # 官方 SQL 的等价语义：按 checkpoint.channel_versions 匹配 blob
            # 行（含 type 为 "empty" 的占位行，由 _load_blobs 统一过滤）。
            versions = value["checkpoint"].get("channel_versions", {})
            value["channel_values"] = [
                (ch, *blobs[(ch, ver)]) for ch, ver in versions.items() if (ch, ver) in blobs
            ]
            await cur.execute(
                _SELECT_PENDING_WRITES, (thread_id, checkpoint_ns, value["checkpoint_id"])
            )
            writes = list(await cur.fetchall())
            writes.sort(key=lambda r: (r["task_id"], r["idx"]))  # 与官方一致
            value["pending_writes"] = [
                (r["task_id"], r["channel"], r["type"], r["blob"]) for r in writes
            ]

    async def _migrate_pending_sends_if_needed(
        self, cur: Any, values: list[dict[str, Any]]
    ) -> None:
        """v<4 的旧 checkpoint 需要 pending sends 迁移（官方逻辑的 5.7 版）。"""
        to_migrate = [v for v in values if v["checkpoint"]["v"] < 4 and v["parent_checkpoint_id"]]
        if not to_migrate:
            return
        # values 可能跨多个 thread_id（如 alist(None) 的线程列表路径）：
        # 按 thread_id 分组查询，否则 A 线程的 parent id 混进 B 线程的
        # 过滤条件里，两条线程都取不到（或取错）迁移行。
        by_thread: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for value in to_migrate:
            by_thread[value["thread_id"]].append(value)
        grouped: dict[tuple[str, str], list[tuple[str, str, str, bytes]]] = defaultdict(list)
        for thread_id, group in by_thread.items():
            placeholders = ",".join(["%s"] * len(group))
            await cur.execute(
                "SELECT checkpoint_id, task_path, task_id, type, `blob`, idx "
                f"FROM checkpoint_writes WHERE thread_id = %s AND checkpoint_id IN ({placeholders}) "
                "AND channel = %s",
                (thread_id, *[v["parent_checkpoint_id"] for v in group], TASKS),
            )
            for row in await cur.fetchall():
                grouped[(thread_id, row["checkpoint_id"])].append(
                    (row["task_path"], row["task_id"], row["type"], row["blob"])
                )
        for value in to_migrate:
            sends = sorted(
                grouped.get((value["thread_id"], value["parent_checkpoint_id"]), []),
                key=lambda s: (s[0], s[1]),  # (task_path, task_id)，与官方一致
            )
            if value["channel_values"] is None:
                value["channel_values"] = []
            self._migrate_pending_sends(
                [(s[2], s[3]) for s in sends], value["checkpoint"], value["channel_values"]
            )


async def probe_mysql_major_version(conn: Any) -> int:
    """返回连接指向的服务器主版本号（5 / 8 / ...）。"""
    async with conn.cursor() as cur:
        await cur.execute("SELECT VERSION()")
        row = await cur.fetchone()
    if row is None:
        # 某些代理/边缘节点可能返回空结果：给一条可读的配置错误，
        # 而不是启动时的 TypeError。
        from agent_base.core.config import SettingsError

        raise SettingsError("could not read server version: SELECT VERSION() returned no rows")
    return int(str(row[0]).split(".")[0])
