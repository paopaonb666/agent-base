"""sqlite / MySQL 记忆后端共享的 DML 与行编解码（P0-3 拆分）。

SQL 一律用 ``?`` 占位符书写（本项目语句不含字面问号），MySQL 执行前
整体替换为 ``%s``；整行覆盖式写入用 sqlite 的 ``INSERT OR REPLACE`` /
MySQL 的 ``REPLACE INTO``（同为"删旧插新"的原子语义）。
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from typing import Any

from agent_base.memory.store.models import (
    KNOWN_MEMORY_KINDS,
    KNOWN_MEMORY_STATUSES,
    DocChunk,
    MemoryBlock,
    MemoryOp,
    MemoryRecord,
    MemoryStoreError,
    MemoryVersion,
    SessionSummary,
    ThreadIndex,
)


def _json_list(value: Any) -> list[str]:
    """容错解析 JSON 字符串列表：畸形/空值一律返回空表（审计可查，不抛错）。"""
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed]


class _SqlMemoryStoreBase:
    """sqlite / MySQL 两个实现共享的 DML。

    子类只需提供四个连接原语（``_execute`` / ``_fetch_all`` / 时间戳互转）
    与建表自举；所有 SQL 用 ``?`` 占位符书写，MySQL 侧在执行前整体替换为
    ``%s``——本项目语句不含字面问号，整体替换是安全的。
    """

    _PLACEHOLDER = "?"
    # memory_ops 列表的同秒并列排序键：sqlite 用 rowid，MySQL 用 seq 列。
    _OPS_TIEBREAK = "rowid DESC"

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

    # -- 行编解码 --------------------------------------------------------------
    @staticmethod
    def _record_params(record: MemoryRecord) -> tuple[Any, ...]:
        return (
            record.memory_id,
            record.user_id,
            record.agent_id,
            record.kind,
            record.content,
            json.dumps(record.tags, ensure_ascii=False),
            record.embedding,
            record.embedding_dim,
            record.salience,
            record.status,
            record.source_thread_id,
            json.dumps(record.source_refs, ensure_ascii=False),
        )

    _MEMORY_COLUMNS = (
        "memory_id, user_id, agent_id, kind, content, tags_json, embedding, embedding_dim,"
        " salience, status, source_thread_id, source_refs_json, created_at, updated_at,"
        " last_accessed_at, access_count"
    )

    def _record_from_row(self, row: tuple[Any, ...]) -> MemoryRecord:
        return MemoryRecord(
            memory_id=row[0],
            user_id=row[1],
            agent_id=row[2],
            kind=row[3],
            content=row[4],
            tags=_json_list(row[5]),
            embedding=bytes(row[6]) if row[6] is not None else None,
            embedding_dim=row[7],
            salience=float(row[8]),
            status=row[9],
            source_thread_id=row[10] or "",
            source_refs=_json_list(row[11]),
            created_at=self._from_sql_ts(row[12]),
            updated_at=self._from_sql_ts(row[13]),
            last_accessed_at=self._from_sql_ts(row[14]) if row[14] is not None else None,
            access_count=int(row[15] or 0),
        )

    @staticmethod
    def _block_params(block: MemoryBlock) -> tuple[Any, ...]:
        return (
            block.user_id,
            block.agent_id,
            block.label,
            block.content,
            block.char_limit,
            block.version,
        )

    _BLOCK_COLUMNS = "user_id, agent_id, label, content, char_limit, version, updated_at"

    def _block_from_row(self, row: tuple[Any, ...]) -> MemoryBlock:
        return MemoryBlock(
            user_id=row[0],
            agent_id=row[1],
            label=row[2],
            content=row[3],
            char_limit=int(row[4]),
            version=int(row[5]),
            updated_at=self._from_sql_ts(row[6]),
        )

    _SUMMARY_COLUMNS = "user_id, thread_id, summary, agent_id, covered_message_count, updated_at"

    def _summary_from_row(self, row: tuple[Any, ...]) -> SessionSummary:
        return SessionSummary(
            user_id=row[0],
            thread_id=row[1],
            summary=row[2],
            agent_id=row[3] or "",
            covered_message_count=int(row[4] or 0),
            updated_at=self._from_sql_ts(row[5]),
        )

    _CHUNK_COLUMNS = (
        "chunk_id, file_id, thread_id, user_id, agent_id, ordinal, text, offsets_json,"
        " embedding, embedding_dim, created_at"
    )

    @staticmethod
    def _encode_offsets(offsets: tuple[tuple[int, int], ...] | None) -> str | None:
        if not offsets:
            return None
        return json.dumps([list(pair) for pair in offsets])

    @staticmethod
    def _decode_offsets(value: Any) -> tuple[tuple[int, int], ...] | None:
        if not value:
            return None
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return None
        if not isinstance(parsed, list):
            return None
        return (
            tuple(
                (int(seg[0]), int(seg[1]))
                for seg in parsed
                if isinstance(seg, list) and len(seg) == 2
            )
            or None
        )

    def _chunk_from_row(self, row: tuple[Any, ...]) -> DocChunk:
        return DocChunk(
            chunk_id=row[0],
            file_id=row[1],
            thread_id=row[2] or "",
            user_id=row[3],
            agent_id=row[4] or "",
            ordinal=int(row[5] or 0),
            text=row[6],
            offsets=self._decode_offsets(row[7]),
            embedding=bytes(row[8]) if row[8] is not None else None,
            embedding_dim=row[9],
            created_at=self._from_sql_ts(row[10]),
        )

    _OP_COLUMNS = (
        "op_id, op, user_id, agent_id, thread_id, detail_json, status, error_text,"
        " duration_ms, created_at"
    )

    def _op_from_row(self, row: tuple[Any, ...]) -> MemoryOp:
        try:
            detail = json.loads(row[5]) if row[5] else {}
        except (TypeError, ValueError):
            detail = {}
        return MemoryOp(
            op_id=row[0],
            op=row[1],
            user_id=row[2] or "",
            agent_id=row[3] or "",
            thread_id=row[4] or "",
            detail=detail if isinstance(detail, dict) else {},
            status=row[6] or "ok",
            error_text=row[7] or "",
            duration_ms=int(row[8] or 0),
            created_at=self._from_sql_ts(row[9]),
        )

    # -- memories --------------------------------------------------------------
    async def upsert_memory(self, record: MemoryRecord) -> None:
        if record.kind not in KNOWN_MEMORY_KINDS:
            raise MemoryStoreError(
                f"unknown memory kind {record.kind!r}; expected one of {KNOWN_MEMORY_KINDS}"
            )
        if record.status not in KNOWN_MEMORY_STATUSES:
            raise MemoryStoreError(
                f"unknown memory status {record.status!r}; expected one of {KNOWN_MEMORY_STATUSES}"
            )
        columns = (
            "memory_id",
            "user_id",
            "agent_id",
            "kind",
            "content",
            "tags_json",
            "embedding",
            "embedding_dim",
            "salience",
            "status",
            "source_thread_id",
            "source_refs_json",
            "created_at",
            "updated_at",
            "last_accessed_at",
            "access_count",
        )
        values = (*self._record_params(record),)
        # 时间戳列与 access_count 不在 _record_params 里：它们由后端的
        # 时间格式决定，逐个追加（None 的 last_accessed_at 保持 NULL）。
        await self._execute(
            self._replace_into("memories", columns),
            (
                *values,
                self._to_sql_ts(record.created_at),
                self._to_sql_ts(record.updated_at),
                self._to_sql_ts(record.last_accessed_at) if record.last_accessed_at else None,
                record.access_count,
            ),
        )

    async def get_memory(self, memory_id: str) -> MemoryRecord | None:
        rows = await self._fetch_all(
            f"SELECT {self._MEMORY_COLUMNS} FROM memories WHERE memory_id = ?", (memory_id,)
        )
        return self._record_from_row(rows[0]) if rows else None

    async def delete_memory(self, memory_id: str) -> bool:
        rowcount = await self._execute("DELETE FROM memories WHERE memory_id = ?", (memory_id,))
        return rowcount > 0

    async def list_memories(
        self,
        user_id: str,
        *,
        agent_id: str | None = None,
        include_global: bool = True,
        kinds: Sequence[str] | None = None,
        statuses: Sequence[str] = ("active",),
        limit: int = 2000,
        exclude_profile: bool = False,
    ) -> list[MemoryRecord]:
        sql = f"SELECT {self._MEMORY_COLUMNS} FROM memories WHERE user_id = ?"
        params: list[Any] = [user_id]
        # 浏览/检索列表默认排除画像：画像由上下文组装与画像页签单独
        # 呈现，混进记忆列表既是实现细节泄漏也是重复展示。
        if exclude_profile:
            sql += " AND memory_id NOT LIKE 'profile:%'"
        if statuses:
            sql += f" AND status IN ({', '.join('?' for _ in statuses)})"
            params.extend(statuses)
        if kinds:
            sql += f" AND kind IN ({', '.join('?' for _ in kinds)})"
            params.extend(kinds)
        if agent_id is not None:
            if include_global:
                sql += " AND (agent_id = ? OR agent_id = '*')"
            else:
                sql += " AND agent_id = ?"
            params.append(agent_id)
        sql += " ORDER BY updated_at DESC LIMIT ?"
        params.append(max(1, limit))
        rows = await self._fetch_all(sql, params)
        return [self._record_from_row(row) for row in rows]

    async def touch_memories(self, memory_ids: Sequence[str]) -> None:
        if not memory_ids:
            return
        placeholders = ", ".join("?" for _ in memory_ids)
        await self._execute(
            "UPDATE memories SET last_accessed_at = ?, access_count = access_count + 1"
            f" WHERE memory_id IN ({placeholders})",
            (self._to_sql_ts(time.time()), *memory_ids),
        )

    async def list_memories_needing_embedding(
        self, dims: int, limit: int = 100
    ) -> list[MemoryRecord]:
        """回填查询：active 且向量缺失/维度不符的非画像记忆（锐评 #3）。

        画像记录（profile:*）刻意排除——它们整体注入不走向量召回，且
        永远没有向量，纳入会让回填死循环。
        """
        rows = await self._fetch_all(
            f"SELECT {self._MEMORY_COLUMNS} FROM memories WHERE status = 'active'"
            " AND memory_id NOT LIKE 'profile:%'"
            " AND (embedding IS NULL OR embedding_dim IS NULL OR embedding_dim != ?)"
            " ORDER BY updated_at DESC LIMIT ?",
            (dims, max(1, limit)),
        )
        return [self._record_from_row(row) for row in rows]

    # -- memory blocks -----------------------------------------------------------
    async def upsert_block(self, block: MemoryBlock) -> None:
        await self._execute(
            self._replace_into(
                "memory_blocks",
                ("user_id", "agent_id", "label", "content", "char_limit", "version", "updated_at"),
            ),
            (*self._block_params(block), self._to_sql_ts(block.updated_at)),
        )

    async def get_block(self, user_id: str, agent_id: str, label: str) -> MemoryBlock | None:
        rows = await self._fetch_all(
            f"SELECT {self._BLOCK_COLUMNS} FROM memory_blocks"
            " WHERE user_id = ? AND agent_id = ? AND label = ?",
            (user_id, agent_id, label),
        )
        return self._block_from_row(rows[0]) if rows else None

    async def list_blocks(self, user_id: str, agent_id: str) -> list[MemoryBlock]:
        rows = await self._fetch_all(
            f"SELECT {self._BLOCK_COLUMNS} FROM memory_blocks"
            " WHERE user_id = ? AND agent_id = ? ORDER BY label ASC",
            (user_id, agent_id),
        )
        return [self._block_from_row(row) for row in rows]

    async def delete_block(self, user_id: str, agent_id: str, label: str) -> bool:
        rowcount = await self._execute(
            "DELETE FROM memory_blocks WHERE user_id = ? AND agent_id = ? AND label = ?",
            (user_id, agent_id, label),
        )
        return rowcount > 0

    # -- session summaries --------------------------------------------------------
    async def get_summary(self, user_id: str, thread_id: str) -> SessionSummary | None:
        rows = await self._fetch_all(
            f"SELECT {self._SUMMARY_COLUMNS} FROM session_summaries"
            " WHERE user_id = ? AND thread_id = ?",
            (user_id, thread_id),
        )
        return self._summary_from_row(rows[0]) if rows else None

    async def upsert_summary(self, summary: SessionSummary) -> None:
        await self._execute(
            self._replace_into(
                "session_summaries",
                (
                    "user_id",
                    "thread_id",
                    "summary",
                    "agent_id",
                    "covered_message_count",
                    "updated_at",
                ),
            ),
            (
                summary.user_id,
                summary.thread_id,
                summary.summary,
                summary.agent_id,
                summary.covered_message_count,
                self._to_sql_ts(summary.updated_at),
            ),
        )

    # -- doc chunks ----------------------------------------------------------------
    async def put_chunks(self, chunks: Sequence[DocChunk]) -> None:
        for chunk in chunks:
            await self._execute(
                self._replace_into(
                    "doc_chunks",
                    (
                        "chunk_id",
                        "file_id",
                        "thread_id",
                        "user_id",
                        "agent_id",
                        "ordinal",
                        "text",
                        "offsets_json",
                        "embedding",
                        "embedding_dim",
                        "created_at",
                    ),
                ),
                (
                    chunk.chunk_id,
                    chunk.file_id,
                    chunk.thread_id,
                    chunk.user_id,
                    chunk.agent_id,
                    chunk.ordinal,
                    chunk.text,
                    self._encode_offsets(chunk.offsets),
                    chunk.embedding,
                    chunk.embedding_dim,
                    self._to_sql_ts(chunk.created_at),
                ),
            )

    async def list_chunks(
        self,
        user_id: str,
        *,
        agent_id: str | None = None,
        file_id: str | None = None,
        limit: int = 2000,
    ) -> list[DocChunk]:
        sql = f"SELECT {self._CHUNK_COLUMNS} FROM doc_chunks WHERE user_id = ?"
        params: list[Any] = [user_id]
        if agent_id is not None:
            sql += " AND (agent_id = ? OR agent_id = '*')"
            params.append(agent_id)
        if file_id is not None:
            sql += " AND file_id = ?"
            params.append(file_id)
        sql += " ORDER BY file_id ASC, ordinal ASC LIMIT ?"
        params.append(max(1, limit))
        rows = await self._fetch_all(sql, params)
        return [self._chunk_from_row(row) for row in rows]

    async def count_chunks_for_file(self, file_id: str) -> int:
        rows = await self._fetch_all(
            "SELECT COUNT(*) FROM doc_chunks WHERE file_id = ?", (file_id,)
        )
        return int(rows[0][0]) if rows else 0

    async def delete_chunks_for_file(self, file_id: str) -> int:
        return await self._execute("DELETE FROM doc_chunks WHERE file_id = ?", (file_id,))

    async def chunk_counts_by_file(self, user_id: str) -> dict[str, int]:
        rows = await self._fetch_all(
            "SELECT file_id, COUNT(*) FROM doc_chunks WHERE user_id = ? GROUP BY file_id",
            (user_id,),
        )
        return {str(row[0]): int(row[1]) for row in rows}

    # -- 线程级联 --------------------------------------------------------------------
    async def delete_for_thread(self, thread_id: str) -> None:
        # memories 刻意不删：跨会话记忆独立于线程存活（出处仍在
        # source_thread_id）；随线程消亡的是它的摘要、知识库分块与会话索引。
        await self._execute("DELETE FROM doc_chunks WHERE thread_id = ?", (thread_id,))
        await self._execute("DELETE FROM session_summaries WHERE thread_id = ?", (thread_id,))
        await self._execute("DELETE FROM thread_index WHERE thread_id = ?", (thread_id,))

    # -- 操作审计 --------------------------------------------------------------------
    async def record_op(self, op: MemoryOp) -> None:
        await self._execute(
            f"INSERT INTO memory_ops ({self._OP_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                op.op_id,
                op.op,
                op.user_id,
                op.agent_id,
                op.thread_id,
                json.dumps(op.detail, ensure_ascii=False, default=str),
                op.status,
                op.error_text,
                op.duration_ms,
                self._to_sql_ts(op.created_at),
            ),
        )

    async def list_ops(self, user_id: str | None = None, limit: int = 100) -> list[MemoryOp]:
        sql = f"SELECT {self._OP_COLUMNS} FROM memory_ops"
        params: list[Any] = []
        if user_id is not None:
            sql += " WHERE user_id = ?"
            params.append(user_id)
        sql += f" ORDER BY created_at DESC, {self._OPS_TIEBREAK} LIMIT ?"
        params.append(max(1, limit))
        rows = await self._fetch_all(sql, params)
        return [self._op_from_row(row) for row in rows]

    # -- 版本史 ---------------------------------------------------------------
    _VERSIONS_TIEBREAK = "rowid DESC"

    _VERSION_COLUMNS = (
        "version_id, memory_id, user_id, op, content, previous_content, status, created_at"
    )

    def _version_from_row(self, row: tuple[Any, ...]) -> MemoryVersion:
        return MemoryVersion(
            version_id=row[0],
            memory_id=row[1],
            user_id=row[2] or "",
            op=row[3],
            content=row[4] or "",
            previous_content=row[5] or "",
            status=row[6] or "active",
            created_at=self._from_sql_ts(row[7]),
        )

    async def record_version(self, version: MemoryVersion) -> None:
        await self._execute(
            f"INSERT INTO memory_versions ({self._VERSION_COLUMNS}) VALUES"
            " (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                version.version_id,
                version.memory_id,
                version.user_id,
                version.op,
                version.content,
                # 空串归一为 NULL：语义上是"没有前值"（create）
                version.previous_content or None,
                version.status,
                self._to_sql_ts(version.created_at),
            ),
        )

    async def list_versions(self, memory_id: str, limit: int = 50) -> list[MemoryVersion]:
        rows = await self._fetch_all(
            f"SELECT {self._VERSION_COLUMNS} FROM memory_versions WHERE memory_id = ?"
            f" ORDER BY created_at DESC, {self._VERSIONS_TIEBREAK} LIMIT ?",
            (memory_id, max(1, limit)),
        )
        return [self._version_from_row(row) for row in rows]

    async def get_version(self, version_id: str) -> MemoryVersion | None:
        rows = await self._fetch_all(
            f"SELECT {self._VERSION_COLUMNS} FROM memory_versions WHERE version_id = ?",
            (version_id,),
        )
        return self._version_from_row(rows[0]) if rows else None

    async def list_chunks_needing_embedding(self, dims: int, limit: int = 100) -> list[DocChunk]:
        """回填查询：向量缺失或维度不符的知识库分块（换 embedding
        模型后 doc_chunks 的修复路径，与 memories 的回填同语义）。"""
        rows = await self._fetch_all(
            f"SELECT {self._CHUNK_COLUMNS} FROM doc_chunks"
            " WHERE embedding IS NULL OR embedding_dim IS NULL OR embedding_dim != ?"
            " ORDER BY created_at DESC LIMIT ?",
            (dims, max(1, limit)),
        )
        return [self._chunk_from_row(row) for row in rows]

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
