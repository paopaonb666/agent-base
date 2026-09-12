"""记忆系统存储层（M6a）：五张自管表 + 三种后端实现。

数据模型（与 alembic 0004 迁移一致，运行时自举幂等建表）：

- ``memories``          —— 跨会话长期记忆（semantic/episodic/procedural），带
                          embedding BLOB、显著度、时间有效性与访问记账；
                          整合管线以 ADD/UPDATE/DELETE/NOOP 维护其一致性。
- ``memory_blocks``     —— 常驻上下文的记忆块（persona/human/自定义，Letta 式），
                          agent 通过工具自我编辑。
- ``session_summaries`` —— 每线程的滚动摘要（短期压缩的载体）。
- ``doc_chunks``        —— 文档知识库分块（文件上传后的文本切片 + embedding）。
- ``memory_ops``        —— 记忆系统全量操作审计（抽取/整合/摘要/画像/手工增删）。

实现遵循 filestore/toollog 的既有套路：

- **Protocol + 三实现**（memory/sqlite/mysql），存储后端跟随
  ``CHECKPOINTER_BACKEND``；sqlite 与对话状态共用同一文件。
- sqlite/MySQL 的 DML 由 ``_SqlMemoryStoreBase`` 统一实现——SQL 一律用
  ``?`` 占位符书写（本项目语句不含字面问号），MySQL 执行前替换为 ``%s``；
  整行覆盖式写入用 sqlite 的 ``INSERT OR REPLACE`` / MySQL 的
  ``REPLACE INTO``（同为"删旧插新"的原子语义）。
- 时间戳在数据类里统一为 epoch 秒（``float``），落库形式由后端决定
  （sqlite 存字符串，MySQL 存 datetime）。
- 删除线程只级联清 ``session_summaries`` 与 ``doc_chunks``；``memories``
  刻意保留——它们本来就是跨会话的，``source_thread_id`` 退化为出处标注。

向量以 float32 小端 BLOB 编码（``array`` 模块，无 numpy 依赖），
``encode_embedding`` / ``decode_embedding`` 负责编解码。
"""

from __future__ import annotations

import json
import logging
import sys
import time
from array import array
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - import avoided at runtime
    from agent_base.core.config import Settings

logger = logging.getLogger(__name__)

# 记忆的合法类别（LangMem 分类法）与状态（整合管线的生命周期）。
KNOWN_MEMORY_KINDS: tuple[str, ...] = ("semantic", "episodic", "procedural")
KNOWN_MEMORY_STATUSES: tuple[str, ...] = ("active", "superseded", "archived")

# 用户画像的确定性 memory_id 前缀：画像复用 memories 表（agent_id="*"、
# tags=["profile"]），以 id 前缀与普通记忆区分——检索时被排除（画像由
# 上下文组装单独注入），管理端点里可见可删。
PROFILE_ID_PREFIX = "profile:"


def profile_memory_id(user_id: str) -> str:
    """某用户画像记录的确定性 id。"""
    return f"{PROFILE_ID_PREFIX}{user_id}"


class MemoryStoreError(ValueError):
    """存储层的快速失败：非法类别/状态等数据错误在写入边界被拒绝。"""


def encode_embedding(vector: Sequence[float]) -> bytes:
    """把向量编码为 float32 小端 BLOB（紧凑、跨后端一致的二进制形态）。"""
    buf = array("f", vector)
    if sys.byteorder == "big":  # pragma: no cover - 大端平台罕见，防御性分支
        buf.byteswap()
    return buf.tobytes()


def decode_embedding(blob: bytes) -> list[float]:
    """把 float32 小端 BLOB 解码回向量；空输入返回空表。"""
    if not blob:
        return []
    buf = array("f")
    buf.frombytes(bytes(blob))
    if sys.byteorder == "big":  # pragma: no cover - 大端平台罕见，防御性分支
        buf.byteswap()
    return list(buf)


@dataclass(frozen=True)
class MemoryRecord:
    """一条跨会话长期记忆。"""

    memory_id: str
    user_id: str
    agent_id: str  # 模块名；"*" 表示全模块共享
    kind: str  # semantic | episodic | procedural
    content: str
    tags: list[str] = field(default_factory=list)
    # float32 小端 BLOB（encode_embedding 的产物）；None = 未向量化。
    embedding: bytes | None = None
    embedding_dim: int | None = None
    salience: float = 0.5
    status: str = "active"  # active | superseded | archived
    source_thread_id: str = ""
    source_refs: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    last_accessed_at: float | None = None
    access_count: int = 0

    def meta(self) -> dict[str, Any]:
        """对外元信息（不含 embedding 字节与审计字段）。"""
        return {
            "memory_id": self.memory_id,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "kind": self.kind,
            "content": self.content,
            "tags": list(self.tags),
            "salience": self.salience,
            "status": self.status,
            "source_thread_id": self.source_thread_id,
            "has_embedding": self.embedding is not None,
            "access_count": self.access_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class MemoryBlock:
    """一个常驻上下文的记忆块（Letta 式，agent 可用工具自我编辑）。"""

    user_id: str
    agent_id: str
    label: str  # persona | human | 自定义标签
    content: str
    char_limit: int = 2000
    version: int = 0
    updated_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class SessionSummary:
    """一个线程的滚动摘要（短期压缩的载体，覆盖已修剪的旧历史）。"""

    user_id: str
    thread_id: str  # 带模块前缀的完整 thread id（模块名已含于其中）
    summary: str
    agent_id: str = ""
    covered_message_count: int = 0
    updated_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class DocChunk:
    """文档知识库的一个分块（上传文件的文本切片 + 可选 embedding）。"""

    chunk_id: str
    file_id: str
    thread_id: str
    user_id: str
    agent_id: str
    ordinal: int
    text: str
    embedding: bytes | None = None
    embedding_dim: int | None = None
    created_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class MemoryOp:
    """一次记忆系统操作的审计记录（成功与失败都记录）。"""

    op_id: str
    op: str  # extract | consolidate | summary | profile | ingest | manual | ...
    user_id: str = ""
    agent_id: str = ""
    thread_id: str = ""
    detail: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"  # ok | error
    error_text: str = ""
    duration_ms: int = 0
    created_at: float = field(default_factory=time.time)


@runtime_checkable
class MemoryStore(Protocol):
    """记忆存储的最小接口；实现必须保证并发安全。"""

    # -- memories ----------------------------------------------------------
    async def upsert_memory(self, record: MemoryRecord) -> None: ...

    async def get_memory(self, memory_id: str) -> MemoryRecord | None: ...

    async def delete_memory(self, memory_id: str) -> bool: ...

    async def list_memories(
        self,
        user_id: str,
        *,
        agent_id: str | None = None,
        include_global: bool = True,
        kinds: Sequence[str] | None = None,
        statuses: Sequence[str] = ("active",),
        limit: int = 2000,
    ) -> list[MemoryRecord]: ...

    async def touch_memories(self, memory_ids: Sequence[str]) -> None: ...

    # -- memory blocks -------------------------------------------------------
    async def upsert_block(self, block: MemoryBlock) -> None: ...

    async def get_block(self, user_id: str, agent_id: str, label: str) -> MemoryBlock | None: ...

    async def list_blocks(self, user_id: str, agent_id: str) -> list[MemoryBlock]: ...

    async def delete_block(self, user_id: str, agent_id: str, label: str) -> bool: ...

    # -- session summaries ---------------------------------------------------
    async def get_summary(self, user_id: str, thread_id: str) -> SessionSummary | None: ...

    async def upsert_summary(self, summary: SessionSummary) -> None: ...

    # -- doc chunks ----------------------------------------------------------
    async def put_chunks(self, chunks: Sequence[DocChunk]) -> None: ...

    async def list_chunks(
        self,
        user_id: str,
        *,
        agent_id: str | None = None,
        file_id: str | None = None,
        limit: int = 2000,
    ) -> list[DocChunk]: ...

    async def count_chunks_for_file(self, file_id: str) -> int: ...

    async def delete_chunks_for_file(self, file_id: str) -> int: ...

    # -- 线程级联 -------------------------------------------------------------
    async def delete_for_thread(self, thread_id: str) -> None: ...

    # -- 操作审计 -------------------------------------------------------------
    async def record_op(self, op: MemoryOp) -> None: ...

    async def list_ops(self, user_id: str | None = None, limit: int = 100) -> list[MemoryOp]: ...


# ─────────────────────────────── DDL ───────────────────────────────
# sqlite 与 MySQL 的建表语句逐表对应；MySQL 5.7 兼容约束：TEXT 不设默认
# 值、索引用内联 KEY、显式 ENGINE/CHARSET（与 filestore/toollog 一致）。

_SQLITE_DDLS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS memories (
      memory_id VARCHAR(40) PRIMARY KEY,
      user_id VARCHAR(64) NOT NULL,
      agent_id VARCHAR(64) NOT NULL DEFAULT '',
      kind VARCHAR(16) NOT NULL,
      content TEXT NOT NULL,
      tags_json TEXT NOT NULL DEFAULT '[]',
      embedding BLOB,
      embedding_dim INTEGER,
      salience REAL NOT NULL DEFAULT 0.5,
      status VARCHAR(16) NOT NULL DEFAULT 'active',
      source_thread_id VARCHAR(190) NOT NULL DEFAULT '',
      source_refs_json TEXT NOT NULL DEFAULT '[]',
      created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      last_accessed_at TIMESTAMP,
      access_count INTEGER NOT NULL DEFAULT 0
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS memory_blocks (
      user_id VARCHAR(64) NOT NULL,
      agent_id VARCHAR(64) NOT NULL,
      label VARCHAR(64) NOT NULL,
      content TEXT NOT NULL,
      char_limit INTEGER NOT NULL DEFAULT 2000,
      version INTEGER NOT NULL DEFAULT 0,
      updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      PRIMARY KEY (user_id, agent_id, label)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS session_summaries (
      user_id VARCHAR(64) NOT NULL,
      thread_id VARCHAR(190) NOT NULL,
      summary TEXT NOT NULL,
      agent_id VARCHAR(64) NOT NULL DEFAULT '',
      covered_message_count INTEGER NOT NULL DEFAULT 0,
      updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      PRIMARY KEY (user_id, thread_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS doc_chunks (
      chunk_id VARCHAR(40) PRIMARY KEY,
      file_id VARCHAR(32) NOT NULL,
      thread_id VARCHAR(190) NOT NULL DEFAULT '',
      user_id VARCHAR(64) NOT NULL,
      agent_id VARCHAR(64) NOT NULL DEFAULT '',
      ordinal INTEGER NOT NULL DEFAULT 0,
      text TEXT NOT NULL,
      embedding BLOB,
      embedding_dim INTEGER,
      created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS memory_ops (
      op_id VARCHAR(40) PRIMARY KEY,
      op VARCHAR(32) NOT NULL,
      user_id VARCHAR(64) NOT NULL DEFAULT '',
      agent_id VARCHAR(64) NOT NULL DEFAULT '',
      thread_id VARCHAR(190) NOT NULL DEFAULT '',
      detail_json TEXT NOT NULL DEFAULT '{}',
      status VARCHAR(16) NOT NULL DEFAULT 'ok',
      error_text TEXT,
      duration_ms INTEGER NOT NULL DEFAULT 0,
      created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
)

_SQLITE_INDEXES: tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS idx_mem_scope ON memories (user_id, status)",
    "CREATE INDEX IF NOT EXISTS idx_mem_agent ON memories (user_id, agent_id)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_user ON doc_chunks (user_id)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_file ON doc_chunks (file_id)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_thread ON doc_chunks (thread_id)",
)

_MYSQL_DDLS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS memories (
      memory_id VARCHAR(40) PRIMARY KEY,
      user_id VARCHAR(64) NOT NULL,
      agent_id VARCHAR(64) NOT NULL DEFAULT '',
      kind VARCHAR(16) NOT NULL,
      content LONGTEXT NOT NULL,
      tags_json LONGTEXT NOT NULL,
      embedding BLOB,
      embedding_dim INT NULL,
      salience DOUBLE NOT NULL DEFAULT 0.5,
      status VARCHAR(16) NOT NULL DEFAULT 'active',
      source_thread_id VARCHAR(190) NOT NULL DEFAULT '',
      source_refs_json LONGTEXT NOT NULL,
      created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      last_accessed_at TIMESTAMP NULL,
      access_count INT NOT NULL DEFAULT 0,
      KEY idx_mem_scope (user_id, status),
      KEY idx_mem_agent (user_id, agent_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS memory_blocks (
      user_id VARCHAR(64) NOT NULL,
      agent_id VARCHAR(64) NOT NULL,
      label VARCHAR(64) NOT NULL,
      content LONGTEXT NOT NULL,
      char_limit INT NOT NULL DEFAULT 2000,
      version INT NOT NULL DEFAULT 0,
      updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      PRIMARY KEY (user_id, agent_id, label)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS session_summaries (
      user_id VARCHAR(64) NOT NULL,
      thread_id VARCHAR(190) NOT NULL,
      summary LONGTEXT NOT NULL,
      agent_id VARCHAR(64) NOT NULL DEFAULT '',
      covered_message_count INT NOT NULL DEFAULT 0,
      updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      PRIMARY KEY (user_id, thread_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS doc_chunks (
      chunk_id VARCHAR(40) PRIMARY KEY,
      file_id VARCHAR(32) NOT NULL,
      thread_id VARCHAR(190) NOT NULL DEFAULT '',
      user_id VARCHAR(64) NOT NULL,
      agent_id VARCHAR(64) NOT NULL DEFAULT '',
      ordinal INT NOT NULL DEFAULT 0,
      text LONGTEXT NOT NULL,
      embedding BLOB,
      embedding_dim INT NULL,
      created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      KEY idx_chunks_user (user_id),
      KEY idx_chunks_file (file_id),
      KEY idx_chunks_thread (thread_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS memory_ops (
      op_id VARCHAR(40) NOT NULL,
      seq BIGINT NOT NULL AUTO_INCREMENT,
      op VARCHAR(32) NOT NULL,
      user_id VARCHAR(64) NOT NULL DEFAULT '',
      agent_id VARCHAR(64) NOT NULL DEFAULT '',
      thread_id VARCHAR(190) NOT NULL DEFAULT '',
      detail_json LONGTEXT NOT NULL,
      status VARCHAR(16) NOT NULL DEFAULT 'ok',
      error_text LONGTEXT,
      duration_ms INT NOT NULL DEFAULT 0,
      created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      PRIMARY KEY (op_id),
      KEY idx_ops_seq (seq)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
)


# ──────────────────── SQL 实现的共享层（DML + 行编解码） ────────────────────


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

    _SUMMARY_COLUMNS = (
        "user_id, thread_id, summary, agent_id, covered_message_count, updated_at"
    )

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
        "chunk_id, file_id, thread_id, user_id, agent_id, ordinal, text, embedding,"
        " embedding_dim, created_at"
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
            embedding=bytes(row[7]) if row[7] is not None else None,
            embedding_dim=row[8],
            created_at=self._from_sql_ts(row[9]),
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
                f"unknown memory status {record.status!r}; "
                f"expected one of {KNOWN_MEMORY_STATUSES}"
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
        rowcount = await self._execute(
            "DELETE FROM memories WHERE memory_id = ?", (memory_id,)
        )
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
    ) -> list[MemoryRecord]:
        sql = f"SELECT {self._MEMORY_COLUMNS} FROM memories WHERE user_id = ?"
        params: list[Any] = [user_id]
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

    # -- 线程级联 --------------------------------------------------------------------
    async def delete_for_thread(self, thread_id: str) -> None:
        # memories 刻意不删：跨会话记忆独立于线程存活（出处仍在
        # source_thread_id）；随线程消亡的是它的摘要与知识库分块。
        await self._execute("DELETE FROM doc_chunks WHERE thread_id = ?", (thread_id,))
        await self._execute("DELETE FROM session_summaries WHERE thread_id = ?", (thread_id,))

    # -- 操作审计 --------------------------------------------------------------------
    async def record_op(self, op: MemoryOp) -> None:
        await self._execute(
            f"INSERT INTO memory_ops ({self._OP_COLUMNS}) VALUES"
            " (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                op.op_id,
                op.op,
                op.user_id,
                op.agent_id,
                op.thread_id,
                json.dumps(op.detail, ensure_ascii=False),
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


class MemoryMemoryStore:
    """进程内字典实现：memory 后端（重启即丢）与测试用。"""

    def __init__(self) -> None:
        self._memories: dict[str, MemoryRecord] = {}
        self._blocks: dict[tuple[str, str, str], MemoryBlock] = {}
        self._summaries: dict[tuple[str, str], SessionSummary] = {}
        self._chunks: dict[str, DocChunk] = {}
        # (op, 插入序号) 二元组：同秒并列时按插入顺序稳定排序。
        self._ops: list[tuple[MemoryOp, int]] = []
        self._op_counter = 0

    async def upsert_memory(self, record: MemoryRecord) -> None:
        if record.kind not in KNOWN_MEMORY_KINDS:
            raise MemoryStoreError(
                f"unknown memory kind {record.kind!r}; expected one of {KNOWN_MEMORY_KINDS}"
            )
        if record.status not in KNOWN_MEMORY_STATUSES:
            raise MemoryStoreError(
                f"unknown memory status {record.status!r}; "
                f"expected one of {KNOWN_MEMORY_STATUSES}"
            )
        self._memories[record.memory_id] = record

    async def get_memory(self, memory_id: str) -> MemoryRecord | None:
        return self._memories.get(memory_id)

    async def delete_memory(self, memory_id: str) -> bool:
        return self._memories.pop(memory_id, None) is not None

    async def list_memories(
        self,
        user_id: str,
        *,
        agent_id: str | None = None,
        include_global: bool = True,
        kinds: Sequence[str] | None = None,
        statuses: Sequence[str] = ("active",),
        limit: int = 2000,
    ) -> list[MemoryRecord]:
        matched = [
            record
            for record in self._memories.values()
            if record.user_id == user_id
            and record.status in statuses
            and (kinds is None or record.kind in kinds)
            and (
                agent_id is None
                or record.agent_id == agent_id
                or (include_global and record.agent_id == "*")
            )
        ]
        matched.sort(key=lambda r: r.updated_at, reverse=True)
        return matched[: max(1, limit)]

    async def touch_memories(self, memory_ids: Sequence[str]) -> None:
        now = time.time()
        for mid in memory_ids:
            record = self._memories.get(mid)
            if record is None:
                continue
            self._memories[mid] = replace(
                record,
                last_accessed_at=now,
                access_count=record.access_count + 1,
            )

    async def upsert_block(self, block: MemoryBlock) -> None:
        self._blocks[(block.user_id, block.agent_id, block.label)] = block

    async def get_block(self, user_id: str, agent_id: str, label: str) -> MemoryBlock | None:
        return self._blocks.get((user_id, agent_id, label))

    async def list_blocks(self, user_id: str, agent_id: str) -> list[MemoryBlock]:
        blocks = [
            block
            for (uid, aid, _), block in self._blocks.items()
            if uid == user_id and aid == agent_id
        ]
        blocks.sort(key=lambda b: b.label)
        return blocks

    async def delete_block(self, user_id: str, agent_id: str, label: str) -> bool:
        return self._blocks.pop((user_id, agent_id, label), None) is not None

    async def get_summary(self, user_id: str, thread_id: str) -> SessionSummary | None:
        return self._summaries.get((user_id, thread_id))

    async def upsert_summary(self, summary: SessionSummary) -> None:
        self._summaries[(summary.user_id, summary.thread_id)] = summary

    async def put_chunks(self, chunks: Sequence[DocChunk]) -> None:
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk

    async def list_chunks(
        self,
        user_id: str,
        *,
        agent_id: str | None = None,
        file_id: str | None = None,
        limit: int = 2000,
    ) -> list[DocChunk]:
        matched = [
            chunk
            for chunk in self._chunks.values()
            if chunk.user_id == user_id
            and (agent_id is None or chunk.agent_id == agent_id or chunk.agent_id == "*")
            and (file_id is None or chunk.file_id == file_id)
        ]
        matched.sort(key=lambda c: (c.file_id, c.ordinal))
        return matched[: max(1, limit)]

    async def count_chunks_for_file(self, file_id: str) -> int:
        return sum(1 for chunk in self._chunks.values() if chunk.file_id == file_id)

    async def delete_chunks_for_file(self, file_id: str) -> int:
        stale = [cid for cid, chunk in self._chunks.items() if chunk.file_id == file_id]
        for cid in stale:
            del self._chunks[cid]
        return len(stale)

    async def delete_for_thread(self, thread_id: str) -> None:
        self._chunks = {
            cid: chunk for cid, chunk in self._chunks.items() if chunk.thread_id != thread_id
        }
        self._summaries = {
            key: summary for key, summary in self._summaries.items()
            if summary.thread_id != thread_id
        }

    async def record_op(self, op: MemoryOp) -> None:
        self._op_counter += 1
        self._ops.append((op, self._op_counter))

    async def list_ops(self, user_id: str | None = None, limit: int = 100) -> list[MemoryOp]:
        pairs = [(op, seq) for op, seq in self._ops if user_id is None or op.user_id == user_id]
        pairs.sort(key=lambda pair: (pair[0].created_at, pair[1]), reverse=True)
        return [op for op, _ in pairs[: max(1, limit)]]


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
            return time.time()

    async def aclose(self) -> None:
        await self._conn.close()


class MysqlMemoryStore(_SqlMemoryStoreBase):
    """MySQL 记忆表：aiomysql 按操作连接（写入频率低，无需连接池）。"""

    _PLACEHOLDER = "%s"
    _OPS_TIEBREAK = "seq DESC"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._table_ready = False

    async def _connect(self) -> Any:
        import aiomysql  # type: ignore[import-untyped]

        s = self._settings
        return await aiomysql.connect(
            host=s.checkpointer_mysql_host,
            port=s.checkpointer_mysql_port,
            user=s.checkpointer_mysql_user,
            password=s.checkpointer_mysql_password.get_secret_value(),
            db=s.checkpointer_mysql_database,
            autocommit=True,
        )

    async def _ensure_tables(self, cursor: Any) -> None:
        if self._table_ready:
            return
        for ddl in _MYSQL_DDLS:
            await cursor.execute(ddl)
        self._table_ready = True

    async def _execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        conn = await self._connect()
        try:
            async with conn.cursor() as cur:
                await self._ensure_tables(cur)
                await cur.execute(self._sql(sql), tuple(params))
                return cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
        finally:
            conn.close()

    async def _fetch_all(self, sql: str, params: Sequence[Any] = ()) -> list[tuple[Any, ...]]:
        conn = await self._connect()
        try:
            async with conn.cursor() as cur:
                await self._ensure_tables(cur)
                await cur.execute(self._sql(sql), tuple(params))
                return list(await cur.fetchall())
        finally:
            conn.close()

    def _to_sql_ts(self, seconds: float) -> Any:
        return datetime.fromtimestamp(seconds)

    def _from_sql_ts(self, value: Any) -> float:
        if isinstance(value, datetime):
            return value.timestamp()
        try:
            return datetime.fromisoformat(str(value)).timestamp()
        except (TypeError, ValueError):
            return time.time()


async def build_memory_store(settings: Settings) -> MemoryStore | None:
    """按 settings 装配记忆存储（跟随 checkpointer 后端）。"""
    backend = settings.checkpointer_backend
    if backend == "memory":
        return MemoryMemoryStore()
    if backend == "sqlite":
        return await SqliteMemoryStore.create(settings.checkpointer_sqlite_path)
    if backend == "mysql":
        return MysqlMemoryStore(settings)
    logger.warning("memory: unknown checkpointer backend %r; memory store disabled", backend)
    return None


__all__ = [
    "KNOWN_MEMORY_KINDS",
    "KNOWN_MEMORY_STATUSES",
    "DocChunk",
    "MemoryBlock",
    "MemoryMemoryStore",
    "MemoryOp",
    "MemoryRecord",
    "MemoryStore",
    "MemoryStoreError",
    "MysqlMemoryStore",
    "SessionSummary",
    "SqliteMemoryStore",
    "build_memory_store",
    "decode_embedding",
    "encode_embedding",
]
