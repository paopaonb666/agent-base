"""上传附件存储（附件对话功能）：原始字节 + 解析文本双持久化。

设计要点：

- **原始字节入库**（BLOB/LONGBLOB）：与提取文本并存——未来可以换更好的
  解析器重新解析、可以扩展下载端点；chat-agent 把原始文件写磁盘却从不
  使用（路径字段是死数据），这里统一收进存储后端，零磁盘管理。
- **thread_id 发送时绑定**：上传发生在对话线程生成之前（新对话的首个
  附件），行先以空 thread_id 入库，invoke 引用时 ``bind_thread`` 回填；
  24 小时未绑定的孤儿行由 ``purge_orphans`` 机会式清理。
- **全部异步接口**：上传/invoke/删除都运行在事件循环里，sqlite 用
  aiosqlite 持久连接、mysql 用 aiomysql 按操作连接，无需线程桥接。
- **存储后端跟随 checkpointer**（memory/sqlite/mysql），与 toollog 同款
  装配语义；sqlite/mysql 的建表 DDL 与 alembic 0003 迁移一致，未跑迁移
  的库也能自举。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - import avoided at runtime
    from agent_base.core.config import Settings

logger = logging.getLogger(__name__)

# 未绑定线程的孤儿附件保留时长（小时），超过后在上传时机会式清理。
ORPHAN_MAX_AGE_HOURS = 24

_SQLITE_DDL = """
CREATE TABLE IF NOT EXISTS uploaded_files (
  file_id VARCHAR(32) PRIMARY KEY,
  thread_id VARCHAR(190) NOT NULL DEFAULT '',
  module VARCHAR(64) NOT NULL DEFAULT '',
  user_id VARCHAR(64) NOT NULL DEFAULT '',
  filename VARCHAR(255) NOT NULL,
  format VARCHAR(16) NOT NULL,
  pages INTEGER,
  paragraphs INTEGER,
  truncated INTEGER NOT NULL DEFAULT 0,
  text_len INTEGER NOT NULL DEFAULT 0,
  warning VARCHAR(500) NOT NULL DEFAULT '',
  extracted_text TEXT,
  content BLOB,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""

_MYSQL_DDL = """
CREATE TABLE IF NOT EXISTS uploaded_files (
  file_id VARCHAR(32) PRIMARY KEY,
  thread_id VARCHAR(190) NOT NULL DEFAULT '',
  module VARCHAR(64) NOT NULL DEFAULT '',
  user_id VARCHAR(64) NOT NULL DEFAULT '',
  filename VARCHAR(255) NOT NULL,
  format VARCHAR(16) NOT NULL,
  pages INT NULL,
  paragraphs INT NULL,
  truncated SMALLINT NOT NULL DEFAULT 0,
  text_len INT NOT NULL DEFAULT 0,
  warning VARCHAR(500) NOT NULL DEFAULT '',
  extracted_text LONGTEXT,
  content LONGBLOB,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""


@dataclass(frozen=True)
class UploadedFileInfo:
    """一条上传附件：原始字节、解析文本与元信息。"""

    file_id: str
    filename: str
    format: str
    pages: int | None = None
    paragraphs: int | None = None
    truncated: bool = False
    text_len: int = 0
    extracted_text: str = ""
    content: bytes = b""
    warning: str = ""
    thread_id: str = ""
    module: str = ""
    # 上传者的记忆身份（X-User-Id；撤销文件时的属主判定依据）。
    # 旧库行缺省为空串：视为"无主遗留"，允许任意已验身份用户撤销。
    user_id: str = ""
    created_at: float = field(default_factory=time.time)

    def meta(self) -> dict[str, Any]:
        """对外元信息（不含提取文本与原始字节）。"""
        return {
            "file_id": self.file_id,
            "filename": self.filename,
            "format": self.format,
            "pages": self.pages,
            "text_len": self.text_len,
            "truncated": self.truncated,
            "user_id": self.user_id,
            **({"warning": self.warning} if self.warning else {}),
        }


@runtime_checkable
class UploadedFileStore(Protocol):
    """附件存储的最小接口；实现必须保证并发安全。"""

    async def save(self, info: UploadedFileInfo) -> None: ...

    async def get_many(self, file_ids: list[str]) -> list[UploadedFileInfo]: ...

    async def list_for_user(
        self, user_id: str, *, module: str | None = None, limit: int = 200
    ) -> list[UploadedFileInfo]: ...

    async def bind_thread(self, file_ids: list[str], thread_id: str) -> None: ...

    async def delete_for_thread(self, thread_id: str) -> None: ...

    async def delete_file(self, file_id: str) -> bool: ...

    async def purge_orphans(self, max_age_hours: int = ORPHAN_MAX_AGE_HOURS) -> int: ...


class MemoryUploadedFileStore:
    """进程内字典；memory 后端（重启即丢）与测试用。"""

    def __init__(self) -> None:
        self._files: dict[str, UploadedFileInfo] = {}

    async def save(self, info: UploadedFileInfo) -> None:
        self._files[info.file_id] = info

    async def get_many(self, file_ids: list[str]) -> list[UploadedFileInfo]:
        found = [self._files[fid] for fid in file_ids if fid in self._files]
        # 按 get_many 的请求顺序返回，注入顺序与前端附件列表一致。
        found.sort(key=lambda info: file_ids.index(info.file_id))
        return found

    async def bind_thread(self, file_ids: list[str], thread_id: str) -> None:
        for fid in file_ids:
            if fid in self._files:
                self._files[fid] = replace(self._files[fid], thread_id=thread_id)

    async def list_for_user(
        self, user_id: str, *, module: str | None = None, limit: int = 200
    ) -> list[UploadedFileInfo]:
        matched = [
            info
            for info in self._files.values()
            if info.user_id == user_id and (module is None or info.module == module)
        ]
        matched.sort(key=lambda info: info.created_at, reverse=True)
        return matched[: max(1, limit)]

    async def delete_for_thread(self, thread_id: str) -> None:
        self._files = {
            fid: info for fid, info in self._files.items() if info.thread_id != thread_id
        }

    async def delete_file(self, file_id: str) -> bool:
        return self._files.pop(file_id, None) is not None

    async def purge_orphans(self, max_age_hours: int = ORPHAN_MAX_AGE_HOURS) -> int:
        cutoff = time.time() - max_age_hours * 3600
        stale = [
            fid
            for fid, info in self._files.items()
            if not info.thread_id and info.created_at < cutoff
        ]
        for fid in stale:
            del self._files[fid]
        return len(stale)


class SqliteUploadedFileStore:
    """sqlite 附件表：与对话状态同一文件，aiosqlite 持久连接。"""

    def __init__(self, connection: Any) -> None:
        self._conn = connection

    @classmethod
    async def create(cls, path: str) -> SqliteUploadedFileStore:
        import aiosqlite

        conn = await aiosqlite.connect(path)
        await conn.execute(_SQLITE_DDL)
        # 旧库补列（sqlite 无 alembic，运行时自迁移）：user_id 是后加的。
        cursor = await conn.execute("PRAGMA table_info(uploaded_files)")
        columns = {row[1] for row in await cursor.fetchall()}
        if "user_id" not in columns:
            await conn.execute(
                "ALTER TABLE uploaded_files"
                " ADD COLUMN user_id VARCHAR(64) NOT NULL DEFAULT ''"
            )
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_uf_thread ON uploaded_files (thread_id)")
        await conn.commit()
        return cls(conn)

    @staticmethod
    def _to_info(row: Any) -> UploadedFileInfo:
        return UploadedFileInfo(
            file_id=row[0],
            thread_id=row[1],
            module=row[2],
            user_id=row[3] or "",
            filename=row[4],
            format=row[5],
            pages=row[6],
            paragraphs=row[7],
            truncated=bool(row[8]),
            text_len=row[9],
            warning=row[10],
            extracted_text=row[11] or "",
            content=bytes(row[12]) if row[12] is not None else b"",
        )

    async def save(self, info: UploadedFileInfo) -> None:
        await self._conn.execute(
            "INSERT OR REPLACE INTO uploaded_files (file_id, thread_id, module, user_id,"
            " filename, format, pages, paragraphs, truncated, text_len, warning,"
            " extracted_text, content, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                info.file_id,
                info.thread_id,
                info.module,
                info.user_id,
                info.filename,
                info.format,
                info.pages,
                info.paragraphs,
                int(info.truncated),
                info.text_len,
                info.warning,
                info.extracted_text,
                info.content,
                datetime.fromtimestamp(info.created_at).isoformat(sep=" ", timespec="seconds"),
            ),
        )
        await self._conn.commit()

    async def get_many(self, file_ids: list[str]) -> list[UploadedFileInfo]:
        placeholders = ",".join("?" for _ in file_ids)
        rows = await self._conn.execute_fetchall(
            f"SELECT file_id, thread_id, module, user_id, filename, format, pages,"
            f" paragraphs, truncated, text_len, warning, extracted_text, content"
            f" FROM uploaded_files WHERE file_id IN ({placeholders})",
            file_ids,
        )
        infos = {info.file_id: info for info in map(self._to_info, rows)}
        found = [infos[fid] for fid in file_ids if fid in infos]
        return found

    async def bind_thread(self, file_ids: list[str], thread_id: str) -> None:
        placeholders = ",".join("?" for _ in file_ids)
        await self._conn.execute(
            f"UPDATE uploaded_files SET thread_id = ? WHERE file_id IN ({placeholders})",
            [thread_id, *file_ids],
        )
        await self._conn.commit()

    async def delete_for_thread(self, thread_id: str) -> None:
        await self._conn.execute("DELETE FROM uploaded_files WHERE thread_id = ?", (thread_id,))
        await self._conn.commit()

    async def purge_orphans(self, max_age_hours: int = ORPHAN_MAX_AGE_HOURS) -> int:
        cutoff = (datetime.now() - timedelta(hours=max_age_hours)).isoformat(
            sep=" ", timespec="seconds"
        )
        cursor = await self._conn.execute(
            "DELETE FROM uploaded_files WHERE thread_id = '' AND created_at < ?",
            (cutoff,),
        )
        await self._conn.commit()
        return cursor.rowcount if cursor.rowcount and cursor.rowcount > 0 else 0

    async def delete_file(self, file_id: str) -> bool:
        cursor = await self._conn.execute(
            "DELETE FROM uploaded_files WHERE file_id = ?", (file_id,)
        )
        await self._conn.commit()
        return bool(cursor.rowcount and cursor.rowcount > 0)

    async def list_for_user(
        self, user_id: str, *, module: str | None = None, limit: int = 200
    ) -> list[UploadedFileInfo]:
        sql = (
            "SELECT file_id, thread_id, module, user_id, filename, format, pages,"
            " paragraphs, truncated, text_len, warning, extracted_text, content"
            " FROM uploaded_files WHERE user_id = ?"
        )
        params: list[Any] = [user_id]
        if module is not None:
            sql += " AND module = ?"
            params.append(module)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(max(1, limit))
        rows = await self._conn.execute_fetchall(sql, tuple(params))
        return [self._to_info(row) for row in rows]

    async def aclose(self) -> None:
        await self._conn.close()


class MysqlUploadedFileStore:
    """MySQL 附件表：aiomysql 按操作连接（写入频率低，无需连接池）。"""

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

    @staticmethod
    def _to_info(row: Any) -> UploadedFileInfo:
        created = row[13]
        return UploadedFileInfo(
            file_id=row[0],
            thread_id=row[1],
            module=row[2],
            user_id=row[3] or "",
            filename=row[4],
            format=row[5],
            pages=row[6],
            paragraphs=row[7],
            truncated=bool(row[8]),
            text_len=row[9],
            warning=row[10] or "",
            extracted_text=row[11] or "",
            content=bytes(row[12]) if row[12] is not None else b"",
            created_at=created.timestamp() if isinstance(created, datetime) else time.time(),
        )

    async def save(self, info: UploadedFileInfo) -> None:
        conn = await self._connect()
        try:
            async with conn.cursor() as cur:
                if not self._table_ready:
                    await cur.execute(_MYSQL_DDL)
                    self._table_ready = True
                await cur.execute(
                    "INSERT INTO uploaded_files (file_id, thread_id, module, user_id,"
                    " filename, format, pages, paragraphs, truncated, text_len, warning,"
                    " extracted_text, content) VALUES (%s, %s, %s, %s, %s, %s, %s, %s,"
                    " %s, %s, %s, %s, %s)",
                    (
                        info.file_id,
                        info.thread_id,
                        info.module,
                        info.user_id,
                        info.filename,
                        info.format,
                        info.pages,
                        info.paragraphs,
                        int(info.truncated),
                        info.text_len,
                        info.warning,
                        info.extracted_text,
                        info.content,
                    ),
                )
        finally:
            conn.close()

    _SELECT = (
        "SELECT file_id, thread_id, module, user_id, filename, format, pages, paragraphs,"
        " truncated, text_len, warning, extracted_text, content, created_at"
        " FROM uploaded_files"
    )

    async def get_many(self, file_ids: list[str]) -> list[UploadedFileInfo]:
        conn = await self._connect()
        try:
            async with conn.cursor() as cur:
                placeholders = ",".join("%s" for _ in file_ids)
                await cur.execute(f"{self._SELECT} WHERE file_id IN ({placeholders})", file_ids)
                rows = await cur.fetchall()
            infos = {info.file_id: info for info in map(self._to_info, rows)}
            found = [infos[fid] for fid in file_ids if fid in infos]
            return found
        finally:
            conn.close()

    async def bind_thread(self, file_ids: list[str], thread_id: str) -> None:
        conn = await self._connect()
        try:
            async with conn.cursor() as cur:
                placeholders = ",".join("%s" for _ in file_ids)
                await cur.execute(
                    f"UPDATE uploaded_files SET thread_id = %s WHERE file_id IN ({placeholders})",
                    [thread_id, *file_ids],
                )
        finally:
            conn.close()

    async def delete_for_thread(self, thread_id: str) -> None:
        conn = await self._connect()
        try:
            async with conn.cursor() as cur:
                await cur.execute("DELETE FROM uploaded_files WHERE thread_id = %s", (thread_id,))
        finally:
            conn.close()

    async def purge_orphans(self, max_age_hours: int = ORPHAN_MAX_AGE_HOURS) -> int:
        conn = await self._connect()
        try:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM uploaded_files WHERE thread_id = '' AND created_at < %s",
                    (datetime.now() - timedelta(hours=max_age_hours),),
                )
                return cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
        finally:
            conn.close()

    async def delete_file(self, file_id: str) -> bool:
        conn = await self._connect()
        try:
            async with conn.cursor() as cur:
                await cur.execute("DELETE FROM uploaded_files WHERE file_id = %s", (file_id,))
                return bool(cur.rowcount and cur.rowcount > 0)
        finally:
            conn.close()

    async def list_for_user(
        self, user_id: str, *, module: str | None = None, limit: int = 200
    ) -> list[UploadedFileInfo]:
        sql = (
            "SELECT file_id, thread_id, module, user_id, filename, format, pages,"
            " paragraphs, truncated, text_len, warning, extracted_text, content, created_at"
            " FROM uploaded_files WHERE user_id = %s"
        )
        params: list[Any] = [user_id]
        if module is not None:
            sql += " AND module = %s"
            params.append(module)
        sql += " ORDER BY created_at DESC LIMIT %s"
        params.append(max(1, limit))
        conn = await self._connect()
        try:
            async with conn.cursor() as cur:
                await cur.execute(sql, tuple(params))
                return [self._to_info(row) for row in await cur.fetchall()]
        finally:
            conn.close()


async def build_uploaded_file_store(settings: Settings) -> UploadedFileStore | None:
    """按 settings 装配附件存储（跟随 checkpointer 后端）。"""
    backend = settings.checkpointer_backend
    if backend == "memory":
        return MemoryUploadedFileStore()
    if backend == "sqlite":
        return await SqliteUploadedFileStore.create(settings.checkpointer_sqlite_path)
    if backend == "mysql":
        return MysqlUploadedFileStore(settings)
    logger.warning("filestore: unknown checkpointer backend %r; attachments disabled", backend)
    return None


__all__ = [
    "MemoryUploadedFileStore",
    "MysqlUploadedFileStore",
    "SqliteUploadedFileStore",
    "UploadedFileInfo",
    "UploadedFileStore",
    "build_uploaded_file_store",
]
