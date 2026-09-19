"""工具调用审计记录（M5 可观测）：成功、超时、异常全量落库。

设计要点：

- **唯一事实源在池的收口**：``core/tools._TimeoutTool`` 在 finally 里组装
  ``ToolCallRecord``（name/args/result/结局/时长）交给 recorder；无论
  工具成功、超时还是异常——超时与异常也会被记录——审计不漏一条。
- **写入绝不拖垮对话**：``record()`` 同步、非阻塞，内部各自消化并发
  （memory 加锁、sqlite 走专属写线程、mysql 经事件循环桥接异步写）；
  写失败只记日志。
- **存储后端跟随 checkpointer**：``memory`` → 进程内环形缓冲（重启即丢，
  与对话语义一致）；``sqlite`` → 与对话状态同一文件里自建
  ``tool_call_records`` 表；``mysql`` → aiomysql 写表（建表 DDL 与
  alembic ``0002_tool_call_records`` 一致，未跑迁移的库也能自举）。
- **截断防膨胀**：result 4k / args 2k / error 500 字符封顶。
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import sqlite3
import threading
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - import avoided at runtime
    from agent_base.core.config import Settings

logger = logging.getLogger(__name__)

# 字段截断上限（字符）：审计是旁路数据，超长内容进日志/全量输出即可。
RESULT_MAX_CHARS = 4000
ARGS_MAX_CHARS = 2000
ERROR_MAX_CHARS = 500
# memory 后端的进程内环形缓冲上限。
MEMORY_MAX_RECORDS = 1000

_SQLITE_DDL = """
CREATE TABLE IF NOT EXISTS tool_call_records (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  call_id VARCHAR(32) NOT NULL DEFAULT '',
  thread_id VARCHAR(190) NOT NULL DEFAULT '',
  module VARCHAR(64) NOT NULL DEFAULT '',
  request_id VARCHAR(64) NOT NULL DEFAULT '',
  tool VARCHAR(128) NOT NULL,
  args_json TEXT,
  result_text TEXT,
  status VARCHAR(16) NOT NULL,
  error_text TEXT,
  duration_ms INTEGER,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""

_MYSQL_DDL = """
CREATE TABLE IF NOT EXISTS tool_call_records (
  id BIGINT NOT NULL AUTO_INCREMENT,
  call_id VARCHAR(32) NOT NULL DEFAULT '',
  thread_id VARCHAR(190) NOT NULL DEFAULT '',
  module VARCHAR(64) NOT NULL DEFAULT '',
  request_id VARCHAR(64) NOT NULL DEFAULT '',
  tool VARCHAR(128) NOT NULL,
  args_json TEXT,
  result_text TEXT,
  status VARCHAR(16) NOT NULL,
  error_text TEXT,
  duration_ms INT,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY idx_tcr_thread (thread_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""


@dataclass(frozen=True)
class ToolCallRecord:
    """一条已完结的工具调用记录（成功/超时/异常都产生一条）。"""

    call_id: str
    tool: str
    status: str  # ok | timeout | error
    args_json: str = ""
    result_text: str = ""
    error_text: str = ""
    duration_ms: int = 0
    thread_id: str = ""
    module: str = ""
    request_id: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "tool": self.tool,
            "status": self.status,
            "args": _loads_or_raw(self.args_json),
            "result": self.result_text,
            "error": self.error_text or None,
            "duration_ms": self.duration_ms,
            "thread_id": self.thread_id,
            "module": self.module,
            "request_id": self.request_id,
        }


def _loads_or_raw(text: str) -> Any:
    try:
        return json.loads(text) if text else {}
    except json.JSONDecodeError:
        return text


@runtime_checkable
class ToolCallRecorder(Protocol):
    """审计记录器的最小接口；实现必须各自保证线程/并发安全。"""

    def record(self, record: ToolCallRecord) -> None: ...

    async def list_for_thread(self, thread_id: str, limit: int = 100) -> list[dict[str, Any]]: ...


class MemoryToolCallRecorder:
    """进程内环形缓冲；memory 后端（重启即丢）与测试用。"""

    def __init__(self) -> None:
        self._records: deque[ToolCallRecord] = deque(maxlen=MEMORY_MAX_RECORDS)
        self._lock = threading.Lock()

    def record(self, record: ToolCallRecord) -> None:
        with self._lock:
            self._records.append(record)

    async def list_for_thread(self, thread_id: str, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            matched = [r for r in self._records if r.thread_id == thread_id]
        return [r.as_dict() for r in matched[-limit:]]


class SqliteToolCallRecorder:
    """sqlite 审计表：与对话状态同一文件，专属写线程消化插入。"""

    def __init__(self, path: str) -> None:
        self._queue: queue.Queue[ToolCallRecord | None] = queue.Queue()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(_SQLITE_DDL)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tcr_thread ON tool_call_records (thread_id)"
        )
        self._conn.commit()
        self._writer = threading.Thread(
            target=self._writer_loop, name="toollog-sqlite", daemon=True
        )
        self._writer.start()

    def record(self, record: ToolCallRecord) -> None:
        self._queue.put(record)

    def _writer_loop(self) -> None:  # pragma: no cover - 线程主体在 aclose 前常驻
        while True:
            record = self._queue.get()
            if record is None:
                return
            try:
                self._conn.execute(
                    "INSERT INTO tool_call_records (call_id, thread_id, module, request_id, tool,"
                    " args_json, result_text, status, error_text, duration_ms)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        record.call_id,
                        record.thread_id,
                        record.module,
                        record.request_id,
                        record.tool,
                        record.args_json,
                        record.result_text,
                        record.status,
                        record.error_text,
                        record.duration_ms,
                    ),
                )
                self._conn.commit()
            except sqlite3.Error:
                logger.exception("toollog: sqlite insert failed")

    async def list_for_thread(self, thread_id: str, limit: int = 100) -> list[dict[str, Any]]:
        def _query() -> list[dict[str, Any]]:
            rows = self._conn.execute(
                "SELECT call_id, thread_id, module, request_id, tool, args_json, result_text,"
                " status, error_text, duration_ms FROM tool_call_records"
                " WHERE thread_id = ? ORDER BY id DESC LIMIT ?",
                (thread_id, limit),
            ).fetchall()
            return [
                {
                    "call_id": r[0],
                    "thread_id": r[1],
                    "module": r[2],
                    "request_id": r[3],
                    "tool": r[4],
                    "args": _loads_or_raw(r[5]),
                    "result": r[6],
                    "status": r[7],
                    "error": r[8] or None,
                    "duration_ms": r[9],
                }
                for r in reversed(rows)
            ]

        return await asyncio.to_thread(_query)

    async def aclose(self) -> None:
        self._queue.put(None)
        await asyncio.to_thread(self._writer.join)


class MySqlToolCallRecorder:
    """MySQL 审计表：写请求桥接到装配时的事件循环异步执行（aiomysql）。"""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._loop = asyncio.get_running_loop()
        self._table_ready = False
        # 持有 task 引用防止被事件循环 GC（asyncio 官方约定）。
        self._tasks: set[asyncio.Task[None]] = set()

    def record(self, record: ToolCallRecord) -> None:
        try:
            self._loop.call_soon_threadsafe(self._spawn_write, record)
        except RuntimeError:
            # 装配时的事件循环已关闭（进程退出中）：丢弃并记日志。
            logger.warning("toollog: event loop closed; dropping record for %s", record.tool)

    def _spawn_write(self, record: ToolCallRecord) -> None:
        task = asyncio.create_task(self._write(record))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

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

    async def _write(self, record: ToolCallRecord) -> None:
        conn = None
        try:
            conn = await self._connect()
            async with conn.cursor() as cur:
                if not self._table_ready:
                    await cur.execute(_MYSQL_DDL)
                    self._table_ready = True
                await cur.execute(
                    "INSERT INTO tool_call_records (call_id, thread_id, module, request_id, tool,"
                    " args_json, result_text, status, error_text, duration_ms)"
                    " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (
                        record.call_id,
                        record.thread_id,
                        record.module,
                        record.request_id,
                        record.tool,
                        record.args_json,
                        record.result_text,
                        record.status,
                        record.error_text,
                        record.duration_ms,
                    ),
                )
        except Exception:
            logger.exception("toollog: mysql insert failed")
        finally:
            if conn is not None:
                conn.close()

    async def list_for_thread(self, thread_id: str, limit: int = 100) -> list[dict[str, Any]]:
        conn = await self._connect()
        try:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT call_id, thread_id, module, request_id, tool, args_json, result_text,"
                    " status, error_text, duration_ms FROM tool_call_records"
                    " WHERE thread_id = %s ORDER BY id DESC LIMIT %s",
                    (thread_id, limit),
                )
                rows = await cur.fetchall()
            return [
                {
                    "call_id": r[0],
                    "thread_id": r[1],
                    "module": r[2],
                    "request_id": r[3],
                    "tool": r[4],
                    "args": _loads_or_raw(r[5]),
                    "result": r[6],
                    "status": r[7],
                    "error": r[8] or None,
                    "duration_ms": r[9],
                }
                for r in reversed(rows)
            ]
        finally:
            conn.close()


def build_tool_call_recorder(settings: Settings) -> ToolCallRecorder | None:
    """按 settings 装配审计记录器；总开关关闭或后端未知时返回 None。"""
    if not settings.toolkit.call_log_enabled:
        return None
    backend = settings.checkpointer.backend
    if backend == "memory":
        return MemoryToolCallRecorder()
    if backend == "sqlite":
        return SqliteToolCallRecorder(settings.checkpointer.sqlite_path)
    if backend == "mysql":
        return MySqlToolCallRecorder(settings)
    logger.warning("toollog: unknown checkpointer backend %r; tool call audit disabled", backend)
    return None


__all__ = [
    "ARGS_MAX_CHARS",
    "ERROR_MAX_CHARS",
    "RESULT_MAX_CHARS",
    "MemoryToolCallRecorder",
    "MySqlToolCallRecorder",
    "SqliteToolCallRecorder",
    "ToolCallRecord",
    "ToolCallRecorder",
    "build_tool_call_recorder",
]
