"""对话状态装配：checkpointer（阶段 3）。

仅包含**短期**对话状态——thread 级别的持久化，让一次对话在轮次之间得以
保留（使用 sqlite 后端时，还能跨进程重启保留）。长期记忆（跨对话、
向量库、抽取）在 ``agent_base.memory`` 子系统落地（M6），其存储后端
同样跟随这里选择的 ``CHECKPOINTER_BACKEND``。

后端（``CHECKPOINTER_BACKEND``）：
- ``memory``  —— ``InMemorySaver``，零依赖，仅进程生命周期内有效。
- ``sqlite``  —— 基于 ``CHECKPOINTER_SQLITE_PATH`` 的 ``AsyncSqliteSaver``。

整个运行时是异步优先的（CLI 把会话包在 ``asyncio.run`` 里，服务器运行在
uvicorn 的循环上），所以 sqlite 后端使用异步 saver：同步的
``SqliteSaver`` 在 ``astream`` 下会抛出 ``NotImplementedError``。
``InMemorySaver`` 则同时支持两种模式。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

if TYPE_CHECKING:  # pragma: no cover - import avoided at runtime
    from agent_base.core.config import Settings


async def build_checkpointer(settings: Settings) -> BaseCheckpointSaver[Any]:
    """装配由 ``settings.checkpointer.backend`` 选定的 checkpointer。

    必须在将要运行这些图的事件循环中被 await——sqlite / mysql 后端会把
    连接绑定到调用它的循环，因此在一个临时循环中构造它会导致后续使用
    出错。``create_runtime`` 之所以是异步的，正是因为这个原因。
    """
    if settings.checkpointer.backend == "memory":
        return InMemorySaver()
    if settings.checkpointer.backend == "sqlite":
        return await _build_sqlite_checkpointer(settings)
    if settings.checkpointer.backend == "mysql":
        return await _build_mysql_checkpointer(settings)
    raise ValueError(f"unknown CHECKPOINTER_BACKEND {settings.checkpointer.backend!r}")


async def _build_sqlite_checkpointer(settings: Settings) -> BaseCheckpointSaver[Any]:
    # 延迟导入：只有选择 sqlite 时才需要 aiosqlite。
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    # 先预检路径可写性：否则坏路径会以原始的 aiosqlite.OperationalError
    # 在会话中途爆出，而不是启动时一条可读的配置错误。
    from agent_base.core.config import SettingsError

    path = Path(settings.checkpointer.sqlite_path)
    parent = path.parent if str(path.parent) else Path(".")
    if not parent.is_dir() or not os.access(parent, os.W_OK):
        raise SettingsError(
            f"CHECKPOINTER_SQLITE_PATH {settings.checkpointer.sqlite_path!r} is not "
            "writable: the parent directory does not exist or denies write access"
        )

    # 注意：刻意不使用 AsyncSqliteSaver.from_conn_string——它的上下文
    # 管理器在连接被垃圾回收时会立即关闭连接，这会在会话中途悄无声息地
    # 杀死 saver。一个直接 await 的连接会一直存活到
    # close_checkpointer()（或进程退出）。
    conn = await aiosqlite.connect(settings.checkpointer.sqlite_path)
    saver = AsyncSqliteSaver(conn)
    await saver.setup()
    return saver


# MySQL 保活间隔：默认 wait_timeout 是 8 小时，30 分钟一次 ping 远在
# 其之前，连接基本不会被服务端掐断；即便被掐，ping(reconnect=True)
# 也会自动重连。
MYSQL_KEEPALIVE_SECONDS = 1800.0


async def _mysql_keepalive(conn: Any, interval_seconds: float) -> None:
    """周期性 ping MySQL 连接（掉线自动重连），直到被取消。"""
    logger = logging.getLogger(__name__)
    was_ok = True
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            await conn.ping(reconnect=True)
        except Exception:
            # ping 失败不致命：下一次循环再试；查询层的失败由调用方处理。
            # 但不能完全静默：状态变化时记一条 WARNING，否则 DB 掉线后
            # 唯一线索是请求层失败，无从诊断。
            if was_ok:
                logger.warning("mysql keepalive: ping failed; will retry next cycle")
                was_ok = False
            continue
        if not was_ok:
            logger.info("mysql keepalive: connection recovered")
            was_ok = True


async def _build_mysql_checkpointer(settings: Settings) -> BaseCheckpointSaver[Any]:
    """构建 MySQL checkpointer（按服务器版本选择官方 saver 或 5.7 兼容 saver）。

    与 sqlite 后端同理，连接直接 await 并保持存活到 ``close_checkpointer()``；
    刻意不使用 ``from_conn_string`` 上下文管理器，以免连接被提前关闭。
    另起一个后台保活任务周期性 ping 连接（掉线自动重连），规避
    ``wait_timeout`` 掐断长驻连接的问题——任务句柄挂在 saver 上，
    由 ``close_checkpointer`` 负责取消。官方 saver 要求 MySQL >= 8.0.19；
    5.7 服务器自动改用 ``MySQL57Saver``（extensions/mysql57.py）。
    """
    # 延迟导入：只有选择 mysql 时才需要 aiomysql / 官方 saver 依赖。
    import aiomysql  # type: ignore[import-untyped]

    from agent_base.extensions.mysql57 import MySQL57Saver, probe_mysql_major_version

    conn = await aiomysql.connect(
        host=settings.checkpointer.mysql_host,
        port=settings.checkpointer.mysql_port,
        user=settings.checkpointer.mysql_user,
        password=settings.checkpointer.mysql_password.get_secret_value(),
        db=settings.checkpointer.mysql_database,
        autocommit=True,
    )
    saver: BaseCheckpointSaver[Any]
    if await probe_mysql_major_version(conn) < 8:
        saver = MySQL57Saver(conn=conn)
    else:
        from langgraph.checkpoint.mysql.aio import AIOMySQLSaver

        saver = AIOMySQLSaver(conn=conn)
    await saver.setup()
    saver._keepalive_task = asyncio.create_task(  # type: ignore[attr-defined]
        _mysql_keepalive(conn, MYSQL_KEEPALIVE_SECONDS)
    )
    return saver


async def close_checkpointer(checkpointer: BaseCheckpointSaver[Any]) -> None:
    """尽力而为地拆除（测试会创建很多 saver；进程可以跳过）。"""
    # 取消 MySQL 保活任务（如果有的话），再关连接。
    task = getattr(checkpointer, "_keepalive_task", None)
    if task is not None and not task.done():
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    conn = getattr(checkpointer, "conn", None)
    # 优先使用异步的 ensure_closed（aiomysql），否则退回 close（aiosqlite）。
    closer = getattr(conn, "ensure_closed", None) or getattr(conn, "close", None)
    if closer is not None:
        result = closer()
        if hasattr(result, "__await__"):
            await result
