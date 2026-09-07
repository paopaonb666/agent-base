"""对话状态装配：checkpointer（阶段 3）。

仅包含**短期**对话状态——thread 级别的持久化，让一次对话在轮次之间得以
保留（使用 sqlite 后端时，还能跨进程重启保留）。长期记忆（跨对话、
向量库、抽取）按设计不属于基座的范围（范围红线），未来会以模块的形式
到来。

后端（``CHECKPOINTER_BACKEND``）：
- ``memory``  —— ``InMemorySaver``，零依赖，仅进程生命周期内有效。
- ``sqlite``  —— 基于 ``CHECKPOINTER_SQLITE_PATH`` 的 ``AsyncSqliteSaver``。

整个运行时是异步优先的（CLI 把会话包在 ``asyncio.run`` 里，服务器运行在
uvicorn 的循环上），所以 sqlite 后端使用异步 saver：同步的
``SqliteSaver`` 在 ``astream`` 下会抛出 ``NotImplementedError``。
``InMemorySaver`` 则同时支持两种模式。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

if TYPE_CHECKING:  # pragma: no cover - import avoided at runtime
    from agent_base.core.config import Settings


async def build_checkpointer(settings: Settings) -> BaseCheckpointSaver[Any]:
    """装配由 ``settings.checkpointer_backend`` 选定的 checkpointer。

    必须在将要运行这些图的事件循环中被 await——sqlite / mysql 后端会把
    连接绑定到调用它的循环，因此在一个临时循环中构造它会导致后续使用
    出错。``create_runtime`` 之所以是异步的，正是因为这个原因。
    """
    if settings.checkpointer_backend == "memory":
        return InMemorySaver()
    if settings.checkpointer_backend == "sqlite":
        return await _build_sqlite_checkpointer(settings)
    if settings.checkpointer_backend == "mysql":
        return await _build_mysql_checkpointer(settings)
    raise ValueError(f"unknown CHECKPOINTER_BACKEND {settings.checkpointer_backend!r}")


async def _build_sqlite_checkpointer(settings: Settings) -> BaseCheckpointSaver[Any]:
    # 延迟导入：只有选择 sqlite 时才需要 aiosqlite。
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    # 先预检路径可写性：否则坏路径会以原始的 aiosqlite.OperationalError
    # 在会话中途爆出，而不是启动时一条可读的配置错误。
    from agent_base.core.config import SettingsError

    path = Path(settings.checkpointer_sqlite_path)
    parent = path.parent if str(path.parent) else Path(".")
    if not parent.is_dir() or not os.access(parent, os.W_OK):
        raise SettingsError(
            f"CHECKPOINTER_SQLITE_PATH {settings.checkpointer_sqlite_path!r} is not "
            "writable: the parent directory does not exist or denies write access"
        )

    # 注意：刻意不使用 AsyncSqliteSaver.from_conn_string——它的上下文
    # 管理器在连接被垃圾回收时会立即关闭连接，这会在会话中途悄无声息地
    # 杀死 saver。一个直接 await 的连接会一直存活到
    # close_checkpointer()（或进程退出）。
    conn = await aiosqlite.connect(settings.checkpointer_sqlite_path)
    saver = AsyncSqliteSaver(conn)
    await saver.setup()
    return saver


class _KeepaliveMySQLConnection:
    """aiomysql 连接的保活代理。

    saver 持有贯穿进程生命周期的单个连接；MySQL 的 ``wait_timeout`` 到期
    后服务端会掐断它，长会话的下一次查询会直接失败。aiomysql 的
    ``ping(reconnect=True)`` 恰好提供"探活 + 重连"——在每次取游标前
    ping 一次即可，代价是每次查询多一个往返（checkpoint 操作频率很低，
    可忽略）。其余属性全部透传（``ensure_closed`` 等关闭路径不受影响）。
    """

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    async def cursor(self, *args: Any, **kwargs: Any) -> Any:
        await self._conn.ping(reconnect=True)
        return await self._conn.cursor(*args, **kwargs)

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._conn, attr)


async def _build_mysql_checkpointer(settings: Settings) -> BaseCheckpointSaver[Any]:
    """构建 MySQL checkpointer（异步 ``AIOMySQLSaver``，要求 MySQL >= 8.0.19）。

    与 sqlite 后端同理，连接直接 await 并保持存活到 ``close_checkpointer()``；
    刻意不使用 ``from_conn_string`` 上下文管理器，以免连接被提前关闭。
    连接用保活代理包裹，规避 wait_timeout 掐断长驻连接的问题。
    """
    # 延迟导入：只有选择 mysql 时才需要 aiomysql。
    import aiomysql  # type: ignore[import-untyped]
    from langgraph.checkpoint.mysql.aio import AIOMySQLSaver

    conn = await aiomysql.connect(
        host=settings.checkpointer_mysql_host,
        port=settings.checkpointer_mysql_port,
        user=settings.checkpointer_mysql_user,
        password=settings.checkpointer_mysql_password.get_secret_value(),
        db=settings.checkpointer_mysql_database,
        autocommit=True,
    )
    saver = AIOMySQLSaver(conn=_KeepaliveMySQLConnection(conn))
    await saver.setup()
    return saver


async def close_checkpointer(checkpointer: BaseCheckpointSaver[Any]) -> None:
    """尽力而为地拆除（测试会创建很多 saver；进程可以跳过）。"""
    conn = getattr(checkpointer, "conn", None)
    # 优先使用异步的 ensure_closed（aiomysql），否则退回 close（aiosqlite）。
    closer = getattr(conn, "ensure_closed", None) or getattr(conn, "close", None)
    if closer is not None:
        result = closer()
        if hasattr(result, "__await__"):
            await result
