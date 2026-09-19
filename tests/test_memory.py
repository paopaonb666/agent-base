"""对话状态装配的测试（阶段 3）：后端 + 恢复。

恢复测试是阶段 3 的验收证据：两个运行时使用同一个 sqlite 文件（进程
重启的缩影），在同一个 thread id 下续接同一次对话。
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agent_base.core.bootstrap import AgentRuntime, create_runtime
from agent_base.core.config import Settings, SettingsError
from agent_base.extensions.memory import build_checkpointer, close_checkpointer
from agent_base.modules.chat.module import ChatModule
from fakes import ScriptedChatModel


def _settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {"llm_api_key": "sk-test"}
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


async def test_memory_backend_returns_inmemory_saver() -> None:
    saver = await build_checkpointer(_settings(checkpointer_backend="memory"))
    assert isinstance(saver, InMemorySaver)


async def test_sqlite_backend_returns_async_saver(tmp_path: Path) -> None:
    db = tmp_path / "state.db"
    saver = await build_checkpointer(
        _settings(checkpointer_backend="sqlite", checkpointer_sqlite_path=str(db))
    )
    try:
        assert isinstance(saver, AsyncSqliteSaver)
        assert db.exists()  # setup() 创建了该文件
    finally:
        await close_checkpointer(saver)


async def test_sqlite_unwritable_path_fails_fast(tmp_path: Path) -> None:
    """父目录不存在时启动即报配置错误，而不是会话中途的原始异常。"""
    settings = _settings(
        checkpointer_backend="sqlite",
        checkpointer_sqlite_path=str(tmp_path / "missing_dir" / "state.db"),
    )
    with pytest.raises(SettingsError, match="not writable"):
        await build_checkpointer(settings)


async def test_thread_state_recovers_across_runtimes(tmp_path: Path) -> None:
    """阶段 3 验收：重启“进程”，续接对话。"""
    db = str(tmp_path / "state.db")
    settings = _settings(checkpointer_backend="sqlite", checkpointer_sqlite_path=db)

    # 进程 1：第 1 轮（"my name is Alice"）。
    runtime1 = await create_runtime(settings)
    runtime1.llm = ScriptedChatModel([AIMessage(content="nice to meet you")])
    graph1 = runtime1.graph("chat")
    await graph1.ainvoke(
        {"messages": [HumanMessage(content="my name is Alice")]},
        {"configurable": {"thread_id": "chat:t1"}},
    )
    await runtime1.close()

    # 进程 2：同一个 sqlite 文件、同一个 thread——历史必须从 checkpointer
    # 回放，而不是在内存中携带。
    runtime2 = await create_runtime(settings)
    runtime2.llm = ScriptedChatModel([AIMessage(content="I remember Alice")])
    graph2 = runtime2.graph("chat")
    result2 = await graph2.ainvoke(
        {"messages": [HumanMessage(content="what is my name?")]},
        {"configurable": {"thread_id": "chat:t1"}},
    )
    await runtime2.close()

    contents = [m.content for m in result2["messages"]]
    assert contents[:2] == ["my name is Alice", "nice to meet you"]  # 已恢复的历史
    assert contents[2:] == ["what is my name?", "I remember Alice"]  # 新的一轮


async def test_threads_are_isolated() -> None:
    """不同的 thread id 永远不会看到彼此的历史。"""
    runtime = AgentRuntime(
        settings=_settings(),
        llm=ScriptedChatModel([AIMessage(content="a"), AIMessage(content="b")]),
        modules={"chat": ChatModule()},
        checkpointer=InMemorySaver(),
    )
    graph = runtime.graph("chat")
    r1 = await graph.ainvoke(
        {"messages": [HumanMessage(content="one")]}, {"configurable": {"thread_id": "chat:t1"}}
    )
    r2 = await graph.ainvoke(
        {"messages": [HumanMessage(content="two")]}, {"configurable": {"thread_id": "chat:t2"}}
    )
    assert [m.content for m in r1["messages"]] == ["one", "a"]
    assert [m.content for m in r2["messages"]] == ["two", "b"]


@pytest.mark.integration
async def test_mysql_backend_builds_and_probes() -> None:
    """MySQL 后端装配测试（本机无 MySQL 时自动跳过）。

    连接参数可通过 MYSQL_HOST / MYSQL_PORT / MYSQL_USER / MYSQL_PASSWORD
    环境变量覆盖，默认 root/空密码连 127.0.0.1:3306。
    """
    pytest.importorskip("aiomysql")
    try:
        from langgraph.checkpoint.mysql.aio import AIOMySQLSaver
    except ImportError:
        pytest.skip("langgraph-checkpoint-mysql[aiomysql] 未安装")

    import aiomysql

    host = os.environ.get("MYSQL_HOST", "127.0.0.1")
    port = int(os.environ.get("MYSQL_PORT", "3306"))
    user = os.environ.get("MYSQL_USER", "root")
    password = os.environ.get("MYSQL_PASSWORD", "")

    try:
        conn = await aiomysql.connect(
            host=host, port=port, user=user, password=password, autocommit=True
        )
    except Exception as exc:
        pytest.skip(f"MySQL 不可用：{exc}")

    db = f"agent_base_test_{uuid.uuid4().hex[:8]}"
    async with conn.cursor() as cur:
        await cur.execute(f"CREATE DATABASE `{db}` CHARACTER SET utf8mb4")
    conn.close()

    saver = None
    try:
        settings = _settings(
            checkpointer_backend="mysql",
            checkpointer_mysql_host=host,
            checkpointer_mysql_port=port,
            checkpointer_mysql_user=user,
            checkpointer_mysql_password=password,
            checkpointer_mysql_database=db,
        )
        saver = await build_checkpointer(settings)
        assert isinstance(saver, AIOMySQLSaver)
        # setup() 已建表；读取不存在的 thread 应返回 None。
        assert await saver.aget_tuple({"configurable": {"thread_id": "nope"}}) is None
    finally:
        if saver is not None:
            await close_checkpointer(saver)
        cleanup = await aiomysql.connect(
            host=host, port=port, user=user, password=password, autocommit=True
        )
        async with cleanup.cursor() as cur:
            await cur.execute(f"DROP DATABASE IF EXISTS `{db}`")
        cleanup.close()
