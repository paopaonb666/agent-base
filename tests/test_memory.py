"""Tests for conversation-state assembly (Stage 3): backends + recovery.

The recovery test is the Stage 3 acceptance evidence: two runtimes on the
same sqlite file (process restart in miniature) continue one conversation
under the same thread id.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agent_base.core.bootstrap import AgentRuntime, create_runtime
from agent_base.core.config import Settings
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
        assert db.exists()  # setup() created the file
    finally:
        await close_checkpointer(saver)


async def test_thread_state_recovers_across_runtimes(tmp_path: Path) -> None:
    """Stage 3 acceptance: restart the 'process', resume the conversation."""
    db = str(tmp_path / "state.db")
    settings = _settings(checkpointer_backend="sqlite", checkpointer_sqlite_path=db)

    # Process 1: turn 1 ("my name is Alice").
    runtime1 = await create_runtime(settings)
    runtime1.llm = ScriptedChatModel([AIMessage(content="nice to meet you")])
    graph1 = runtime1.graph("chat")
    await graph1.ainvoke(
        {"messages": [HumanMessage(content="my name is Alice")]},
        {"configurable": {"thread_id": "chat:t1"}},
    )
    await runtime1.close()

    # Process 2: same sqlite file, same thread — history must be replayed
    # from the checkpointer, not carried in memory.
    runtime2 = await create_runtime(settings)
    runtime2.llm = ScriptedChatModel([AIMessage(content="I remember Alice")])
    graph2 = runtime2.graph("chat")
    result2 = await graph2.ainvoke(
        {"messages": [HumanMessage(content="what is my name?")]},
        {"configurable": {"thread_id": "chat:t1"}},
    )
    await runtime2.close()

    contents = [m.content for m in result2["messages"]]
    assert contents[:2] == ["my name is Alice", "nice to meet you"]  # recovered history
    assert contents[2:] == ["what is my name?", "I remember Alice"]  # new turn


async def test_threads_are_isolated() -> None:
    """Different thread ids never see each other's history."""
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
