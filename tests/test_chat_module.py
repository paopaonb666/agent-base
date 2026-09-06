"""Tests for the chat sample module (mock LLM, no network)."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from agent_base.core.config import Settings
from agent_base.core.contracts import ModuleContext
from agent_base.modules.chat.module import ChatModule
from fakes import ScriptedChatModel


def _context(*responses: AIMessage | str) -> ModuleContext:
    scripted = [r if isinstance(r, AIMessage) else AIMessage(content=r) for r in responses]
    return ModuleContext(
        settings=Settings(_env_file=None, llm_api_key="sk-test"),
        llm=ScriptedChatModel(scripted),
        tools=[],
    )


def test_chat_module_contract() -> None:
    module = ChatModule()
    assert module.name == "chat"
    assert module.description
    assert [t.name for t in module.get_tools()] == ["echo"]


async def test_chat_graph_responds() -> None:
    graph = ChatModule().build_graph(_context("hello from mock"))
    result = await graph.ainvoke({"messages": [HumanMessage(content="hi")]})
    assert [m.content for m in result["messages"]] == ["hi", "hello from mock"]


async def test_chat_graph_accumulates_history() -> None:
    graph = ChatModule().build_graph(_context("reply 1", "reply 2"))
    first = await graph.ainvoke({"messages": [HumanMessage(content="turn 1")]})
    assert len(first["messages"]) == 2
    second = await graph.ainvoke({"messages": first["messages"] + [HumanMessage(content="turn 2")]})
    assert [m.content for m in second["messages"]] == [
        "turn 1",
        "reply 1",
        "turn 2",
        "reply 2",
    ]


async def test_chat_graph_compiled_with_name_for_supervisor() -> None:
    graph = ChatModule().build_graph(_context("ok"))
    assert graph.name == "chat"


async def test_chat_graph_tool_roundtrip() -> None:
    """A tool-calling turn flows: model calls echo -> ToolMessage -> reply."""
    scripted = [
        AIMessage(
            content="",
            tool_calls=[{"name": "echo", "args": {"text": "hi"}, "id": "call-1"}],
        ),
        AIMessage(content="heard echo"),
    ]
    ctx = ModuleContext(
        settings=Settings(_env_file=None, llm_api_key="sk-test"),
        llm=ScriptedChatModel(scripted),
        tools=ChatModule().get_tools(),
    )
    graph = ChatModule().build_graph(ctx)
    result = await graph.ainvoke({"messages": [HumanMessage(content="call echo")]})
    kinds = [type(m).__name__ for m in result["messages"]]
    assert kinds == ["HumanMessage", "AIMessage", "ToolMessage", "AIMessage"]
    assert result["messages"][2].content == "echo: hi"


async def test_broken_tool_does_not_crash_conversation() -> None:
    """Stage 3 acceptance: a tool exception becomes ToolMessage feedback."""
    from langchain_core.tools import tool

    @tool
    def explode() -> str:
        """Always fails."""
        raise RuntimeError("boom")

    scripted = [
        AIMessage(
            content="",
            tool_calls=[{"name": "explode", "args": {}, "id": "call-1"}],
        ),
        AIMessage(content="recovered from tool failure"),
    ]
    ctx = ModuleContext(
        settings=Settings(_env_file=None, llm_api_key="sk-test"),
        llm=ScriptedChatModel(scripted),
        tools=[explode],
    )
    graph = ChatModule().build_graph(ctx)
    result = await graph.ainvoke({"messages": [HumanMessage(content="use explode")]})
    tool_message = result["messages"][2]
    assert type(tool_message).__name__ == "ToolMessage"
    assert "boom" in str(tool_message.content)  # error text fed back to the model
    assert result["messages"][-1].content == "recovered from tool failure"
