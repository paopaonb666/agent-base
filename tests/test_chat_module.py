"""Tests for the chat sample module (mock LLM, no network)."""

from __future__ import annotations

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import HumanMessage

from agent_base.core.config import Settings
from agent_base.core.contracts import ModuleContext
from agent_base.modules.chat.module import ChatModule


def _context(*responses: str) -> ModuleContext:
    llm = GenericFakeChatModel(messages=iter(responses))
    return ModuleContext(settings=Settings(llm_api_key="sk-test"), llm=llm)


def test_chat_module_contract() -> None:
    module = ChatModule()
    assert module.name == "chat"
    assert module.description
    assert module.get_tools() == []


def test_chat_graph_responds() -> None:
    graph = ChatModule().build_graph(_context("hello from mock"))
    result = graph.invoke({"messages": [HumanMessage(content="hi")]})
    assert [m.content for m in result["messages"]] == ["hi", "hello from mock"]


def test_chat_graph_accumulates_history() -> None:
    graph = ChatModule().build_graph(_context("reply 1", "reply 2"))
    first = graph.invoke({"messages": [HumanMessage(content="turn 1")]})
    assert len(first["messages"]) == 2
    second = graph.invoke({"messages": first["messages"] + [HumanMessage(content="turn 2")]})
    assert [m.content for m in second["messages"]] == ["turn 1", "reply 1", "turn 2", "reply 2"]
