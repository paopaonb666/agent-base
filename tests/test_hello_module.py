"""Stage 5 acceptance: the module-guide's five-step recipe plugs in for real.

The ``hello`` module under ``src/agent_base/modules/hello/`` was added by
following the guide verbatim — no base code was touched. These tests prove it
works like any built-in module: contract, mock-LLM conversation, and full
runtime assembly (CLI / server share this path).
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent_base.core.bootstrap import create_runtime
from agent_base.core.config import Settings
from agent_base.core.contracts import ModuleContext
from agent_base.core.registry import load_modules
from agent_base.modules.hello.module import HelloModule
from fakes import ScriptedChatModel


def test_hello_module_contract() -> None:
    module = HelloModule()
    assert module.name == "hello"
    assert module.description
    assert module.get_tools() == []


def test_registry_loads_hello() -> None:
    modules = load_modules(["hello"])
    assert list(modules) == ["hello"]
    assert modules["hello"].name == "hello"


async def test_hello_graph_responds() -> None:
    ctx = ModuleContext(
        settings=Settings(_env_file=None, llm_api_key="sk-test"),
        llm=ScriptedChatModel([AIMessage(content="hello back")]),
        tools=[],
    )
    graph = HelloModule().build_graph(ctx)
    result = await graph.ainvoke({"messages": [HumanMessage(content="hi")]})
    assert [m.content for m in result["messages"]] == ["hi", "hello back"]


async def test_hello_assembles_in_full_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """The full assembly path (create_runtime) accepts hello alongside chat/writer."""
    runtime = await create_runtime(
        Settings(_env_file=None, agent_modules="chat,writer,hello", llm_api_key="sk-test")
    )
    try:
        graph = runtime.graph("hello")
        assert graph.name == "hello"
    finally:
        await runtime.close()
