"""阶段 5 验收：module-guide 的五步配方在真实代码中接入。

``src/agent_base/modules/hello/`` 下的 ``hello`` 模块是通过逐字遵循手册
添加的——没有触碰任何基座代码。这些测试证明它像任何内置模块一样工作：
契约、mock-LLM 对话、以及完整的运行时装配（CLI / server 共用这条路径）。
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
    """完整装配路径（create_runtime）接受 hello 与 chat/writer 并存。"""
    runtime = await create_runtime(
        Settings(_env_file=None, agent_modules="chat,writer,hello", llm_api_key="sk-test")
    )
    try:
        graph = runtime.graph("hello")
        assert graph.name == "hello"
    finally:
        await runtime.close()
