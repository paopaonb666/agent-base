"""planner 模块的契约与装配测试（B6，registry 契约风格）。"""

from __future__ import annotations

from agent_base.core.config import Settings
from agent_base.core.contracts import ModuleContext
from agent_base.core.registry import load_modules
from agent_base.modules.planner import module as planner_module
from fakes import ScriptedChatModel


def test_planner_contract() -> None:
    assert planner_module.name == "planner"
    assert planner_module.description
    assert planner_module.get_tools() == []  # 复用共享工具池


def test_build_graph_compiles_with_module_name() -> None:
    from agent_base.modules.planner.graph import build_planner_graph

    ctx = ModuleContext(
        settings=Settings(llm_api_key="k", memory_enabled=False),
        llm=ScriptedChatModel([]),
        checkpointer=None,
        tools=[],
    )
    graph = planner_module.build_graph(ctx)
    assert graph is not None
    # 以 name="planner" 编译：supervisor 要求 sub-agent 图携带模块名。
    assert build_planner_graph(ctx).name == "planner"


def test_planner_loadable_via_registry() -> None:
    modules = load_modules(["chat", "planner"])
    assert set(modules) == {"chat", "planner"}
    assert modules["planner"].name == "planner"
