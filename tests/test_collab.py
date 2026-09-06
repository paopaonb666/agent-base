"""supervisor 模板的测试（阶段 4）。

脚本化模型同时扮演两个角色：supervisor 节点（它的第一条响应是一次交给
``writer`` 的 handoff 工具调用）以及 writer sub-agent（它的第二条响应是
答案）。因此路由可以在没有真实 LLM 的情况下端到端地断言。
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent_base.core.bootstrap import SUPERVISOR_MODULE, AgentRuntime
from agent_base.core.config import Settings
from agent_base.core.contracts import ModuleContext
from agent_base.extensions.collab import build_supervisor_graph
from agent_base.modules.chat.module import ChatModule
from agent_base.modules.writer.module import WriterModule
from fakes import ScriptedChatModel


def _runtime(model: ScriptedChatModel) -> AgentRuntime:
    return AgentRuntime(
        settings=Settings(_env_file=None, llm_api_key="sk-test"),
        llm=model,
        modules={"chat": ChatModule(), "writer": WriterModule()},
        tools=[],
        checkpointer=InMemorySaver(),
    )


async def test_supervisor_graph_builds_over_registered_modules() -> None:
    runtime = _runtime(ScriptedChatModel([]))
    graph = runtime.graph(SUPERVISOR_MODULE)
    # Sub-agent 以各自模块命名的节点形式出现。
    node_names = set(graph.get_graph().nodes)
    assert {"chat", "writer"} <= node_names


async def test_supervisor_routes_to_writer() -> None:
    # 第 1 次调用 = supervisor 决策：通过自动生成的
    # `transfer_to_<module>` 工具交接。第 2 次调用 = writer 起草答案。
    # 第 3 次调用 = supervisor 合成最终回复。
    model = ScriptedChatModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "transfer_to_writer", "args": {}, "id": "handoff-1"},
                ],
            ),
            AIMessage(content="draft: the essay body"),
            AIMessage(content="final: your essay is ready"),
        ]
    )
    runtime = _runtime(model)
    graph = runtime.graph(SUPERVISOR_MODULE)
    config: dict[str, Any] = {"configurable": {"thread_id": "sup:t1"}}
    result = await graph.ainvoke({"messages": [HumanMessage(content="write me an essay")]}, config)
    contents = [m.content for m in result["messages"]]
    # 对话经过了 writer sub-agent（它的回复出现在最终状态里），
    # 且 supervisor 产出了一条收尾消息。
    assert "draft: the essay body" in contents
    assert contents[-1] == "final: your essay is ready"


def test_supervisor_requires_modules() -> None:
    import pytest

    ctx = ModuleContext(
        settings=Settings(_env_file=None, llm_api_key="sk-test"),
        llm=ScriptedChatModel([]),
        checkpointer=InMemorySaver(),
    )
    with pytest.raises(ValueError, match="at least one module"):
        build_supervisor_graph(ctx, {})
