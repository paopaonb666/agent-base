"""跨图恢复 + 缺失键兜底的测试（B1 / 决策 6）。"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent_base.core.config import Settings
from agent_base.core.contracts import ModuleContext
from agent_base.core.graphs import build_single_agent_graph
from agent_base.modules.planner.graph import build_planner_graph
from agent_base.modules.planner.state import PlanState, _cursor, _replans, _tasks
from fakes import ScriptedChatModel


def test_accessors_tolerate_missing_keys() -> None:
    state: dict[str, Any] = {}
    assert _tasks(state) == []  # type: ignore[arg-type]
    assert _cursor(state) == 0  # type: ignore[arg-type]
    assert _replans(state) == 0  # type: ignore[arg-type]
    assert _tasks({"tasks": None}) == []  # type: ignore[arg-type]


def _ctx(model: Any, checkpointer: Any) -> ModuleContext:
    return ModuleContext(
        settings=Settings(llm_api_key="k", memory_enabled=False),
        llm=model,
        checkpointer=checkpointer,
        tools=[],
    )


async def test_planner_resumes_chat_thread_without_error() -> None:
    # 决策 6：chat 图只写 messages 通道；planner 图恢复时 tasks/cursor/replans
    # 缺失，节点必须经访问助手兜底而不是 KeyError。
    saver = InMemorySaver()
    config = {"configurable": {"thread_id": "chat:t1"}}
    chat_ctx = _ctx(ScriptedChatModel([AIMessage(content="好的")]), saver)
    await build_single_agent_graph(chat_ctx, name="chat").ainvoke(
        {"messages": [("human", "你好")]}, config
    )
    planner_ctx = _ctx(
        ScriptedChatModel(
            [
                AIMessage(content='[{"id": 1, "goal": "做 A"}]'),
                AIMessage(content="子任务结果"),
                AIMessage(content="全部完成"),
            ]
        ),
        saver,
    )
    result = await build_planner_graph(planner_ctx).ainvoke(
        {"messages": [("human", "帮我规划：先做 A")]}, config
    )
    assert result["tasks"] and result["tasks"][0]["goal"] == "做 A"


def _plan_state(**kwargs: Any) -> PlanState:
    return PlanState(messages=[], **kwargs)  # type: ignore[arg-type]
