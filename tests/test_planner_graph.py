"""planner 图端到端测试（B4）：四条执行路径。

1. 全成功：plan → 逐条 execute → synthesize；
2. 失败 → replan → 成功（replans=1）；
3. replan 预算耗尽：仍走完 → best-effort synthesize；
4. 空输出 → 空输出门改判 failed → replan（评审修正 3 守护）。
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk

from agent_base.core.config import Settings
from agent_base.core.contracts import ModuleContext
from agent_base.modules.planner.graph import build_planner_graph
from fakes import ScriptedChatModel


def _ctx(model: Any, **planner: Any) -> ModuleContext:
    kwargs: dict[str, Any] = {"llm_api_key": "k", "memory_enabled": False, **planner}
    return ModuleContext(settings=Settings(**kwargs), llm=model, checkpointer=None, tools=[])


def _plan(raw: str) -> AIMessage:
    return AIMessage(content=raw)


async def test_all_success_path() -> None:
    model = ScriptedChatModel(
        [
            _plan('[{"id": 1, "goal": "A"}, {"id": 2, "goal": "B"}]'),
            AIMessage(content="A 的结果"),
            AIMessage(content="B 的结果"),
            AIMessage(content="综合答复：A 与 B 都完成了"),
        ]
    )
    graph = build_planner_graph(_ctx(model))
    result = await graph.ainvoke({"messages": [("human", "做 A 和 B")]}, {"configurable": {}})
    assert [t["status"] for t in result["tasks"]] == ["done", "done"]
    assert result["tasks"][0]["summary"] == "A 的结果"
    assert result["replans"] == 0
    assert result["messages"][-1].content == "综合答复：A 与 B 都完成了"


async def test_failure_then_replan_succeeds() -> None:
    model = ScriptedChatModel(
        [
            _plan('[{"id": 1, "goal": "A"}]'),
            # 第一次 execute：模型不产出任何内容 → 空输出门改判 failed。
            AIMessage(content=""),
            # replan：重新规划。
            _plan('[{"id": 1, "goal": "重试 A"}]'),
            AIMessage(content="重试成功"),
            AIMessage(content="最终：完成"),
        ]
    )
    graph = build_planner_graph(_ctx(model))
    result = await graph.ainvoke({"messages": [("human", "做 A")]}, {"configurable": {}})
    assert result["replans"] == 1
    assert result["tasks"][-1]["status"] == "done"
    assert result["messages"][-1].content == "最终：完成"


async def test_replan_budget_exhausted_synthesizes_best_effort() -> None:
    model = ScriptedChatModel(
        [
            _plan('[{"id": 1, "goal": "A"}]'),
            AIMessage(content=""),  # execute：空输出 → failed
            # replan 1（parse 失败 → 以 failed 目标兜底单任务）
            AIMessage(content="不是 JSON"),
            AIMessage(content=""),  # execute：又空输出 → failed
            # replan 2（预算 max_replans=2 的最后一次）
            AIMessage(content="还不是 JSON"),
            AIMessage(content=""),  # execute：仍空输出 → failed
            # replans=2 == max_replans → 直接 synthesize（不再 replan）
            AIMessage(content="如实汇报：没有完成任何子任务"),
        ]
    )
    graph = build_planner_graph(_ctx(model, planner_max_replans=2))
    result = await graph.ainvoke({"messages": [("human", "做 A")]}, {"configurable": {}})
    assert result["replans"] == 2
    assert result["messages"][-1].content == "如实汇报：没有完成任何子任务"
    assert any(t["status"] == "failed" for t in result["tasks"])


async def test_empty_output_gate_triggers_replan() -> None:
    # 评审修正 3：done 且 summary 空白 → check 改判 failed，触发 replan。
    model = ScriptedChatModel(
        [
            _plan('[{"id": 1, "goal": "A"}]'),
            AIMessageChunk(content=""),  # execute 停止但 summary 为空
            _plan('[{"id": 1, "goal": "重做 A"}]'),
            AIMessage(content="这次有产出"),
            AIMessage(content="最终答复"),
        ]
    )
    graph = build_planner_graph(_ctx(model))
    result = await graph.ainvoke({"messages": [("human", "做 A")]}, {"configurable": {}})
    assert result["replans"] == 1
    assert result["messages"][-1].content == "最终答复"
