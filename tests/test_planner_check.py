"""check / replan / synthesize 节点（B4）的单元测试。"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from agent_base.core.config import Settings
from agent_base.core.contracts import ModuleContext
from agent_base.modules.planner.nodes_check import (
    make_check_node,
    make_replan_node,
    route_after_check,
)
from fakes import ScriptedChatModel


def _ctx(model: Any, **planner: Any) -> ModuleContext:
    kwargs: dict[str, Any] = {"llm_api_key": "k", "memory_enabled": False, **planner}
    return ModuleContext(settings=Settings(**kwargs), llm=model, checkpointer=None, tools=[])


def _task(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": 1,
        "goal": "g",
        "status": "done",
        "summary": "有产出",
        "attempts": 1,
    }
    base.update(overrides)
    return base


def _state(tasks: list[dict[str, Any]], cursor: int = 0, replans: int = 0) -> dict[str, Any]:
    return {
        "messages": [HumanMessage(content="总目标")],
        "tasks": tasks,
        "cursor": cursor,
        "replans": replans,
    }


async def test_empty_summary_done_fails_gate() -> None:
    node = make_check_node(_ctx(ScriptedChatModel([])))
    result = await node(_state([_task(summary="   ")]), {"configurable": {}})
    assert result["tasks"][0]["status"] == "failed"
    assert result["tasks"][0]["summary"] == "输出为空"


async def test_min_summary_chars_gate() -> None:
    node = make_check_node(_ctx(ScriptedChatModel([]), planner_min_summary_chars=10))
    result = await node(_state([_task(summary="太短")]), {"configurable": {}})
    assert result["tasks"][0]["status"] == "failed"
    assert "不足" in result["tasks"][0]["summary"]


async def test_check_advances_cursor_to_next_pending() -> None:
    tasks = [_task(id=1, status="done"), _task(id=2, status="pending"), _task(id=3, status="done")]
    node = make_check_node(_ctx(ScriptedChatModel([])))
    result = await node(_state(tasks, cursor=0), {"configurable": {}})
    assert result["cursor"] == 1


async def test_route_priorities() -> None:
    route = route_after_check(_ctx(ScriptedChatModel([])))
    failed = _task(status="failed")
    pending = _task(id=2, status="pending")
    assert route(_state([failed])) == "replan"  # failed 且预算未耗尽
    assert route(_state([failed], replans=2)) == "synthesize"  # replan 预算耗尽
    assert route(_state([_task(), pending])) == "execute"  # 有 pending
    assert route(_state([_task()])) == "synthesize"  # 全部完成


async def test_replan_merges_done_and_new_pending() -> None:
    model = ScriptedChatModel([AIMessage(content='[{"id": 1, "goal": "重试 A"}]')])
    node = make_replan_node(_ctx(model))
    tasks = [
        _task(id=1, status="done", goal="已完成"),
        _task(id=2, status="failed", goal="失败项", summary="工具轮数耗尽"),
    ]
    result = await node(_state(tasks, cursor=1), {"configurable": {}})
    merged = result["tasks"]
    assert merged[0]["goal"] == "已完成" and merged[0]["status"] == "done"
    assert merged[1]["goal"] == "重试 A" and merged[1]["status"] == "pending"
    assert result["cursor"] == 1
    assert result["replans"] == 1


async def test_replan_parse_failure_falls_back_to_failed_goal() -> None:
    model = ScriptedChatModel([AIMessage(content="无法解析")])
    node = make_replan_node(_ctx(model))
    tasks = [_task(id=2, status="failed", goal="失败项")]
    result = await node(_state(tasks), {"configurable": {}})
    assert result["tasks"][0]["goal"] == "失败项"
    assert result["tasks"][0]["status"] == "pending"
