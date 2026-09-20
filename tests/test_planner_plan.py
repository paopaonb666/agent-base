"""plan 节点（B2）的单元测试：解析容错、截断、降级。"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from agent_base.core.config import Settings
from agent_base.core.contracts import ModuleContext
from agent_base.modules.planner.nodes_plan import make_plan_node
from fakes import ScriptedChatModel


def _ctx(model: Any) -> ModuleContext:
    return ModuleContext(
        settings=Settings(llm_api_key="k", memory_enabled=False),
        llm=model,
        checkpointer=None,
        tools=[],
    )


def _human(text: str) -> HumanMessage:
    return HumanMessage(content=text)


def _config() -> dict[str, Any]:
    return {"configurable": {"thread_id": "chat:t1"}}


async def test_plan_parses_json_list() -> None:
    node = make_plan_node(
        _ctx(
            ScriptedChatModel(
                [
                    AIMessage(
                        content='前缀文字 [ {"id": 9, "goal": "A"}, {"id": 2, "goal": "B"} ] 后缀'
                    )
                ]
            )
        )
    )
    result = await node({"messages": [_human("先 A 后 B")]}, _config())
    assert [t["goal"] for t in result["tasks"]] == ["A", "B"]
    assert result["tasks"][0]["status"] == "pending" and result["cursor"] == 0


async def test_plan_reindexes_ids_and_caps_subtasks() -> None:
    raw = (
        '[{"id": 5, "goal": "1"}, {"id": 3, "goal": "2"}, {"id": 8, "goal": "3"},'
        ' {"id": 1, "goal": "4"}, {"id": 2, "goal": "5"}, {"id": 6, "goal": "6"}]'
    )
    node = make_plan_node(_ctx(ScriptedChatModel([AIMessage(content=raw)])))
    result = await node({"messages": [_human("做六件事")]}, _config())
    assert [t["id"] for t in result["tasks"]] == [1, 2, 3, 4, 5]  # 默认 max_subtasks=5 截断
    assert all(t["status"] == "pending" for t in result["tasks"])


async def test_plan_degrades_after_retry() -> None:
    node = make_plan_node(
        _ctx(ScriptedChatModel([AIMessage(content="不是 JSON"), AIMessage(content="还不是 JSON")]))
    )
    result = await node({"messages": [_human("做点事")]}, _config())
    assert len(result["tasks"]) == 1  # 降级单任务
    assert result["tasks"][0]["goal"] == "做点事"


async def test_plan_tolerates_json_fence() -> None:
    node = make_plan_node(
        _ctx(ScriptedChatModel([AIMessage(content='```json\n[{"id": 1, "goal": "拆好"}]\n```')]))
    )
    result = await node({"messages": [_human("目标")]}, _config())
    assert result["tasks"][0]["goal"] == "拆好"
