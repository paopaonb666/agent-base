"""execute 节点（B3）的单元测试：工具循环、轮数耗尽、越界防御。"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.tools import tool

from agent_base.core.config import Settings
from agent_base.core.contracts import ModuleContext
from agent_base.modules.planner.nodes_execute import make_execute_node
from fakes import ChunkedChatModel, ScriptedChatModel


@tool
def echo(text: str) -> str:
    """回显输入。"""
    return text


def _tool_call_batch(args: str = '{"text": "hi"}') -> list[AIMessageChunk]:
    return [
        AIMessageChunk(
            content="",
            tool_call_chunks=[{"name": "echo", "args": args, "id": "call_1", "index": 0}],
        )
    ]


def _ctx(model: Any, *, tools: list[Any] | None = ..., rounds: int = 8) -> ModuleContext:
    kwargs: dict[str, Any] = {"llm_api_key": "k", "memory_enabled": False}
    if rounds != 8:
        kwargs["planner_subtask_tool_rounds"] = rounds
    return ModuleContext(
        settings=Settings(**kwargs),
        llm=model,
        checkpointer=None,
        tools=[echo] if tools is ... else (tools or []),
    )


def _state(tasks: list[dict[str, Any]], cursor: int = 0) -> dict[str, Any]:
    return {"messages": [HumanMessage(content="总目标")], "tasks": tasks, "cursor": cursor}


def _tasks(n: int = 1) -> list[dict[str, Any]]:
    return [
        {"id": i + 1, "goal": f"子任务{i + 1}", "status": "pending", "summary": "", "attempts": 0}
        for i in range(n)
    ]


async def test_tool_calls_then_text_completes_subtask() -> None:
    model = ChunkedChatModel(
        batches=[_tool_call_batch(), _tool_call_batch(), [AIMessageChunk(content="结果汇总")]]
    )
    node = make_execute_node(_ctx(model, tools=...))
    result = await node(_state(_tasks()), {"configurable": {}})
    produced = result["messages"]
    assert any(isinstance(m, ToolMessage) and m.content == "hi" for m in produced)
    task = result["tasks"][0]
    assert task["status"] == "done"
    assert task["summary"] == "结果汇总"
    assert task["attempts"] == 1


async def test_tool_rounds_exhaustion_fails_subtask() -> None:
    batches = [_tool_call_batch() for _ in range(3)]  # rounds=2：两轮后仍未产纯文本
    model = ChunkedChatModel(batches=batches)
    node = make_execute_node(_ctx(model, tools=..., rounds=2))
    result = await node(_state(_tasks()), {"configurable": {}})
    task = result["tasks"][0]
    assert task["status"] == "failed"
    assert task["summary"] == "工具轮数耗尽"


async def test_cursor_out_of_bounds_is_noop() -> None:
    model = ScriptedChatModel([AIMessageChunk(content="不应被消费")])
    node = make_execute_node(_ctx(model))
    result = await node(_state(_tasks(), cursor=5), {"configurable": {}})
    assert result == {}


async def test_no_tool_pool_plain_text_completes() -> None:
    model = ScriptedChatModel([AIMessageChunk(content="直接完成")])
    node = make_execute_node(_ctx(model))
    result = await node(_state(_tasks()), {"configurable": {}})
    assert result["tasks"][0]["status"] == "done"
    assert result["tasks"][0]["summary"] == "直接完成"
