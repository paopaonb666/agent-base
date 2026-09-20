"""共享图工厂（core/graphs.py）的单元测试。

A1（roadmap T0.1）：流式累积的 tool_calls 聚合——模型回复分多块到达时，
累积结果必须归一为 AIMessage（tool_calls 完成聚合校验），而不是把
AIMessageChunk 原样塞进图状态。

注：``GenericFakeChatModel`` 无法构造该场景（空 content 直接抛错、
不携带 tool_call_chunks），分块流式用 fakes.ChunkedChatModel。
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.tools import tool

from agent_base.core.config import Settings
from agent_base.core.contracts import ModuleContext
from agent_base.core.graphs import _stream_and_accumulate, build_single_agent_graph
from fakes import ChunkedChatModel


@tool
def echo(text: str) -> str:
    """回显输入。"""
    return text


def _ctx(model: Any, *, tools: list[Any] | None = None) -> ModuleContext:
    return ModuleContext(
        settings=Settings(llm_api_key="k", memory_enabled=False),
        llm=model,
        checkpointer=None,
        tools=[echo] if tools is None else tools,
    )


def _split_tool_call_chunks(*, index: int | None) -> list[dict[str, Any]]:
    """同一条 tool_call 的参数跨两块拆分（真实 provider 的流式形态）。"""
    return [
        {"name": "echo", "args": '{"text":', "id": "call_1", "index": index},
        {"name": None, "args": '"hi"}', "id": None, "index": index},
    ]


async def test_stream_and_accumulate_returns_normalized_aimessage() -> None:
    """累积结果必须是 AIMessage 而非 AIMessageChunk（A1 契约）。"""
    model = ChunkedChatModel(
        batches=[
            [
                AIMessageChunk(content="", tool_call_chunks=_split_tool_call_chunks(index=0)),
                AIMessageChunk(content="完成"),
            ]
        ]
    )
    final = await _stream_and_accumulate(model, [HumanMessage(content="调用 echo")])
    assert final is not None
    assert not isinstance(final, AIMessageChunk)
    assert isinstance(final, AIMessage)
    assert [tc["name"] for tc in final.tool_calls] == ["echo"]


async def test_streamed_tool_calls_trigger_tool_loop() -> None:
    """分块产 tool_calls 时工具真的被执行（历史里出现 ToolMessage，循环走了第二圈）。"""
    model = ChunkedChatModel(
        batches=[
            [
                AIMessageChunk(content="", tool_call_chunks=_split_tool_call_chunks(index=0)),
                AIMessageChunk(
                    content="",
                    tool_call_chunks=[{"name": None, "args": '"hi"}', "id": None, "index": 0}],
                ),
            ],
            [AIMessageChunk(content="完成")],
        ]
    )
    graph = build_single_agent_graph(_ctx(model), name="t")
    result = await graph.ainvoke({"messages": [HumanMessage(content="调用 echo")]})
    assert any(isinstance(m, ToolMessage) and m.content == "hi" for m in result["messages"])


@pytest.mark.parametrize("index", [0, None])
async def test_accumulate_with_split_args_yields_callable_tool_call(index: int | None) -> None:
    """参数跨块拆分（含真实 provider 常见的无 index 形态）累积后必须可解析。"""
    model = ChunkedChatModel(
        batches=[
            [AIMessageChunk(content="", tool_call_chunks=_split_tool_call_chunks(index=index))]
        ]
    )
    final = await _stream_and_accumulate(model, [])
    assert final is not None
    assert final.tool_calls, "分块聚合后 tool_calls 不应为空"
