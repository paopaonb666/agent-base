"""M3.5 的测试：工具经 custom stream 发的 UI 事件到达 SSE 契约层。

覆盖两端：服务器侧 `_decode_tool_event` 的封闭校验（畸形载荷丢弃、
绝不透传未校验数据），以及"chat 图里执行工具 → 事件出现在 SSE 流"
的端到端往返。
"""

from __future__ import annotations

import contextlib
import json
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer

from agent_base.core.bootstrap import AgentRuntime
from agent_base.core.config import Settings
from agent_base.entrypoints.server import _decode_tool_event, create_app
from agent_base.extensions.events import SourcesEvent, StepEvent
from agent_base.modules.chat.module import ChatModule
from agent_base.tools.streaming import emit_sources, emit_step
from fakes import ScriptedChatModel


@tool
def announce() -> str:
    """发进度与来源事件后返回。"""
    emit_step("web_search", "running", "搜索：测试")
    emit_sources([{"title": "标题", "url": "https://example.com/a"}])
    # 服务器必须丢弃未知类型：直接经 writer 发一个契约外载荷。
    with contextlib.suppress(Exception):
        get_stream_writer()({"type": "bogus", "evil": "<script>"})
    emit_step("web_search", "completed", "找到 1 条结果")
    return "announced"


def _runtime() -> AgentRuntime:
    model = ScriptedChatModel(
        [
            AIMessage(content="", tool_calls=[{"name": "announce", "args": {}, "id": "call-1"}]),
            AIMessage(content="done"),
        ]
    )
    return AgentRuntime(
        settings=Settings(_env_file=None, llm_api_key="sk-test"),
        llm=model,
        modules={"chat": ChatModule()},
        tools=[announce],
        checkpointer=InMemorySaver(),
    )


def _frames(raw: str) -> list[dict[str, Any]]:
    events = []
    for block in raw.strip().split("\n\n"):
        lines = block.split("\n")
        events.append(
            {
                "event": lines[0].removeprefix("event: "),
                **json.loads(lines[1].removeprefix("data: ")),
            }
        )
    return events


def test_tool_events_reach_sse_stream() -> None:
    with (
        TestClient(create_app(runtime=_runtime())) as client,
        client.stream("POST", "/v1/agents/chat/invoke", json={"message": "查一下"}) as response,
    ):
        body = "".join(response.iter_text())
    frames = _frames(body)

    steps = [f for f in frames if f["event"] == "step" and f.get("name") == "web_search"]
    assert [s["status"] for s in steps] == ["running", "completed"]
    assert steps[0]["detail"] == "搜索：测试"

    sources = [f for f in frames if f["event"] == "sources"]
    assert sources[0]["sources"] == [{"title": "标题", "url": "https://example.com/a"}]

    # 契约外载荷必须被丢弃，不会以任何形式出现在 SSE 流里。
    assert all(f["event"] != "bogus" for f in frames)
    assert not any("evil" in json.dumps(f) for f in frames)


def test_decode_accepts_contract_events() -> None:
    step = _decode_tool_event({"type": "step", "name": "web_search", "status": "running"})
    assert isinstance(step, StepEvent)
    sources = _decode_tool_event({"type": "sources", "sources": [{"title": "T"}]})
    assert isinstance(sources, SourcesEvent)


def test_decode_drops_malformed_and_unknown() -> None:
    assert _decode_tool_event({"type": "step", "name": "x", "status": "bogus"}) is None
    assert _decode_tool_event({"type": "sources", "sources": [{"url": 42}]}) is None
    assert _decode_tool_event({"type": "mystery"}) is None
    assert _decode_tool_event("not a dict") is None
    assert _decode_tool_event(None) is None


def test_emit_helpers_outside_graph_are_noop() -> None:
    # 不在图执行上下文（直调工具、单测）：事件静默丢弃，绝不抛错。
    assert emit_step("web_search", "running") is None
    assert emit_sources([{"title": "T"}]) is None


def test_emit_helpers_forward_contract_payloads(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[dict[str, Any]] = []
    monkeypatch.setattr("agent_base.tools.streaming.get_stream_writer", lambda: sent.append)
    emit_step("s", "completed", "详情")
    emit_sources([{"title": "T", "url": "https://x"}])
    assert {"type": "step", "name": "s", "status": "completed", "detail": "详情"} in sent
    assert {"type": "sources", "sources": [{"title": "T", "url": "https://x"}]} in sent
    # detail 为空时省略字段，避免把空值灌给前端。
    emit_step("s2", "error")
    assert {"type": "step", "name": "s2", "status": "error"} in sent
