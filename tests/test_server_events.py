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
from agent_base.entrypoints.server import (
    _decode_tool_event,
    _serialize_message,
    create_app,
)
from agent_base.extensions.events import SourcesEvent, StepEvent
from agent_base.extensions.toollog import MemoryToolCallRecorder, ToolCallRecord
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


# ── 工具调用事件契约 + 历史序列化（M5） ─────────────────────────────


def test_decode_accepts_tool_call_events() -> None:
    start = _decode_tool_event(
        {
            "type": "tool_call",
            "call_id": "abc",
            "name": "web_search",
            "phase": "start",
            "args": {"query": "hi"},
        }
    )
    assert start is not None and start.phase == "start" and start.args == {"query": "hi"}
    end = _decode_tool_event(
        {
            "type": "tool_call",
            "call_id": "abc",
            "name": "web_search",
            "phase": "end",
            "status": "ok",
            "result": "r",
            "duration_ms": 12,
        }
    )
    assert end is not None and end.status == "ok" and end.duration_ms == 12
    bad = _decode_tool_event(
        {"type": "tool_call", "call_id": "abc", "name": "x", "phase": "middle"}
    )
    assert bad is None  # phase 不在契约内 → 丢弃


def test_serialize_message_exposes_tool_calls() -> None:
    from langchain_core.messages import AIMessage, ToolMessage

    ai = AIMessage(
        content="",
        tool_calls=[{"name": "calculator", "args": {"expression": "1+1"}, "id": "call-9"}],
    )
    out = _serialize_message(ai)
    assert out is not None and out["role"] == "assistant"
    assert out["tool_calls"] == [
        {"id": "call-9", "name": "calculator", "args": {"expression": "1+1"}}
    ]

    tool = ToolMessage(content="2", name="calculator", tool_call_id="call-9", status="error")
    out_tool = _serialize_message(tool)
    assert out_tool == {
        "role": "tool",
        "name": "calculator",
        "content": "2",
        "tool_call_id": "call-9",
        "status": "error",
    }
    # 空内容且无调用的 AI 消息仍然丢弃。
    assert _serialize_message(AIMessage(content="")) is None


def test_tool_calls_endpoint_returns_audit_records() -> None:
    recorder = MemoryToolCallRecorder()
    runtime = _runtime()
    runtime.tool_recorder = recorder
    # 记录一条属于 chat:t-history 线程的调用

    recorder.record(
        ToolCallRecord(
            call_id="c42",
            tool="calculator",
            status="error",
            args_json='{"expression": "1/0"}',
            error_text="tool execution failed: 除数为零",
            duration_ms=3,
            thread_id="chat:t-history",
            module="chat",
        )
    )
    with TestClient(create_app(runtime=runtime)) as client:
        body = client.get("/v1/agents/chat/threads/t-history/tool-calls").json()
    assert body["tool_calls"][0]["call_id"] == "c42"
    assert body["tool_calls"][0]["args"] == {"expression": "1/0"}
    assert body["tool_calls"][0]["status"] == "error"
    # 无 recorder 的 runtime → 空列表而非报错
    runtime2 = _runtime()
    with TestClient(create_app(runtime=runtime2)) as client:
        body = client.get("/v1/agents/chat/threads/t-history/tool-calls").json()
    assert body == {"tool_calls": []}


# ── 文件上传端点（M4a 解析层的 HTTP 入口） ──────────────────────────


def test_upload_file_parses_real_pdf() -> None:
    import sys

    from fastapi.testclient import TestClient as _TC

    sys.path.insert(0, "tests")
    from test_parsing import _make_pdf

    runtime = _runtime()
    with _TC(create_app(runtime=runtime)) as client:
        response = client.post(
            "/v1/agents/chat/files",
            files={
                "file": (
                    "resume.pdf",
                    _make_pdf(["resume line one", "page two content"]),
                    "application/pdf",
                )
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["format"] == "pdf"
    assert body["pages"] == 2
    assert body["truncated"] is False
    assert body["text_len"] > 0
    assert "resume line one" in body["text"]


def test_upload_rejects_bad_input() -> None:
    from fastapi.testclient import TestClient as _TC

    runtime = _runtime()
    runtime.settings = Settings(_env_file=None, llm_api_key="sk-test", doc_parse_max_input_bytes=64)
    with _TC(create_app(runtime=runtime)) as client:
        # 不支持的扩展名 → 400
        r1 = client.post(
            "/v1/agents/chat/files", files={"file": ("x.exe", b"MZ", "application/x-exe")}
        )
        assert r1.status_code == 400
        # 损坏的 PDF → 400 可读错误
        r2 = client.post(
            "/v1/agents/chat/files",
            files={"file": ("bad.pdf", b"%PDF-1.4 broken", "application/pdf")},
        )
        assert r2.status_code == 400
        # 超出大小上限 → 413
        r3 = client.post(
            "/v1/agents/chat/files", files={"file": ("big.txt", b"x" * 100, "text/plain")}
        )
        assert r3.status_code == 413
        # 空文件 → 400
        r4 = client.post("/v1/agents/chat/files", files={"file": ("e.txt", b"", "text/plain")})
        assert r4.status_code == 400
    # 未知模块 → 404
    runtime2 = _runtime()
    with _TC(create_app(runtime=runtime2)) as client:
        r5 = client.post("/v1/agents/nobody/files", files={"file": ("a.txt", b"hi", "text/plain")})
        assert r5.status_code == 404
