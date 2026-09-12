"""M3.5 的测试：工具经 custom stream 发的 UI 事件到达 SSE 契约层。

覆盖两端：服务器侧 `_decode_tool_event` 的封闭校验（畸形载荷丢弃、
绝不透传未校验数据），以及"chat 图里执行工具 → 事件出现在 SSE 流"
的端到端往返。
"""

from __future__ import annotations

import asyncio
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
from agent_base.extensions.filestore import MemoryUploadedFileStore, UploadedFileInfo
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


def test_upload_file_persists_and_returns_meta() -> None:
    from test_parsing import _make_pdf

    runtime = _runtime()
    runtime.file_store = MemoryUploadedFileStore()
    with TestClient(create_app(runtime=runtime)) as client:
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
    # 响应只回元信息：全文留在服务端，由 invoke 时注入上下文
    assert body["format"] == "pdf"
    assert body["pages"] == 2
    assert body["truncated"] is False
    assert body["text_len"] > 0
    assert "text" not in body
    assert "extracted_text" not in body


def test_upload_rejects_bad_input() -> None:
    runtime = _runtime()
    runtime.file_store = MemoryUploadedFileStore()
    runtime.settings = Settings(_env_file=None, llm_api_key="sk-test", doc_parse_max_input_bytes=64)
    with TestClient(create_app(runtime=runtime)) as client:
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
    with TestClient(create_app(runtime=runtime2)) as client:
        r5 = client.post("/v1/agents/nobody/files", files={"file": ("a.txt", b"hi", "text/plain")})
        assert r5.status_code == 404


# ── 文件上传端点（M4a 解析层的 HTTP 入口） ──────────────────────────


def _runtime_with_files() -> tuple[AgentRuntime, MemoryUploadedFileStore]:
    runtime = _runtime()
    store = MemoryUploadedFileStore()
    runtime.file_store = store
    return runtime, store


def test_invoke_injects_attachments_as_system_message() -> None:
    from langchain_core.messages import SystemMessage

    captured: list[list[Any]] = []

    class _CapturingModel(ScriptedChatModel):
        async def _astream(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[no-untyped-def]
            captured.append(list(messages))
            async for chunk in super()._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            ):
                yield chunk

    runtime, store = _runtime_with_files()
    model = _CapturingModel([AIMessage(content="收到")])
    runtime.llm = model
    asyncio.run(
        store.save(
            UploadedFileInfo(
                file_id="f1",
                filename="简历.pdf",
                format="pdf",
                pages=2,
                text_len=6,
                extracted_text="刘骁铖 简历正文",
                content=b"%PDF",
                thread_id="",
                module="chat",
            )
        )
    )
    with TestClient(create_app(runtime=runtime)) as client:
        response = client.post(
            "/v1/agents/chat/invoke",
            json={"message": "读一下附件", "thread_id": "att-1", "attachments": ["f1"]},
        )
    if response.status_code != 200:
        raise AssertionError(f"invoke 失败: {response.status_code} {response.text[:200]}")
    # 模型收到：SystemMessage（附件全文）+ HumanMessage（干净正文）
    assert captured, "模型没有被调用"
    system_texts = [m for m in captured[-1] if isinstance(m, SystemMessage)]
    assert any("简历正文" in str(m.content) for m in system_texts)
    human = [m for m in captured[-1] if m.type == "human"]
    assert human and human[0].content == "读一下附件"
    assert human[0].additional_kwargs["attachments"][0]["filename"] == "简历.pdf"
    # 历史回放：human 带附件元数据、无 system、无 blob
    history = client.get("/v1/agents/chat/threads/att-1").json()
    roles = [m["role"] for m in history["messages"]]
    assert "system" not in roles
    human_out = history["messages"][0]
    assert human_out["attachments"][0]["file_id"] == "f1"
    assert "简历正文" not in json.dumps(history["messages"], ensure_ascii=False)


def test_invoke_unknown_file_id_rejected() -> None:
    runtime, _ = _runtime_with_files()
    with TestClient(create_app(runtime=runtime)) as client:
        response = client.post(
            "/v1/agents/chat/invoke",
            json={"message": "hi", "thread_id": "att-2", "attachments": ["ghost"]},
        )
    assert response.status_code == 400
    assert "ghost" in response.json()["detail"]


async def test_delete_thread_cascades_files() -> None:
    runtime, store = _runtime_with_files()
    await store.save(
        UploadedFileInfo(
            file_id="f9",
            filename="a.pdf",
            format="pdf",
            text_len=3,
            extracted_text="abc",
            content=b"%PDF",
            thread_id="chat:del-1",
            module="chat",
        )
    )
    with TestClient(create_app(runtime=runtime)) as client:
        response = client.delete("/v1/agents/chat/threads/del-1")
    assert response.json() == {"deleted": True}
    assert await store.get_many(["f9"]) == []


# ── 图片附件多模态（M4c） ────────────────────────────────────────────

_PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"fake png body for tests"


def test_upload_image_sniffs_magic() -> None:
    runtime, store = _runtime_with_files()
    with TestClient(create_app(runtime=runtime)) as client:
        response = client.post(
            "/v1/agents/chat/files",
            files={"file": ("pic.png", _PNG_BYTES, "image/png")},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["format"] == "png"
    assert body["text_len"] == 0
    assert "warning" not in body  # 图片是有意无文本，不算异常
    # 原始字节入库（可重解析/下载）
    rows = asyncio.run(store.get_many([body["file_id"]]))
    assert rows[0].content.startswith(b"\x89PNG")


def test_upload_rejects_disguised_image() -> None:
    runtime, _ = _runtime_with_files()
    with TestClient(create_app(runtime=runtime)) as client:
        # 扩展名 .png 但内容不是图片（magic 校验失败）
        response = client.post(
            "/v1/agents/chat/files",
            files={"file": ("evil.png", b"MZ not an image", "image/png")},
        )
    assert response.status_code == 400
    assert "magic" in response.json()["detail"]


def test_invoke_image_injection_is_multimodal() -> None:
    captured: list[list[Any]] = []

    class _Cap(ScriptedChatModel):
        async def _astream(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[no-untyped-def]
            captured.append(list(messages))
            async for chunk in super()._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            ):
                yield chunk

    runtime, store = _runtime_with_files()
    runtime.llm = _Cap([AIMessage(content="看到图片了")])
    asyncio.run(
        store.save(
            UploadedFileInfo(
                file_id="img1",
                filename="photo.png",
                format="png",
                text_len=0,
                extracted_text="",
                content=_PNG_BYTES,
                thread_id="",
                module="chat",
            )
        )
    )
    with TestClient(create_app(runtime=runtime)) as client:
        response = client.post(
            "/v1/agents/chat/invoke",
            json={"message": "图里有什么", "thread_id": "img-1", "attachments": ["img1"]},
        )
    if response.status_code != 200:
        raise AssertionError(f"invoke 失败: {response.status_code} {response.text[:200]}")
    human = [m for m in captured[-1] if m.type == "human"]
    assert human, "模型没有被调用"
    content = human[0].content
    assert isinstance(content, list), "图片应走多模态 content blocks"
    assert content[0] == {"type": "text", "text": "图里有什么"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    # 纯图片不产生 SystemMessage（没有文档文本可注入）
    assert not any(m.type == "system" for m in captured[-1])


def test_invoke_mixed_doc_and_image() -> None:
    from langchain_core.messages import SystemMessage

    captured: list[list[Any]] = []

    class _Cap2(ScriptedChatModel):
        async def _astream(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[no-untyped-def]
            captured.append(list(messages))
            async for chunk in super()._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            ):
                yield chunk

    runtime, store = _runtime_with_files()
    runtime.llm = _Cap2([AIMessage(content="收到")])
    asyncio.run(
        store.save(
            UploadedFileInfo(
                file_id="doc1",
                filename="doc.pdf",
                format="pdf",
                text_len=5,
                extracted_text="文档正文",
                content=b"%PDF",
                thread_id="",
                module="chat",
            )
        )
    )
    asyncio.run(
        store.save(
            UploadedFileInfo(
                file_id="img2",
                filename="photo.jpg",
                format="jpeg",
                text_len=0,
                extracted_text="",
                content=b"\xff\xd8\xffjpegbody",
                thread_id="",
                module="chat",
            )
        )
    )
    with TestClient(create_app(runtime=runtime)) as client:
        response = client.post(
            "/v1/agents/chat/invoke",
            json={"message": "对比一下", "thread_id": "mix-1", "attachments": ["doc1", "img2"]},
        )
    if response.status_code != 200:
        raise AssertionError(f"invoke 失败: {response.status_code} {response.text[:200]}")
    messages = captured[-1]
    # 文档 → SystemMessage；图片 → 多模态 human content
    system = [m for m in messages if isinstance(m, SystemMessage)]
    assert len(system) == 1 and "文档正文" in system[0].content
    human = [m for m in messages if m.type == "human"]
    content = human[0].content
    assert isinstance(content, list)
    assert any(p.get("type") == "image_url" for p in content)
    # 历史序列化：human 的 content 提取为纯文本，图片块不上屏
    with TestClient(create_app(runtime=runtime)) as client:
        history = client.get("/v1/agents/chat/threads/mix-1").json()
    human_out = next(m for m in history["messages"] if m["role"] == "human")
    assert human_out["content"] == "对比一下"
    assert [a["format"] for a in human_out["attachments"]] == ["pdf", "jpeg"]
