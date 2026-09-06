"""SSE 事件契约的测试（阶段 4，A2 继承）。"""

from __future__ import annotations

import json

from agent_base.extensions.events import (
    DeltaEvent,
    DoneEvent,
    ErrorEvent,
    PingEvent,
    SourcesEvent,
    StepEvent,
    encode_sse,
)


def test_step_event_encodes_with_event_type() -> None:
    frame = encode_sse(StepEvent(name="agent", status="completed"))
    assert frame.startswith("event: step\n")
    payload = json.loads(frame.split("data: ", 1)[1])
    assert payload == {
        "type": "step",
        "name": "agent",
        "status": "completed",
        "detail": None,
    }


def test_delta_event_roundtrip() -> None:
    frame = encode_sse(DeltaEvent(content="he"))
    payload = json.loads(frame.split("data: ", 1)[1])
    assert payload["type"] == "delta"
    assert payload["content"] == "he"


def test_done_event_carries_thread_id() -> None:
    frame = encode_sse(DoneEvent(thread_id="t-42"))
    payload = json.loads(frame.split("data: ", 1)[1])
    assert payload["thread_id"] == "t-42"


def test_error_event_message() -> None:
    frame = encode_sse(ErrorEvent(message="boom"))
    payload = json.loads(frame.split("data: ", 1)[1])
    assert payload == {"type": "error", "message": "boom"}


def test_ping_event_is_minimal() -> None:
    assert encode_sse(PingEvent()) == 'event: ping\ndata: {"type":"ping"}\n\n'


def test_sources_event_reserved_for_future_modules() -> None:
    frame = encode_sse(SourcesEvent(sources=[{"title": "doc", "url": "https://x"}]))
    payload = json.loads(frame.split("data: ", 1)[1])
    assert payload["sources"][0]["title"] == "doc"


def test_every_frame_terminates_stream() -> None:
    for event in (
        StepEvent(name="n", status="running"),
        DeltaEvent(content="x"),
        DoneEvent(),
        ErrorEvent(message="m"),
        PingEvent(),
    ):
        assert encode_sse(event).endswith("\n\n")
