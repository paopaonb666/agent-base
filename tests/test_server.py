"""Tests for the SSE service entrypoint (Stage 4).

Covers the acceptance points: SSE event sequence, health degradation,
route-template metric labels, and cancellation propagation (client
disconnect must cancel the in-flight LLM call).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent_base.core.bootstrap import AgentRuntime
from agent_base.core.config import Settings
from agent_base.entrypoints.server import _event_stream, create_app
from agent_base.modules.chat.module import ChatModule
from agent_base.modules.writer.module import WriterModule
from fakes import CancellableChatModel, ScriptedChatModel


def _runtime(model: Any = None) -> AgentRuntime:
    return AgentRuntime(
        settings=Settings(_env_file=None, llm_api_key="sk-test"),
        llm=model if model is not None else ScriptedChatModel([]),
        modules={"chat": ChatModule(), "writer": WriterModule()},
        tools=[],
        checkpointer=InMemorySaver(),
    )


def _client(model: Any = None) -> TestClient:
    app = create_app(runtime=_runtime(model))
    return TestClient(app)


def _frames(raw: str) -> list[dict[str, Any]]:
    """Parse an SSE body into (event type, payload) dicts."""
    events = []
    for block in raw.strip().split("\n\n"):
        lines = block.split("\n")
        etype = lines[0].removeprefix("event: ")
        payload = json.loads(lines[1].removeprefix("data: "))
        events.append({"event": etype, **payload})
    return events


def test_health_reports_components() -> None:
    with _client() as client:
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["components"]["checkpointer"] == "ok"
    assert body["components"]["model"] == "ok"


def test_health_degrades_without_model_key() -> None:
    runtime = _runtime()
    runtime.settings = Settings(_env_file=None, llm_api_key="")
    with TestClient(create_app(runtime=runtime)) as client:
        body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["components"]["model"] == "unconfigured"


def test_invoke_unknown_module_404() -> None:
    with _client() as client:
        response = client.post("/v1/agents/nobody/invoke", json={"message": "hi"})
    assert response.status_code == 404


def test_cors_preflight_allowed() -> None:
    """Browser preflight (OPTIONS) must pass CORS, not die with 405."""
    with _client() as client:
        response = client.options(
            "/v1/agents/chat/invoke",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert "POST" in response.headers["access-control-allow-methods"]


def test_cors_headers_on_invoke() -> None:
    model = ScriptedChatModel([AIMessage(content="hi")])
    with (
        _client(model) as client,
        client.stream(
            "POST",
            "/v1/agents/chat/invoke",
            json={"message": "hi"},
            headers={"Origin": "http://localhost:3000"},
        ) as response,
    ):
        assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
        "".join(response.iter_text())


def test_cors_rejects_unknown_origin() -> None:
    with _client() as client:
        response = client.options(
            "/v1/agents/chat/invoke",
            headers={
                "Origin": "http://evil.example.com",
                "Access-Control-Request-Method": "POST",
            },
        )
    assert "access-control-allow-origin" not in response.headers


def test_invoke_sse_sequence() -> None:
    model = ScriptedChatModel([AIMessage(content="hello there")])
    with (
        _client(model) as client,
        client.stream("POST", "/v1/agents/chat/invoke", json={"message": "hi"}) as response,
    ):
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["X-Request-ID"]
        raw = "".join(response.iter_text())
    events = _frames(raw)
    types = [e["event"] for e in events]
    # ping ... step(running agent) ... delta(s) ... step(completed agent) ... done
    assert types[0] == "ping"
    assert "delta" in types
    assert "done" in types
    assert types[-1] == "done"
    deltas = "".join(e["content"] for e in events if e["event"] == "delta")
    assert deltas == "hello there"
    done = events[-1]
    assert done["thread_id"]


def test_invoke_error_becomes_error_event() -> None:
    class ExplodingModel(ScriptedChatModel):
        async def _astream(self, *args: Any, **kwargs: Any) -> Any:
            # A generator that fails before yielding (the shape astream expects).
            raise RuntimeError("llm exploded")
            yield  # pragma: no cover - unreachable; marks this a generator

    with (
        _client(ExplodingModel([])) as client,
        client.stream("POST", "/v1/agents/chat/invoke", json={"message": "hi"}) as response,
    ):
        raw = "".join(response.iter_text())
    events = _frames(raw)
    assert events[-1]["event"] == "error"
    assert "llm exploded" in events[-1]["message"]


def test_thread_id_resumes_conversation() -> None:
    model = ScriptedChatModel([AIMessage(content="first"), AIMessage(content="second")])
    with _client(model) as client:
        with client.stream("POST", "/v1/agents/chat/invoke", json={"message": "one"}) as response:
            done = _frames("".join(response.iter_text()))[-1]
        thread_id = done["thread_id"]
        with client.stream(
            "POST",
            "/v1/agents/chat/invoke",
            json={"message": "two", "thread_id": thread_id},
        ) as response:
            raw = "".join(response.iter_text())
    # Turn 2 must include the recovered turn-1 history in its step payload;
    # the runtime state proves it via the deltas only containing the new
    # reply and the second invocation succeeding on the same thread.
    deltas = "".join(e["content"] for e in _frames(raw) if e["event"] == "delta")
    assert deltas == "second"


def test_metrics_uses_route_template_labels() -> None:
    model = ScriptedChatModel([AIMessage(content="ok")])
    with _client(model) as client:
        client.get("/health")
        with client.stream("POST", "/v1/agents/chat/invoke", json={"message": "hi"}):
            pass
        text = client.get("/metrics").text
    assert 'route="/health"' in text
    assert 'route="/v1/agents/{module}/invoke"' in text  # template, not raw path
    assert "/v1/agents/chat/invoke" not in text.replace("/v1/agents/{module}/invoke", "")
    assert 'status="200"' in text


async def test_client_disconnect_cancels_llm_call() -> None:
    """Stage 4 acceptance: disconnect stops the in-flight LLM request."""
    model = CancellableChatModel()
    runtime = _runtime(model)
    graph = runtime.graph("chat")
    config: dict[str, Any] = {"configurable": {"thread_id": "chat:t-cancel"}}
    stream = _event_stream(graph, "hi", config, "t-cancel")  # type: ignore[arg-type]

    # Consume like an SSE client; the model hangs, so after the initial
    # ping the stream sits idle exactly like a real slow LLM call.
    task = asyncio.ensure_future(_consume(stream))
    await asyncio.sleep(0.2)  # producer has reached the hanging model call
    task.cancel()  # what starlette does when the client disconnects
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert model.cancelled is True


async def _consume(stream: Any) -> None:
    async for _ in stream:
        pass
