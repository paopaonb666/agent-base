"""FastAPI + SSE service entrypoint (Stage 4).

Run::

    uvicorn agent_base.entrypoints.server:app --reload

Endpoints:
- ``POST /v1/agents/{module}/invoke`` — one conversation turn as an SSE
  stream of contract events (ping / step / delta / done / error); the
  response carries the thread id in the ``done`` event for resumption.
- ``GET /health`` — component health (checkpointer probe + model config),
  ``degraded`` instead of crashing (A4 inheritance).
- ``GET /metrics`` — Prometheus text format, route-template labels (B4/B5).

Cancellation: when the client disconnects, the SSE generator's ``finally``
cancels the producer task, which cancels the in-flight ``astream`` and
with it the LLM request — token spend stops instead of finishing in the
background. That is why the whole runtime is async-first.

Observability: every request runs under a ``request_id`` (``X-Request-ID``
header honored, echoed back) so logs can be traced end to end (A1).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from agent_base import __version__
from agent_base.core.bootstrap import AgentRuntime, create_runtime
from agent_base.core.config import Settings
from agent_base.extensions.events import (
    AgentEvent,
    DeltaEvent,
    DoneEvent,
    ErrorEvent,
    PingEvent,
    StepEvent,
    encode_sse,
)
from agent_base.extensions.metrics import Metrics
from agent_base.extensions.observability import new_request_id, request_id

logger = logging.getLogger(__name__)

HEARTBEAT_SECONDS = 15.0

_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


class InvokeRequest(BaseModel):
    """One conversation turn."""

    message: str
    thread_id: str | None = None  # None -> a new thread is created


async def _produce(
    queue: asyncio.Queue[AgentEvent | None],
    graph: Any,
    message: str,
    config: RunnableConfig,
    user_thread_id: str,
) -> None:
    """Push graph events onto the queue; sentinel-terminate."""
    running: set[str] = set()
    try:
        stream: Any = graph.astream(
            {"messages": [HumanMessage(content=message)]},
            config,
            stream_mode=["messages", "updates"],
        )
        async for mode, payload in stream:
            if mode == "messages":
                chunk, metadata = payload
                node = str(metadata.get("langgraph_node") or "")
                if node and node not in running:
                    running.add(node)
                    await queue.put(StepEvent(name=node, status="running"))
                if isinstance(chunk, AIMessageChunk) and chunk.content:
                    await queue.put(DeltaEvent(content=str(chunk.content)))
            elif mode == "updates":
                for node in payload:
                    if str(node).startswith("__"):
                        continue
                    await queue.put(StepEvent(name=str(node), status="completed"))
        await queue.put(DoneEvent(thread_id=user_thread_id))
    except Exception as exc:
        logger.exception("sse: stream failed")
        await queue.put(ErrorEvent(message=str(exc)))
    finally:
        queue.put_nowait(None)


async def _heartbeat(queue: asyncio.Queue[AgentEvent | None]) -> None:
    """Emit a ping every interval so idle streams stay open."""
    while True:
        await asyncio.sleep(HEARTBEAT_SECONDS)
        await queue.put(PingEvent())


async def _event_stream(
    graph: Any, message: str, config: RunnableConfig, user_thread_id: str
) -> AsyncIterator[str]:
    """SSE frames for one conversation turn (contract events, encoded)."""
    queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()
    producer = asyncio.create_task(_produce(queue, graph, message, config, user_thread_id))
    heartbeat = asyncio.create_task(_heartbeat(queue))
    try:
        # Immediate liveness signal so clients (and proxies) see the
        # stream is open before the first model token arrives.
        yield encode_sse(PingEvent())
        while True:
            event = await queue.get()
            if event is None:
                break
            yield encode_sse(event)
    finally:
        # Client disconnect lands here: cancel both tasks so the
        # in-flight LLM request is aborted (token spend stops).
        for task in (producer, heartbeat):
            task.cancel()
        await asyncio.gather(producer, heartbeat, return_exceptions=True)


def create_app(runtime: AgentRuntime | None = None) -> FastAPI:
    """Build the service app; tests inject a runtime, production builds one."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.runtime = runtime if runtime is not None else await create_runtime()
        yield

    app = FastAPI(title="agent-base", version=__version__, lifespan=lifespan)
    metrics = Metrics()
    app.state.metrics = metrics

    # CORS: the SSE endpoint serves cross-origin browser clients
    # (agent-base-ui). The middleware answers the OPTIONS preflight and
    # stamps Access-Control-Allow-Origin on responses; without it the
    # preflight reaches the router and dies with 405 Method Not Allowed.
    # Origins come from config (CORS_ORIGINS, default localhost:3000).
    cors_settings = runtime.settings if runtime is not None else Settings()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=False,
    )

    @app.middleware("http")
    async def observability_middleware(request: Request, call_next: Any) -> Any:
        rid = request.headers.get("X-Request-ID") or new_request_id()
        started = time.perf_counter()
        with request_id(rid):
            response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        # B4/B5 lesson: label by ROUTE TEMPLATE, never the raw path —
        # raw paths (one per thread id) would explode metric cardinality.
        route = request.scope.get("route")
        template = getattr(route, "path", request.url.path)
        metrics.observe(
            request.method, str(template), response.status_code, time.perf_counter() - started
        )
        return response

    @app.post("/v1/agents/{module}/invoke")
    async def invoke(module: str, body: InvokeRequest, request: Request) -> StreamingResponse:
        runtime: AgentRuntime = request.app.state.runtime
        try:
            graph = runtime.graph(module)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        user_thread_id = body.thread_id or new_request_id()
        # Module-scoped thread id: graphs share one checkpointer; an
        # un-namespaced id would mix their states.
        config: RunnableConfig = {"configurable": {"thread_id": f"{module}:{user_thread_id}"}}
        return StreamingResponse(
            _event_stream(graph, body.message, config, user_thread_id),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        """Component health; ``degraded`` reports partial failure (A4)."""
        runtime: AgentRuntime = request.app.state.runtime
        components: dict[str, str] = {}
        if runtime.checkpointer is None:
            components["checkpointer"] = "unconfigured"
        else:
            try:
                await runtime.checkpointer.aget_tuple({"configurable": {"thread_id": "__health__"}})
                components["checkpointer"] = "ok"
            except Exception:
                components["checkpointer"] = "error"
        key = runtime.settings.llm_api_key.get_secret_value().strip()
        components["model"] = "ok" if key else "unconfigured"
        status = "ok" if all(v == "ok" for v in components.values()) else "degraded"
        return {"status": status, "components": components}

    @app.get("/metrics")
    async def metrics_endpoint(request: Request) -> PlainTextResponse:
        metrics: Metrics = request.app.state.metrics
        return PlainTextResponse(
            metrics.render(), media_type="text/plain; version=0.0.4; charset=utf-8"
        )

    return app


app = create_app()
