"""FastAPI + SSE 服务入口（阶段 4）。

运行::

    uvicorn agent_base.entrypoints.server:app --reload

端点：
- ``POST /v1/agents/{module}/invoke`` —— 一次对话轮，以 SSE 流的形式
  输出契约事件（ping / step / delta / done / error）；响应的 ``done``
  事件携带 thread id 以便恢复会话。
- ``GET /health`` —— 组件健康（checkpointer 探针 + 模型配置），
  出问题时返回 ``degraded`` 而不是崩溃（A4 继承）。
- ``GET /metrics`` —— Prometheus 文本格式，使用路由模板 label（B4/B5）。

取消：当客户端断开连接时，SSE 生成器的 ``finally`` 会取消生产者任务，
从而取消正在进行的 ``astream`` 以及随之而来的 LLM 请求——token 消耗
随即停止，而不是在后台跑完。这正是整个运行时异步优先的原因。

可观测性：每个请求都在一个 ``request_id`` 下运行（会尊重并回显
``X-Request-ID`` 头），因此日志可以端到端追踪（A1）。
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
    """一次对话轮。"""

    message: str
    thread_id: str | None = None  # None -> 创建一个新 thread


def _serialize_message(message: Any) -> dict[str, Any] | None:
    """把一条 LangChain 消息序列化为前端可渲染的简单结构。

    返回 ``None`` 表示这条消息不该显示（例如仅含 tool_calls 的 AI 消息）。
    前端 agent-base-ui 用它与 invoke 流式事件对齐，以便回放历史会话。
    """
    mtype = getattr(message, "type", "")
    content = getattr(message, "content", "")
    if isinstance(content, str):
        text = content
    elif content:
        text = "".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content
        )
    else:
        text = ""
    if mtype == "human":
        return {"role": "human", "content": text}
    if mtype == "ai":
        if not text:
            return None  # 仅含 tool_calls 的 AI 消息，没有可展示文本
        return {"role": "assistant", "content": text}
    if mtype == "tool":
        return {
            "role": "tool",
            "name": str(getattr(message, "name", "") or ""),
            "content": text,
        }
    return None


async def _produce(
    queue: asyncio.Queue[AgentEvent | None],
    graph: Any,
    message: str,
    config: RunnableConfig,
    user_thread_id: str,
) -> None:
    """把图事件推入队列；用哨兵值终止。"""
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
    """每隔一段时间发出一个 ping，让空闲的流保持打开。"""
    while True:
        await asyncio.sleep(HEARTBEAT_SECONDS)
        await queue.put(PingEvent())


async def _event_stream(
    graph: Any, message: str, config: RunnableConfig, user_thread_id: str
) -> AsyncIterator[str]:
    """一次对话轮的 SSE 帧（契约事件，已编码）。"""
    queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()
    producer = asyncio.create_task(_produce(queue, graph, message, config, user_thread_id))
    heartbeat = asyncio.create_task(_heartbeat(queue))
    try:
        # 立即发出存活信号，让客户端（和代理）在第一个模型 token
        # 到达之前就知道流已打开。
        yield encode_sse(PingEvent())
        while True:
            event = await queue.get()
            if event is None:
                break
            yield encode_sse(event)
    finally:
        # 客户端断开会走到这里：取消两个任务，使正在进行的 LLM 请求
        # 中止（token 消耗停止）。
        for task in (producer, heartbeat):
            task.cancel()
        await asyncio.gather(producer, heartbeat, return_exceptions=True)


def create_app(runtime: AgentRuntime | None = None) -> FastAPI:
    """构建服务应用；测试注入一个 runtime，生产环境则现场构建。"""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.runtime = runtime if runtime is not None else await create_runtime()
        yield

    app = FastAPI(title="agent-base", version=__version__, lifespan=lifespan)
    metrics = Metrics()
    app.state.metrics = metrics

    # CORS：SSE 端点服务于跨域浏览器客户端（agent-base-ui）。中间件会
    # 应答 OPTIONS 预检请求，并在响应上打上 Access-Control-Allow-Origin；
    # 没有它，预检请求会到达路由器并以 405 Method Not Allowed 失败。
    # 来源来自配置（CORS_ORIGINS，默认 localhost:3000）。
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
        # B4/B5 经验：用路由模板打 label，绝不用原始路径——
        # 原始路径（每个 thread id 一条）会让指标基数爆炸。
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
        # 按模块划分的 thread id：图共用一个 checkpointer；未划分命名空间
        # 的 id 会把它们的状态混在一起。
        config: RunnableConfig = {"configurable": {"thread_id": f"{module}:{user_thread_id}"}}
        return StreamingResponse(
            _event_stream(graph, body.message, config, user_thread_id),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )

    @app.get("/v1/agents/{module}/threads/{thread_id}")
    async def get_thread_history(module: str, thread_id: str, request: Request) -> dict[str, Any]:
        """返回某会话已持久化的消息历史（供前端恢复会话显示）。

        agent-base 没有 thread-list 端点；这个只读端点读取 checkpointer 里
        该 thread 的最新状态，并把其中的消息序列化成与 invoke 事件对齐的
        简单结构。前端 agent-base-ui 点击历史时用它回放对话。
        """
        runtime: AgentRuntime = request.app.state.runtime
        try:
            graph = runtime.graph(module)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        # thread id 按模块划分命名空间，与 invoke 端点保持一致。
        config: RunnableConfig = {"configurable": {"thread_id": f"{module}:{thread_id}"}}
        snapshot = await graph.aget_state(config)
        raw_messages = (snapshot.values or {}).get("messages", []) if snapshot else []
        messages = [
            serialized
            for serialized in (_serialize_message(m) for m in raw_messages)
            if serialized is not None
        ]
        return {"thread_id": thread_id, "module": module, "messages": messages}

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        """组件健康；``degraded`` 表示部分失败（A4）。"""
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
