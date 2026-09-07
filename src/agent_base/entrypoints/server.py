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
import contextlib
import logging
import re
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from agent_base import __version__
from agent_base.core.bootstrap import SUPERVISOR_MODULE, AgentRuntime, create_runtime
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
from agent_base.extensions.observability import (
    new_request_id,
    request_id,
    setup_logging,
)

logger = logging.getLogger(__name__)

HEARTBEAT_SECONDS = 15.0

# SSE 事件队列上限：慢客户端（半开连接、不读数据）停止消费时，生产者在
# 队列满后阻塞等待——内存有界，背压自然传导到 LLM 流。ping 允许丢弃。
SSE_QUEUE_MAXSIZE = 256

# 模型端点探活（/health）：结果缓存，避免每个探活请求都打真实网络。
MODEL_PROBE_TTL_SECONDS = 30.0
MODEL_PROBE_TIMEOUT_SECONDS = 3.0

_URL_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+.-]*://\S+")

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


def _safe_error_text(exc: Exception) -> str:
    """给客户端的错误摘要：类型名 + 消息，URL 脱敏、长度封顶。

    原始 ``str(exc)`` 可能携带内部端点、文件路径等部署细节；完整堆栈
    已经带着 request_id 进了服务端日志，客户端只需要可行动的摘要。
    """
    text = _URL_RE.sub("<redacted-url>", f"{type(exc).__name__}: {exc}")
    return text[:300]


# 探活结果缓存：{base_url: (monotonic 时间, 结果)}。
_model_probe_cache: dict[str, tuple[float, str]] = {}


async def _probe_model(settings: Settings) -> str:
    """探测 openai 兼容端点的网络可达性（带 TTL 缓存）。

    只证明"端点在网络层可达"——任何 HTTP 应答（含 401/404）都算 ok；
    配额、鉴权属于业务语义，不由 /health 判定。网络错误/超时 = error。
    """
    cache_key = settings.llm_base_url
    now = time.monotonic()
    hit = _model_probe_cache.get(cache_key)
    if hit is not None and now - hit[0] < MODEL_PROBE_TTL_SECONDS:
        return hit[1]
    status = "error"
    try:
        async with httpx.AsyncClient(timeout=MODEL_PROBE_TIMEOUT_SECONDS) as client:
            await client.get(
                f"{cache_key.rstrip('/')}/models",
                headers={"Authorization": f"Bearer {settings.llm_api_key.get_secret_value()}"},
            )
        status = "ok"  # 任何 HTTP 应答都证明可达
    except Exception:
        logger.warning("health: model endpoint probe failed for %s", cache_key)
    _model_probe_cache[cache_key] = (now, status)
    return status


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
        await queue.put(ErrorEvent(message=_safe_error_text(exc)))
    finally:
        # 哨兵必须送达：队列暂时满就等消费者排空。若任务已被取消（客户端
        # 断开），sleep 抛 CancelledError，循环随之终止。
        while True:
            try:
                queue.put_nowait(None)
                break
            except asyncio.QueueFull:
                await asyncio.sleep(0.05)


async def _heartbeat(queue: asyncio.Queue[AgentEvent | None]) -> None:
    """每隔一段时间发出一个 ping，让空闲的流保持打开。

    ping 是可丢弃的保活信号：队列满（慢客户端背压）时直接丢，
    不与关键事件争抢容量。
    """
    while True:
        await asyncio.sleep(HEARTBEAT_SECONDS)
        with contextlib.suppress(asyncio.QueueFull):
            queue.put_nowait(PingEvent())


async def _event_stream(
    graph: Any, message: str, config: RunnableConfig, user_thread_id: str
) -> AsyncIterator[str]:
    """一次对话轮的 SSE 帧（契约事件，已编码）。"""
    queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue(maxsize=SSE_QUEUE_MAXSIZE)
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
        rt = runtime if runtime is not None else await create_runtime()
        # 日志契约（LOG_JSON / request_id 过滤器）在 server 路径同样生效，
        # 而不是只在 CLI——否则生产日志既非结构化也无法端到端追踪（A1）。
        setup_logging(json_lines=rt.settings.log_json)
        app.state.runtime = rt
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
        # X-Request-ID 是非简单响应头：不 expose 的话浏览器 JS 读不到，
        # 前端无法用它做端到端追踪。
        expose_headers=["X-Request-ID"],
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

    @app.get("/v1/modules")
    async def list_modules(request: Request) -> dict[str, Any]:
        """列出已注册的模块（供前端配置面板选择，免去试错模块名）。"""
        rt: AgentRuntime = request.app.state.runtime
        modules = [
            {"name": m.name, "description": str(getattr(m, "description", ""))}
            for m in rt.modules.values()
        ]
        modules.append(
            {
                "name": SUPERVISOR_MODULE,
                "description": "多 Agent 协作：把所有已注册模块编排为 sub-agent",
            }
        )
        return {"modules": modules}

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        """组件健康；``degraded`` 表示部分失败（A4）。"""
        rt: AgentRuntime = request.app.state.runtime
        components: dict[str, str] = {}
        if rt.checkpointer is None:
            components["checkpointer"] = "unconfigured"
        else:
            try:
                await rt.checkpointer.aget_tuple({"configurable": {"thread_id": "__health__"}})
                components["checkpointer"] = "ok"
            except Exception:
                components["checkpointer"] = "error"
        key = rt.settings.llm_api_key.get_secret_value().strip()
        if not key:
            components["model"] = "unconfigured"
        elif not rt.settings.llm_base_url.startswith(("http://", "https://")):
            components["model"] = "misconfigured"
        elif rt.settings.health_probe_model:
            # 真实探活由开关控制（LLM_HEALTH_PROBE_MODEL=true）：探活会打
            # 真实网络，默认关闭——负载均衡器的主动检查通常已覆盖此需求。
            components["model"] = await _probe_model(rt.settings)
        else:
            components["model"] = "ok"
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
