"""FastAPI + SSE 服务入口（阶段 4）。

运行::

    uvicorn agent_base.entrypoints.server:app --reload

端点：
- ``POST /v1/agents/{module}/invoke`` —— 一次对话轮，以 SSE 流的形式
  输出契约事件（ping / step / delta / sources / done / error）；响应的
  ``done`` 事件携带 thread id 以便恢复会话。``sources`` 与工具发出的
  进度 ``step`` 来自工具经 custom stream 发的载荷（M3.5），经封闭的
  事件模型校验后透传。
- ``POST/GET/PATCH/DELETE /v1/memory`` —— 长期记忆的管理端点（M6b），
  以 ``X-User-Id`` 头为作用域（缺省 ``default``）。
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
import base64
import contextlib
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator, Coroutine
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse
from langchain_core.messages import AIMessageChunk, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from agent_base import __version__
from agent_base.core.bootstrap import SUPERVISOR_MODULE, AgentRuntime, create_runtime
from agent_base.core.config import Settings
from agent_base.extensions.events import (
    AgentEvent,
    DeltaEvent,
    DoneEvent,
    ErrorEvent,
    PingEvent,
    SourcesEvent,
    StepEvent,
    ToolCallEvent,
    encode_sse,
)
from agent_base.extensions.filestore import UploadedFileInfo
from agent_base.extensions.metrics import MEMORY_METRICS, TOOL_METRICS, Metrics
from agent_base.extensions.observability import (
    new_request_id,
    request_id,
    setup_logging,
)
from agent_base.memory.service import MemoryService
from agent_base.memory.store import KNOWN_MEMORY_KINDS, KNOWN_MEMORY_STATUSES
from agent_base.tools.parsing import DocumentParseError, parse_document

logger = logging.getLogger(__name__)

HEARTBEAT_SECONDS = 15.0

# SSE 事件队列上限：慢客户端（半开连接、不读数据）停止消费时，生产者在
# 队列满后阻塞等待——内存有界，背压自然传导到 LLM 流。ping 允许丢弃。
SSE_QUEUE_MAXSIZE = 256

# 线程列表端点的扫描/数量上限：alist 按 checkpoint 粒度迭代（一个线程
# 有多轮 checkpoint），无上限会在大库上退化成全表遍历。扫描预算只计
# 目标模块的 checkpoint，另设一个覆盖所有模块的总行数上限兜底。
_THREAD_LIST_SCAN_LIMIT = 2000
_THREAD_LIST_TOTAL_LIMIT = 20_000
_THREAD_LIST_LIMIT = 50

# 模型端点探活（/health）：结果缓存，避免每个探活请求都打真实网络。
MODEL_PROBE_TTL_SECONDS = 30.0
MODEL_PROBE_TIMEOUT_SECONDS = 3.0

_URL_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+.-]*://\S+")

_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

# 客户端提供的 X-Request-ID 必须通过此白名单，否则拒绝并另发新 id：
# 该值会被回显到响应头并写进每条日志，放行任意字符串等于允许
# 伪造日志行 / 触发非法响应头。
_REQUEST_ID_RE = re.compile(r"[A-Za-z0-9._-]{1,64}")


class InvokeRequest(BaseModel):
    """一次对话轮。"""

    message: str = Field(min_length=1, max_length=100_000)
    # thread_id 是 checkpointer 的一部分键：限定字符集与长度，避免任意
    # 字符串直接落库/进日志。
    thread_id: str | None = Field(
        default=None, max_length=128, pattern=r"^[\w.-]+$"
    )  # None -> 创建一个新 thread
    # 附件对话（M4b）：本条消息引用的上传文件 id（≤5 个），服务端把解析
    # 文本作为 SystemMessage 注入上下文——用户气泡保持干净。
    attachments: list[str] = Field(default_factory=list, max_length=5)


class MemoryCreateRequest(BaseModel):
    """手工写入一条长期记忆（M6b 管理端点）。"""

    content: str = Field(min_length=1, max_length=20_000)
    kind: str = "semantic"
    tags: list[str] = Field(default_factory=list, max_length=10)
    # None → "*"（全模块共享）；模块名不校验（模块清单可动态变化）。
    agent_id: str | None = Field(default=None, max_length=64)
    salience: float = Field(default=0.5, ge=0.0, le=1.0)


class MemoryPatchRequest(BaseModel):
    """部分更新一条长期记忆；content 变化会触发重新向量化。"""

    content: str | None = Field(default=None, min_length=1, max_length=20_000)
    tags: list[str] | None = Field(default=None, max_length=10)
    salience: float | None = Field(default=None, ge=0.0, le=1.0)
    status: str | None = None


def _extract_text(content: Any) -> str:
    """提取消息 content 里的纯文本：str 直接返回；列表型 content
    （部分 provider 的多段内容）拼接各段的 text 字段，避免把
    Python repr 原样发给客户端。"""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        return "".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content
        )
    return str(content)


# 图片附件（多模态对话）：按 magic bytes 嗅探真实类型——扩展名可伪装，
# 内容头不会。webp 的 RIFF/WEBP 头单独检查。
_IMAGE_SIGNATURES: tuple[tuple[bytes, str, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png", "png"),
    (b"\xff\xd8\xff", "image/jpeg", "jpeg"),
    (b"GIF87a", "image/gif", "gif"),
    (b"GIF89a", "image/gif", "gif"),
)
IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "gif"}
# filestore 中图片附件的归一化 format 值（mime 去前缀）。
IMAGE_STORED_FORMATS = {"png", "jpeg", "webp", "gif"}


def _sniff_image_mime(data: bytes) -> str | None:
    """按 magic bytes 识别图片真实类型；无法识别返回 None。"""
    for signature, mime, _ in _IMAGE_SIGNATURES:
        if data.startswith(signature):
            return mime
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _known_module(rt: AgentRuntime, module: str) -> bool:
    """invoke 之外的线程端点也必须接受 supervisor 保留名：supervisor 的
    invoke 走 ``runtime.graph`` 的特例路径，能正常产生 checkpointer 记录；
    若 list/history/delete 只认 ``rt.modules``，supervisor 的对话就会
    "能聊但不能列历史、不能删"——前后端语义不一致。
    """
    return module in rt.modules or module == SUPERVISOR_MODULE


def _memory_user_id(request: Request) -> str:
    """解析记忆作用域的 user_id（M6）：X-User-Id 头，缺省 "default"。

    与 X-Request-ID 同一白名单正则：该值会进记忆表与日志，放行任意
    字符串等于允许伪造日志行 / 注入脏数据。非法值以 400 拒绝而不是
    静默替换——调用方应当修自己的头，而不是猜服务端用了什么值。
    """
    raw = request.headers.get("X-User-Id")
    if not raw:
        return "default"
    if not _REQUEST_ID_RE.fullmatch(raw):
        raise HTTPException(
            status_code=400,
            detail="X-User-Id 非法：仅允许字母/数字/点/下划线/连字符，1-64 字符",
        )
    return raw


def _require_memory(rt: AgentRuntime) -> MemoryService:
    """取记忆服务；未启用时以 503 给出可读原因。"""
    if rt.memory is None:
        raise HTTPException(status_code=503, detail="记忆系统未启用（MEMORY_ENABLED=false）")
    return rt.memory


def _build_capture_hook(
    runtime: AgentRuntime, module: str, user_thread_id: str, graph: Any, config: RunnableConfig
) -> Coroutine[Any, Any, None] | None:
    """构造轮次完成后的记忆形成回调（M6c）；管线不可用时返回 None。

    回调从 checkpointer 快照取本轮完整消息（含工具轮次），交给
    ``capture_turn``——形成管线内部失败安全，这里再兜一层异常。
    user_id 取自 config（invoke 端点已从 X-User-Id 头写入 configurable）。
    """
    memory = runtime.memory
    if memory is None or memory.pipeline is None:
        return None

    async def _capture() -> None:
        try:
            snapshot = await graph.aget_state(config)
            messages = (snapshot.values or {}).get("messages", []) if snapshot else []
            if not messages:
                return
            raw_user = (config.get("configurable") or {}).get("user_id")
            user_id = raw_user if isinstance(raw_user, str) and raw_user else "default"
            await memory.capture_turn(
                user_id=user_id,
                agent_id=module,
                thread_id=f"{module}:{user_thread_id}",
                messages=messages,
            )
        except Exception:
            logger.warning("memory: 后台记忆形成失败（不影响对话）", exc_info=True)

    return _capture()


def _serialize_message(message: Any) -> dict[str, Any] | None:
    """把一条 LangChain 消息序列化为前端可渲染的简单结构。

    返回 ``None`` 表示这条消息不该显示（例如仅含 tool_calls 的 AI 消息）。
    前端 agent-base-ui 用它与 invoke 流式事件对齐，以便回放历史会话。

    工具调用可观测（M5）：AI 消息携带 ``tool_calls``（id/name/args——
    模型给定的调用参数），tool 消息携带 ``tool_call_id``/``status``——
    前端按 id 把两者配对，回放出完整的"参数 + 结果"调用面板。仅含
    tool_calls 的 AI 消息也保留（它是调用参数的唯一持久化位置）。
    """
    mtype = getattr(message, "type", "")
    text = _extract_text(getattr(message, "content", ""))
    if mtype == "system":
        # 附件注入的系统消息：内容只在当轮给模型，历史回放由 human 消息
        # 上的附件元数据承载——blob 永不上屏。
        return None
    if mtype == "human":
        attachments = (getattr(message, "additional_kwargs", {}) or {}).get("attachments")
        out: dict[str, Any] = {"role": "human", "content": text}
        if isinstance(attachments, list) and attachments:
            out["attachments"] = attachments
        return out
    if mtype == "ai":
        tool_calls: list[dict[str, Any]] = [
            {
                "id": str(tc.get("id") or ""),
                "name": str(tc.get("name") or ""),
                "args": tc.get("args") if isinstance(tc.get("args"), dict) else {},
            }
            for tc in (getattr(message, "tool_calls", None) or [])
            if isinstance(tc, dict)
        ]
        if not text and not tool_calls:
            return None  # 既无文本也无调用的空 AI 消息，没有可展示内容
        return {
            "role": "assistant",
            "content": text,
            **({"tool_calls": tool_calls} if tool_calls else {}),
        }
    if mtype == "tool":
        raw_status = str(getattr(message, "status", "") or "success")
        return {
            "role": "tool",
            "name": str(getattr(message, "name", "") or ""),
            "content": text,
            "tool_call_id": str(getattr(message, "tool_call_id", "") or ""),
            "status": "error" if raw_status == "error" else "ok",
        }
    return None


def _safe_error_text(exc: Exception) -> str:
    """给客户端的错误摘要：类型名 + 消息，URL 脱敏、长度封顶。

    原始 ``str(exc)`` 可能携带内部端点、文件路径等部署细节；完整堆栈
    已经带着 request_id 进了服务端日志，客户端只需要可行动的摘要。
    """
    text = _URL_RE.sub("<redacted-url>", f"{type(exc).__name__}: {exc}")
    return text[:300]


def _decode_tool_event(payload: Any) -> AgentEvent | None:
    """把工具经 custom stream 发来的载荷校验成契约事件。

    工具与前端之间的每一个事件都必须经过封闭的 Pydantic 模型（M3.5
    的安全前提：坏掉/恶意的工具不能把未校验数据直接透给浏览器）。
    未知类型或畸形载荷被丢弃并记日志——事件是尽力而为的旁路信号，
    绝不让它打断对话流。
    """
    if not isinstance(payload, dict):
        logger.warning("sse: dropping non-dict custom event: %r", type(payload).__name__)
        return None
    kind = payload.get("type")
    try:
        if kind == "step":
            return StepEvent.model_validate(payload)
        if kind == "sources":
            return SourcesEvent.model_validate(payload)
        if kind == "tool_call":
            return ToolCallEvent.model_validate(payload)
    except ValueError as exc:
        logger.warning("sse: dropping malformed %r event: %s", kind, exc)
        return None
    logger.warning("sse: dropping unknown custom event type: %r", kind)
    return None


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
    messages_input: list[Any],
    config: RunnableConfig,
    user_thread_id: str,
    on_complete: Coroutine[Any, Any, None] | None = None,
) -> None:
    """把图事件推入队列；用哨兵值终止。

    ``on_complete``（M6c）：轮次正常完成后触发的后台回调（记忆形成管
    线）。fire-and-forget 的独立任务——不阻塞哨兵送达，客户端中途断开
    或本轮出错时不会被调用。
    """
    running: set[str] = set()
    try:
        stream: Any = graph.astream(
            {"messages": messages_input},
            config,
            # custom：工具通过 get_stream_writer 发的 UI 事件（M3.5）——
            # 联网搜索的进度步骤与来源引用由此到达前端。
            stream_mode=["messages", "updates", "custom"],
        )
        async for mode, payload in stream:
            if mode == "messages":
                chunk, metadata = payload
                node = str(metadata.get("langgraph_node") or "")
                if node and node not in running:
                    running.add(node)
                    await queue.put(StepEvent(name=node, status="running"))
                if isinstance(chunk, AIMessageChunk) and chunk.content:
                    await queue.put(DeltaEvent(content=_extract_text(chunk.content)))
            elif mode == "custom":
                event = _decode_tool_event(payload)
                if event is not None:
                    await queue.put(event)
            elif mode == "updates":
                for node in payload:
                    if str(node).startswith("__"):
                        continue
                    await queue.put(StepEvent(name=str(node), status="completed"))
        await queue.put(DoneEvent(thread_id=user_thread_id))
        if on_complete is not None:
            # 独立任务：随后的哨兵与 finally 取消都不影响它跑完。
            asyncio.get_running_loop().create_task(on_complete)
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
    graph: Any,
    messages_input: list[Any],
    config: RunnableConfig,
    user_thread_id: str,
    on_complete: Coroutine[Any, Any, None] | None = None,
) -> AsyncIterator[str]:
    """一次对话轮的 SSE 帧（契约事件，已编码）。"""
    queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue(maxsize=SSE_QUEUE_MAXSIZE)
    producer = asyncio.create_task(
        _produce(queue, graph, messages_input, config, user_thread_id, on_complete)
    )
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
        client_rid = request.headers.get("X-Request-ID") or ""
        rid = client_rid if _REQUEST_ID_RE.fullmatch(client_rid) else new_request_id()
        started = time.perf_counter()
        with request_id(rid):
            response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        # B4/B5 经验：用路由模板打 label，绝不用原始路径——
        # 原始路径（每个 thread id 一条）会让指标基数爆炸。404/405 没有
        # 匹配路由，用常量兜底而不是原始路径。
        route = request.scope.get("route")
        template = getattr(route, "path", None) or "unmatched"
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
            # UnknownModuleError 是 KeyError 子类：str() 会带 KeyError 的
            # 引号包装，取 args[0] 给客户端一条干净的报错。
            detail = str(exc.args[0]) if exc.args else "unknown module"
            raise HTTPException(status_code=404, detail=detail) from exc
        user_thread_id = body.thread_id or new_request_id()
        # 按模块划分的 thread id：图共用一个 checkpointer；未划分命名空间
        # 的 id 会把它们的状态混在一起。user_id（M6）进 configurable：
        # 记忆工具经 RunnableConfig 注入读取（M6e）。
        scope_user_id = _memory_user_id(request)
        config: RunnableConfig = {
            "configurable": {
                "thread_id": f"{module}:{user_thread_id}",
                "user_id": scope_user_id,
            }
        }
        # 附件（M4b）：解析引用的 file_id → 取记录 → 绑定线程 → 注入。
        messages_input: list[Any] = []
        if body.attachments:
            file_store = runtime.file_store
            if file_store is None:
                raise HTTPException(status_code=503, detail="附件存储未启用（存储后端不可用）")
            attachment_infos = await file_store.get_many(body.attachments)
            missing = sorted(set(body.attachments) - {info.file_id for info in attachment_infos})
            if missing:
                raise HTTPException(
                    status_code=400,
                    detail=f"附件不存在或已过期：{', '.join(missing)}；请重新上传",
                )
            await file_store.bind_thread(body.attachments, f"{module}:{user_thread_id}")
            blocks = []
            image_parts: list[UploadedFileInfo] = []
            for info in attachment_infos:
                if info.format in IMAGE_STORED_FORMATS:
                    image_parts.append(info)
                    continue
                if info.extracted_text.strip():
                    pages = f"{info.pages} 页，" if info.pages else ""
                    blocks.append(
                        f"[附件文件：{info.filename}（{info.format}，{pages}"
                        f"{info.text_len} 字符）]\n{info.extracted_text}"
                    )
                else:
                    note = info.warning or "无法提取文本"
                    blocks.append(f"[附件文件：{info.filename}]（{note}）")
            if blocks:
                messages_input.append(
                    SystemMessage(
                        content="以下是用户上传的附件内容，供回答时参考：\n\n" + "\n\n".join(blocks)
                    )
                )
            # 图片：多模态 content blocks（vision 模型直接看图）。
            human_content: Any = body.message
            if image_parts:
                content: list[dict[str, Any]] = [{"type": "text", "text": body.message}]
                for img in image_parts:
                    encoded = base64.b64encode(img.content).decode("ascii")
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/{img.format};base64,{encoded}"},
                        }
                    )
                human_content = content
            attachments_meta = [info.meta() for info in attachment_infos]
            messages_input.append(
                HumanMessage(
                    content=human_content,
                    additional_kwargs={"attachments": attachments_meta},
                )
            )
        else:
            messages_input.append(HumanMessage(content=body.message))
        # 记忆形成（M6c）：轮次正常完成后在后台跑抽取/整合/画像/摘要。
        # 需要形成管线可用（llm 已装配）且记忆系统启用。
        on_complete = _build_capture_hook(runtime, module, user_thread_id, graph, config)
        return StreamingResponse(
            _event_stream(graph, messages_input, config, user_thread_id, on_complete),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )

    @app.get("/v1/agents/{module}/threads")
    async def list_threads(module: str, request: Request) -> dict[str, Any]:
        """列出某模块命名空间下的所有已持久化线程（供前端侧栏显示）。

        只依赖 checkpointer 的公开 ``alist`` 接口（跨后端通用），按
        ``module:`` 前缀过滤、每线程取最新 checkpoint。标题取该线程
        第一条用户消息（与前端本地保存规则一致）。迭代设上限，防止
        大库上无界扫描。
        """
        rt: AgentRuntime = request.app.state.runtime
        if not _known_module(rt, module):
            raise HTTPException(status_code=404, detail=f"unknown module {module!r}")
        if rt.checkpointer is None:
            return {"threads": []}
        prefix = f"{module}:"
        threads: dict[str, dict[str, Any]] = {}
        scanned = 0
        total = 0
        async for tp in rt.checkpointer.alist(None):
            # 扫描预算只计目标模块的 checkpoint：否则繁忙模块的行会耗尽
            # 预算，让目标模块在大库里返回空列表。总行数上限兜底，避免
            # 无界遍历。
            total += 1
            if total > _THREAD_LIST_TOTAL_LIMIT:
                break
            full_id = tp.config["configurable"].get("thread_id", "")
            if not full_id.startswith(prefix):
                continue
            scanned += 1
            if scanned > _THREAD_LIST_SCAN_LIMIT:
                break
            short_id = full_id[len(prefix) :]
            if short_id in threads or not short_id:
                continue
            messages = tp.checkpoint.get("channel_values", {}).get("messages", [])
            title = next(
                (str(m.content)[:60] for m in messages if getattr(m, "type", "") == "human"),
                short_id,
            )
            try:
                updated_at = int(
                    datetime.fromisoformat(str(tp.checkpoint.get("ts", ""))).timestamp() * 1000
                )
            except ValueError:
                updated_at = 0
            threads[short_id] = {
                "thread_id": short_id,
                "module": module,
                "title": title,
                "updated_at": updated_at,
            }
            if len(threads) >= _THREAD_LIST_LIMIT:
                break
        ordered = sorted(threads.values(), key=lambda t: t["updated_at"], reverse=True)
        return {"threads": ordered}

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
            detail = str(exc.args[0]) if exc.args else "unknown module"
            raise HTTPException(status_code=404, detail=detail) from exc
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

    @app.get("/v1/agents/{module}/threads/{thread_id}/tool-calls")
    async def list_tool_calls(
        module: str, thread_id: str, request: Request, limit: int = 100
    ) -> dict[str, Any]:
        """返回某会话的工具调用审计记录（按时间升序，最新在末尾）。

        数据来自 ``tool_call_records`` 审计表（跟随 checkpointer 后端落库）：
        无论成功、超时还是异常都有记录，含参数 JSON、结果文本与耗时。
        ``limit`` 由 FastAPI 做数值校验（1-500）。
        """
        rt: AgentRuntime = request.app.state.runtime
        if not _known_module(rt, module):
            raise HTTPException(status_code=404, detail=f"unknown module {module!r}")
        recorder = rt.tool_recorder
        if recorder is None:
            return {"tool_calls": []}
        records = await recorder.list_for_thread(
            f"{module}:{thread_id}", limit=max(1, min(limit, 500))
        )
        return {"tool_calls": records}

    @app.delete("/v1/agents/{module}/threads/{thread_id}")
    async def delete_thread(module: str, thread_id: str, request: Request) -> dict[str, Any]:
        """删除某模块命名空间下的一个已持久化线程。

        前端删除会话时调用：只删本地索引会让线程在清缓存/换设备后
        "复活"。按 invoke 端点相同的 ``module:thread_id`` 命名空间删除
        checkpointer 里的全部 checkpoint。
        """
        rt: AgentRuntime = request.app.state.runtime
        if not _known_module(rt, module):
            raise HTTPException(status_code=404, detail=f"unknown module {module!r}")
        if rt.checkpointer is None:
            return {"deleted": False, "reason": "checkpointer unconfigured"}
        # adelete_thread 接收原始 thread_id 字符串；与 invoke/get 一致地
        # 使用 module:thread_id 命名空间。
        await rt.checkpointer.adelete_thread(f"{module}:{thread_id}")
        if rt.file_store is not None:
            await rt.file_store.delete_for_thread(f"{module}:{thread_id}")
        if rt.memory is not None:
            # 记忆级联（M6）：清该线程的滚动摘要与知识库分块；跨会话
            # 记忆刻意保留（出处仍在 source_thread_id）。
            await rt.memory.purge_thread(f"{module}:{thread_id}")
        return {"deleted": True}

    @app.post("/v1/agents/{module}/files")
    async def upload_file(module: str, request: Request, file: UploadFile) -> dict[str, Any]:
        """上传附件（M4a 解析层 + M4c 图片多模态的 HTTP 入口）。

        文档（pdf/docx/txt/md）走 ``tools/parsing`` 提取文本，注入对话
        上下文；图片（png/jpg/webp/gif）按 magic bytes 嗅探后整字节入库，
        invoke 时以多模态 content blocks 注入 vision 模型。两类都返回
        元信息（不含全文与字节）；解析/校验失败以 400 返回可读原因。
        """
        rt: AgentRuntime = request.app.state.runtime
        if not _known_module(rt, module):
            raise HTTPException(status_code=404, detail=f"unknown module {module!r}")
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="上传文件为空")
        max_bytes = rt.settings.doc_parse_max_input_bytes
        if len(data) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"文件超出大小上限：{len(data)} 字节 > {max_bytes} 字节",
            )
        suffix = (file.filename or "").rsplit(".", 1)[-1].lower()
        file_store = rt.file_store
        if file_store is None:
            raise HTTPException(status_code=503, detail="附件存储未启用（存储后端不可用）")
        if suffix in IMAGE_EXTENSIONS:
            # 图片：不走文本解析，按 magic bytes 校验后整字节入库，
            # invoke 时以多模态 content blocks 注入 vision 模型。
            mime = _sniff_image_mime(data)
            if mime is None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"文件内容不是有效的图片：扩展名 {suffix!r} 与实际"
                        "内容不符（magic 校验失败）"
                    ),
                )
            info = UploadedFileInfo(
                file_id=uuid.uuid4().hex[:12],
                filename=file.filename or "",
                format=mime.split("/", 1)[1],
                pages=None,
                paragraphs=None,
                truncated=False,
                text_len=0,
                extracted_text="",
                content=data,
            )
            await file_store.save(info)
            asyncio.get_running_loop().create_task(file_store.purge_orphans())
            return info.meta()
        try:
            doc = parse_document(data, filename=file.filename or "")
        except DocumentParseError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        # 空提取文本（扫描件/纯图片 PDF）不是错误，但要显式告知前端。
        warning = ""
        if not doc.text.strip():
            warning = "未能从文件中提取到文本：可能是扫描件或纯图片 PDF"
        info = UploadedFileInfo(
            file_id=uuid.uuid4().hex[:12],
            filename=file.filename or "",
            format=doc.format,
            pages=doc.pages,
            paragraphs=doc.paragraphs,
            truncated=doc.truncated,
            text_len=len(doc.text),
            extracted_text=doc.text,
            content=data,
            warning=warning,
        )
        await file_store.save(info)
        # 机会式清理 24h 未绑定的孤儿附件（fire-and-forget）。
        asyncio.get_running_loop().create_task(file_store.purge_orphans())
        return info.meta()

    @app.post("/v1/memory")
    async def create_memory(body: MemoryCreateRequest, request: Request) -> dict[str, Any]:
        """手工写入一条长期记忆（M6b 管理端点）。

        与管线写入（M6c）同一条路径：内容落库前尽力向量化，embedding
        不可用时存纯文本、检索自动走关键词路径。响应不含 embedding 字节。
        """
        rt: AgentRuntime = request.app.state.runtime
        memory = _require_memory(rt)
        user_id = _memory_user_id(request)
        if body.kind not in KNOWN_MEMORY_KINDS:
            raise HTTPException(
                status_code=400,
                detail=f"kind 非法：{body.kind!r}；允许 {list(KNOWN_MEMORY_KINDS)}",
            )
        try:
            record = await memory.add_memory(
                user_id=user_id,
                agent_id=body.agent_id or "*",
                content=body.content,
                kind=body.kind,
                tags=body.tags,
                salience=body.salience,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return record.meta()

    @app.get("/v1/memory")
    async def search_memory(
        request: Request,
        q: str | None = None,
        module: str | None = None,
        kind: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """检索/浏览长期记忆。

        ``q`` 存在 → 混合检索（向量 + BM25 + 时间衰减 + 显著度），按相关
        度降序；不存在 → 按更新时间浏览。``module`` 限定 agent 作用域
        （该模块 + 全局共享；缺省查全部）。``kind`` 在两种模式下都生效。
        """
        rt: AgentRuntime = request.app.state.runtime
        memory = _require_memory(rt)
        user_id = _memory_user_id(request)
        if kind is not None and kind not in KNOWN_MEMORY_KINDS:
            raise HTTPException(
                status_code=400, detail=f"kind 非法：{kind!r}；允许 {list(KNOWN_MEMORY_KINDS)}"
            )
        limit = max(1, min(limit, 100))
        if q:
            scored = await memory.search(user_id=user_id, agent_id=module, query=q, top_k=limit)
            results = [
                {"score": round(item.score, 4), **item.record.meta()}
                for item in scored
                if kind is None or item.record.kind == kind
            ]
            return {"query": q, "memories": results}
        records = await memory.list_memories(
            user_id, agent_id=module, kinds=[kind] if kind else None, limit=limit
        )
        return {"memories": [record.meta() for record in records]}

    @app.patch("/v1/memory/{memory_id}")
    async def patch_memory(
        memory_id: str, body: MemoryPatchRequest, request: Request
    ) -> dict[str, Any]:
        """部分更新一条长期记忆（改内容会重新向量化）。"""
        rt: AgentRuntime = request.app.state.runtime
        memory = _require_memory(rt)
        if body.status is not None and body.status not in KNOWN_MEMORY_STATUSES:
            raise HTTPException(
                status_code=400,
                detail=f"status 非法：{body.status!r}；允许 {list(KNOWN_MEMORY_STATUSES)}",
            )
        existing = await memory.get_memory(memory_id)
        if existing is None:
            raise HTTPException(status_code=404, detail=f"记忆不存在：{memory_id}")
        try:
            updated = await memory.update_memory(
                memory_id,
                content=body.content,
                tags=body.tags,
                salience=body.salience,
                status=body.status,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return updated.meta() if updated is not None else {}

    @app.delete("/v1/memory/{memory_id}")
    async def delete_memory(memory_id: str, request: Request) -> dict[str, Any]:
        """删除一条长期记忆（硬删除；审计仍在 memory_ops）。"""
        rt: AgentRuntime = request.app.state.runtime
        memory = _require_memory(rt)
        deleted = await memory.delete_memory(memory_id)
        return {"deleted": deleted}

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
        # 记忆系统（M6）：未启用不是降级（组件缺席即可），启用时探存储。
        if rt.memory is not None:
            components["memory"] = await rt.memory.health_probe()
        status = "ok" if all(v == "ok" for v in components.values()) else "degraded"
        return {"status": status, "components": components}

    @app.get("/metrics")
    async def metrics_endpoint(request: Request) -> PlainTextResponse:
        metrics: Metrics = request.app.state.metrics
        body = (
            metrics.render()
            + TOOL_METRICS.render_tool_metrics()
            + MEMORY_METRICS.render_memory_metrics()
        )
        return PlainTextResponse(body, media_type="text/plain; version=0.0.4; charset=utf-8")

    return app


app = create_app()
