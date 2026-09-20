"""FastAPI + SSE 服务入口（H1 拆分后的包形态）。

运行::

    uvicorn agent_base.entrypoints.server:app --reload

对外契约与拆分前的单文件 ``server.py`` 完全一致；内部按关注点分层：

- ``auth``      —— 可插拔身份后端（``AuthBackend`` + 默认 HMAC 头实现）
- ``deps``      —— 路由共享依赖（runtime / 指标 / 统一身份依赖）
- ``serializers`` —— 消息序列化与图片内容嗅探
- ``sse``       —— SSE 生产者 / 心跳 / 事件流 / 记忆形成钩子
- ``background`` —— 后台任务助手（持引用 + 异常落日志，H4）
- ``routes/``   —— agents（对话与线程）/ memory / files / system 四个
  ``APIRouter``；路由层只做参数校验、依赖注入与编排，无业务规则

本模块（包根）保留应用装配职责：lifespan、CORS、可观测性中间件，
以及测试依赖的模块级符号（``_event_stream`` / ``_serialize_message`` /
``_probe_model`` 等）。
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from agent_base import __version__
from agent_base.core.bootstrap import AgentRuntime, create_runtime
from agent_base.core.config import Settings
from agent_base.entrypoints.server.auth import REQUEST_ID_RE, AuthBackend, HmacHeaderAuth
from agent_base.entrypoints.server.routes import agents as agents_routes
from agent_base.entrypoints.server.routes import files as files_routes
from agent_base.entrypoints.server.routes import memory as memory_routes
from agent_base.entrypoints.server.routes import system as system_routes
from agent_base.entrypoints.server.routes.system import _probe_model
from agent_base.entrypoints.server.serializers import (
    IMAGE_EXTENSIONS,
    IMAGE_STORED_FORMATS,
    _extract_text,
    _serialize_message,
    _sniff_image_mime,
)
from agent_base.entrypoints.server.sse import (
    _SSE_HEADERS,
    _decode_custom_event,
    _event_stream,
)
from agent_base.extensions.metrics import Metrics
from agent_base.extensions.observability import (
    new_request_id,
    request_id,
    setup_logging,
)

__all__ = [
    "IMAGE_EXTENSIONS",
    "IMAGE_STORED_FORMATS",
    "REQUEST_ID_RE",
    "SSE_QUEUE_MAXSIZE",
    "_SSE_HEADERS",
    "AuthBackend",
    "HmacHeaderAuth",
    "_decode_custom_event",
    "_event_stream",
    "_extract_text",
    "_probe_model",
    "_serialize_message",
    "_sniff_image_mime",
    "app",
    "create_app",
    "new_request_id",
    "request_id",
    "setup_logging",
]


def create_app(
    runtime: AgentRuntime | None = None, *, auth_backend: AuthBackend | None = None
) -> FastAPI:
    """构建服务应用；测试注入一个 runtime，生产环境则现场构建。

    ``auth_backend``（P0-5）：可插拔身份后端，缺省 ``HmacHeaderAuth``；
    公网多用户部署可在网关层实现真实鉴权后经此注入，无需改路由。
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        rt = runtime if runtime is not None else await create_runtime()
        # 日志契约（LOG_JSON / request_id 过滤器）在 server 路径同样生效，
        # 而不是只在 CLI——否则生产日志既非结构化也无法端到端追踪（A1）。
        setup_logging(json_lines=rt.settings.observability.log_json)
        app.state.runtime = rt
        try:
            yield
        finally:
            # H4：优雅关闭必须释放运行时资源（MySQL 池、httpx client、
            # sqlite 连接、toollog 写线程），否则全部泄漏到进程退出。
            await rt.close()

    app = FastAPI(title="agent-base", version=__version__, lifespan=lifespan)
    metrics = Metrics()
    app.state.metrics = metrics
    app.state.auth_backend = auth_backend if auth_backend is not None else HmacHeaderAuth()

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
        rid = client_rid if REQUEST_ID_RE.fullmatch(client_rid) else new_request_id()
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

    app.include_router(agents_routes.router)
    app.include_router(memory_routes.router)
    app.include_router(files_routes.router)
    app.include_router(system_routes.router)
    return app


app = create_app()
