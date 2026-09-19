"""路由层的共享依赖（H1 拆分）：runtime / 指标 / 身份 / 记忆门面。"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from fastapi import HTTPException, Request

from agent_base.core.bootstrap import SUPERVISOR_MODULE, AgentRuntime
from agent_base.entrypoints.server.auth import AuthBackend
from agent_base.memory.service import MemoryService

if TYPE_CHECKING:  # pragma: no cover - import avoided at runtime
    from agent_base.extensions.metrics import Metrics


def get_runtime(request: Request) -> AgentRuntime:
    """当前请求绑定的运行时（lifespan 装配）。"""
    return cast(AgentRuntime, request.app.state.runtime)


def get_metrics(request: Request) -> Metrics:
    """应用级 HTTP 指标收集器。"""
    return cast("Metrics", request.app.state.metrics)


def get_current_user(request: Request) -> str:
    """用户作用域身份：经可插拔的 AuthBackend 解析（S1 统一依赖）。

    所有用户作用域端点必须显式声明该依赖——身份解析只有这一条路径。
    """
    rt = get_runtime(request)
    backend: AuthBackend = request.app.state.auth_backend
    return str(backend.authenticate(request, rt.settings))


def require_memory(rt: AgentRuntime) -> MemoryService:
    """取记忆服务；未启用时以 503 给出可读原因。

    ``rt.memory`` 的静态类型是契约端口（MemoryPort）；入口层需要完整
    门面（store/管线/管理面），运行时对象实为 MemoryService 实例。
    """
    if rt.memory is None:
        raise HTTPException(status_code=503, detail="记忆系统未启用（MEMORY_ENABLED=false）")
    return cast(MemoryService, rt.memory)


def memory_or_none(rt: AgentRuntime) -> MemoryService | None:
    """完整记忆门面或 None（静态端口类型在此收窄为门面）。"""
    return cast("MemoryService | None", rt.memory)


def known_module(rt: AgentRuntime, module: str) -> bool:
    """invoke 之外的线程端点也必须接受 supervisor 保留名：supervisor 的
    invoke 走 ``runtime.graph`` 的特例路径，能正常产生 checkpointer 记录；
    若 list/history/delete 只认 ``rt.modules``，supervisor 的对话就会
    "能聊但不能列历史、不能删"——前后端语义不一致。
    """
    return module in rt.modules or module == SUPERVISOR_MODULE
