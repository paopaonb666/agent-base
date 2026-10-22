"""记忆工具的作用域 ContextVar（入口层写入，记忆工具读取）。

为什么放在基座：每轮请求开始时由 SSE 入口设置 ``(user_id, thread_id)``，
记忆工具（属于可选的记忆包）在工具执行时读取。作用域传递是**入口层
语义**，不属于记忆实现——把 ContextVar 留在记忆包里会让基座反向依赖
记忆包，拆分就失去意义。
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

_memory_scope: ContextVar[dict[str, str] | None] = ContextVar("memory_scope", default=None)


def set_memory_scope(user_id: str, thread_id: str) -> None:
    """绑定当前请求的记忆作用域（同一协程内后续工具调用可见）。"""
    _memory_scope.set({"user_id": user_id, "thread_id": thread_id})


def current_memory_scope() -> dict[str, Any] | None:
    """读取当前请求的记忆作用域；未设置返回 None。"""
    return _memory_scope.get()


__all__ = ["current_memory_scope", "set_memory_scope"]
