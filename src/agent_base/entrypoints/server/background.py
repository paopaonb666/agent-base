"""后台任务助手（H4）：集中持引用 + 统一异常落日志。

裸 ``asyncio.create_task`` 的任务只被事件循环弱引用——官方文档明确
记载了任务被 GC 中途回收的风险（本项目在 toollog 写线程处早已遵守该
规则，server 却一直裸用）。所有 fire-and-forget 任务都经
``spawn_background`` 创建：强引用登记、完成即清理、异常必落日志。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any

logger = logging.getLogger(__name__)

# 进程级强引用注册表：任务完成后由回调移除。
_TASKS: set[asyncio.Task[None]] = set()


def spawn_background(
    coro: Coroutine[Any, Any, Any], *, name: str | None = None
) -> asyncio.Task[Any]:
    """创建并登记一个后台任务；异常不会无声丢失。"""
    task = asyncio.get_running_loop().create_task(coro, name=name)
    _TASKS.add(task)

    def _done(done: asyncio.Task[Any]) -> None:
        _TASKS.discard(done)
        if not done.cancelled() and done.exception() is not None:
            logger.error("background task %s failed", done.get_name(), exc_info=done.exception())

    task.add_done_callback(_done)
    return task


def pending_background_tasks() -> int:
    """仍在运行的后台任务数（测试收尾断言用）。"""
    return len(_TASKS)
