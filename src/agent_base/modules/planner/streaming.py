"""planner 向 UI 发计划事件（B5/M9）。载荷与 extensions.events.PlanEvent
一一对应，服务器侧按该模型校验后透传；不在图执行上下文时静默放弃。"""

from __future__ import annotations

import contextlib
from typing import Any

from langgraph.config import get_stream_writer

from agent_base.modules.planner.state import Task


def _emit(payload: dict[str, Any]) -> None:
    """尽力而为的旁路信号：上下文缺失 / 写入失败都静默放弃。"""
    with contextlib.suppress(Exception):
        writer = get_stream_writer()
        if callable(writer):
            writer(payload)


def emit_plan(tasks: list[Task], status: str, *, detail: str | None = None) -> None:
    """发送全量任务清单（前端整表替换）。status 见 PlanEvent 契约。"""
    payload: dict[str, Any] = {
        "type": "plan",
        "status": status,
        "plan": [{"id": t["id"], "goal": t["goal"], "status": t["status"]} for t in tasks],
    }
    if detail:
        payload["detail"] = detail
    _emit(payload)
