"""工具向 UI 发事件的帮助函数（M3.5）。

基座的 SSE 事件契约（``extensions/events.py``）从 chat-agent 评审继承：
``StepEvent`` 表达具名阶段的 running/completed/error，``SourcesEvent``
附着引用来源——两者都是为"工具在执行中"预留的表达。工具通过
LangGraph 的 custom stream（``get_stream_writer``）发原始 dict，服务器
侧按契约模型校验后编码为 SSE 帧。

约定：

- 事件是**尽力而为**的：``get_stream_writer`` 只在图执行上下文里可用
  （服务器与 CLI 的图内调用都满足；单元测试直调工具时不满足）。在
  上下文之外静默丢弃——事件是锦上添花，绝不能让工具本体失败。
- 载荷是**封闭契约**的子集：只发 ``step`` / ``sources``；服务器对
  未知或畸形载荷丢弃并记日志，绝不会把未校验的数据透给前端。
"""

from __future__ import annotations

import contextlib
from typing import Any, Literal

from langgraph.config import get_stream_writer

StepStatus = Literal["running", "completed", "error"]


def _writer() -> Any | None:
    try:
        return get_stream_writer()
    except Exception:
        # 不在图执行上下文（直调工具、纯单测）：没有消费者，直接放弃。
        return None


def _emit(payload: dict[str, Any]) -> None:
    """发送一个 custom 事件；任何发送期异常都吞掉——事件是尽力而为的
    旁路信号（如流已被取消），绝不能让工具本体失败。"""
    writer = _writer()
    if writer is None:
        return
    with contextlib.suppress(Exception):
        writer(payload)


def emit_step(name: str, status: StepStatus, detail: str | None = None) -> None:
    """报告一个具名执行阶段的状态变化（前端显示为进度步骤）。"""
    payload: dict[str, str] = {"type": "step", "name": name, "status": status}
    if detail:
        payload["detail"] = detail
    _emit(payload)


def emit_sources(sources: list[dict[str, str]]) -> None:
    """把引用来源附着到回复上（前端渲染为可点击的来源列表）。"""
    _emit({"type": "sources", "sources": sources})


def emit_tool_call(
    call_id: str,
    name: str,
    phase: Literal["start", "end"],
    *,
    args: dict[str, Any] | None = None,
    result: str | None = None,
    status: str | None = None,
    duration_ms: int | None = None,
    error: str | None = None,
) -> None:
    """报告一次工具调用的开始/结束（前端渲染为参数+结果折叠面板）。

    ``start`` 必带 ``args``；``end`` 必带 ``status``，成功带 ``result``，
    失败/超时带 ``error``。载荷字段与 ``extensions.events.ToolCallEvent``
    契约一一对应，服务器侧会按该模型校验后透传。
    """
    payload: dict[str, Any] = {
        "type": "tool_call",
        "call_id": call_id,
        "name": name,
        "phase": phase,
    }
    if args is not None:
        payload["args"] = args
    if result is not None:
        payload["result"] = result
    if status is not None:
        payload["status"] = status
    if duration_ms is not None:
        payload["duration_ms"] = duration_ms
    if error is not None:
        payload["error"] = error
    _emit(payload)


__all__ = ["emit_sources", "emit_step", "emit_tool_call"]
