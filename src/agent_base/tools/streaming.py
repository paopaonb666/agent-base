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

from typing import Any, Literal

from langgraph.config import get_stream_writer

StepStatus = Literal["running", "completed", "error"]


def _writer() -> Any | None:
    try:
        return get_stream_writer()
    except Exception:
        # 不在图执行上下文（直调工具、纯单测）：没有消费者，直接放弃。
        return None


def emit_step(name: str, status: StepStatus, detail: str | None = None) -> None:
    """报告一个具名执行阶段的状态变化（前端显示为进度步骤）。"""
    writer = _writer()
    if writer is None:
        return
    payload: dict[str, str] = {"type": "step", "name": name, "status": status}
    if detail:
        payload["detail"] = detail
    writer(payload)


def emit_sources(sources: list[dict[str, str]]) -> None:
    """把引用来源附着到回复上（前端渲染为可点击的来源列表）。"""
    writer = _writer()
    if writer is None:
        return
    writer({"type": "sources", "sources": sources})


__all__ = ["emit_sources", "emit_step"]
