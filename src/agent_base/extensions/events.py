"""SSE 事件契约（阶段 4）。

继承自 chat-agent 评审中的 A2：agent 的进展是显式的——step 转换、token
增量、来源、终止——以一组封闭的 Pydantic 模型呈现，因此线上格式是经过
校验且可版本化的，而不是临时的字符串。``SourcesEvent`` 刻意作为契约的
一部分，尽管基座从不发出它：未来模块（如 RAG）可以通过它发布引用，
而不必改动编码器。

线上格式：Server-Sent Events，每个模型一个事件::

    event: delta
    data: {"type":"delta","content":"he"}
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class StepEvent(BaseModel):
    """agent 运行中的一个具名阶段（节点 running / completed / error）。"""

    type: Literal["step"] = "step"
    name: str
    status: Literal["running", "completed", "error"]
    detail: str | None = None


class DeltaEvent(BaseModel):
    """assistant 回复中 token 级别的一小段（流式）。"""

    type: Literal["delta"] = "delta"
    content: str


class Source(BaseModel):
    """一条被引用的来源（未来的 RAG 模块会发布这些）。"""

    title: str
    url: str | None = None


class SourcesEvent(BaseModel):
    """附着在回复上的引用（基座从不发出；契约预留）。"""

    type: Literal["sources"] = "sources"
    sources: list[Source] = Field(default_factory=list)


class DoneEvent(BaseModel):
    """终止性的成功标记；携带 thread id 以便恢复会话。"""

    type: Literal["done"] = "done"
    thread_id: str | None = None


class ErrorEvent(BaseModel):
    """终止性的失败标记，带有人类可读的消息。"""

    type: Literal["error"] = "error"
    message: str


class PingEvent(BaseModel):
    """心跳，防止中间环节关闭空闲的流。"""

    type: Literal["ping"] = "ping"


class ToolCallEvent(BaseModel):
    """一次工具调用的开始/结束记录（M5 工具调用可观测）。

    由池的 ``_TimeoutTool`` 在执行前后经 custom stream 发出：``start``
    携带模型给定的参数，``end`` 携带结果/状态/耗时——成功、超时、异常
    三种结局都会发。前端按 ``call_id`` 把两半配对成完整的调用面板。
    """

    type: Literal["tool_call"] = "tool_call"
    call_id: str
    name: str
    phase: Literal["start", "end"]
    args: dict[str, Any] = Field(default_factory=dict)
    result: str | None = None
    status: Literal["ok", "timeout", "error"] | None = None
    duration_ms: int | None = None
    error: str | None = None


AgentEvent = (
    StepEvent | DeltaEvent | SourcesEvent | DoneEvent | ErrorEvent | PingEvent | ToolCallEvent
)


def encode_sse(event: AgentEvent) -> str:
    """把一个契约事件渲染为 Server-Sent Events 帧。"""
    return f"event: {event.type}\ndata: {event.model_dump_json()}\n\n"
