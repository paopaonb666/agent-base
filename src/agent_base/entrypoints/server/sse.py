"""SSE 流式契约的生产端（H1 拆分）：生产者 / 心跳 / 事件流 / 记忆形成钩子。

取消传播：客户端断开会取消生产者任务，从而取消正在进行的 ``astream``
与随之而来的 LLM 请求——token 消耗随即停止，而不是在后台跑完。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
from collections.abc import AsyncIterator, Coroutine
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from agent_base.entrypoints.server.background import spawn_background
from agent_base.entrypoints.server.serializers import _extract_text
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
from agent_base.memory.service import MemoryService
from agent_base.memory.tools import set_memory_scope

logger = logging.getLogger(__name__)

HEARTBEAT_SECONDS = 15.0

# SSE 事件队列上限：慢客户端（半开连接、不读数据）停止消费时，生产者在
# 队列满后阻塞等待——内存有界，背压自然传导到 LLM 流。ping 允许丢弃。
SSE_QUEUE_MAXSIZE = 256

_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

_URL_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+.-]*://\S+")


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


def _build_capture_hook(
    runtime_memory: MemoryService | None,
    module: str,
    user_thread_id: str,
    graph: Any,
    config: RunnableConfig,
) -> Coroutine[Any, Any, None] | None:
    """构造轮次完成后的记忆形成回调（M6c）；管线不可用时返回 None。

    回调从 checkpointer 快照取本轮完整消息（含工具轮次），交给
    ``capture_turn``——形成管线内部失败安全，这里再兜一层异常。
    user_id 取自 config（invoke 端点已从 X-User-Id 头写入 configurable）。

    返回的是**已创建的协程对象**：仅当本轮正常完成时才由 ``_produce``
    经 ``spawn_background`` 调度——流异常/客户端断开路径下它从未 await，
    也不会产生 "coroutine was never awaited" 警告（H4）。
    """
    memory = runtime_memory
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
    线）。经 ``spawn_background`` 独立调度（持强引用、异常落日志）——
    不阻塞哨兵送达，客户端中途断开或本轮出错时不会被调用。

    记忆作用域（M6e）：在这里设置 ContextVar——图节点与工具都运行在
    本任务上下文里，记忆工具据此拿到 user/thread（工具池的
    ``_TimeoutTool`` 现已转发 RunnableConfig，ContextVar 是双保险）。
    """
    _configurable = config.get("configurable") or {}
    _raw_user = _configurable.get("user_id")
    set_memory_scope(
        _raw_user if isinstance(_raw_user, str) and _raw_user else "default",
        str(_configurable.get("thread_id") or ""),
    )
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
                # supervisor 模板（react agent）的模型节点是非流式调用，
                # 回复以完整 AIMessage（非逐 token 的 AIMessageChunk）经
                # messages 模式的节点输出路径到达——必须一并接收，否则
                # supervisor 对话一个 delta 都发不出去（前端只能靠事后
                # 对账补内容，短回复会整条丢失）。chunk 是 AIMessage 的
                # 子类；同一消息的完整发射已被按 id 去重，不会双发。
                if isinstance(chunk, AIMessage) and chunk.content:
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
            spawn_background(on_complete, name="memory-capture")
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
