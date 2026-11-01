"""共享的测试替身（通过 pytest rootdir 模式供测试模块导入）。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
import time
from collections.abc import AsyncIterator, Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel, ChatResult
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolCallChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk
from langchain_core.tools import BaseTool
from pydantic import PrivateAttr


class HashEmbedding:
    """确定性假 embedding（M6 测试替身）：把 token hash 进固定维度向量。

    用 hashlib 而不是内置 hash（PYTHONHASHSEED 会让后者跨进程不稳定），
    保证同一文本永远得到同一向量——检索测试因此可以断言"相关者得分
    更高"。分词复用 retrieval.tokenize（中文二元组），相似文本天然
    向量相近。
    """

    def __init__(self, dims: int = 32) -> None:
        self.dims: int | None = dims

    async def embed(self, texts: Sequence[str]) -> list[list[float]] | None:
        from agent_base_memory.retrieval import tokenize

        vectors: list[list[float]] = []
        for text in texts:
            vec = [0.0] * (self.dims or 32)
            for token in tokenize(text):
                digest = hashlib.md5(token.encode("utf-8")).digest()
                vec[digest[0] % len(vec)] += 1.0
            norm = sum(x * x for x in vec) ** 0.5
            vectors.append([x / norm for x in vec] if norm > 0 else vec)
        return vectors


def _to_chunk(msg: AIMessage) -> AIMessageChunk:
    """把一个脚本化的 AIMessage 转换成流式分块，保留 tool_calls。"""
    if msg.tool_calls:
        return AIMessageChunk(
            content=msg.content,
            tool_call_chunks=[
                ToolCallChunk(
                    name=tc["name"],
                    args=json.dumps(tc["args"]),
                    id=tc["id"],
                    type="tool_call_chunk",
                )
                for tc in msg.tool_calls
            ],
        )
    return AIMessageChunk(content=msg.content)


class ScriptedChatModel(BaseChatModel):
    """每次模型调用弹出一条脚本化的 ``AIMessage``；把它作为单个分块流式输出。

    与 ``GenericFakeChatModel`` 不同，它能扛过工具调用响应（空 content +
    tool_calls）的 ``astream``，这正是 chat 节点的流式累积所需要的。
    ``bind_tools`` 原样通过，因此用共享工具池构建的图可以原封不动地工作。
    """

    _queue: list[AIMessage] = PrivateAttr(default_factory=list)
    # 收到的模型输入录制（M6d 上下文工程测试用）。
    _received: list[list[BaseMessage]] = PrivateAttr(default_factory=list)

    def __init__(self, responses: list[AIMessage] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._queue = list(responses or [])
        self._received = []

    @property
    def received(self) -> list[list[BaseMessage]]:
        """每次模型调用收到的完整输入（诊断/断言用）。"""
        return self._received

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    def _next(self) -> AIMessage:
        if not self._queue:
            raise RuntimeError("ScriptedChatModel: scripted responses exhausted")
        return self._queue.pop(0)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        # 注意：generations 必须是关键字参数——pydantic v2 的 BaseModel 会拒绝
        # 位置式初始化参数（ChatResult([...]) 会抛出 TypeError）。
        self._received.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self._next())])

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        self._received.append(list(messages))
        yield ChatGenerationChunk(message=_to_chunk(self._next()))

    def bind_tools(  # type: ignore[override]
        self, tools: list[BaseTool], **kwargs: Any
    ) -> ScriptedChatModel:
        return self


class ChunkedChatModel(BaseChatModel):
    """按"分块序列"流式输出每条脚本回复（A1/工具循环测试替身）。

    ``GenericFakeChatModel`` 对空 content 直接抛错且不携带
    tool_call_chunks，构造不出"一条回复分多块到达"的真实流式场景
    （OpenAI 兼容 provider 的 tool_call 参数就是跨块拆分的）。每个
    条目是一段 ``AIMessageChunk`` 序列，按序产出、整体算一次模型调用。
    """

    _batches: list[list[AIMessageChunk]] = PrivateAttr(default_factory=list)

    def __init__(self, batches: list[list[AIMessageChunk]] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._batches = [list(batch) for batch in (batches or [])]

    @property
    def _llm_type(self) -> str:
        return "chunked-fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        batch = self._batches.pop(0) if self._batches else [AIMessageChunk(content="")]
        final: AIMessageChunk | None = None
        for chunk in batch:
            final = chunk if final is None else final + chunk
        # 空批次（模型无产出）也必须返回一条消息：ainvoke 契约不允许
        # 空结果；调用方用 content 判断即可。
        return ChatResult(generations=[ChatGeneration(message=final or AIMessage(content=""))])

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        batch = self._batches.pop(0) if self._batches else [AIMessageChunk(content="")]
        for chunk in batch:
            yield ChatGenerationChunk(message=chunk)

    def bind_tools(  # type: ignore[override]
        self, tools: list[BaseTool], **kwargs: Any
    ) -> ChunkedChatModel:
        return self


class FakeRedisService:
    """手写 Redis 替身（ADR-012 测试用）：内存 dict + TTL 记账 + 队列 Pub/Sub。

    与 ``RedisService`` 同一原语面（含键前缀与锁 token 语义），消费方
    （成本计数 / nonce / 分布式锁 / 快照缓存 / SSE 镜像）的核心测试不
    需要真实 Redis；真连语义走 integration 标记（见 test_redis_service.py）。
    """

    def __init__(self, *, key_prefix: str = "agent_base:") -> None:
        self._prefix = key_prefix
        self._data: dict[str, str] = {}
        # 键 → 过期时刻（单调时钟）；无记录 = 永不过期。
        self._expiry: dict[str, float] = {}
        # 频道 → 订阅者队列（publish 扇出）。
        self._channels: dict[str, list[asyncio.Queue[str]]] = {}
        # 置 True 后 ping 抛错（/health 的 error 分支与失败路径测试）。
        self.fail_ping = False
        self.closed = False

    def _k(self, key: str) -> str:
        return f"{self._prefix}{key}"

    def _alive(self, full_key: str) -> bool:
        expire_at = self._expiry.get(full_key)
        if expire_at is not None and expire_at <= time.monotonic():
            self._data.pop(full_key, None)
            self._expiry.pop(full_key, None)
            return False
        return full_key in self._data

    async def ping(self) -> None:
        if self.fail_ping:
            raise RuntimeError("fake redis: ping failed (fail_ping=True)")

    async def aclose(self) -> None:
        self.closed = True

    async def get(self, key: str) -> str | None:
        full = self._k(key)
        return self._data.get(full) if self._alive(full) else None

    async def set_ex(self, key: str, value: str, *, ttl_seconds: int) -> None:
        full = self._k(key)
        self._data[full] = value
        self._expiry[full] = time.monotonic() + max(ttl_seconds, 0)

    async def set_nx(self, key: str, value: str, *, ttl_seconds: int) -> bool:
        full = self._k(key)
        if self._alive(full):
            return False
        self._data[full] = value
        self._expiry[full] = time.monotonic() + max(ttl_seconds, 0)
        return True

    async def delete(self, *keys: str) -> int:
        removed = 0
        for key in keys:
            full = self._k(key)
            if full in self._data:
                self._data.pop(full, None)
                self._expiry.pop(full, None)
                removed += 1
        return removed

    async def delete_prefix(self, prefix: str) -> int:
        matched = [k for k in self._data if k.startswith(f"{self._prefix}{prefix}")]
        for full in matched:
            self._data.pop(full, None)
            self._expiry.pop(full, None)
        return len(matched)

    async def incr_by_float(self, key: str, amount: float) -> float:
        full = self._k(key)
        current = float(self._data[full]) if self._alive(full) else 0.0
        current += amount
        self._data[full] = repr(current)
        return current

    async def expire(self, key: str, *, ttl_seconds: int) -> None:
        full = self._k(key)
        if full in self._data:
            self._expiry[full] = time.monotonic() + max(ttl_seconds, 0)

    async def publish(self, channel: str, message: str) -> int:
        subscribers = self._channels.get(self._k(channel), [])
        for queue in list(subscribers):
            queue.put_nowait(message)
        return len(subscribers)

    async def pubsub_messages(self, channel: str) -> AsyncIterator[str]:
        full = self._k(channel)
        queue: asyncio.Queue[str] = asyncio.Queue()
        subscribers = self._channels.setdefault(full, [])
        subscribers.append(queue)
        try:
            while True:
                yield await queue.get()
        finally:
            if queue in subscribers:
                subscribers.remove(queue)

    async def acquire_lock(
        self, key: str, *, ttl_seconds: int, acquire_timeout_seconds: float
    ) -> str | None:
        token = secrets.token_hex(16)
        deadline = time.monotonic() + acquire_timeout_seconds
        while True:
            if await self.set_nx(key, token, ttl_seconds=ttl_seconds):
                return token
            if time.monotonic() >= deadline:
                return None
            await asyncio.sleep(0.001)

    async def release_lock(self, key: str, token: str) -> bool:
        full = self._k(key)
        if self._alive(full) and self._data.get(full) == token:
            self._data.pop(full, None)
            self._expiry.pop(full, None)
            return True
        return False


class CancellableChatModel(BaseChatModel):
    """挂起直到被取消；记录取消是否到达了模型。

    供 SSE 取消测试使用：如果客户端断开没有被传播，``cancelled`` 会一直
    为 False，测试就会失败。
    """

    cancelled: bool = False

    @property
    def _llm_type(self) -> str:
        return "cancellable-fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="done"))])

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        try:
            await asyncio.sleep(10)
            yield ChatGenerationChunk(message=AIMessageChunk(content="late"))
        except asyncio.CancelledError:
            self.cancelled = True
            raise
