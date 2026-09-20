"""共享的测试替身（通过 pytest rootdir 模式供测试模块导入）。"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Sequence
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
        from agent_base.memory.retrieval import tokenize

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
