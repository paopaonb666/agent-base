"""共享的测试替身（通过 pytest rootdir 模式供测试模块导入）。"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel, ChatResult
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolCallChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk
from langchain_core.tools import BaseTool
from pydantic import PrivateAttr


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

    def __init__(self, responses: list[AIMessage] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._queue = list(responses or [])

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
        return ChatResult(generations=[ChatGeneration(message=self._next())])

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        yield ChatGenerationChunk(message=_to_chunk(self._next()))

    def bind_tools(  # type: ignore[override]
        self, tools: list[BaseTool], **kwargs: Any
    ) -> ScriptedChatModel:
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
