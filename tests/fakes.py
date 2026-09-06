"""Shared test fakes (importable by test modules via pytest rootdir mode)."""

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
    """Convert a scripted AIMessage into a stream chunk, tool_calls intact."""
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
    """Pops one scripted ``AIMessage`` per model call; streams it as one chunk.

    Unlike ``GenericFakeChatModel`` this survives ``astream`` for tool-call
    responses (empty content + tool_calls), which the chat node's streaming
    accumulation needs. ``bind_tools`` passes through so graphs built with
    the shared tool pool work unchanged.
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
        # NOTE: generations must be a keyword — pydantic v2 BaseModel rejects
        # positional init args (ChatResult([...]) raises TypeError).
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
    """Hangs until cancelled; records whether cancellation reached the model.

    Used by the SSE cancellation test: if client disconnect does NOT
    propagate, ``cancelled`` stays False and the test fails.
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
