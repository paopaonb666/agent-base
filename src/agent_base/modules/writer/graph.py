"""writer 模块的图构建。

一个带写作取向系统提示词的 LLM 节点。结构上与 chat 模块完全相同
（流式累积 + checkpointer + 具名编译）——唯一的区别就是提示词，而这
正是要点所在：模块的区别在于行为，而不在于接线。
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessageChunk, SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from agent_base.core.contracts import Graph, ModuleContext

MODULE_NAME = "writer"

WRITER_PROMPT = (
    "You are a professional writing agent. Draft, rewrite, summarize, or "
    "polish the requested text with clear structure and a consistent voice. "
    "Ask for nothing except the writing task itself."
)


def build_writer_graph(ctx: ModuleContext) -> Graph:
    """构建并编译 writer 图。"""

    async def call_model(state: MessagesState) -> dict[str, Any]:
        final: AIMessageChunk | None = None
        messages = [SystemMessage(content=WRITER_PROMPT), *state["messages"]]
        async for chunk in ctx.llm.astream(messages):
            final = chunk if final is None else final + chunk
        if final is None:
            return {"messages": []}
        return {"messages": [final]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    return graph.compile(checkpointer=ctx.checkpointer, name=MODULE_NAME)
