"""Graph construction for the writer module.

One LLM node with a writing-focused system prompt. Structurally identical
to the chat module (streaming accumulation + checkpointer + named compile)
— the only difference is the prompt, which is exactly the point: modules
differ in behavior, not in wiring.
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
    """Build and compile the writer graph."""

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
