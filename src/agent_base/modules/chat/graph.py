"""Graph construction for the chat module.

A single LLM node over the conversation history, plus a ``ToolNode`` loop
when the shared tool pool is non-empty (Stage 3). The node streams the
model and accumulates the chunks, so:

- the server's SSE layer sees token-level ``delta`` events (it taps the
  same stream), and
- cancellation propagates — aborting the graph run cancels the in-flight
  LLM request instead of letting it finish in the background (Stage 4).

State uses the standard ``MessagesState`` channel whose ``add_messages``
reducer accumulates history across turns; compiled with the runtime's
checkpointer, history persists per ``thread_id`` (in memory or sqlite).

The graph is compiled with ``name="chat"``: the supervisor (Stage 4)
requires every sub-agent graph to carry its module's name.
"""

from __future__ import annotations

from typing import Any, cast

from langchain_core.messages import AIMessageChunk
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent_base.core.contracts import Graph, ModuleContext
from agent_base.core.tools import handle_tool_error

MODULE_NAME = "chat"


def build_chat_graph(ctx: ModuleContext) -> Graph:
    """Build and compile the chat graph (LLM node + optional tool loop)."""
    tools = ctx.tools
    # bind_tools on every call would also be fine; the pool is rebuilt per
    # graph so binding once here is enough.
    model = ctx.llm.bind_tools(tools) if tools else ctx.llm

    async def call_model(state: MessagesState) -> dict[str, Any]:
        final: AIMessageChunk | None = None
        async for chunk in model.astream(state["messages"]):
            # astream's declared yield type is a message union; the runtime
            # chunks are AIMessageChunk (chat models only).
            piece = cast(AIMessageChunk, chunk)
            final = piece if final is None else final + piece
        if final is None:
            return {"messages": []}
        return {"messages": [final]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    if tools:
        # handle_tool_error normalizes tool failures into ToolMessage
        # feedback, so a broken tool never crashes the conversation
        # (Stage 3 acceptance).
        graph.add_node("tools", ToolNode(tools, handle_tool_errors=handle_tool_error))
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", tools_condition)
        graph.add_edge("tools", "agent")
    else:
        graph.add_edge(START, "agent")
        graph.add_edge("agent", END)
    return graph.compile(checkpointer=ctx.checkpointer, name=MODULE_NAME)
