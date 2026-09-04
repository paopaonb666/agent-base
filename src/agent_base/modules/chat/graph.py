"""Graph construction for the chat module.

The simplest possible LangGraph agent: one node that calls the LLM on the
conversation so far. State uses the standard ``MessagesState`` channel whose
``add_messages`` reducer accumulates history across turns, so multi-turn
conversations work without the base doing anything special.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, MessagesState, StateGraph

from agent_base.core.contracts import Graph, ModuleContext


def build_chat_graph(ctx: ModuleContext) -> Graph:
    """Build and compile the chat graph."""
    llm = ctx.llm

    def call_model(state: MessagesState) -> dict[str, Any]:
        return {"messages": [llm.invoke(state["messages"])]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    return graph.compile()
