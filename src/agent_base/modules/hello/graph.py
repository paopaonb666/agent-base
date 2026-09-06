"""Graph construction for the hello module.

This is the module-guide's five-step recipe implemented for real: the
smallest possible module (one LLM node, no tools). It exists to prove the
guide's promise — a new module plugs in without touching base code.

The graph is compiled with the runtime's checkpointer (conversation state
per thread) and ``name="hello"`` so the supervisor can orchestrate it as a
sub-agent like any other registered module.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, MessagesState, StateGraph

from agent_base.core.contracts import Graph, ModuleContext

MODULE_NAME = "hello"


def build_hello_graph(ctx: ModuleContext) -> Graph:
    """Build and compile the hello graph: a single LLM node."""
    llm = ctx.llm

    def reply(state: MessagesState) -> dict[str, Any]:
        return {"messages": [llm.invoke(state["messages"])]}

    graph = StateGraph(MessagesState)
    graph.add_node("reply", reply)
    graph.add_edge(START, "reply")
    graph.add_edge("reply", END)
    return graph.compile(checkpointer=ctx.checkpointer, name=MODULE_NAME)
