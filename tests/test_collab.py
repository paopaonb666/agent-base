"""Tests for the supervisor template (Stage 4).

The scripted model plays BOTH roles: the supervisor node (its first
response is a handoff tool call to ``writer``) and the writer sub-agent
(its second response is the answer). Routing is therefore asserted
end-to-end without a real LLM.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent_base.core.bootstrap import SUPERVISOR_MODULE, AgentRuntime
from agent_base.core.config import Settings
from agent_base.core.contracts import ModuleContext
from agent_base.extensions.collab import build_supervisor_graph
from agent_base.modules.chat.module import ChatModule
from agent_base.modules.writer.module import WriterModule
from fakes import ScriptedChatModel


def _runtime(model: ScriptedChatModel) -> AgentRuntime:
    return AgentRuntime(
        settings=Settings(_env_file=None, llm_api_key="sk-test"),
        llm=model,
        modules={"chat": ChatModule(), "writer": WriterModule()},
        tools=[],
        checkpointer=InMemorySaver(),
    )


async def test_supervisor_graph_builds_over_registered_modules() -> None:
    runtime = _runtime(ScriptedChatModel([]))
    graph = runtime.graph(SUPERVISOR_MODULE)
    # Sub-agents appear as nodes named after their modules.
    node_names = set(graph.get_graph().nodes)
    assert {"chat", "writer"} <= node_names


async def test_supervisor_routes_to_writer() -> None:
    # 1st call = supervisor decides: hand off via the auto-generated
    # `transfer_to_<module>` tool. 2nd call = writer drafts the answer.
    # 3rd call = supervisor synthesizes the final reply.
    model = ScriptedChatModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "transfer_to_writer", "args": {}, "id": "handoff-1"},
                ],
            ),
            AIMessage(content="draft: the essay body"),
            AIMessage(content="final: your essay is ready"),
        ]
    )
    runtime = _runtime(model)
    graph = runtime.graph(SUPERVISOR_MODULE)
    config: dict[str, Any] = {"configurable": {"thread_id": "sup:t1"}}
    result = await graph.ainvoke({"messages": [HumanMessage(content="write me an essay")]}, config)
    contents = [m.content for m in result["messages"]]
    # The conversation passed through the writer sub-agent (its reply is in
    # the final state) and the supervisor produced a closing message.
    assert "draft: the essay body" in contents
    assert contents[-1] == "final: your essay is ready"


def test_supervisor_requires_modules() -> None:
    import pytest

    ctx = ModuleContext(
        settings=Settings(_env_file=None, llm_api_key="sk-test"),
        llm=ScriptedChatModel([]),
        checkpointer=InMemorySaver(),
    )
    with pytest.raises(ValueError, match="at least one module"):
        build_supervisor_graph(ctx, {})
