"""Multi-agent collaboration: the supervisor template (Stage 4).

Wraps LangGraph's native ``create_supervisor`` (langgraph-supervisor): every
registered module becomes a sub-agent, routed by a supervisor node that
reads each module's ``name`` + ``description`` (the same contract fields
used for discovery). The base adds no orchestration logic of its own —
that is the whole point of the thin-base principle.

Requirements on modules (enforced here as early failures):
- every module graph must be compiled with ``name=<module name>`` so the
  supervisor can hand off to it (checked by langgraph-supervisor itself);
- ``description`` is what the supervisor model sees when routing, so it
  must be a meaningful summary (the registry already validates non-empty).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph_supervisor import create_handoff_tool, create_supervisor

from agent_base.core.contracts import Graph, ModuleContext

if TYPE_CHECKING:  # pragma: no cover - import avoided at runtime
    from agent_base.core.contracts import AgentModule

DEFAULT_PROMPT = (
    "You are a team supervisor coordinating specialist agents. "
    "Route each user request to the most suitable agent using the handoff "
    "tools; relay their results and synthesize a final answer to the user. "
    "Do not call an agent more than necessary."
)


def build_supervisor_graph(ctx: ModuleContext, modules: dict[str, AgentModule]) -> Graph:
    """Orchestrate every registered module as a supervisor-led sub-agent."""
    if not modules:
        raise ValueError("supervisor requires at least one module in AGENT_MODULES")

    # Each sub-agent is the module's own compiled graph; the names come
    # from the modules (and match their AGENT_MODULES entries).
    agents = [module.build_graph(ctx) for module in modules.values()]

    # Handoff tools carry the module descriptions so the supervisor model
    # can route on what each agent actually does.
    handoffs = [
        create_handoff_tool(agent_name=module.name, description=module.description)
        for module in modules.values()
    ]

    workflow = create_supervisor(
        # Graph alias vs Pregel / list-invariance mismatches in the library's
        # stubs; runtime compatibility is covered by tests/test_collab.py.
        agents=agents,  # type: ignore[arg-type]
        model=ctx.llm,
        tools=handoffs,  # type: ignore[arg-type]
        prompt=DEFAULT_PROMPT,
    )
    return workflow.compile(checkpointer=ctx.checkpointer)
