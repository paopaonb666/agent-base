"""The ``AgentModule`` contract + ``ModuleContext``.

This is the single extension point every agent module implements. The base
never imports business logic; modules implement this protocol and the
registry (``core/registry.py``) wires them into the runtime.

The contract is intentionally minimal and grows by stage:
- Stage 1: ``name`` / ``description`` / ``build_graph`` / ``get_tools``
- Stage 3: the tool pool and checkpointer are wired into ``ModuleContext``;
  modules bind ``ctx.tools`` and compile with ``ctx.checkpointer``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, runtime_checkable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

if TYPE_CHECKING:  # pragma: no cover - import avoided at runtime
    from agent_base.core.config import Settings

# CompiledStateGraph is generic over (StateT, ContextT, InputT, OutputT).
# The base treats compiled graphs opaquely — it only passes them back to
# entrypoints — so all four parameters are bound to Any.
Graph: TypeAlias = CompiledStateGraph[Any, Any, Any, Any]


@dataclass
class ModuleContext:
    """The runtime services the base hands to every module.

    ``settings``     — validated configuration
    ``llm``          — assembled chat model (openai-compatible)
    ``checkpointer`` — conversation-state saver (Stage 3); ``None`` only
                       when a caller constructs the context by hand (tests)
    ``tools``        — the shared tool pool (Stage 3): every module's
                       ``get_tools()`` contribution, timeout-wrapped; bind
                       to the LLM and execute via ``ToolNode``
    """

    settings: Settings
    llm: BaseChatModel
    checkpointer: BaseCheckpointSaver[Any] | None = None
    tools: list[BaseTool] = field(default_factory=list)


@runtime_checkable
class AgentModule(Protocol):
    """A runnable agent module.

    ``name``        — stable identifier; must match the AGENT_MODULES entry
                      and the module's directory name (registry enforces it)
    ``description`` — human-readable summary for discovery / supervisor
    ``build_graph`` — construct the module's compiled LangGraph; compile
                      with ``name=<module name>`` so the supervisor can
                      orchestrate the graph as a sub-agent
    ``get_tools``   — tools this module contributes to the shared pool
                      (Stage 3: collected into ``ModuleContext.tools``)
    """

    name: str
    description: str

    def build_graph(self, ctx: ModuleContext) -> Graph: ...

    def get_tools(self) -> list[BaseTool]: ...
