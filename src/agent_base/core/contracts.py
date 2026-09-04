"""The ``AgentModule`` contract + ``ModuleContext``.

This is the single extension point every agent module implements. The base
never imports business logic; modules implement this protocol and the
registry (``core/registry.py``) wires them into the runtime.

The contract is intentionally minimal and grows by stage:
- Stage 1: ``name`` / ``description`` / ``build_graph`` / ``get_tools``
  (``get_tools`` returns an empty list until the tool pool lands in Stage 3)
- Stage 3: tool pool + checkpointer wiring
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, runtime_checkable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
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
    ``checkpointer`` — conversation-state saver (wired in Stage 3; ``None``
                       until then)
    """

    settings: Settings
    llm: BaseChatModel
    checkpointer: object | None = None


@runtime_checkable
class AgentModule(Protocol):
    """A runnable agent module.

    ``name``        — stable identifier; must match the AGENT_MODULES entry
                      and the module's directory name (registry enforces it)
    ``description`` — human-readable summary for discovery / supervisor
    ``build_graph`` — construct the module's compiled LangGraph
    ``get_tools``   — tools this module contributes to the shared pool
                      (empty in Stage 1; executed in Stage 3)
    """

    name: str
    description: str

    def build_graph(self, ctx: ModuleContext) -> Graph: ...

    def get_tools(self) -> list[BaseTool]: ...
