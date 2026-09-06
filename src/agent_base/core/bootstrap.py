"""Assemble the runtime: load modules, build the LLM, wire state + tools.

This is the thin "wiring" layer between config / registry / llm / memory /
tools and the entrypoints. It contains no business logic — it only composes
the pieces the base owns and hands a ready runtime to the CLI / server.

Stage 3 additions: the checkpointer (``extensions/memory``) and the shared
tool pool (``core/tools``) are assembled here and flow into every module's
``ModuleContext``. ``create_runtime`` is async because the sqlite
checkpointer binds to the calling event loop.

Stage 4 addition: the reserved module name ``supervisor`` builds a
supervisor graph (``extensions/collab``) that orchestrates every registered
module as a sub-agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver

from agent_base.core.config import Settings
from agent_base.core.contracts import AgentModule, Graph, ModuleContext
from agent_base.core.llm import build_llm
from agent_base.core.registry import load_modules
from agent_base.core.tools import build_tool_pool
from agent_base.extensions.memory import build_checkpointer, close_checkpointer

# Reserved module name that builds the supervisor graph over all loaded
# modules instead of a single module's graph (Stage 4).
SUPERVISOR_MODULE = "supervisor"


@dataclass
class AgentRuntime:
    """A fully-assembled runtime: settings + llm + modules + state + tools."""

    settings: Settings
    llm: BaseChatModel
    modules: dict[str, AgentModule]
    tools: list[BaseTool] = field(default_factory=list)
    checkpointer: BaseCheckpointSaver[Any] | None = None
    _supervisor: Graph | None = field(default=None, repr=False)

    def context(self) -> ModuleContext:
        """The ModuleContext handed to every module's build_graph."""
        return ModuleContext(
            settings=self.settings,
            llm=self.llm,
            checkpointer=self.checkpointer,
            tools=self.tools,
        )

    def graph(self, module_name: str) -> Graph:
        """Build (and compile) the named module's graph.

        ``supervisor`` is reserved: it returns the multi-agent supervisor
        graph orchestrating every registered module (Stage 4).
        """
        if module_name == SUPERVISOR_MODULE:
            return self.supervisor_graph()
        try:
            module = self.modules[module_name]
        except KeyError:
            available = ", ".join([*sorted(self.modules), SUPERVISOR_MODULE]) or "(none)"
            raise KeyError(
                f"module {module_name!r} is not enabled; "
                f"available: {available}. Add it to AGENT_MODULES."
            ) from None
        return module.build_graph(self.context())

    def supervisor_graph(self) -> Graph:
        """Lazily build the supervisor graph over all registered modules."""
        if self._supervisor is None:
            # Deferred import: extensions may grow; core should not depend on
            # the import graph of every extension at module load time.
            from agent_base.extensions.collab import build_supervisor_graph

            self._supervisor = build_supervisor_graph(self.context(), self.modules)
        return self._supervisor

    async def close(self) -> None:
        """Release runtime-held resources (the sqlite connection, if any)."""
        if self.checkpointer is not None:
            await close_checkpointer(self.checkpointer)


async def create_runtime(settings: Settings | None = None) -> AgentRuntime:
    """Create a runtime from settings (or defaults). Fails fast on bad config.

    Must run inside the event loop that will execute the graphs (the sqlite
    checkpointer binds aiosqlite to the calling loop).
    """
    resolved = settings if settings is not None else Settings()
    resolved.ensure_production_ready()
    llm = build_llm(resolved)
    modules = load_modules(resolved.agent_modules)
    tools = build_tool_pool(modules, timeout=resolved.tool_timeout_seconds)
    checkpointer = await build_checkpointer(resolved)
    return AgentRuntime(
        settings=resolved, llm=llm, modules=modules, tools=tools, checkpointer=checkpointer
    )
