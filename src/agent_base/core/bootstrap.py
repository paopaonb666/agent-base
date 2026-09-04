"""Assemble the runtime: load modules, build the LLM, expose compiled graphs.

This is the thin "wiring" layer between config / registry / llm and the
entrypoints. It contains no business logic — it only composes the pieces the
base owns and hands a ready runtime to the CLI / server.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel

from agent_base.core.config import Settings
from agent_base.core.contracts import AgentModule, Graph, ModuleContext
from agent_base.core.llm import build_llm
from agent_base.core.registry import load_modules


@dataclass
class AgentRuntime:
    """A fully-assembled runtime: settings + llm + validated modules."""

    settings: Settings
    llm: BaseChatModel
    modules: dict[str, AgentModule]

    def context(self) -> ModuleContext:
        """The ModuleContext handed to every module's build_graph."""
        return ModuleContext(settings=self.settings, llm=self.llm)

    def graph(self, module_name: str) -> Graph:
        """Build (and compile) the named module's graph."""
        try:
            module = self.modules[module_name]
        except KeyError:
            available = ", ".join(sorted(self.modules)) or "(none)"
            raise KeyError(
                f"module {module_name!r} is not enabled; "
                f"available: {available}. Add it to AGENT_MODULES."
            ) from None
        return module.build_graph(self.context())


def create_runtime(settings: Settings | None = None) -> AgentRuntime:
    """Create a runtime from settings (or defaults). Fails fast on bad config."""
    resolved = settings if settings is not None else Settings()
    resolved.ensure_production_ready()
    llm = build_llm(resolved)
    modules = load_modules(resolved.agent_modules)
    return AgentRuntime(settings=resolved, llm=llm, modules=modules)
