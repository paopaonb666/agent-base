"""The ``ChatModule`` AgentModule implementation.

Reference implementation of the full contract: ``name`` / ``description`` /
``build_graph`` / ``get_tools``. Graph construction is delegated to
``graph.py`` and tools to ``tools.py``, keeping this file a thin adapter.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from agent_base.core.contracts import Graph, ModuleContext
from agent_base.modules.chat.graph import build_chat_graph
from agent_base.modules.chat.tools import get_tools


class ChatModule:
    """A minimal conversational agent — the template every module copies."""

    def __init__(self) -> None:
        self.name = "chat"
        self.description = (
            "Minimal conversational agent: a single LLM node over the message "
            "history. Reference implementation of the AgentModule contract."
        )

    def build_graph(self, ctx: ModuleContext) -> Graph:
        return build_chat_graph(ctx)

    def get_tools(self) -> list[BaseTool]:
        return get_tools()
