"""The ``HelloModule`` AgentModule implementation (module-guide step 3).

The smallest possible AgentModule: graph construction delegated to
``graph.py``, no contributed tools.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from agent_base.core.contracts import Graph, ModuleContext
from agent_base.modules.hello.graph import build_hello_graph


class HelloModule:
    """A hello-world sample module — the guide's five-step recipe, verbatim."""

    def __init__(self) -> None:
        self.name = "hello"
        self.description = "Hello-world sample module from the module-guide five-step recipe."

    def build_graph(self, ctx: ModuleContext) -> Graph:
        return build_hello_graph(ctx)

    def get_tools(self) -> list[BaseTool]:
        return []
