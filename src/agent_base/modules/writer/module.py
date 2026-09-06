"""``WriterModule`` 的 AgentModule 实现。"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from agent_base.core.contracts import Graph, ModuleContext
from agent_base.modules.writer.graph import build_writer_graph


class WriterModule:
    """一个文本起草专家——第二个参考模块。"""

    def __init__(self) -> None:
        self.name = "writer"
        self.description = (
            "Writing specialist: drafts, rewrites, summarizes, and polishes "
            "text. Route any request that is about producing or improving a "
            "piece of writing to this agent."
        )

    def build_graph(self, ctx: ModuleContext) -> Graph:
        return build_writer_graph(ctx)

    def get_tools(self) -> list[BaseTool]:
        return []
