"""``HelloModule`` 的 AgentModule 实现（module-guide 第 3 步）。

最小的可行 AgentModule：图构建委托给 ``graph.py``，不贡献任何工具。
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from agent_base.core.contracts import Graph, ModuleContext
from agent_base.modules.hello.graph import build_hello_graph


class HelloModule:
    """一个 hello-world 样板模块——手册五步配方的逐字实现。"""

    def __init__(self) -> None:
        self.name = "hello"
        self.description = "Hello-world sample module from the module-guide five-step recipe."

    def build_graph(self, ctx: ModuleContext) -> Graph:
        return build_hello_graph(ctx)

    def get_tools(self) -> list[BaseTool]:
        return []
