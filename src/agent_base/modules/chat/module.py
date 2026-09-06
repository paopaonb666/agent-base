"""``ChatModule`` 的 AgentModule 实现。

完整契约的参考实现：``name`` / ``description`` / ``build_graph`` /
``get_tools``。图构建委托给 ``graph.py``，工具委托给 ``tools.py``，
让本文件保持为一个薄适配层。
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from agent_base.core.contracts import Graph, ModuleContext
from agent_base.modules.chat.graph import build_chat_graph
from agent_base.modules.chat.tools import get_tools


class ChatModule:
    """一个最小化的对话 agent——每个模块都要照抄的模板。"""

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
