"""chat 模块的图构建。

单 LLM 节点 + 可选工具循环，经基座共享图工厂装配（``core/graphs.py``）：
上下文工程（注入去重 + token 预算修剪）、工具失败归一、流式累积与取消
传播都是工厂的公共行为，本模块只剩差异化的声明（模块名，无提示词）。

该图以 ``name="chat"`` 编译：supervisor（Stage 4）要求每个 sub-agent 图
都携带其模块的名字。
"""

from __future__ import annotations

from agent_base.core.contracts import Graph, ModuleContext
from agent_base.core.graphs import build_single_agent_graph

MODULE_NAME = "chat"


def build_chat_graph(ctx: ModuleContext) -> Graph:
    """构建并编译 chat 图（LLM 节点 + 可选的工具循环）。"""
    return build_single_agent_graph(ctx, name=MODULE_NAME)
