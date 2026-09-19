"""hello 模块的图构建。

这是 module-guide 五步配方在真实代码中的落地：最小的可行模块（一个
LLM 节点，无工具），经基座共享图工厂一行装配（``core/graphs.py``）。
它的存在是为了证明手册的承诺——新模块接入时无需触碰基座代码。

该图用运行时的 checkpointer 编译（按 thread 持久化对话状态），并以
``name="hello"`` 编译，这样 supervisor 就能像编排其他已注册模块一样
把它作为 sub-agent 编排。
"""

from __future__ import annotations

from agent_base.core.contracts import Graph, ModuleContext
from agent_base.core.graphs import build_single_agent_graph

MODULE_NAME = "hello"


def build_hello_graph(ctx: ModuleContext) -> Graph:
    """构建并编译 hello 图：单个 LLM 节点。"""
    return build_single_agent_graph(ctx, name=MODULE_NAME)
