"""writer 模块的图构建。

一个带写作取向系统提示词的 LLM 节点，经基座共享图工厂装配
（``core/graphs.py``）——结构与 chat 完全一致，唯一的区别就是提示词，
而这正是要点所在：模块的区别在于行为，而不在于接线。工厂同时带来
基座级的上下文保护（注入去重 + token 预算修剪），注入块不再随轮次
无界累积（H3）。
"""

from __future__ import annotations

from agent_base.core.contracts import Graph, ModuleContext
from agent_base.core.graphs import build_single_agent_graph

MODULE_NAME = "writer"

WRITER_PROMPT = (
    "You are a professional writing agent. Draft, rewrite, summarize, or "
    "polish the requested text with clear structure and a consistent voice. "
    "Ask for nothing except the writing task itself."
)


def build_writer_graph(ctx: ModuleContext) -> Graph:
    """构建并编译 writer 图。"""
    return build_single_agent_graph(ctx, name=MODULE_NAME, system_prompt=WRITER_PROMPT)
