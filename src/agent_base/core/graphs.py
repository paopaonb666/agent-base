"""共享图工厂（P1-4 / H3）：单 LLM 节点 + 可选工具循环的公共装配。

chat / writer / hello 这类"单 LLM 节点（可选工具循环）"模块过去各自
复制同一套 ``StateGraph → ToolNode → compile`` 样板，且上下文工程
（注入去重 + token 预算修剪）只在 chat 里生效——writer/hello 的历史里
注入块逐轮累积，模型输入持续膨胀。工厂把样板与上下文工程一并收敛：

- 模块只声明 ``name`` / ``system_prompt`` / ``tools``，几行即可接入；
- 发给模型的输入统一经过 ``core.context.build_model_input``（注入去重
  + 预算修剪，工具配对不拆散）——**对所有模块生效**；
- 模块级 ``system_prompt`` 是临时拼装（不落库、不参与修剪去重）；
- 流式累积保持不变：SSE 层看到 token 级 delta，取消传播到 LLM 请求。

需要自定义拓扑的模块保留 escape hatch：直接使用 ``core.context`` 与
LangGraph 原生 API 自行组装。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

from langchain_core.messages import AIMessageChunk, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent_base.core.context import build_model_input
from agent_base.core.contracts import Graph, ModuleContext
from agent_base.core.tools import handle_tool_error


async def _stream_and_accumulate(model: Any, model_input: list[Any]) -> AIMessageChunk | None:
    """流式调用模型并累积分块（取消传播依赖 astream 不被吞掉）。"""
    final: AIMessageChunk | None = None
    async for chunk in model.astream(model_input):
        # astream 声明的产出类型是消息联合体；运行时的分块是
        # AIMessageChunk（仅对话模型）。
        piece = cast(AIMessageChunk, chunk)
        final = piece if final is None else final + piece
    return final


def build_single_agent_graph(
    ctx: ModuleContext,
    *,
    name: str,
    system_prompt: str | None = None,
    tools: Sequence[BaseTool] | None = None,
) -> Graph:
    """构建并编译"单 LLM 节点 + 可选工具循环"的图。

    ``tools`` 缺省用共享工具池（``ctx.tools``）；传空列表可显式禁用
    工具循环。图以 ``name=<模块名>`` 编译——supervisor 要求每个
    sub-agent 图携带其模块名。
    """
    pool = list(ctx.tools) if tools is None else list(tools)
    model = ctx.llm.bind_tools(pool) if pool else ctx.llm
    max_tokens = ctx.settings.memory.context_max_tokens

    async def call_model(state: MessagesState, config: RunnableConfig) -> dict[str, Any]:
        # 上下文工程（H3）：注入去重 + token 预算修剪，对所有模块生效。
        model_input = build_model_input(state["messages"], max_tokens=max_tokens)
        if system_prompt is not None:
            # 模块提示词临时拼装在最前：不落库，也不属于"注入型系统消息"。
            model_input = [SystemMessage(content=system_prompt), *model_input]
        final = await _stream_and_accumulate(model, model_input)
        if final is None:
            return {"messages": []}
        return {"messages": [final]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    if pool:
        # handle_tool_error 把工具失败归一成 ToolMessage 反馈，
        # 这样坏掉的工具绝不会打断对话（阶段 3 验收）。
        graph.add_node("tools", ToolNode(pool, handle_tool_errors=handle_tool_error))
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", tools_condition)
        graph.add_edge("tools", "agent")
    else:
        graph.add_edge(START, "agent")
        graph.add_edge("agent", END)
    return graph.compile(checkpointer=ctx.checkpointer, name=name)
