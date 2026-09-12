"""chat 模块的图构建。

在对话历史上运行单个 LLM 节点，当共享工具池非空时再外加一个 ``ToolNode`` 循环
（Stage 3）。该节点以流式输出模型并累积各个分块，因此：

- 服务端的 SSE 层能看到 token 级别的 ``delta`` 事件（它监听的是同一股流），并且
- 取消能够传播——中止图运行会取消正在进行的 LLM 请求，而不是让它在后台跑完
  （Stage 4）。

状态使用标准的 ``MessagesState`` 通道，其 ``add_messages`` reducer 会跨轮次累积
历史；用运行时的 checkpointer 编译后，历史会按 ``thread_id`` 持久化（内存或
sqlite）。

该图以 ``name="chat"`` 编译：supervisor（Stage 4）要求每个 sub-agent 图都携带
其模块的名字。
"""

from __future__ import annotations

from typing import Any, cast

from langchain_core.messages import AIMessageChunk
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent_base.core.contracts import Graph, ModuleContext
from agent_base.core.tools import handle_tool_error
from agent_base.memory.context import build_model_input

MODULE_NAME = "chat"


def build_chat_graph(ctx: ModuleContext) -> Graph:
    """构建并编译 chat 图（LLM 节点 + 可选的工具循环）。"""
    tools = ctx.tools
    # 每次调用都 bind_tools 也可以；池是每个图重建一次，所以这里绑定
    # 一次就够了。
    model = ctx.llm.bind_tools(tools) if tools else ctx.llm

    async def call_model(state: MessagesState, config: RunnableConfig) -> dict[str, Any]:
        # 上下文工程（M6d）：注入型系统消息去重 + token 预算修剪（保尾部、
        # 工具配对不拆散）。只影响发给模型的输入——checkpointer 里的全量
        # 历史不动（recall 语义）；没有记忆系统时这是纯上下文保护。
        model_input = build_model_input(
            state["messages"], max_tokens=ctx.settings.memory_context_max_tokens
        )
        final: AIMessageChunk | None = None
        async for chunk in model.astream(model_input):
            # astream 声明的产出类型是消息联合体；运行时的分块是
            # AIMessageChunk（仅对话模型）。
            piece = cast(AIMessageChunk, chunk)
            final = piece if final is None else final + piece
        if final is None:
            return {"messages": []}
        return {"messages": [final]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    if tools:
        # handle_tool_error 把工具失败归一成 ToolMessage 反馈，
        # 这样坏掉的工具绝不会打断对话（阶段 3 验收）。
        graph.add_node("tools", ToolNode(tools, handle_tool_errors=handle_tool_error))
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", tools_condition)
        graph.add_edge("tools", "agent")
    else:
        graph.add_edge(START, "agent")
        graph.add_edge("agent", END)
    return graph.compile(checkpointer=ctx.checkpointer, name=MODULE_NAME)
