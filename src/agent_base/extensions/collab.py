"""多 Agent 协作：supervisor 模板（阶段 4）。

包装 LangGraph 原生的 ``create_supervisor``（langgraph-supervisor）：每个
已注册的模块都成为一个 sub-agent，由一个 supervisor 节点读取每个模块的
``name`` + ``description``（与用于发现的契约字段相同）来路由。基座自身
不添加任何编排逻辑——这正是“薄基座”原则的全部意义所在。

对模块的要求（在这里作为早期失败强制执行）：
- 每个模块图都必须以 ``name=<模块名>`` 编译，这样 supervisor 才能把
  控制权交给它（由 langgraph-supervisor 本身检查）；
- ``description`` 是 supervisor 模型在路由时看到的内容，因此必须是有
  意义的摘要（registry 已校验其非空）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph_supervisor import create_handoff_tool, create_supervisor

from agent_base.core.contracts import Graph, ModuleContext

if TYPE_CHECKING:  # pragma: no cover - import avoided at runtime
    from agent_base.core.contracts import AgentModule

DEFAULT_PROMPT = (
    "You are a team supervisor coordinating specialist agents. "
    "Break the user request into steps and dispatch them to suitable agents "
    "in sequence using the handoff tools; call as many agents as the task "
    "requires. Relay intermediate results between agents when one step "
    "depends on another, then synthesize a final answer. "
    "The supervisor itself has no memory tools: if the user asks to recall, "
    "save or update long-term memory, hand off to the conversational agent "
    "(e.g. chat) that owns the memory_search/memory_save tools, then relay "
    "the tool results verbatim."
)


def build_supervisor_graph(ctx: ModuleContext, modules: dict[str, AgentModule]) -> Graph:
    """把每个已注册的模块编排为由 supervisor 主导的 sub-agent。"""
    if not modules:
        raise ValueError("supervisor requires at least one module in AGENT_MODULES")

    # 每个 sub-agent 都是该模块自己的编译图；名字来自模块
    # （并且与它们的 AGENT_MODULES 条目一致）。
    agents = [module.build_graph(ctx) for module in modules.values()]

    # Handoff 工具携带模块描述，这样 supervisor 模型就能根据每个 agent
    # 实际做什么来路由。
    handoffs = [
        create_handoff_tool(agent_name=module.name, description=module.description)
        for module in modules.values()
    ]

    workflow = create_supervisor(
        # 该库的类型桩里 Graph 别名 vs Pregel / 列表不变性存在不匹配；
        # 运行时兼容性由 tests/test_collab.py 覆盖。
        agents=agents,  # type: ignore[arg-type]
        model=ctx.llm,
        tools=handoffs,  # type: ignore[arg-type]
        prompt=DEFAULT_PROMPT,
    )
    return workflow.compile(checkpointer=ctx.checkpointer)
