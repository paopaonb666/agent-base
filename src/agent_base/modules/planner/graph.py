"""planner 模块的图构建（M9）：plan → execute → check →(replan)* → synthesize。

五节点拓扑（决策 1）：execute 内联 ReAct 循环，外层图每子任务恒定
2 个 super-step；完成判定用规则（轮数耗尽 / 空输出门），不用额外 LLM
verdict；replan 有预算上限（``PLANNER_MAX_REPLANS``），耗尽后走
best-effort synthesize。
"""

from __future__ import annotations

from typing import Any, cast

from langgraph.graph import END, START, StateGraph

from agent_base.core.contracts import Graph, ModuleContext
from agent_base.modules.planner.state import PlanState


def build_planner_graph(ctx: ModuleContext) -> Graph:
    """构建并编译 planner 图（以 name="planner" 编译）。"""
    from agent_base.modules.planner.nodes_check import (
        make_check_node,
        make_replan_node,
        make_synthesize_node,
        route_after_check,
    )
    from agent_base.modules.planner.nodes_execute import make_execute_node
    from agent_base.modules.planner.nodes_plan import make_plan_node

    graph = StateGraph(PlanState)
    # add_node 的重载只认内联闭包的字面类型；工厂返回的节点函数是
    # NodeFn 别名，在此边界显式收窄（与 collab.py 的边界处理一致），
    # 避免 mypy 结果随上游 langgraph 版本在 arg-type 与 unused-ignore
    # 之间抖动。
    for name, make in (
        ("plan", make_plan_node),
        ("execute", make_execute_node),
        ("check", make_check_node),
        ("replan", make_replan_node),
        ("synthesize", make_synthesize_node),
    ):
        graph.add_node(name, cast(Any, make(ctx)))
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "check")
    graph.add_conditional_edges(
        "check",
        route_after_check(ctx),
        {"execute": "execute", "replan": "replan", "synthesize": "synthesize"},
    )
    graph.add_edge("replan", "execute")
    graph.add_edge("synthesize", END)
    return graph.compile(checkpointer=ctx.checkpointer, name="planner")
