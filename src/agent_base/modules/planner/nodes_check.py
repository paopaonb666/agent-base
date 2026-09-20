"""check / replan / synthesize 节点与路由（B4）。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.runnables import RunnableConfig

from agent_base.core.contracts import ModuleContext
from agent_base.modules.planner.state import PlanState, _replans, _tasks

# LangGraph 节点的统一签名：状态进、状态更新字典出。
NodeFn = Callable[[PlanState, RunnableConfig], Awaitable[dict[str, Any]]]


def make_check_node(ctx: ModuleContext) -> NodeFn:
    async def check_node(state: PlanState, config: RunnableConfig) -> dict[str, Any]:
        # 占位（B1）：空输出门与 cursor 推进在 B4 落地。
        tasks = _tasks(state)
        next_idx = next(
            (i for i, t in enumerate(tasks) if t.get("status") == "pending"), len(tasks)
        )
        return {"tasks": tasks, "cursor": next_idx}

    return check_node


def make_replan_node(ctx: ModuleContext) -> NodeFn:
    async def replan_node(state: PlanState, config: RunnableConfig) -> dict[str, Any]:
        # 占位（B1）：预算内重规划在 B4 落地。
        return {"replans": _replans(state) + 1}

    return replan_node


def make_synthesize_node(ctx: ModuleContext) -> NodeFn:
    async def synthesize_node(state: PlanState, config: RunnableConfig) -> dict[str, Any]:
        # 占位（B1）：best-effort 综合在 B4 落地。
        return {"messages": []}

    return synthesize_node


def route_after_check(ctx: ModuleContext) -> Callable[[PlanState], str]:
    max_replans = ctx.settings.planner.max_replans

    def route(state: PlanState) -> str:
        tasks = _tasks(state)
        if any(t.get("status") == "failed" for t in tasks) and _replans(state) < max_replans:
            return "replan"
        if any(t.get("status") == "pending" for t in tasks):
            return "execute"
        return "synthesize"

    return route
