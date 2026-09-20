"""execute 节点（B3）：子任务内联 ReAct 循环。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.runnables import RunnableConfig

from agent_base.core.contracts import ModuleContext
from agent_base.modules.planner.state import PlanState, _cursor, _tasks

# LangGraph 节点的统一签名：状态进、状态更新字典出。
NodeFn = Callable[[PlanState, RunnableConfig], Awaitable[dict[str, Any]]]


def make_execute_node(ctx: ModuleContext) -> NodeFn:
    async def execute_node(state: PlanState, config: RunnableConfig) -> dict[str, Any]:
        # 占位（B1）：真实执行在 B3 落地。
        tasks = _tasks(state)
        cursor = _cursor(state)
        if cursor < len(tasks):
            tasks[cursor] = {
                **tasks[cursor],
                "status": "done",
                "summary": str(tasks[cursor].get("goal", "")),
            }
        return {"messages": [], "tasks": tasks}

    return execute_node
