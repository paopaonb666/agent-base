"""plan 节点（B2）：把用户目标拆解为有界 JSON 子任务清单。

解析（``_parse_tasks``）容忍模型输出的常见噪声（```json 围栏、前后散
文），规范化为 ``Task`` 列表（重排 id 1..n、截断到 max_subtasks、全部
pending）；两次解析失败降级为单任务——规划失败不能让整轮对话失败。
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.runnables import RunnableConfig

from agent_base.core.context import content_text
from agent_base.core.contracts import ModuleContext
from agent_base.modules.planner.prompts import PLAN_PROMPT
from agent_base.modules.planner.state import PlanState, Task, _messages
from agent_base.modules.planner.streaming import emit_plan

_JSON_ARRAY = re.compile(r"\[.*\]", re.DOTALL)

# LangGraph 节点的统一签名：状态进、状态更新字典出。
NodeFn = Callable[[PlanState, RunnableConfig], Awaitable[dict[str, Any]]]


def make_plan_node(ctx: ModuleContext) -> NodeFn:
    model = ctx.llm
    max_subtasks = ctx.settings.planner.max_subtasks

    async def plan_node(state: PlanState, config: RunnableConfig) -> dict[str, Any]:
        goal = _last_human_text(_messages(state)) or "（未提供目标）"
        raw = str(
            (
                await model.ainvoke(
                    PLAN_PROMPT.format(max_subtasks=max_subtasks, goal=goal), config
                )
            ).content
        )
        tasks = _parse_tasks(raw, max_subtasks=max_subtasks, fallback_goal=goal)
        if not tasks:
            # 第一次解析失败：带错误反馈重试一次。
            retry = str(
                (
                    await model.ainvoke(
                        PLAN_PROMPT.format(max_subtasks=max_subtasks, goal=goal)
                        + f"\n\n你上次的输出无法解析为 JSON 数组：{raw[:200]}",
                        config,
                    )
                ).content
            )
            tasks = _parse_tasks(retry, max_subtasks=max_subtasks, fallback_goal=goal)
        emit_plan(tasks, "created", detail=f"拆解出 {len(tasks)} 个子任务")
        return {"tasks": tasks, "cursor": 0, "replans": 0}

    return plan_node


def _last_human_text(messages: list[Any]) -> str:
    """最后一条 human 消息的文本（兼容多模态 content blocks）。"""
    for message in reversed(messages):
        if getattr(message, "type", "") == "human":
            return content_text(message.content).strip()
    return ""


def _parse_tasks(raw: str, *, max_subtasks: int, fallback_goal: str) -> list[Task]:
    """模型输出 → 规范化 Task 列表；任何解析失败都降级为单任务。"""
    try:
        match = _JSON_ARRAY.search(raw)
        if match is None:
            raise ValueError("no JSON array found")
        parsed = json.loads(match.group(0))
        if not isinstance(parsed, list) or not parsed:
            raise ValueError("JSON array empty or not a list")
        goals = [str(item.get("goal", "")).strip() for item in parsed if isinstance(item, dict)]
        goals = [goal for goal in goals if goal]
        if not goals:
            raise ValueError("no usable goals")
        goals = goals[:max_subtasks]
        return [
            {"id": i + 1, "goal": goal, "status": "pending", "summary": "", "attempts": 0}
            for i, goal in enumerate(goals)
        ]
    except (ValueError, TypeError, json.JSONDecodeError, AttributeError):
        return [
            {
                "id": 1,
                "goal": fallback_goal,
                "status": "pending",
                "summary": "",
                "attempts": 0,
            }
        ]
