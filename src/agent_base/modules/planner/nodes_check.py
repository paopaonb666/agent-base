"""check / replan / synthesize 节点与路由（B4）。

- check：空输出一致性门（评审修正 3——模型中途放弃不得静默放行）+
  cursor 推进到下一个 pending；
- replan：预算内（``PLANNER_MAX_REPLANS``）重规划剩余工作，保留 done
  结果，解析失败以 failed 任务目标兜底；
- synthesize：无工具的 best-effort 综合——replan 预算之外绝不再开第二条
  执行路径（决策 2），如实标注未完成项；
- 路由：有 failed 且预算未耗尽 → replan；有 pending → execute；
  否则 → synthesize。

所有节点与路由禁止对非 messages 键 ``state[...]`` 直读（决策 6）。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from agent_base.core.context import build_model_input
from agent_base.core.contracts import ModuleContext
from agent_base.core.graphs import _stream_and_accumulate
from agent_base.modules.planner.prompts import REPLAN_PROMPT, SYNTH_PROMPT
from agent_base.modules.planner.state import PlanState, _cursor, _messages, _replans, _tasks
from agent_base.modules.planner.streaming import emit_plan

# LangGraph 节点的统一签名：状态进、状态更新字典出。
NodeFn = Callable[[PlanState, RunnableConfig], Awaitable[dict[str, Any]]]


def make_check_node(ctx: ModuleContext) -> NodeFn:
    min_chars = ctx.settings.planner.min_summary_chars

    async def check_node(state: PlanState, config: RunnableConfig) -> dict[str, Any]:
        tasks = _tasks(state)
        cur = _cursor(state)
        if cur < len(tasks):
            t = tasks[cur]
            if t["status"] == "done" and not t["summary"].strip():
                # 评审修正 3：模型中途放弃（无产出就停止调工具）不得静默放行。
                tasks[cur] = {**t, "status": "failed", "summary": "输出为空"}
            elif min_chars and t["status"] == "done" and len(t["summary"]) < min_chars:
                tasks[cur] = {
                    **t,
                    "status": "failed",
                    "summary": f"输出不足 {min_chars} 字",
                }
        next_idx = next((i for i, t in enumerate(tasks) if t["status"] == "pending"), len(tasks))
        return {"tasks": tasks, "cursor": next_idx}

    return check_node


def make_replan_node(ctx: ModuleContext) -> NodeFn:
    model = ctx.llm
    max_subtasks = ctx.settings.planner.max_subtasks
    from agent_base.modules.planner.nodes_plan import _parse_tasks

    async def replan_node(state: PlanState, config: RunnableConfig) -> dict[str, Any]:
        tasks = _tasks(state)
        done = [t for t in tasks if t["status"] == "done"]
        failed = [t for t in tasks if t["status"] == "failed"]
        goal = _last_human_text(_messages(state)) or "（未提供目标）"
        raw = str(
            (
                await model.ainvoke(
                    REPLAN_PROMPT.format(
                        goal=goal,
                        done="\n".join(
                            f"#{t['id']} {t['goal']}：{t['summary'][:150]}" for t in done
                        )
                        or "（无）",
                        failed="\n".join(
                            f"#{t['id']} {t['goal']}：{t['summary'][:150]}" for t in failed
                        )
                        or "（无）",
                        max_subtasks=max_subtasks,
                    ),
                    config,
                )
            ).content
        )
        fallback_goal = failed[-1]["goal"] if failed else goal
        new_tasks = _parse_tasks(raw, max_subtasks=max_subtasks, fallback_goal=fallback_goal)
        merged = [*done, *new_tasks][:max_subtasks]
        emit_plan(merged, "replanned", detail=f"重规划为 {len(merged)} 个子任务")
        return {"tasks": merged, "cursor": len(done), "replans": _replans(state) + 1}

    return replan_node


def make_synthesize_node(ctx: ModuleContext) -> NodeFn:
    model = ctx.llm
    max_tokens = ctx.settings.memory.context_max_tokens

    async def synthesize_node(state: PlanState, config: RunnableConfig) -> dict[str, Any]:
        # 保持无工具（决策 2）：不在 replan 预算之外开第二条执行路径。
        plan = "\n".join(
            f"#{t['id']} [{t['status']}] {t['goal']}：{t['summary'][:300]}" for t in _tasks(state)
        )
        final = await _stream_and_accumulate(
            model,
            [
                SystemMessage(content=SYNTH_PROMPT.format(plan=plan)),
                *build_model_input(_messages(state), max_tokens=max_tokens),
            ],
        )
        emit_plan(_tasks(state), "done")
        return {"messages": [final] if final is not None else []}

    return synthesize_node


def route_after_check(ctx: ModuleContext) -> Callable[[PlanState], str]:
    max_replans = ctx.settings.planner.max_replans

    def route(state: PlanState) -> str:
        tasks = _tasks(state)
        if any(t["status"] == "failed" for t in tasks) and _replans(state) < max_replans:
            return "replan"
        if any(t["status"] == "pending" for t in tasks):
            return "execute"
        return "synthesize"

    return route


def _last_human_text(messages: list[Any]) -> str:
    from agent_base.core.context import content_text

    for message in reversed(messages):
        if isinstance(message, HumanMessage) or getattr(message, "type", "") == "human":
            return content_text(message.content).strip()
    return ""
