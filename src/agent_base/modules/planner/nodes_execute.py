"""execute 节点（B3）：子任务内联 ReAct 循环。

决策 1：循环内联在节点里而不是 LangGraph 子图——外层图每子任务恒定
2 个 super-step（execute+check），recursion_limit 不随子任务数和工具
轮数膨胀。复用基座件：``_stream_and_accumulate``（A1 修复后的 AIMessage
归一）、``handle_tool_error``（工具失败归一为模型可读反馈，模型自愈）、
``build_model_input``（注入去重 + 预算修剪）、共享工具池（超时 + 审计
包装器在直调路径上同样生效——本节点经 ``_execute_tool_calls`` 直接调
池工具，而非 ToolNode：langgraph 1.2 的 ToolNode 不支持在图外直接
ainvoke，直调反而少了这层不确定性）。

子任务级失败只有两种：工具轮数耗尽、空输出（check 的空输出门改判）。
循环局部累积 ``produced``，结束时整体进 messages 通道（checkpointer /
记忆形成 / 历史回放都能看到完整过程）。任何状态变更都拷贝重建 Task
（决策 7：不得原地修改）。
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from langchain_core.messages import SystemMessage, ToolCall, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from agent_base.core.context import build_model_input, content_text
from agent_base.core.contracts import ModuleContext
from agent_base.core.graphs import _stream_and_accumulate
from agent_base.core.tools import handle_tool_error
from agent_base.modules.planner.prompts import EXECUTE_SYSTEM
from agent_base.modules.planner.state import PlanState, Task, _cursor, _messages, _tasks
from agent_base.modules.planner.streaming import emit_plan

# LangGraph 节点的统一签名：状态进、状态更新字典出。
NodeFn = Callable[[PlanState, RunnableConfig], Awaitable[dict[str, Any]]]

# 子任务结果摘要截断（写回 Task.summary 的长度上限）。
_SUMMARY_MAX_CHARS = 500


def _make_tool_runner(
    pool: list[BaseTool],
) -> Callable[[list[ToolCall], RunnableConfig], Awaitable[list[ToolMessage]]]:
    tool_map = {t.name: t for t in pool}

    async def _execute_tool_calls(
        tool_calls: list[ToolCall], config: RunnableConfig
    ) -> list[ToolMessage]:
        """执行一轮 tool_calls，每个结果/失败都归一为 ToolMessage。"""
        messages: list[ToolMessage] = []
        for call in tool_calls:
            name = call.get("name", "")
            instance = tool_map.get(name)
            if instance is None:
                content = f"tool execution failed: unknown tool {name!r}"
            else:
                try:
                    result = await instance.ainvoke(call.get("args") or {}, config=config)
                    content = (
                        result
                        if isinstance(result, str)
                        else json.dumps(result, ensure_ascii=False, default=str)
                    )
                except Exception as exc:
                    content = handle_tool_error(exc)
            messages.append(
                ToolMessage(content=content, tool_call_id=call.get("id") or "", name=name)
            )
        return messages

    return _execute_tool_calls


def make_execute_node(ctx: ModuleContext) -> NodeFn:
    pool = list(ctx.tools)
    model = ctx.llm.bind_tools(pool) if pool else ctx.llm
    run_tool_calls = _make_tool_runner(pool) if pool else None
    p = ctx.settings.planner
    max_tokens = ctx.settings.memory.context_max_tokens

    def _subtask_input(state: PlanState, task: Task, tasks: list[Task], cursor: int) -> list[Any]:
        done = (
            "\n".join(
                f"#{t['id']} {t['goal']}：{t['summary'][:200]}"
                for t in tasks[:cursor]
                if t.get("status") == "done"
            )
            or "（无）"
        )
        messages = _messages(state)
        goal = messages[-1].content if messages else "（未知）"
        system = SystemMessage(
            content=EXECUTE_SYSTEM.format(goal=goal, done=done, current=task["goal"])
        )
        return [system, *build_model_input(messages, max_tokens=max_tokens)]

    async def execute_node(state: PlanState, config: RunnableConfig) -> dict[str, Any]:
        tasks = _tasks(state)
        cursor = _cursor(state)
        if cursor >= len(tasks):
            return {}  # 防御：replan 可能清空剩余任务
        task: Task = {
            **tasks[cursor],
            "status": "running",
            "attempts": int(tasks[cursor].get("attempts") or 0) + 1,
        }
        tasks[cursor] = task
        emit_plan(tasks, "progress", detail=f"子任务 {cursor + 1}/{len(tasks)} 执行中")

        inputs = _subtask_input(state, task, tasks, cursor)
        produced: list[Any] = []
        exhausted = True
        summary = ""
        for _ in range(p.subtask_tool_rounds):
            final = await _stream_and_accumulate(model, [*inputs, *produced])
            if final is None:
                exhausted = False  # 模型无产出：交给 check 的空输出门改判
                break
            produced.append(final)
            if not final.tool_calls:
                summary = content_text(final.content).strip()[:_SUMMARY_MAX_CHARS]
                exhausted = False
                break
            if run_tool_calls is None:
                break
            produced.extend(await run_tool_calls(final.tool_calls, config))
        status: Literal["done", "failed"] = "failed" if exhausted else "done"
        updated: Task = {
            **task,
            "status": status,
            "summary": "工具轮数耗尽" if exhausted else summary,
        }
        tasks[cursor] = updated
        emit_plan(tasks, "progress", detail=f"子任务 {cursor + 1} {status}")
        return {"messages": produced, "tasks": tasks}

    return execute_node
