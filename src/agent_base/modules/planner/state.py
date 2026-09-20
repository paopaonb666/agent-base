"""planner 状态模型与跨图访问助手（决策 6）。

planner 图可能恢复在 chat 写过的线程上（决策 5：图选择与线程命名空间
解耦），checkpoint 里没有 tasks/cursor/replans 键。本模块内的强制约定：
**任何节点与路由函数都不得对非 messages 键做 ``state[...]`` 直读**，
一律走这里的访问助手（``state.get`` 兜底默认值）。
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from langgraph.graph import MessagesState

TaskStatus = Literal["pending", "running", "done", "failed"]


class Task(TypedDict):
    id: int
    goal: str
    status: TaskStatus
    # 子任务执行结果摘要（execute 结束时写入；check 的空输出门据此改判）
    summary: str
    attempts: int


class PlanState(MessagesState):
    tasks: list[Task]
    cursor: int  # 下一个待执行子任务的下标（check 负责推进）
    replans: int  # 已发生的重规划次数（预算 = planner.max_replans）


def _messages(state: PlanState) -> list[Any]:
    return list(state.get("messages") or [])


def _tasks(state: PlanState) -> list[Task]:
    return list(state.get("tasks") or [])


def _cursor(state: PlanState) -> int:
    return int(state.get("cursor") or 0)


def _replans(state: PlanState) -> int:
    return int(state.get("replans") or 0)
