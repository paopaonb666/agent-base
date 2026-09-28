"""计划快照的旁路持久化（M10 跨轮延续）。

planner 图在每个状态变更点（拆解 / check 推进 / 重规划 / 完成）把
tasks/cursor/replans upsert 进 ``plan_snapshots`` 表；plan 检查端点读表
而非图状态——chat 轮穿插（chat 图的 checkpoint 只含 messages 通道）
不再重置计划。

旁路语义：写失败只记日志，绝不影响图执行（与记忆审计同款约定）。
``plan_store`` 缺席（存储后端不可用）时退化为只写图状态，跨轮延续
不可用但图行为不变。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.runnables import RunnableConfig

from agent_base.core.threads import ThreadId
from agent_base.modules.planner.state import Task

if TYPE_CHECKING:  # pragma: no cover - import avoided at runtime
    from agent_base.core.contracts import ModuleContext

logger = logging.getLogger(__name__)


async def persist_plan(
    ctx: ModuleContext,
    config: RunnableConfig,
    *,
    tasks: list[Task],
    cursor: int,
    replans: int,
) -> None:
    """把当前计划快照落表（旁路；失败只记日志）。"""
    store = ctx.plan_store
    if store is None:
        return
    thread_id = str((config.get("configurable") or {}).get("thread_id") or "")
    parsed = ThreadId.try_parse(thread_id)
    if parsed is None:
        return
    try:
        await store.save_plan(
            thread_id=thread_id,
            module=parsed.module,
            tasks=[dict(task) for task in tasks],
            cursor=cursor,
            replans=replans,
        )
    except Exception:
        logger.warning("planner: 计划快照写入失败（不影响图执行）", exc_info=True)
