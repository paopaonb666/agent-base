"""进程内字典实现的基座存储：memory 后端（重启即丢）与测试用。"""

from __future__ import annotations

import time
from typing import Any

from agent_base.core.storage.models import ThreadIndex


class CoreMemoryStore:
    """进程内字典实现：memory 后端（重启即丢）与测试用。"""

    def __init__(self) -> None:
        self._threads: dict[str, ThreadIndex] = {}
        self._plans: dict[str, dict[str, Any]] = {}

    # -- 会话索引（thread_index；S1 线程作用域） --------------------------------
    async def upsert_thread_index(self, entry: ThreadIndex) -> None:
        self._threads[entry.thread_id] = entry

    async def get_thread_index(self, thread_id: str) -> ThreadIndex | None:
        return self._threads.get(thread_id)

    async def list_thread_index(
        self, user_id: str, *, module: str | None = None, limit: int = 50
    ) -> list[ThreadIndex]:
        matched = [
            entry
            for entry in self._threads.values()
            if entry.user_id == user_id and (module is None or entry.module == module)
        ]
        matched.sort(key=lambda entry: entry.updated_at, reverse=True)
        return matched[: max(1, limit)]

    async def delete_thread_index(self, thread_id: str) -> None:
        self._threads.pop(thread_id, None)

    # -- 计划快照（plan_snapshots；M10 跨轮延续） --------------------------------
    async def save_plan(
        self,
        *,
        thread_id: str,
        module: str,
        tasks: list[dict[str, Any]],
        cursor: int,
        replans: int,
    ) -> None:
        self._plans[thread_id] = {
            "thread_id": thread_id,
            "module": module,
            "tasks": [dict(task) for task in tasks],
            "cursor": int(cursor),
            "replans": int(replans),
            "updated_at": time.time(),
        }

    async def get_plan(self, thread_id: str) -> dict[str, Any] | None:
        snapshot = self._plans.get(thread_id)
        return dict(snapshot) if snapshot is not None else None

    async def delete_plan(self, thread_id: str) -> None:
        self._plans.pop(thread_id, None)

    async def aclose(self) -> None:
        self._threads.clear()
        self._plans.clear()
