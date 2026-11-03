"""进程内字典实现的基座存储：memory 后端（重启即丢）与测试用。"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

from agent_base.core.storage.models import MessageNode, ThreadIndex


class CoreMemoryStore:
    """进程内字典实现：memory 后端（重启即丢）与测试用。

    消息树与指针的语义跟 SQL 实现对齐：``seq`` 按线程独立分配；
    ``list_nodes`` 按 seq 稳定排序。
    """

    def __init__(self) -> None:
        self._threads: dict[str, ThreadIndex] = {}
        self._plans: dict[str, dict[str, Any]] = {}
        self._nodes: dict[str, dict[str, MessageNode]] = {}
        self._pointers: dict[str, str] = {}

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

    # -- 消息树（message_nodes + tree_pointers；ADR-013） ------------------------
    async def next_node_seq(self, thread_id: str) -> int:
        nodes = self._nodes.get(thread_id, {})
        return max((node.seq for node in nodes.values()), default=0) + 1

    async def append_node(self, node: MessageNode) -> None:
        nodes = self._nodes.setdefault(node.thread_id, {})
        seq = node.seq or await self.next_node_seq(node.thread_id)
        nodes[node.node_id] = replace(node, seq=seq)

    async def get_node(self, thread_id: str, node_id: str) -> MessageNode | None:
        return self._nodes.get(thread_id, {}).get(node_id)

    async def list_nodes(
        self, thread_id: str, *, include_deleted: bool = False
    ) -> list[MessageNode]:
        nodes = self._nodes.get(thread_id, {}).values()
        ordered = sorted(nodes, key=lambda node: node.seq)
        if include_deleted:
            return list(ordered)
        return [node for node in ordered if not node.deleted]

    async def update_node_meta(
        self,
        thread_id: str,
        node_id: str,
        *,
        label: str | None = None,
        starred: bool | None = None,
        archived: bool | None = None,
    ) -> bool:
        node = self._nodes.get(thread_id, {}).get(node_id)
        if node is None:
            return False
        updated = replace(
            node,
            label=label[:120] if label is not None else node.label,
            starred=bool(starred) if starred is not None else node.starred,
            archived=bool(archived) if archived is not None else node.archived,
        )
        self._nodes[thread_id][node_id] = updated
        return True

    async def set_nodes_deleted(
        self, thread_id: str, node_ids: list[str], *, deleted: bool
    ) -> int:
        nodes = self._nodes.get(thread_id, {})
        changed = 0
        for node_id in node_ids:
            node = nodes.get(node_id)
            if node is None:
                continue
            nodes[node_id] = replace(node, deleted_at=time.time() if deleted else None)
            changed += 1
        return changed

    async def set_current_node(self, thread_id: str, node_id: str | None) -> bool:
        if node_id is None:
            self._pointers.pop(thread_id, None)
            return True
        self._pointers[thread_id] = node_id
        return True

    async def get_current_node(self, thread_id: str) -> str | None:
        return self._pointers.get(thread_id)

    async def delete_nodes_for_thread(self, thread_id: str) -> None:
        self._nodes.pop(thread_id, None)
        self._pointers.pop(thread_id, None)

    async def aclose(self) -> None:
        self._threads.clear()
        self._plans.clear()
        self._nodes.clear()
        self._pointers.clear()
