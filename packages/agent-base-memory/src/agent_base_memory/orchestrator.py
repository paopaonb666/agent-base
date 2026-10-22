"""形成管线编排与上下文组装（M6c/M6d，P0-2 拆分自 MemoryService）。

数据服务不自编排：抽取/整合/画像/摘要的管线调度（``capture_turn``）、
每用户捕获锁、线程级冻结注入段缓存与逐轮召回注入都集中在这里。
``MemoryService`` 门面委托本模块；管线的破坏性决策（整合裁决）仍在
``pipeline.py`` 内部。
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any

from agent_base.memory.context import compose_frozen_context, compose_recall_context
from agent_base.memory.pipeline import MemoryPipeline, render_transcript
from agent_base.memory.retrieval import ScoredMemory

if TYPE_CHECKING:  # pragma: no cover - import avoided at runtime
    from agent_base.core.config import MemorySettings
    from agent_base.memory.profile import ProfileService
    from agent_base.memory.store import MemoryStore

logger = logging.getLogger(__name__)

# 门面的混合检索（user_id/agent_id/query 关键字参数）。
SearchFn = Callable[..., Coroutine[Any, Any, list[ScoredMemory]]]

# 管线的惰性提供者：门面的 ``pipeline`` 属性在构造后仍可替换（测试与
# 装配路径都会换桩），编排层每次捕获时取当前值。
PipelineProvider = Callable[[], MemoryPipeline | None]


class CaptureLockRegistry:
    """按 user_id 的捕获串行化锁（锐评 #10）。

    同一用户并发轮次的后台捕获会互相看不到对方未写入的记忆，导致
    重复 ADD——串行化同一用户的捕获。注册表跨服务实例的进程级唯一
    由组合根保证（运行时单实例装配，多 worker 各自为政）。
    """

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}

    def for_user(self, user_id: str) -> asyncio.Lock:
        return self._locks.setdefault(user_id, asyncio.Lock())


class ContextSnapshotCache:
    """线程级冻结注入段的 LRU 缓存（成本治理 T2.1）。

    键 (user_id, agent_id, thread_id)。单进程口径（与 /metrics 同款
    局限，多 worker 不保证）。
    """

    def __init__(self) -> None:
        self._entries: OrderedDict[tuple[str, str, str], str] = OrderedDict()

    def get(self, key: tuple[str, str, str]) -> str | None:
        cached = self._entries.get(key)
        if cached is not None:
            self._entries.move_to_end(key)
        return cached

    def put(self, key: tuple[str, str, str], value: str, *, max_entries: int) -> None:
        self._entries[key] = value
        while len(self._entries) > max_entries:
            self._entries.popitem(last=False)

    def purge_thread(self, thread_id: str) -> int:
        """删除该线程的全部快照（线程删除的级联清理）；返回清理数。"""
        keys = [key for key in self._entries if key[2] == thread_id]
        for key in keys:
            self._entries.pop(key, None)
        return len(keys)


class MemoryOrchestrator:
    """记忆形成与上下文注入的编排层：依赖收敛到 store + MemorySettings 节
    + 管线 + 画像服务 + 检索回调。"""

    def __init__(
        self,
        *,
        store: MemoryStore,
        settings: MemorySettings,
        pipeline: PipelineProvider,
        profile: ProfileService,
        search: SearchFn,
        locks: CaptureLockRegistry | None = None,
        snapshots: ContextSnapshotCache | None = None,
    ) -> None:
        self._store = store
        self._settings = settings
        self._pipeline = pipeline
        self._profile = profile
        self._search = search
        self._locks = locks if locks is not None else CaptureLockRegistry()
        self._snapshots = snapshots if snapshots is not None else ContextSnapshotCache()

    async def capture_turn(
        self,
        *,
        user_id: str,
        agent_id: str,
        thread_id: str,
        messages: list[Any],
        force: bool = False,
    ) -> dict[str, Any] | None:
        """一轮对话结束后的记忆形成（后台调用；绝不抛异常）。

        ``messages`` 是该线程的全部持久化消息（来自 checkpointer 快照）：
        转写渲染、轮次计数与触发阈值都在这里统一处理。``force=True``
        旁路 ``MEMORY_CAPTURE_ENABLED`` 门控（夜间批脚本 T3.2）。
        """
        pipeline = self._pipeline()
        if pipeline is None:
            return None
        human_count = sum(1 for m in messages if getattr(m, "type", "") == "human")
        transcript = render_transcript(
            messages[-self._settings.extraction_max_messages :],
            self._settings.extraction_max_input_chars,
        )
        if not transcript.strip():
            return None
        lock = self._locks.for_user(user_id)
        async with lock:
            try:
                return await pipeline.capture_turn(
                    user_id=user_id,
                    agent_id=agent_id,
                    thread_id=thread_id,
                    transcript=transcript,
                    human_count=human_count,
                    force=force,
                )
            except Exception:
                # 管线内部已逐步失败安全；这里兜底防任何漏网异常干扰调用方。
                logger.exception("memory: capture_turn 意外失败")
                return None

    async def compose_context(
        self, *, user_id: str, agent_id: str, thread_id: str, query: str
    ) -> str | None:
        """线程级**冻结**注入段（M6d + T2.1）：画像 + 记忆块 + 会话摘要。

        快照在线程首轮定型并缓存（LRU 上限 ``context_snapshot_max``）：
        同线程后续调用返回逐字节相同的内容，即使画像/记忆块/摘要中途
        已更新——供应商前缀缓存按最长公共前缀命中，冻结头部是输入缓存
        的地基；数据变更在**下个会话**进入注入（接受的行为权衡，见
        ``compose_frozen_context``）。``purge_thread`` 会级联清理。

        逐轮变化的查询相关召回走 ``recall_context``（贴尾注入）。
        任何一步失败都降级为"缺那一节"；整体无内容时返回 None。
        """
        key = (user_id, agent_id, thread_id)
        cached = self._snapshots.get(key)
        if cached is not None:
            return cached
        try:
            global_blocks = await self._store.list_blocks(user_id, "*")
            agent_blocks = await self._store.list_blocks(user_id, agent_id)
            profile = (
                await self._profile.get_profile(user_id) if self._settings.profile_enabled else None
            )
            summary_record = await self._store.get_summary(user_id, thread_id)
            frozen = compose_frozen_context(
                profile=profile,
                blocks=[*global_blocks, *agent_blocks],
                summary=summary_record.summary if summary_record else None,
                max_chars=self._settings.context_max_chars,
            )
        except Exception:
            logger.warning("memory: 冻结注入段组装失败（本轮不注入）", exc_info=True)
            return None
        if frozen is not None:
            self._snapshots.put(key, frozen, max_entries=self._settings.context_snapshot_max)
        return frozen

    async def recall_context(self, *, user_id: str, query: str) -> str | None:
        """本轮**召回**注入段（M6d + T2.1）：按查询召回的相关记忆。

        用户级召回（与 memory_search 工具同契约：跨模块可见）。逐轮
        变化，由 server 紧贴本轮 human 注入——尾部变化只牺牲自身之后
        的缓存，不动冻结头部与历史体。失败降级为 None。
        """
        try:
            recalled = await self._search(user_id=user_id, agent_id=None, query=query)
            return compose_recall_context(
                recalled=recalled,
                max_chars=self._settings.context_max_chars,
            )
        except Exception:
            logger.warning("memory: 召回注入段组装失败（本轮不注入召回）", exc_info=True)
            return None

    async def purge_thread(self, thread_id: str) -> None:
        """线程删除的级联清理（滚动摘要 + 知识库分块 + 注入快照；跨会话记忆保留）。"""
        await self._store.delete_for_thread(thread_id)
        self._snapshots.purge_thread(thread_id)
