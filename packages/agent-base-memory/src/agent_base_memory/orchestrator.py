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
from collections.abc import AsyncIterator, Callable, Coroutine, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import TYPE_CHECKING, Any, Protocol

from agent_base_memory.context import compose_frozen_context, compose_recall_context
from agent_base_memory.pipeline import MemoryPipeline, render_transcript
from agent_base_memory.retrieval import ScoredMemory

if TYPE_CHECKING:  # pragma: no cover - import avoided at runtime
    from agent_base.core.config import MemorySettings
    from agent_base.extensions.redissvc import RedisService
    from agent_base_memory.profile import ProfileService
    from agent_base_memory.store import MemoryStore

logger = logging.getLogger(__name__)

# 门面的混合检索（user_id/agent_id/query 关键字参数）。
SearchFn = Callable[..., Coroutine[Any, Any, list[ScoredMemory]]]

# 管线的惰性提供者：门面的 ``pipeline`` 属性在构造后仍可替换（测试与
# 装配路径都会换桩），编排层每次捕获时取当前值。
PipelineProvider = Callable[[], MemoryPipeline | None]


class CaptureLocks(Protocol):
    """按 user_id 串行化记忆捕获（进程内 asyncio.Lock 或 Redis 分布式锁）。

    同一用户并发轮次的后台捕获会互相看不到对方未写入的记忆，导致
    重复 ADD——串行化同一用户的捕获（锐评 #10）。0.2 起锁面抽象化：
    单进程装配用 ``CaptureLockRegistry``，多 worker（REDIS_ENABLED=true）
    用 ``RedisCaptureLocks``（ADR-012）。

    上下文**产出是否获得锁**（``async with ... as acquired``）：进程内实现
    恒为 True；Redis 实现在等待超时时产出 False（调用方据此跳过本轮，
    而不是死等——捕获每轮读线程全量消息，跳过的一轮由下一轮补上）。
    """

    def for_user(self, user_id: str) -> AbstractAsyncContextManager[bool]: ...


class CaptureLockRegistry:
    """进程内实现：按 user_id 的 ``asyncio.Lock`` 注册表。

    注册表跨服务实例的进程级唯一由组合根保证；多 worker 各自为政的
    局限由 ``RedisCaptureLocks`` 解决（本实现保持 0.1 行为零变化）。
    """

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}

    def for_user(self, user_id: str) -> AbstractAsyncContextManager[bool]:
        lock = self._locks.setdefault(user_id, asyncio.Lock())

        @asynccontextmanager
        async def _ctx() -> AsyncIterator[bool]:
            async with lock:
                yield True

        return _ctx()


class RedisCaptureLocks:
    """Redis 分布式捕获锁（ADR-012）：多 worker 下同一用户的捕获仍串行。

    - 抢锁等待超过 ``acquire_timeout_seconds`` → 产出 False（调用方跳过
      本轮）：重复 ADD 竞态与死等二选一，选前者；
    - Redis 故障 → 回退进程内锁（单进程串行化仍在，跨 worker 串行化
      暂时缺席）；
    - ``ttl_seconds`` 兜底持有者崩溃后的死锁。
    """

    def __init__(
        self,
        redis: RedisService,
        *,
        ttl_seconds: int,
        acquire_timeout_seconds: float,
    ) -> None:
        self._redis = redis
        self._ttl_seconds = ttl_seconds
        self._acquire_timeout_seconds = acquire_timeout_seconds
        self._fallback = CaptureLockRegistry()

    def for_user(self, user_id: str) -> AbstractAsyncContextManager[bool]:
        @asynccontextmanager
        async def _ctx() -> AsyncIterator[bool]:
            key = f"lock:capture:{user_id}"
            try:
                token = await self._redis.acquire_lock(
                    key,
                    ttl_seconds=self._ttl_seconds,
                    acquire_timeout_seconds=self._acquire_timeout_seconds,
                )
            except Exception:
                logger.warning("memory: Redis 捕获锁不可用，回退进程内锁", exc_info=True)
                async with self._fallback.for_user(user_id):
                    yield True
                return
            if token is None:
                logger.warning("memory: 用户 %s 的捕获锁等待超时（本轮跳过，下一轮补上）", user_id)
                yield False
                return
            try:
                yield True
            finally:
                try:
                    await self._redis.release_lock(key, token)
                except Exception:
                    logger.warning("memory: 捕获锁释放失败（TTL 兜底过期）", exc_info=True)

        return _ctx()


class SnapshotCache(Protocol):
    """线程级冻结注入段缓存（进程内 LRU 或 Redis 共享，ADR-012）。

    键 (user_id, agent_id, thread_id, summary_node)：第 4 位是冻结段实际
    命中的**摘要节点 id**（树模式，ADR-013）或空串（线程口径）——同一
    分支的摘要节点不变 → 缓存持续命中；切分支 / 摘要推进到新叶子 →
    新键重建。
    """

    async def get(self, key: tuple[str, str, str, str]) -> str | None: ...

    async def put(
        self, key: tuple[str, str, str, str], value: str, *, max_entries: int
    ) -> None: ...

    async def purge_thread(self, thread_id: str) -> int: ...


class ContextSnapshotCache:
    """线程级冻结注入段的 LRU 缓存（成本治理 T2.1；进程内实现）。

    键 (user_id, agent_id, thread_id, summary_node)。单进程口径（与
    /metrics 同款局限）；多 worker 共享见 ``RedisSnapshotCache``。
    """

    def __init__(self) -> None:
        self._entries: OrderedDict[tuple[str, str, str, str], str] = OrderedDict()

    async def get(self, key: tuple[str, str, str, str]) -> str | None:
        cached = self._entries.get(key)
        if cached is not None:
            self._entries.move_to_end(key)
        return cached

    async def put(self, key: tuple[str, str, str, str], value: str, *, max_entries: int) -> None:
        self._entries[key] = value
        while len(self._entries) > max_entries:
            self._entries.popitem(last=False)

    async def purge_thread(self, thread_id: str) -> int:
        """删除该线程的全部快照（线程删除的级联清理）；返回清理数。"""
        keys = [key for key in self._entries if key[2] == thread_id]
        for key in keys:
            self._entries.pop(key, None)
        return len(keys)


class RedisSnapshotCache:
    """Redis 共享的冻结注入段缓存（ADR-012）：多 worker 逐字节一致。

    键 ``snap:{thread_id}/{user_id}/{agent_id}``（线程在前，
    ``purge_thread`` 按前缀清理）；TTL 限界——跨进程共享后由 TTL 而非
    LRU 容量兜底（``max_entries`` 形参为协议兼容保留，本实现忽略）。
    get/put 的 Redis 故障降级为"未命中/放弃写回"：缓存层绝不破坏注入
    路径，下轮重建即可。
    """

    def __init__(self, redis: RedisService, *, ttl_seconds: int) -> None:
        self._redis = redis
        self._ttl_seconds = ttl_seconds

    @staticmethod
    def _key(key: tuple[str, str, str, str]) -> str:
        # 线程口径（summary_node 为空）保持历史键格式不变（存量部署兼容）；
        # 树模式（ADR-013）追加命中的摘要节点段，切分支自然换键。
        user_id, agent_id, thread_id = key[0], key[1], key[2]
        summary_node = key[3] if len(key) > 3 else ""
        base = f"snap:{thread_id}/{user_id}/{agent_id}"
        return f"{base}/{summary_node}" if summary_node else base

    async def get(self, key: tuple[str, str, str, str]) -> str | None:
        try:
            return await self._redis.get(self._key(key))
        except Exception:
            logger.warning("memory: 快照缓存读取失败（按未命中处理）", exc_info=True)
            return None

    async def put(self, key: tuple[str, str, str, str], value: str, *, max_entries: int) -> None:
        try:
            await self._redis.set_ex(self._key(key), value, ttl_seconds=self._ttl_seconds)
        except Exception:
            logger.warning("memory: 快照缓存写回失败（下轮重建）", exc_info=True)

    async def purge_thread(self, thread_id: str) -> int:
        try:
            return await self._redis.delete_prefix(f"snap:{thread_id}/")
        except Exception:
            logger.warning("memory: 快照缓存线程清理失败（TTL 兜底过期）", exc_info=True)
            return 0


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
        locks: CaptureLocks | None = None,
        snapshots: SnapshotCache | None = None,
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
        summary_node: str | None = None,
        prior_summary_nodes: Sequence[str] = (),
        covered_message_count: int | None = None,
    ) -> dict[str, Any] | None:
        """一轮对话结束后的记忆形成（后台调用；绝不抛异常）。

        ``messages`` 是该线程的全部持久化消息（来自 checkpointer 快照）：
        转写渲染、轮次计数与触发阈值都在这里统一处理。``force=True``
        旁路 ``MEMORY_CAPTURE_ENABLED`` 门控（夜间批脚本 T3.2）。

        树模式（ADR-013）：``summary_node`` 是本轮叶子节点——摘要键控到
        它；``prior_summary_nodes`` 是先近后远的祖先链（继承最近的祖先
        摘要继续滚动）；``covered_message_count`` 是路径消息数。
        """
        pipeline = self._pipeline()
        if pipeline is None:
            return None
        human_count = sum(1 for m in messages if getattr(m, "type", "") == "human")
        # 每 N 轮门控（M0）：``capture_every_turns``>1 时仅在本轮人类消息数
        # 是 N 的倍数时才跑管线（消费此前声明但未接线的配置项）。``force``
        # 旁路（夜间批语义与 ``capture_enabled`` 门控一致）；摘要触发有独立
        # 阈值（summary_trigger_messages），配置 N 后会随门控顺延到倍数轮——
        # 默认 N=1 时行为与历史完全一致。
        every = self._settings.capture_every_turns
        if not force and every > 1 and human_count % every != 0:
            logger.debug("memory: 跳过本轮形成管线（human_count=%d, every=%d）", human_count, every)
            return None
        transcript = render_transcript(
            messages[-self._settings.extraction_max_messages :],
            self._settings.extraction_max_input_chars,
        )
        if not transcript.strip():
            return None
        async with self._locks.for_user(user_id) as acquired:
            if not acquired:
                # Redis 锁等待超时：跳过本轮（下一轮读全量消息补上）。
                return None
            try:
                return await pipeline.capture_turn(
                    user_id=user_id,
                    agent_id=agent_id,
                    thread_id=thread_id,
                    transcript=transcript,
                    human_count=human_count,
                    force=force,
                    summary_node=summary_node,
                    prior_summary_nodes=prior_summary_nodes,
                    covered_message_count=covered_message_count,
                )
            except Exception:
                # 管线内部已逐步失败安全；这里兜底防任何漏网异常干扰调用方。
                logger.exception("memory: capture_turn 意外失败")
                return None

    async def compose_context(
        self,
        *,
        user_id: str,
        agent_id: str,
        thread_id: str,
        query: str,
        summary_nodes: Sequence[str] = (),
    ) -> str | None:
        """线程级**冻结**注入段（M6d + T2.1）：画像 + 记忆块 + 会话摘要。

        快照在线程首轮定型并缓存（LRU 上限 ``context_snapshot_max``）：
        同线程同分支返回逐字节相同的内容，即使画像/记忆块/摘要中途
        已更新——供应商前缀缓存按最长公共前缀命中，冻结头部是输入缓存
        的地基；数据变更在**下个会话**进入注入（接受的行为权衡，见
        ``compose_frozen_context``）。``purge_thread`` 会级联清理。

        树模式（ADR-013）：``summary_nodes`` 是先近后远的祖先节点 id 候
        选——取**最近一个有摘要的祖先**（分支共享祖先摘要、独立续写）；
        缓存键带命中的摘要节点，同分支持续命中，切分支自动重建。候选
        为空（非树模式）走线程口径 ``session_summaries``。逐轮变化的查
        询相关召回走 ``recall_context``（贴尾注入）。任何一步失败都降级
        为"缺那一节"；整体无内容时返回 None。
        """
        candidates = [node_id for node_id in summary_nodes if node_id]
        try:
            # 先近后远找最近的祖先摘要；候选为空或全部未命中时回退线程
            # 口径（非树模式的原路径 / 新分岔的树干内容回退）。
            summary_text: str | None = None
            hit_node = ""
            for node_id in candidates:
                record = await self._store.get_node_summary(user_id, node_id)
                if record is not None:
                    summary_text, hit_node = record.summary, node_id
                    break
            key = (user_id, agent_id, thread_id, hit_node)
            cached = await self._snapshots.get(key)
            if cached is not None:
                return cached
            if summary_text is None:
                thread_record = await self._store.get_summary(user_id, thread_id)
                summary_text = thread_record.summary if thread_record else None
            global_blocks = await self._store.list_blocks(user_id, "*")
            agent_blocks = await self._store.list_blocks(user_id, agent_id)
            profile = (
                await self._profile.get_profile(user_id) if self._settings.profile_enabled else None
            )
            frozen = compose_frozen_context(
                profile=profile,
                blocks=[*global_blocks, *agent_blocks],
                summary=summary_text,
                max_chars=self._settings.context_max_chars,
            )
        except Exception:
            logger.warning("memory: 冻结注入段组装失败（本轮不注入）", exc_info=True)
            return None
        if frozen is not None:
            await self._snapshots.put(key, frozen, max_entries=self._settings.context_snapshot_max)
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
        await self._snapshots.purge_thread(thread_id)

    async def purge_nodes(self, node_ids: Sequence[str]) -> int:
        """会话消息树删除的摘要级联（ADR-013）：按节点 id 清 node_summaries。

        返回清理的摘要条数；失败安全（树已删，遗留摘要只是孤儿行）。
        """
        if not node_ids:
            return 0
        try:
            return await self._store.purge_nodes(list(node_ids))
        except Exception:
            logger.warning("memory: 节点摘要清理失败（遗留孤儿行）", exc_info=True)
            return 0
