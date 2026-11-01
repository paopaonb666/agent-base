"""进程内事件触发器（ADR-007，循环工程 Loop 3）：定时任务 + webhook 手动触发。

零新依赖：调度只支持 ``{"every_seconds": N}`` 与 ``{"daily_at": "HH:MM"}``
两种表达（覆盖健康检查、夜间批、日报类场景）；复杂 cron 表达式仍走外部
cron 调 webhook 入口。任务类型是进程内注册表（``register_job_type``，
组合根装配，不做动态 import 插件）。

失败退避：连续失败按 ``base * 2^(n-1)`` 指数退避（封顶 1h），成功重置；
连续失败计数进 ``status()``（/health 消费）。单进程口径（与 /metrics
一致）：多 worker 部署会重复触发，用 ``SCHEDULER_ENABLED=false`` 关闭
并保留外部 cron + ``POST /v1/triggers/{job}``。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import avoided at runtime
    from agent_base.core.bootstrap import AgentRuntime
    from agent_base.core.config import Settings

logger = logging.getLogger(__name__)

# 失败退避参数：基数 30s，指数增长，封顶 1h。
_BACKOFF_BASE_SECONDS = 30.0
_BACKOFF_MAX_SECONDS = 3600.0

_DAILY_AT_RE = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")

# 任务运行器：``async def runner(rt, params) -> Any``。返回值进状态摘要。
JobRunner = Callable[["AgentRuntime", dict[str, Any]], Awaitable[Any]]


class SchedulerError(ValueError):
    """调度器装配失败（任务声明非法、类型未注册、名字重复——启用即承诺）。"""


@dataclass(frozen=True)
class SchedulerJob:
    """一个声明式任务条目（SCHEDULER_JOBS_JSON 的元素）。"""

    name: str
    type: str
    schedule: dict[str, Any]
    params: dict[str, Any] = field(default_factory=dict)


def parse_jobs(jobs_json: str) -> list[SchedulerJob]:
    """解析 ``SCHEDULER_JOBS_JSON``（JSON 数组）；空串返回空表。

    声明非法（非 JSON / 非数组 / 缺字段 / 名字重复 / 调度表达非法 /
    类型未注册）一律抛 ``SchedulerError``——启动快速失败，不静默跳过。
    """
    text = jobs_json.strip()
    if not text:
        return []
    try:
        raw = json.loads(text)
    except ValueError as exc:
        raise SchedulerError(f"SCHEDULER_JOBS_JSON 不是合法 JSON：{exc}") from exc
    if not isinstance(raw, list):
        raise SchedulerError("SCHEDULER_JOBS_JSON 必须是 JSON 数组")
    jobs: list[SchedulerJob] = []
    seen: set[str] = set()
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise SchedulerError(f"任务 #{i} 不是对象：{item!r}")
        name = str(item.get("name") or "").strip()
        job_type = str(item.get("type") or "").strip()
        schedule = item.get("schedule") or {}
        params = item.get("params") or {}
        if not name or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise SchedulerError(f"任务 #{i} 名字非法（^[a-z][a-z0-9_]*$）：{name!r}")
        if name in seen:
            raise SchedulerError(f"任务名字重复：{name!r}")
        if job_type not in JOB_RUNNERS:
            raise SchedulerError(
                f"任务 {name!r} 的类型未注册：{job_type!r}；已注册：{sorted(JOB_RUNNERS)}"
            )
        if not isinstance(schedule, dict) or not _validate_schedule(schedule):
            raise SchedulerError(
                f"任务 {name!r} 的调度非法：{schedule!r}"
                '（应为 {"every_seconds": N} 或 {"daily_at": "HH:MM"}）'
            )
        if not isinstance(params, dict):
            raise SchedulerError(f"任务 {name!r} 的 params 必须是对象：{params!r}")
        seen.add(name)
        jobs.append(
            SchedulerJob(name=name, type=job_type, schedule=dict(schedule), params=dict(params))
        )
    return jobs


def _validate_schedule(schedule: dict[str, Any]) -> bool:
    every = schedule.get("every_seconds")
    daily = schedule.get("daily_at")
    if isinstance(every, (int, float)) and not isinstance(every, bool) and every > 0:
        return True
    return bool(isinstance(daily, str) and _DAILY_AT_RE.fullmatch(daily.strip()))


def next_delay(schedule: dict[str, Any], *, now: datetime | None = None) -> float:
    """距离下一次触发的秒数（interval 直读；daily 算到下一个 HH:MM）。"""
    current = now or datetime.now()
    every = schedule.get("every_seconds")
    if isinstance(every, (int, float)):
        return float(every)
    hour, minute = (int(part) for part in str(schedule["daily_at"]).strip().split(":"))
    candidate = current.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= current:
        candidate += timedelta(days=1)
    return max(0.0, (candidate - current).total_seconds())


# ─────────────────────────── 任务类型注册表 ───────────────────────────

JOB_RUNNERS: dict[str, JobRunner] = {}


def register_job_type(type_name: str, runner: JobRunner) -> None:
    """注册任务类型（组合根扩展点；名字冲突拒绝——注册即覆盖太容易埋雷）。"""
    if type_name in JOB_RUNNERS:
        raise SchedulerError(f"任务类型已注册：{type_name!r}")
    JOB_RUNNERS[type_name] = runner


async def _run_noop(rt: AgentRuntime, params: dict[str, Any]) -> str:
    """演示/冒烟任务：立即成功。"""
    return "ok"


async def run_memory_capture(
    rt: AgentRuntime, *, user_ids: list[str], since_ts: float, dry_run: bool = False
) -> dict[str, Any]:
    """批量补跑记忆形成（夜间批语义，force=True 旁路门控）。

    从 ``scripts/memory_nightly_capture.py`` 收编：线程来源 thread_index、
    消息来源 checkpointer 快照；单线程失败不中断批次（失败安全语义与
    每轮路径一致）。
    """
    memory = rt.memory
    if memory is None or not memory.capture_ready():
        logger.warning("scheduler: 记忆形成管线不可用，memory_nightly_capture 无事可做")
        return {"processed": 0, "skipped": 0, "failed": 0, "reason": "pipeline_unavailable"}
    # 多实例去重（ADR-012）：共享 Redis 在场时用全局锁保证"同一时刻只有
    # 一个实例跑夜间批"——不等待，抢不到即让位（另一实例正在跑）。
    lock_token: str | None = None
    if rt.redis is not None:
        lock_token = await rt.redis.acquire_lock(
            "lock:nightly_capture",
            ttl_seconds=rt.settings.redis.lock_ttl_seconds,
            acquire_timeout_seconds=0.0,
        )
        if lock_token is None:
            logger.info("scheduler: 夜间批全局锁被持有（其他实例在跑），本实例让位")
            return {"processed": 0, "skipped": 0, "failed": 0, "reason": "locked"}
    try:
        assert rt.thread_index is not None
        processed = skipped = failed = 0
        for user_id in user_ids:
            entries = await rt.thread_index.list_thread_index(user_id, limit=500)
            for entry in entries:
                if entry.updated_at < since_ts:
                    skipped += 1
                    continue
                if dry_run:
                    processed += 1
                    continue
                try:
                    graph = rt.graph(entry.module or "chat")
                    snapshot = await graph.aget_state(
                        {"configurable": {"thread_id": entry.thread_id}}
                    )
                    messages: list[Any] = (
                        (snapshot.values or {}).get("messages", []) if snapshot else []
                    )
                    if not messages:
                        skipped += 1
                        continue
                    detail = await memory.capture_turn(
                        user_id=user_id,
                        agent_id=entry.module or "chat",
                        thread_id=entry.thread_id,
                        messages=messages,
                        force=True,
                    )
                except Exception as exc:  # 单线程失败不中断批次
                    failed += 1
                    logger.warning(
                        "scheduler: 捕获失败 %s：%s: %s", entry.thread_id, type(exc).__name__, exc
                    )
                    continue
                if detail is None:
                    skipped += 1
                    continue
                processed += 1
    finally:
        if lock_token is not None and rt.redis is not None:
            try:
                await rt.redis.release_lock("lock:nightly_capture", lock_token)
            except Exception:
                logger.warning("scheduler: 夜间批全局锁释放失败（TTL 兜底过期）", exc_info=True)
    summary = {"processed": processed, "skipped": skipped, "failed": failed}
    logger.info("scheduler: memory_nightly_capture 完成：%s", summary)
    return summary


async def _run_memory_nightly_capture(rt: AgentRuntime, params: dict[str, Any]) -> dict[str, Any]:
    user_ids = [str(u) for u in params.get("user_ids", ["default"]) if str(u).strip()]
    since_hours = float(params.get("since_hours", 24))
    return await run_memory_capture(
        rt, user_ids=user_ids, since_ts=time.time() - since_hours * 3600.0
    )


register_job_type("noop", _run_noop)
register_job_type("memory_nightly_capture", _run_memory_nightly_capture)


# ─────────────────────────── 调度器 ───────────────────────────


@dataclass
class _JobState:
    job: SchedulerJob
    consecutive_failures: int = 0
    last_run_at: float | None = None
    last_outcome: str = "pending"  # pending | ok | error
    last_error: str = ""


class Scheduler:
    """按声明跑定时任务；每任务独立循环 + 指数退避 + 手动触发。"""

    def __init__(self, runtime: AgentRuntime, jobs: list[SchedulerJob]) -> None:
        self._rt = runtime
        self._states = {job.name: _JobState(job=job) for job in jobs}
        self._tasks: list[asyncio.Task[None]] = []
        self._running = False

    @property
    def jobs(self) -> list[SchedulerJob]:
        return [state.job for state in self._states.values()]

    def start(self) -> None:
        """启动各任务循环（幂等：重复调用不叠加任务）。"""
        if self._running:
            return
        self._running = True
        for state in self._states.values():
            self._tasks.append(asyncio.create_task(self._loop(state)))
        logger.info(
            "scheduler: 已启动 %d 个任务：%s",
            len(self._states),
            [s.job.name for s in self._states.values()],
        )

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    def trigger(self, job_name: str) -> bool:
        """立即执行一次（webhook 入口用）；返回是否命中任务。

        经 ``spawn_background`` 调度并立即返回——执行结果看日志与
        ``status()``。刻意是同步方法：只做注册表查找 + 任务投递。
        """
        state = self._states.get(job_name)
        if state is None:
            return False
        from agent_base.entrypoints.server.background import spawn_background

        spawn_background(self._execute(state), name=f"scheduler:{job_name}")
        return True

    def status(self) -> dict[str, Any]:
        """任务状态摘要（/health 消费：连续失败>0 即非 ok）。"""
        return {
            "running": self._running,
            "jobs": {
                state.job.name: {
                    "type": state.job.type,
                    "last_outcome": state.last_outcome,
                    "last_run_at": state.last_run_at,
                    "consecutive_failures": state.consecutive_failures,
                    "last_error": state.last_error[:200],
                }
                for state in self._states.values()
            },
        }

    def health(self) -> str:
        """/health 组件口径：ok（全部正常）/ degraded（有连续失败）。"""
        if not self._running and self._states:
            return "degraded"
        if any(state.consecutive_failures > 0 for state in self._states.values()):
            return "degraded"
        return "ok"

    async def _loop(self, state: _JobState) -> None:
        while True:
            if not self._running:
                return
            delay = next_delay(state.job.schedule)
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                return
            await self._execute(state)
            if state.consecutive_failures:
                backoff = min(
                    _BACKOFF_BASE_SECONDS * (2 ** (state.consecutive_failures - 1)),
                    _BACKOFF_MAX_SECONDS,
                )
                logger.warning(
                    "scheduler: 任务 %s 连续失败 %d 次，退避 %.0fs",
                    state.job.name,
                    state.consecutive_failures,
                    backoff,
                )
                try:
                    await asyncio.sleep(backoff)
                except asyncio.CancelledError:
                    return

    async def _execute(self, state: _JobState) -> None:
        job = state.job
        runner = JOB_RUNNERS[job.type]
        state.last_run_at = time.time()
        try:
            result = await runner(self._rt, job.params)
        except Exception as exc:
            state.consecutive_failures += 1
            state.last_outcome = "error"
            state.last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("scheduler: 任务 %s 失败：%s", job.name, state.last_error)
            return
        state.consecutive_failures = 0
        state.last_outcome = "ok"
        state.last_error = ""
        logger.info("scheduler: 任务 %s 完成：%s", job.name, result)


def build_scheduler(settings: Settings, runtime: AgentRuntime) -> Scheduler | None:
    """按 settings 装配调度器：未启用返回 None；声明非法启动失败。"""
    if not settings.scheduler.enabled:
        return None
    jobs = parse_jobs(settings.scheduler.jobs_json)
    if not jobs:
        logger.info("scheduler: SCHEDULER_ENABLED=true 但没有任务声明，空转关闭")
        return None
    return Scheduler(runtime, jobs)
