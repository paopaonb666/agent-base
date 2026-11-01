"""事件触发器（ADR-007，Loop 3）的测试：声明解析、调度计算、循环与退避、
HTTP 触发端点。"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from agent_base.core.bootstrap import AgentRuntime
from agent_base.core.config import Settings
from agent_base.entrypoints.server import create_app
from agent_base.extensions.scheduler import (
    JOB_RUNNERS,
    Scheduler,
    SchedulerError,
    SchedulerJob,
    build_scheduler,
    next_delay,
    parse_jobs,
)
from agent_base.modules.chat.module import ChatModule
from fakes import ScriptedChatModel


def _settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {"llm_api_key": "sk-test"}
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


def _runtime() -> AgentRuntime:
    return AgentRuntime(
        settings=_settings(),
        llm=ScriptedChatModel([]),
        modules={"chat": ChatModule()},
        tools=[],
        checkpointer=InMemorySaver(),
    )


# ─────────────────────────── 声明解析 ───────────────────────────


def test_parse_jobs_empty() -> None:
    assert parse_jobs("") == []
    assert parse_jobs("   ") == []


def test_parse_jobs_valid() -> None:
    jobs = parse_jobs(
        json.dumps(
            [
                {
                    "name": "nightly",
                    "type": "memory_nightly_capture",
                    "schedule": {"daily_at": "03:00"},
                    "params": {"user_ids": ["default"]},
                },
                {"name": "tick", "type": "noop", "schedule": {"every_seconds": 60}},
            ]
        )
    )
    assert [j.name for j in jobs] == ["nightly", "tick"]
    assert jobs[0].schedule == {"daily_at": "03:00"}
    assert jobs[1].params == {}


@pytest.mark.parametrize(
    "bad",
    [
        "not-json",
        '{"name": "x"}',  # 非数组
        json.dumps([{"type": "noop", "schedule": {"every_seconds": 1}}]),  # 缺名
        json.dumps([{"name": "Bad", "type": "noop"}]),  # 名字大写非法
        json.dumps([{"name": "a", "type": "nope"}]),  # 类型未注册
        json.dumps([{"name": "a", "type": "noop"}]),  # 缺调度
        json.dumps([{"name": "a", "type": "noop", "schedule": {"every_seconds": 0}}]),
        json.dumps([{"name": "a", "type": "noop", "schedule": {"daily_at": "25:00"}}]),
        json.dumps(
            [
                {"name": "a", "type": "noop", "schedule": {"every_seconds": 5}},
                {"name": "a", "type": "noop", "schedule": {"every_seconds": 5}},
            ]
        ),  # 重名
        json.dumps([{"name": "a", "type": "noop", "schedule": {"every_seconds": 5}, "params": 1}]),
    ],
)
def test_parse_jobs_rejects_bad_declarations(bad: str) -> None:
    with pytest.raises(SchedulerError):
        parse_jobs(bad)


# ─────────────────────────── 调度计算 ───────────────────────────


def test_next_delay_interval() -> None:
    assert next_delay({"every_seconds": 30}) == 30.0


def test_next_delay_daily_at() -> None:
    now = datetime(2026, 11, 1, 10, 0, 0)
    # 今天 23:30 还有 13.5h。
    assert next_delay({"daily_at": "23:30"}, now=now) == pytest.approx(13.5 * 3600)
    # 09:00 已过 → 明天 09:00（23h）。
    assert next_delay({"daily_at": "09:00"}, now=now) == pytest.approx(23 * 3600)
    # 恰好当前时刻 → 明天同刻。
    assert next_delay({"daily_at": "10:00"}, now=now) == pytest.approx(24 * 3600)


# ─────────────────────────── 调度器行为 ───────────────────────────


class CountingRunner:
    calls: int = 0

    async def __call__(self, rt: Any, params: dict[str, Any]) -> str:
        CountingRunner.calls += 1
        if params.get("fail", False) and CountingRunner.calls < 3:
            raise RuntimeError(f"boom-{CountingRunner.calls}")
        return "ok"


@pytest.fixture()
def counting_runner() -> CountingRunner:
    CountingRunner.calls = 0
    runner = CountingRunner()
    JOB_RUNNERS["counting"] = runner
    yield runner
    JOB_RUNNERS.pop("counting", None)


async def test_scheduler_loop_executes_and_retries(counting_runner: CountingRunner) -> None:
    """interval 任务按周期执行；失败计数成功后清零；health 随之恢复。"""
    rt = _runtime()
    sched = Scheduler(
        rt,
        [SchedulerJob(name="c1", type="counting", schedule={"every_seconds": 0.05}, params={})],
    )
    sched.start()
    try:
        await asyncio.sleep(0.3)
        assert CountingRunner.calls >= 3
        assert sched.health() == "ok"
        assert sched.status()["jobs"]["c1"]["last_outcome"] == "ok"
    finally:
        await sched.stop()


async def test_scheduler_backoff_on_failures(counting_runner: CountingRunner) -> None:
    """连续失败 → health degraded + 状态里带连续失败计数与错误。"""
    rt = _runtime()
    sched = Scheduler(
        rt,
        [
            SchedulerJob(
                name="f1", type="counting", schedule={"every_seconds": 0.05}, params={"fail": True}
            )
        ],
    )
    sched.start()
    try:
        await asyncio.sleep(0.15)
        state = sched.status()["jobs"]["f1"]
        assert state["last_outcome"] == "error"
        assert state["consecutive_failures"] >= 1
        assert "boom" in state["last_error"]
        assert sched.health() == "degraded"
    finally:
        await sched.stop()


async def test_scheduler_trigger_and_unknown(counting_runner: CountingRunner) -> None:
    rt = _runtime()
    sched = Scheduler(
        rt, [SchedulerJob(name="c1", type="counting", schedule={"every_seconds": 60})]
    )
    assert sched.trigger("nope") is False
    assert sched.trigger("c1") is True
    # spawn_background 异步执行：给事件循环一个调度机会。
    for _ in range(20):
        await asyncio.sleep(0.01)
        if CountingRunner.calls >= 1:
            break
    assert CountingRunner.calls >= 1


def test_build_scheduler_selection() -> None:
    rt = _runtime()
    assert build_scheduler(_settings(), rt) is None  # 未启用
    assert build_scheduler(_settings(scheduler_enabled=True), rt) is None  # 无声明 → 空转关闭
    sched = build_scheduler(
        _settings(
            scheduler_enabled=True,
            scheduler_jobs_json=json.dumps(
                [{"name": "t", "type": "noop", "schedule": {"every_seconds": 60}}]
            ),
        ),
        rt,
    )
    assert sched is not None and [j.name for j in sched.jobs] == ["t"]


# ─────────────────────────── HTTP 触发端点 ───────────────────────────


def _client(**settings_overrides: object) -> TestClient:
    runtime = AgentRuntime(
        settings=_settings(**settings_overrides),
        llm=ScriptedChatModel([]),
        modules={"chat": ChatModule()},
        tools=[],
        checkpointer=InMemorySaver(),
    )
    return TestClient(create_app(runtime=runtime))


def test_trigger_endpoint_disabled_503() -> None:
    with _client() as client:
        response = client.post("/v1/triggers/anything")
    assert response.status_code == 503


def test_trigger_endpoint_unknown_job_404() -> None:
    jobs = json.dumps([{"name": "tick", "type": "noop", "schedule": {"every_seconds": 60}}])
    with _client(scheduler_enabled=True, scheduler_jobs_json=jobs) as client:
        response = client.post("/v1/triggers/nope")
    assert response.status_code == 404


def test_trigger_endpoint_fires_and_health_reports() -> None:
    jobs = json.dumps([{"name": "tick", "type": "noop", "schedule": {"every_seconds": 3600}}])
    with _client(scheduler_enabled=True, scheduler_jobs_json=jobs) as client:
        assert client.post("/v1/triggers/tick").json() == {
            "triggered": "tick",
            "status": "scheduled",
        }
        health = client.get("/health").json()
    assert health["components"]["scheduler"] == "ok"


def test_health_no_scheduler_component_when_disabled() -> None:
    with _client() as client:
        health = client.get("/health").json()
    assert "scheduler" not in health["components"]
