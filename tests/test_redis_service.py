"""Redis 共享状态服务（ADR-012）：配置节、原语面与装配语义。

核心集（无 Redis 运行）覆盖配置校验、FakeRedisService 上的原语语义
与"启用即承诺"的 fail-fast；真连语义（roundtrip / 互斥锁 / Pub/Sub）
在 integration 标记下执行——CI 的 test job 挂 redis:7.4 服务容器，
本地需在 127.0.0.1:6379 起一个 Redis（或设 REDIS_URL）。
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from datetime import datetime

import pytest
from pydantic import ValidationError

from agent_base.core.config import Settings
from agent_base.extensions.redissvc import RedisService, RedisServiceError
from fakes import FakeRedisService


def _startup_settings(tmp_path: object, **redis_flat: object) -> Settings:
    """启动用例的最小 Settings：不读 .env、不要求真实 key、状态落 tmp_path
    （与 test_memory_plugin 的启动用例同款形态）；redis 默认关闭，覆盖项
    以平铺 REDIS_* 字段名传入。"""
    from pathlib import Path

    kwargs: dict[str, object] = {
        "_env_file": None,
        "llm": {"api_key": "sk-test"},
        "checkpointer": {"sqlite_path": str(Path(str(tmp_path)) / "state.db")},
        "redis_enabled": False,
    }
    kwargs.update(redis_flat)
    return Settings(**kwargs)  # type: ignore[arg-type]


# -- 配置节（REDIS_*） ------------------------------------------------------


def test_redis_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """默认开启、指向本机、键前缀隔离——与 .env.example 声明一致。"""
    monkeypatch.delenv("REDIS_ENABLED", raising=False)
    redis = Settings(_env_file=None).redis
    assert redis.enabled is True
    assert redis.url == "redis://127.0.0.1:6379/0"
    assert redis.key_prefix == "agent_base:"
    assert redis.socket_timeout_seconds == 3.0


def test_redis_flat_kwargs_lift_and_legacy_read() -> None:
    """平铺关键字经 sections 映射归位；历史平铺读取经 __getattr__ 代理。"""
    settings = Settings(
        _env_file=None,
        redis_enabled=False,
        redis_url="redis://redis:6379/1",
        redis_key_prefix="app2:",
    )
    assert settings.redis.enabled is False
    assert settings.redis.url == "redis://redis:6379/1"
    # 兼容垫片（读取侧）：既有调用方的平铺访问不断。
    assert settings.redis_enabled is False
    assert settings.redis_key_prefix == "app2:"


def test_redis_url_scheme_rejected() -> None:
    with pytest.raises(ValidationError, match="REDIS_URL"):
        Settings(_env_file=None, redis_url="http://127.0.0.1:6379/0")


def test_redis_tls_and_unix_schemes_accepted() -> None:
    settings = Settings(
        _env_file=None,
        redis_url="rediss://example.com:6379/0",
        redis_key_prefix="p:",
    )
    assert settings.redis.url.startswith("rediss://")
    assert Settings(_env_file=None, redis_url="unix:///tmp/redis.sock").redis.url.startswith(
        "unix://"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("redis_socket_timeout_seconds", 0),
        ("redis_lock_acquire_timeout_seconds", -1),
        ("redis_pool_max_connections", 0),
        ("redis_lock_ttl_seconds", 0),
        ("redis_snapshot_ttl_seconds", 0),
    ],
)
def test_redis_positive_validators(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})


def test_redis_empty_key_prefix_rejected() -> None:
    with pytest.raises(ValidationError, match="REDIS_KEY_PREFIX"):
        Settings(_env_file=None, redis_key_prefix="  ")


# -- FakeRedisService 原语语义（核心集） -------------------------------------


async def test_fake_set_nx_exclusivity_and_expiry() -> None:
    fake = FakeRedisService()
    assert await fake.set_nx("nonce:u1", "1", ttl_seconds=60) is True
    assert await fake.set_nx("nonce:u1", "1", ttl_seconds=60) is False
    # TTL=0 → 立即过期，指纹视作新请求（窗口语义的边界）。
    assert await fake.set_nx("nonce:u2", "1", ttl_seconds=0) is True
    assert await fake.set_nx("nonce:u2", "1", ttl_seconds=0) is True


async def test_fake_prefix_isolation() -> None:
    """键前缀是多应用共享同一实例时的命名空间边界。"""
    app_a = FakeRedisService(key_prefix="a:")
    app_b = FakeRedisService(key_prefix="b:")
    await app_a.set_ex("k", "v", ttl_seconds=60)
    assert await app_b.get("k") is None
    assert await app_a.get("k") == "v"


async def test_fake_delete_prefix_purges_only_matching() -> None:
    fake = FakeRedisService()
    await fake.set_ex("snap:t1/u1/chat", "x", ttl_seconds=60)
    await fake.set_ex("snap:t1/u1/planner", "y", ttl_seconds=60)
    await fake.set_ex("snap:t2/u1/chat", "z", ttl_seconds=60)
    assert await fake.delete_prefix("snap:t1/") == 2
    assert await fake.get("snap:t2/u1/chat") == "z"
    assert await fake.delete_prefix("snap:t1/") == 0


async def test_fake_incr_by_float_accumulates() -> None:
    fake = FakeRedisService()
    assert await fake.incr_by_float("cost:day", 1.25) == pytest.approx(1.25)
    assert await fake.incr_by_float("cost:day", 0.5) == pytest.approx(1.75)


async def test_fake_pubsub_roundtrip() -> None:
    fake = FakeRedisService()
    received: list[str] = []

    async def consume() -> None:
        async for message in fake.pubsub_messages("sse:t1"):
            received.append(message)
            if len(received) >= 2:
                return

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.01)  # 让订阅注册进频道
    assert await fake.publish("sse:t1", "m1") == 1
    assert await fake.publish("sse:t1", "m2") == 1
    await asyncio.wait_for(task, timeout=2.0)
    assert received == ["m1", "m2"]


async def test_fake_lock_token_semantics() -> None:
    """互斥 + 安全释放：错 token（锁已易主）绝不删除他人的锁。"""
    fake = FakeRedisService()
    token1 = await fake.acquire_lock("lock:capture:u1", ttl_seconds=60, acquire_timeout_seconds=0.1)
    assert token1
    assert (
        await fake.acquire_lock("lock:capture:u1", ttl_seconds=60, acquire_timeout_seconds=0.05)
        is None
    )
    assert await fake.release_lock("lock:capture:u1", "wrong-token") is False
    assert (
        await fake.acquire_lock("lock:capture:u1", ttl_seconds=60, acquire_timeout_seconds=0.05)
        is None
    )
    assert await fake.release_lock("lock:capture:u1", token1) is True
    token2 = await fake.acquire_lock("lock:capture:u1", ttl_seconds=60, acquire_timeout_seconds=0.1)
    assert token2 and token2 != token1


# -- 装配语义（启用即承诺） --------------------------------------------------


async def test_runtime_without_redis_is_none(tmp_path: object) -> None:
    """REDIS_ENABLED=false：不装配、不导入，全部能力回退进程内实现。"""
    from agent_base.core.bootstrap import create_runtime

    settings = _startup_settings(tmp_path, redis_enabled=False)
    rt = await create_runtime(settings)
    try:
        assert rt.redis is None
    finally:
        await rt.close()


async def test_create_fails_fast_when_unreachable() -> None:
    """启用但不可达 → RedisServiceError（启动 fail-fast，不静默降级）。"""
    pytest.importorskip("redis")
    settings = Settings(
        _env_file=None,
        redis_url="redis://127.0.0.1:1/0",
        redis_socket_timeout_seconds=0.5,
    )
    with pytest.raises(RedisServiceError, match="unreachable"):
        await RedisService.create(settings.redis)


async def test_runtime_fails_fast_when_redis_enabled_unreachable(tmp_path: object) -> None:
    pytest.importorskip("redis")
    from agent_base.core.bootstrap import create_runtime

    settings = _startup_settings(
        tmp_path,
        redis_enabled=True,
        redis_url="redis://127.0.0.1:1/0",
        redis_socket_timeout_seconds=0.5,
    )
    with pytest.raises(RedisServiceError):
        await create_runtime(settings)


# -- 服务端集成（FakeRedis：跨 worker 语义 + /health） ------------------------


def _sig_headers(user_id: str, secret: str, *, nonce: str) -> dict[str, str]:
    import hashlib
    import hmac as hmac_mod
    import time as time_mod

    ts = str(int(time_mod.time()))
    message = f"{user_id}\n{ts}\n{nonce}"
    sig = hmac_mod.new(secret.encode(), message.encode(), hashlib.sha256).hexdigest()
    return {
        "X-User-Id": user_id,
        "X-User-Timestamp": ts,
        "X-User-Nonce": nonce,
        "X-User-Sig": sig,
    }


def _runtime_with_redis(
    fake: FakeRedisService | None, *, llm: object | None = None, **memory: object
):  # type: ignore[no-untyped-def]
    """带共享 Redis 的最小运行时（记忆服务启用，鉴权头才参与防重放路径）。"""
    from langgraph.checkpoint.memory import InMemorySaver

    from agent_base.core.bootstrap import AgentRuntime
    from agent_base.modules.chat.module import ChatModule
    from agent_base_memory.service import MemoryService
    from agent_base_memory.store import MemoryMemoryStore
    from fakes import HashEmbedding, ScriptedChatModel

    memory_opts: dict[str, object] = {"enabled": True, "auth_secret": "s3cret-key"}
    memory_opts.update(memory)
    settings = Settings(
        _env_file=None,
        llm={"api_key": "sk-test"},
        memory=memory_opts,  # type: ignore[arg-type]
    )
    rt = AgentRuntime(
        settings=settings,
        llm=llm if llm is not None else ScriptedChatModel([]),  # type: ignore[arg-type]
        modules={"chat": ChatModule()},
        tools=[],
        checkpointer=InMemorySaver(),
        redis=fake,
    )
    rt.memory = MemoryService(
        store=MemoryMemoryStore(), embedder=HashEmbedding(), settings=settings
    )
    return rt


def _app_with_redis(fake: FakeRedisService | None, *, llm: object | None = None):  # type: ignore[no-untyped-def]
    """带共享 Redis 的最小应用（TestClient 口径；lifespan 会装上 runtime）。"""
    from fastapi.testclient import TestClient

    from agent_base.entrypoints.server import create_app

    return TestClient(create_app(runtime=_runtime_with_redis(fake, llm=llm)))


async def test_nonce_replay_shared_across_workers_via_redis() -> None:
    """两个"worker"（各自 app/各自 LRU）经共享 Redis 拦下同一 nonce 重放。"""
    shared = FakeRedisService()
    worker_a = _app_with_redis(shared)
    worker_b = _app_with_redis(shared)
    headers = _sig_headers("alice", "s3cret-key", nonce="fixed-nonce-redis-1")
    with worker_a as ca, worker_b as cb:
        assert ca.get("/v1/memory", headers=headers).status_code == 200
        # 进程内 LRU 互相看不见；没有共享层时这一发会漏过。
        assert cb.get("/v1/memory", headers=headers).status_code == 401
        # 指纹确实落在共享 Redis（键含 user_id，TTL 随窗口）。
        assert await shared.get("nonce:alice:fixed-nonce-redis-1") == "1"


def test_health_reports_redis_component() -> None:
    """启用 → components["redis"] 随 ping 结果 ok/error；未启用 → 缺席。"""
    ok_fake = FakeRedisService()
    with _app_with_redis(ok_fake) as client:
        body = client.get("/health").json()
        assert body["components"]["redis"] == "ok"
        assert body["status"] == "ok"

    broken_fake = FakeRedisService()
    broken_fake.fail_ping = True
    with _app_with_redis(broken_fake) as client:
        body = client.get("/health").json()
        assert body["components"]["redis"] == "error"
        assert body["status"] == "degraded"


# -- 成本计数器（跨 worker 预算门） ------------------------------------------


def _usage(cost_cny: float, ts: float) -> object:
    from agent_base.extensions.costmeter import UsageRecord

    return UsageRecord(
        ts=ts,
        request_id="r",
        model="m",
        profile="main",
        prompt_tokens=0,
        completion_tokens=0,
        cache_read_tokens=0,
        cost_cny=cost_cny,
    )


async def test_cost_counters_shared_budget_gate() -> None:
    """共享口径与进程内口径的分野：跨 worker 总和触发熔断，单进程视角看不到。"""
    from agent_base.extensions.costmeter import (
        CostBudgetExceeded,
        CostGovernor,
        CostMeter,
        RedisCostCounters,
    )

    fake = FakeRedisService()
    counters = RedisCostCounters(fake)  # type: ignore[arg-type]
    now = time.time()
    today = datetime.fromtimestamp(now).date().isoformat()

    # "另一 worker"已累计 2.5 元（经计数器写入共享状态）。
    await counters.add([_usage(2.5, now)])
    assert float(await fake.get(f"cost:day:{today}")) == pytest.approx(2.5)

    meter = CostMeter()
    meter.restore_day(today, 1.0)  # 本进程视角只见到自己的 1.0

    shared_gov = CostGovernor(
        daily_cny=2.0, monthly_cny=0.0, action="block", meter=meter, counters=counters
    )
    with pytest.raises(CostBudgetExceeded):
        await shared_gov.acheck()  # 共享 2.5 ≥ 2 → 熔断
    assert await shared_gov.astatus() == "blocked"

    # 进程内口径（无 counters）：1.0 < 2 不熔断——acheck 委托同步路径。
    local_gov = CostGovernor(daily_cny=2.0, monthly_cny=0.0, action="block", meter=meter)
    local_gov.check()
    await local_gov.acheck()
    assert local_gov.status() == "ok"

    # flush 批次跨午夜的归位：行按自身 ts 落到所属日期键。
    two_days_ago = now - 2 * 86400
    await counters.add([_usage(10.0, two_days_ago)])
    other_day = datetime.fromtimestamp(two_days_ago).date().isoformat()
    assert float(await fake.get(f"cost:day:{other_day}")) == pytest.approx(10.0)
    assert float(await fake.get(f"cost:day:{today}")) == pytest.approx(2.5)


async def test_cost_counters_totals_fallback_on_redis_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Redis 读失败 → totals() 返回 None，预算门回退进程内值（方向安全）。"""
    from agent_base.extensions.costmeter import CostGovernor, CostMeter, RedisCostCounters

    fake = FakeRedisService()

    async def boom(key: str) -> str:
        raise RuntimeError("redis down")

    monkeypatch.setattr(fake, "get", boom)
    counters = RedisCostCounters(fake)  # type: ignore[arg-type]
    assert await counters.totals() is None

    meter = CostMeter()
    today = datetime.now().date().isoformat()
    meter.restore_day(today, 1.0)
    gov = CostGovernor(
        daily_cny=2.0, monthly_cny=0.0, action="block", meter=meter, counters=counters
    )
    await gov.acheck()  # 回退进程内口径（1.0 < 2），不抛
    assert await gov.astatus() == "ok"


# -- 记忆域共享状态（memory 包 0.2） ------------------------------------------


async def test_memory_capture_locks_cross_worker() -> None:
    """两个"worker"共享同一 Redis：同一用户的捕获互斥；抢不到的拿到 False。"""
    from agent_base_memory.orchestrator import RedisCaptureLocks

    shared = FakeRedisService()
    worker_a = RedisCaptureLocks(shared, ttl_seconds=60, acquire_timeout_seconds=0.1)  # type: ignore[arg-type]
    worker_b = RedisCaptureLocks(shared, ttl_seconds=60, acquire_timeout_seconds=0.05)  # type: ignore[arg-type]

    async with worker_a.for_user("u1") as acquired_a:
        assert acquired_a is True
        # 锁被 A 持有：B 等待超时后拿到 False（编排层据此跳过本轮）。
        async with worker_b.for_user("u1") as acquired_b:
            assert acquired_b is False

    # A 已退出并释放：B 现在能抢到。
    async with worker_b.for_user("u1") as acquired_b2:
        assert acquired_b2 is True


async def test_memory_redis_snapshot_cache_semantics() -> None:
    """快照按线程前缀存取；purge_thread 只清该线程；TTL 限界。"""
    from agent_base_memory.orchestrator import RedisSnapshotCache

    fake = FakeRedisService()
    cache = RedisSnapshotCache(fake, ttl_seconds=60)  # type: ignore[arg-type]
    key_t1 = ("u1", "chat", "t1")
    await cache.put(key_t1, "frozen-段", max_entries=256)  # max_entries 协议兼容，忽略
    await cache.put(("u1", "chat", "t2"), "另一线程", max_entries=256)
    assert await cache.get(key_t1) == "frozen-段"
    assert await cache.purge_thread("t1") == 1
    assert await cache.get(key_t1) is None
    assert await cache.get(("u1", "chat", "t2")) == "另一线程"


async def test_memory_snapshot_cache_degrades_on_redis_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Redis 故障时快照缓存降级为未命中/放弃写回——绝不破坏注入路径。"""
    from agent_base_memory.orchestrator import RedisSnapshotCache

    fake = FakeRedisService()
    cache = RedisSnapshotCache(fake, ttl_seconds=60)  # type: ignore[arg-type]

    async def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("redis down")

    monkeypatch.setattr(fake, "get", boom)
    monkeypatch.setattr(fake, "set_ex", boom)
    assert await cache.get(("u", "chat", "t")) is None
    await cache.put(("u", "chat", "t"), "v", max_entries=10)  # 不抛


async def test_memory_plugin_wires_redis_shared_state(tmp_path: object) -> None:
    """插件装配：redis 在场 → 快照进共享缓存（跨 worker 逐字节一致）。"""
    from pathlib import Path

    from agent_base.core.memory_plugin import build_memory_plugin

    settings = Settings(
        _env_file=None,
        llm={"api_key": "sk-test"},
        checkpointer={"sqlite_path": str(Path(str(tmp_path)) / "memory.db")},
    )
    fake = FakeRedisService()
    plugin = await build_memory_plugin(settings=settings, llm=None, fast_llm=None, redis=fake)
    assert plugin is not None
    service = plugin.service
    await service.save_profile("u1", {"偏好": ["简洁回答"]})
    first = await service.compose_context(user_id="u1", agent_id="chat", thread_id="t9", query="q")
    assert first is not None
    # 快照落在共享 Redis（键 snap:{thread}/{user}/{agent}），第二"worker"命中同一份。
    assert await fake.get("snap:t9/u1/chat") == first
    second = await service.compose_context(user_id="u1", agent_id="chat", thread_id="t9", query="q")
    assert second == first
    await service.purge_thread("t9")
    assert await fake.get("snap:t9/u1/chat") is None
    await service.aclose()


async def test_nightly_capture_yields_when_lock_held() -> None:
    """夜间批全局锁被持有时，本实例让位（reason=locked），不碰线程索引。"""
    from types import SimpleNamespace

    from agent_base.core.bootstrap import AgentRuntime
    from agent_base.extensions.scheduler import run_memory_capture

    fake = FakeRedisService()
    token = await fake.acquire_lock(
        "lock:nightly_capture", ttl_seconds=60, acquire_timeout_seconds=0.1
    )
    assert token
    rt = AgentRuntime(
        settings=Settings(_env_file=None, llm={"api_key": "sk-test"}),
        llm=SimpleNamespace(),
        modules={},
        tools=[],
        redis=fake,
    )
    rt.memory = SimpleNamespace(capture_ready=lambda: True)
    summary = await run_memory_capture(rt, user_ids=["u1"], since_ts=0.0)
    assert summary["reason"] == "locked"
    # 让位路径不释放他人的锁。
    assert await fake.release_lock("lock:nightly_capture", token) is True


# -- SSE 事件镜像（ADR-012） --------------------------------------------------


async def test_sse_emit_dual_writes_local_and_mirror() -> None:
    """_emit：事件进本地队列（主链路）并镜像发布；订阅侧可还原为契约事件。"""
    import asyncio as aio

    from agent_base.entrypoints.server.sse import _emit, sse_channel
    from agent_base.extensions.events import DeltaEvent, DoneEvent, decode_event

    fake = FakeRedisService()
    queue: aio.Queue = aio.Queue()
    received: list[str] = []

    async def consume() -> None:
        async for payload in fake.pubsub_messages(sse_channel("t1")):
            received.append(payload)
            if len(received) >= 2:
                return

    task = aio.create_task(consume())
    await aio.sleep(0.01)
    await _emit(queue, DeltaEvent(content="你好"), fake, sse_channel("t1"))
    await _emit(queue, DoneEvent(thread_id="t1"), fake, sse_channel("t1"))
    assert queue.qsize() == 2
    await aio.wait_for(task, timeout=2.0)
    assert decode_event(received[0]) == DeltaEvent(content="你好")
    assert decode_event(received[1]) == DoneEvent(thread_id="t1")


async def test_decode_event_drops_malformed_payload() -> None:
    from agent_base.extensions.events import decode_event

    assert decode_event('{"type":"no-such-type"}') is None
    assert decode_event("not json") is None


def test_observe_events_endpoint_requires_redis() -> None:
    """REDIS_ENABLED=false（redis 缺席）→ 观察端点 503（能力未开启）。"""
    with _app_with_redis(None) as client:
        resp = client.get(
            "/v1/agents/chat/threads/t-x/events",
            headers=_sig_headers("default", "s3cret-key", nonce="observe-503"),
        )
        assert resp.status_code == 503


async def test_sse_mirror_observer_stream_receives_published_events() -> None:
    """镜像 e2e（ASGI 层）：观察流收到频道上发布（= invoke 双写）的事件。

    TestClient 的传输会把 app 跑完才返回（starlette 1.6 起），永不终止
    的观察流没法走它——这里直接驱动端点协程的 body_iterator，与
    ``_produce`` 同款的 ``publish`` 调用并发。
    """
    import contextlib
    import json

    from agent_base.entrypoints.server import create_app
    from agent_base.entrypoints.server.routes.agents import observe_thread_events
    from agent_base.extensions.events import DeltaEvent, DoneEvent

    fake = FakeRedisService()
    rt = _runtime_with_redis(fake)
    app = create_app(runtime=rt)
    # 直调端点函数（绕过 DI 与 lifespan）：手动补上 lifespan 才会装的
    # runtime 引用（get_runtime 依赖）。
    app.state.runtime = rt

    request = type(
        "R",
        (),
        {"app": app},
    )()
    response = await observe_thread_events(
        module="chat",
        thread_id="t-mirror",
        request=request,
        user_id="default",  # type: ignore[arg-type]
    )
    assert response.status_code == 200

    async def publish_later() -> None:
        await asyncio.sleep(0.05)  # 等订阅注册
        await fake.publish("sse:t-mirror", DeltaEvent(content="hi").model_dump_json())
        await fake.publish("sse:t-mirror", DoneEvent(thread_id="t-mirror").model_dump_json())

    publisher = asyncio.create_task(publish_later())
    frames: list[str] = []
    async with contextlib.aclosing(response.body_iterator):  # type: ignore[attr-defined]
        async for frame in response.body_iterator:  # type: ignore[attr-defined]
            frames.append(frame if isinstance(frame, str) else frame.decode())
            if '"type":"done"' in frames[-1]:
                break
    await publisher
    # 帧是多行字符串（event: ...\ndata: {...}\n\n）：按行拆出 data 载荷。
    data_lines = [
        line for frame in frames for line in frame.splitlines() if line.startswith("data:")
    ]
    types = [json.loads(line[len("data:") :].strip())["type"] for line in data_lines]
    assert types[0] == "ping"  # 流打开即发的存活信号
    assert "delta" in types
    assert types[-1] == "done"


# -- 真连语义（integration；CI test job 的 redis:7.4 服务容器） --------------


def _redis_url() -> str:
    return os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")


async def _service_or_skip(prefix: str) -> RedisService:
    """按唯一前缀构建连接；Redis 不可达时跳过（与 MySQL 集成测试同款）。"""
    pytest.importorskip("redis")
    settings = Settings(
        _env_file=None, redis_url=_redis_url(), redis_key_prefix=prefix, redis_enabled=True
    )
    try:
        return await RedisService.create(settings.redis)
    except RedisServiceError:
        pytest.skip(f"redis not reachable at {_redis_url()}")


@pytest.mark.integration
async def test_integration_roundtrip_and_prefix_cleanup() -> None:
    prefix = f"test:{uuid.uuid4().hex[:8]}:"
    service = await _service_or_skip(prefix)
    try:
        await service.set_ex("k", "v", ttl_seconds=60)
        assert await service.get("k") == "v"
        assert await service.set_nx("k", "x", ttl_seconds=60) is False
        assert await service.incr_by_float("cost:day", 1.5) == pytest.approx(1.5)
        assert await service.delete_prefix("") == 2  # k 与 cost:day
        assert await service.get("k") is None
    finally:
        await service.delete_prefix("")
        await service.aclose()


@pytest.mark.integration
async def test_integration_lock_mutual_exclusion() -> None:
    """两个连接（两个"worker"）对同一把锁互斥；token 安全释放。"""
    prefix = f"test:{uuid.uuid4().hex[:8]}:"
    worker_a = await _service_or_skip(prefix)
    worker_b = await _service_or_skip(prefix)
    try:
        token = await worker_a.acquire_lock("job", ttl_seconds=60, acquire_timeout_seconds=0.2)
        assert token
        locked = await worker_b.acquire_lock("job", ttl_seconds=60, acquire_timeout_seconds=0.2)
        assert locked is None
        # 错 token（锁值不匹配）绝不删除——compare-and-delete 的安全语义
        # 是校验 token 值而非连接身份。
        assert await worker_b.release_lock("job", "wrong-token") is False
        assert await worker_a.release_lock("job", token) is True
        assert await worker_b.acquire_lock("job", ttl_seconds=60, acquire_timeout_seconds=0.2)
    finally:
        await worker_a.delete_prefix("")
        await worker_a.aclose()
        await worker_b.aclose()


@pytest.mark.integration
async def test_integration_pubsub_roundtrip() -> None:
    prefix = f"test:{uuid.uuid4().hex[:8]}:"
    service = await _service_or_skip(prefix)
    received: list[str] = []

    async def consume() -> None:
        async for message in service.pubsub_messages("chan"):
            received.append(message)
            return

    try:
        task = asyncio.create_task(consume())
        await asyncio.sleep(0.3)  # 订阅在服务端生效（真实传播延迟）
        assert await service.publish("chan", "hello") >= 1
        await asyncio.wait_for(task, timeout=3.0)
        assert received == ["hello"]
    finally:
        await service.delete_prefix("")
        await service.aclose()
