"""extensions/toollog 的测试（M5）：memory/sqlite 记录、装配开关、mysql 桥接。"""

from __future__ import annotations

import asyncio
import sys
import types

import pytest

from agent_base.core.config import Settings
from agent_base.extensions.toollog import (
    ARGS_MAX_CHARS,
    ERROR_MAX_CHARS,
    RESULT_MAX_CHARS,
    MemoryToolCallRecorder,
    MySqlToolCallRecorder,
    SqliteToolCallRecorder,
    ToolCallRecord,
    build_tool_call_recorder,
)


def _record(thread_id: str = "chat:t1", **overrides: object) -> ToolCallRecord:
    fields: dict[str, object] = {
        "call_id": "c1",
        "tool": "calculator",
        "status": "ok",
        "args_json": '{"expression": "1+1"}',
        "result_text": "2",
        "error_text": "",
        "duration_ms": 5,
        "thread_id": thread_id,
        "module": "chat",
        "request_id": "req-1",
    }
    fields.update(overrides)
    return ToolCallRecord(**fields)  # type: ignore[arg-type]


# ── memory 后端 ──────────────────────────────────────────────────────


async def test_memory_records_filters_and_maps() -> None:
    recorder = MemoryToolCallRecorder()
    recorder.record(_record("chat:t1"))
    recorder.record(_record("chat:t2"))
    rows = await recorder.list_for_thread("chat:t1")
    assert len(rows) == 1
    assert rows[0]["call_id"] == "c1"
    assert rows[0]["tool"] == "calculator"
    assert rows[0]["args"] == {"expression": "1+1"}
    assert rows[0]["status"] == "ok"
    assert rows[0]["duration_ms"] == 5
    assert await recorder.list_for_thread("chat:none") == []


async def test_memory_ring_buffer_caps() -> None:
    recorder = MemoryToolCallRecorder()
    for i in range(1500):
        recorder.record(_record(thread_id=f"chat:t{i}"))
    # 环形缓冲淘汰最旧的：t0 已被挤出。
    assert await recorder.list_for_thread("chat:t0") == []


# ── sqlite 后端 ──────────────────────────────────────────────────────


async def test_sqlite_roundtrip_and_order(tmp_path: object) -> None:
    recorder = SqliteToolCallRecorder(str(tmp_path / "state.db"))  # type: ignore[arg-type]
    recorder.record(_record("chat:t1", call_id="a", result_text="first"))
    recorder.record(_record("chat:t1", call_id="b", result_text="second"))
    recorder.record(_record("chat:t2", call_id="c"))
    rows: list[dict[str, object]] = []
    for _ in range(100):
        rows = await recorder.list_for_thread("chat:t1")
        rows_t2 = await recorder.list_for_thread("chat:t2")
        if len(rows) == 2 and len(rows_t2) == 1:
            break
        await asyncio.sleep(0.05)
    # 升序返回（最新在末尾）；跨线程隔离。
    assert [r["call_id"] for r in rows] == ["a", "b"]
    assert rows[0]["result"] == "first"
    assert [r["call_id"] for r in await recorder.list_for_thread("chat:t2")] == ["c"]
    await recorder.aclose()


async def test_sqlite_truncation_is_applied_at_capture(tmp_path: object) -> None:
    # 截断发生在 _TimeoutTool._track（捕获侧），工具日志按原样存储；
    # 这里验证常量契约仍然成立（防止有人把上限改丢）。
    assert RESULT_MAX_CHARS == 4000
    assert ARGS_MAX_CHARS == 2000
    assert ERROR_MAX_CHARS == 500


# ── mysql 后端（桥接逻辑，aiomysql 用替身） ─────────────────────────


class _FakeCursor:
    def __init__(self, sink: list[tuple[str, object]]) -> None:
        self._sink = sink

    async def __aenter__(self) -> _FakeCursor:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def execute(self, sql: str, params: object = None) -> None:
        self._sink.append((sql, params))

    async def fetchall(self) -> list[object]:
        return []


class _FakeConnection:
    def __init__(self, sink: list[tuple[str, object]]) -> None:
        self._sink = sink
        self.closed = False

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self._sink)

    def close(self) -> None:
        self.closed = True


async def test_mysql_record_bridges_to_loop_and_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executed: list[tuple[str, object]] = []

    async def fake_connect(**kwargs: object) -> _FakeConnection:
        return _FakeConnection(executed)

    monkeypatch.setitem(sys.modules, "aiomysql", types.SimpleNamespace(connect=fake_connect))
    settings = Settings(_env_file=None, checkpointer_backend="mysql")
    recorder = MySqlToolCallRecorder(settings)
    recorder.record(_record("chat:t1"))
    # record 经 loop.call_soon_threadsafe 异步写入——让出控制权等待完成。
    for _ in range(50):
        if executed:
            break
        await asyncio.sleep(0.02)
    assert executed, "mysql 写入没有发生"
    assert "INSERT INTO tool_call_records" in executed[-1][0]
    rows = await recorder.list_for_thread("chat:t1")
    assert rows == []  # 替身查询返回空行，仅验证查询路径可执行


async def test_mysql_record_survives_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_connect(**kwargs: object) -> _FakeConnection:
        raise RuntimeError("db down")

    monkeypatch.setitem(sys.modules, "aiomysql", types.SimpleNamespace(connect=fake_connect))
    settings = Settings(_env_file=None, checkpointer_backend="mysql")
    recorder = MySqlToolCallRecorder(settings)
    recorder.record(_record("chat:t1"))
    await asyncio.sleep(0.1)  # 写失败只记日志，不向调用方传播
    # 不抛异常即通过。


# ── 装配开关 ─────────────────────────────────────────────────────────


def test_build_recorder_follows_backend() -> None:
    assert isinstance(build_tool_call_recorder(Settings(_env_file=None)), MemoryToolCallRecorder)


def test_build_recorder_sqlite_backend(tmp_path: object) -> None:
    settings = Settings(
        _env_file=None,
        checkpointer_backend="sqlite",
        checkpointer_sqlite_path=str(tmp_path / "s.db"),  # type: ignore[arg-type]
    )
    assert isinstance(build_tool_call_recorder(settings), SqliteToolCallRecorder)


def test_build_recorder_disabled_returns_none() -> None:
    settings = Settings(_env_file=None, tool_call_log_enabled=False)
    assert build_tool_call_recorder(settings) is None
