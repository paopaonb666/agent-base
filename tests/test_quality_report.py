"""运行质量报告（ADR-008，Loop 4 分析半环）的测试：聚合、检测规则、
经协议取数的端到端（sqlite 临时库）。"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "quality_report.py"
_spec = importlib.util.spec_from_file_location("quality_report", _SCRIPT)
qr = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(qr)

from agent_base.extensions.costmeter import SqliteCostLedger, UsageRecord  # noqa: E402
from agent_base.extensions.toollog import (  # noqa: E402
    MemoryToolCallRecorder,
    SqliteToolCallRecorder,
    ToolCallRecord,
)

# ─────────────────────────── 纯函数：聚合 ───────────────────────────


def test_classify_error_patterns() -> None:
    assert qr.classify_error("Connection to x timed out") == "timed out"
    assert qr.classify_error("HTTP 429 rate limit exceeded") == "rate limit"
    assert qr.classify_error("permission denied: /etc/shadow") == "permission denied"
    assert qr.classify_error("奇奇怪怪的失败") == "other"
    assert qr.classify_error("") == "other"


def test_analyze_tool_calls_empty() -> None:
    section = qr.analyze_tool_calls([])
    assert section["total"] == 0
    assert section["success_rate"] is None


def test_analyze_tool_calls_metrics() -> None:
    rows = [
        {"tool": "search", "status": "ok", "duration_ms": 100, "error_text": ""},
        {"tool": "search", "status": "ok", "duration_ms": 300, "error_text": ""},
        {"tool": "search", "status": "error", "duration_ms": 0, "error_text": "rate limit"},
        {"tool": "repl", "status": "timeout", "duration_ms": 30_000, "error_text": "timed out"},
    ]
    section = qr.analyze_tool_calls(rows)
    assert section["total"] == 4
    assert section["success_rate"] == pytest.approx(0.5)
    assert section["avg_duration_ms"] == pytest.approx(7600.0)
    assert {e["pattern"] for e in section["top_errors"]} == {"rate limit", "timed out"}
    assert section["tools_with_errors"]["repl"] == 1
    slow = section["slow_tools"][0]
    assert slow["tool"] == "repl" and slow["p95_ms"] == pytest.approx(30_000.0)


def test_analyze_costs_grouping() -> None:
    day = 1_700_000_000.0
    rows = [
        {"ts": day, "model": "m1", "prompt_tokens": 100, "completion_tokens": 50, "cost_cny": 0.1},
        {
            "ts": day + 1,
            "model": "m1",
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "cost_cny": 0.2,
        },
        {
            "ts": day + 90_000,
            "model": "m2",
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "cost_cny": 0.5,
        },
    ]
    section = qr.analyze_costs(rows)
    assert section["calls"] == 3
    assert section["total_cny"] == pytest.approx(0.8)
    assert section["total_tokens"] == 315
    assert next(iter(section["by_model"])) == "m2"  # 成本降序
    assert len(section["by_day"]) >= 1


def test_analyze_memory_ops_steps() -> None:
    rows = [
        {"op": "extract", "status": "ok"},
        {"op": "extract", "status": "ok"},
        {"op": "extract", "status": "error"},
        {"op": "consolidate", "status": "ok"},
    ]
    section = qr.analyze_memory_ops(rows)
    assert section["total"] == 4 and section["errors"] == 1
    assert section["steps"]["extract"]["fail_rate"] == pytest.approx(1 / 3)


# ─────────────────────────── 纯函数：检测规则 ───────────────────────────


def _tools_section(rate: float, total: int, avg_ms: float = 0.0) -> dict[str, Any]:
    ok = round(rate * total)
    return {
        "total": total,
        "success_rate": rate,
        "avg_duration_ms": avg_ms,
        "top_errors": [],
        "slow_tools": [],
        "tools_with_errors": {"search": total - ok},
    }


def test_rules_no_alert_on_small_samples() -> None:
    issues = qr.apply_rules(_tools_section(0.0, 3), {"steps": {}})
    assert issues == []  # 样本不足不告警


def test_rules_success_rate_drop() -> None:
    issues = qr.apply_rules(_tools_section(0.5, 10), {"steps": {}})
    assert [i["rule"] for i in issues] == ["success_rate_drop"]
    assert issues[0]["severity"] == "critical"


def test_rules_latency_spike() -> None:
    issues = qr.apply_rules(_tools_section(1.0, 10, avg_ms=40_000), {"steps": {}})
    assert [i["rule"] for i in issues] == ["latency_spike"]
    assert issues[0]["severity"] == "high"


def test_rules_tool_error_count() -> None:
    issues = qr.apply_rules(_tools_section(0.5, 10), {"steps": {}})
    # 0.5×10 → search 错 5 次：恰好等于阈值不告警（> 5 才触发）。
    assert [i["rule"] for i in issues] == ["success_rate_drop"]
    section = _tools_section(0.3, 10)
    section["tools_with_errors"] = {"search": 8}
    issues = qr.apply_rules(section, {"steps": {}})
    assert "tool_error_rate" in [i["rule"] for i in issues]


def test_rules_pipeline_fail_spike() -> None:
    memory = {"steps": {"extract": {"ok": 3, "error": 7, "fail_rate": 0.7}}}
    issues = qr.apply_rules(_tools_section(1.0, 10), memory)
    assert [i["rule"] for i in issues] == ["pipeline_fail_spike"]
    assert issues[0]["severity"] == "high"


def test_build_report_summary() -> None:
    report = qr.build_report([], [], [], since_ts=time.time() - 3600)
    assert report["summary"] == {"issue_count": 0, "worst_severity": "none"}
    assert report["thresholds"]["success_rate"] == 0.8
    text = qr.render_text(report)
    assert "质量报告" in text and "未命中检测规则" in text


# ─────────────────────────── list_recent 新读取 ───────────────────────────


async def test_memory_recorder_list_recent_window() -> None:
    recorder = MemoryToolCallRecorder()
    old = time.time() - 7200
    recorder._records.append((ToolCallRecord(call_id="c1", tool="t", status="ok"), old))
    recorder.record(ToolCallRecord(call_id="c2", tool="t", status="error", error_text="boom"))
    rows = await recorder.list_recent(since_ts=time.time() - 60)
    assert [r["call_id"] for r in rows] == ["c2"]
    assert rows[0]["created_at"] >= time.time() - 60


async def test_sqlite_recorder_list_recent(tmp_path: Path) -> None:
    recorder = SqliteToolCallRecorder(str(tmp_path / "tcr.db"))
    try:
        recorder.record(ToolCallRecord(call_id="a", tool="search", status="ok", duration_ms=10))
        recorder.record(
            ToolCallRecord(call_id="b", tool="repl", status="error", error_text="kaboom")
        )
        await _drain(recorder)
        rows = await recorder.list_recent(since_ts=time.time() - 60)
        assert {r["tool"] for r in rows} == {"search", "repl"}
        error_row = next(r for r in rows if r["tool"] == "repl")
        assert error_row["error_text"] == "kaboom"
        assert error_row["created_at"] >= time.time() - 60
    finally:
        await recorder.aclose()


async def _drain(recorder: SqliteToolCallRecorder) -> None:
    """写线程是异步消化的：等队列清空（简单轮询）。"""
    for _ in range(50):
        if recorder._queue.empty():
            await _sleep(0.02)
            return
        await _sleep(0.02)


async def _sleep(seconds: float) -> None:
    import asyncio

    await asyncio.sleep(seconds)


async def test_sqlite_cost_ledger_list_recent(tmp_path: Path) -> None:
    ledger = await SqliteCostLedger.create(str(tmp_path / "cost.db"))
    try:
        now = time.time()
        await ledger.insert(
            [
                UsageRecord(
                    ts=now,
                    request_id="r1",
                    model="deepseek-chat",
                    profile="main",
                    prompt_tokens=100,
                    completion_tokens=20,
                    cache_read_tokens=0,
                    cost_cny=0.05,
                ),
                UsageRecord(
                    ts=now - 100_000,  # 窗口外
                    request_id="r0",
                    model="old",
                    profile="main",
                    prompt_tokens=1,
                    completion_tokens=1,
                    cache_read_tokens=0,
                    cost_cny=9.0,
                ),
            ]
        )
        rows = await ledger.list_recent(since_ts=now - 10)
        assert [r.request_id for r in rows] == ["r1"]
        assert rows[0].model == "deepseek-chat"
    finally:
        await ledger.aclose()


# ─────────────────────────── 端到端（sqlite 后端） ───────────────────────────


async def test_collect_rows_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """经协议取数 → 报告：三数据源中工具与成本两节可端到端验证。"""
    db = tmp_path / "state.db"
    monkeypatch.setenv("CHECKPOINTER_BACKEND", "sqlite")
    monkeypatch.setenv("CHECKPOINTER_SQLITE_PATH", str(db))
    recorder = SqliteToolCallRecorder(str(db))
    ledger = await SqliteCostLedger.create(str(db))
    try:
        recorder.record(ToolCallRecord(call_id="a", tool="search", status="ok", duration_ms=50))
        recorder.record(
            ToolCallRecord(call_id="b", tool="search", status="error", error_text="rate limit")
        )
        await _drain(recorder)
        now = time.time()
        await ledger.insert(
            [
                UsageRecord(
                    ts=now,
                    request_id="r",
                    model="deepseek-chat",
                    profile="main",
                    prompt_tokens=10,
                    completion_tokens=5,
                    cache_read_tokens=0,
                    cost_cny=0.01,
                )
            ]
        )
    finally:
        await recorder.aclose()
        await ledger.aclose()

    rows = await qr.collect_rows(time.time() - 60, limit=100)
    assert {r["tool"] for r in rows["tools"]} == {"search"}
    assert [r["model"] for r in rows["costs"]] == ["deepseek-chat"]
    report = qr.build_report(
        rows["tools"], rows["costs"], rows["memory"], since_ts=time.time() - 60
    )
    assert report["tools"]["total"] == 2
    assert report["costs"]["calls"] == 1
    # 记忆节缺席（未装记忆包的形态）不炸、如实标注。
    assert report["memory_pipeline"]["total"] == 0
