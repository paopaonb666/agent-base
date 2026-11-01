"""运行质量分析报告（ADR-008，循环工程 Loop 4 的"分析"半环）。

聚合三份既有审计数据（零 schema 变更、零裸 SQL——全部经组件协议取数）：

- ``ToolCallRecorder.list_recent``（toollog）→ 成功率、错误关键词归因
  Top-K、慢工具 Top-K；
- ``CostLedger.list_recent``（costmeter）→ 日趋势、模型分布、总成本；
- 记忆包 store 的 ``list_ops``（未装记忆包时该节置空并说明）→
  形成管线各步骤 ok/error 分布。

四条检测规则（阈值可覆盖；样本不足不告警——避免小样本误报）：
tool 成功率 < 0.8（critical）、平均工具延迟 > 30s（high）、单工具错误
> 5（medium）、记忆管线失败率 > 0.3（high）。命中输出 issues +
recommendations；分析产物进人审流程，不做任何自动改配置（ADR-008 被
否方案）。

用法::

    python scripts/quality_report.py [--since-hours 24] [--format json|text]

后端跟随 ``CHECKPOINTER_*`` 环境变量；只读，不写任何数据。分析逻辑
（``build_report`` 等）是纯函数——tests 直接导入测。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

# 检测规则默认阈值（外部方法论 LangSmith Engine 同款起点）。
DEFAULT_THRESHOLDS = {
    "success_rate": 0.8,  # 低于 → critical
    "avg_duration_ms": 30_000,  # 高于 → high
    "tool_errors": 5,  # 单工具错误数高于 → medium
    "pipeline_fail_rate": 0.3,  # 记忆管线失败率高于 → high
    "min_samples": 5,  # 样本不足不告警
}

_ERROR_PATTERNS = (
    "timeout",
    "timed out",
    "permission denied",
    "not found",
    "rate limit",
    "authentication",
    "connection refused",
    "out of memory",
    "syntax error",
)


def classify_error(text: str) -> str:
    """错误文本关键词归因（不命中归 other；照外部方法论模式）。"""
    lowered = (text or "").lower()
    for pattern in _ERROR_PATTERNS:
        if pattern in lowered:
            return pattern
    return "other"


def analyze_tool_calls(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """工具侧聚合：成功率、错误归因、慢工具（P95 最近邻近似）。"""
    total = len(rows)
    if not total:
        return {
            "total": 0,
            "success_rate": None,
            "avg_duration_ms": 0.0,
            "top_errors": [],
            "slow_tools": [],
            "tools_with_errors": {},
        }
    ok = sum(1 for r in rows if r.get("status") == "ok")
    durations = [float(r.get("duration_ms") or 0) for r in rows]
    by_tool: dict[str, list[float]] = defaultdict(list)
    error_kinds: Counter[str] = Counter()
    per_tool_errors: Counter[str] = Counter()
    for r in rows:
        duration = float(r.get("duration_ms") or 0)
        name = str(r.get("tool") or "?")
        by_tool[name].append(duration)
        if r.get("status") != "ok":
            per_tool_errors[name] += 1
            error_kinds[classify_error(str(r.get("error_text") or ""))] += 1

    def _p95(values: list[float]) -> float:
        ordered = sorted(values)
        idx = max(0, min(len(ordered) - 1, round(0.95 * (len(ordered) - 1))))
        return ordered[idx]

    slow_tools = sorted(
        (
            {"tool": name, "p95_ms": _p95(values), "calls": len(values)}
            for name, values in by_tool.items()
        ),
        key=lambda item: item["p95_ms"],
        reverse=True,
    )[:5]
    top_errors = [{"pattern": k, "count": v} for k, v in error_kinds.most_common(5)]
    return {
        "total": total,
        "success_rate": ok / total,
        "avg_duration_ms": sum(durations) / len(durations),
        "top_errors": top_errors,
        "slow_tools": slow_tools,
        "tools_with_errors": dict(per_tool_errors.most_common()),
    }


def analyze_costs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """成本侧聚合：总量、日趋势、模型分布。"""
    by_day: dict[str, float] = defaultdict(float)
    by_model: dict[str, float] = defaultdict(float)
    total = 0.0
    tokens = 0
    for r in rows:
        cost = float(r.get("cost_cny") or 0)
        total += cost
        tokens += int(r.get("prompt_tokens") or 0) + int(r.get("completion_tokens") or 0)
        ts = float(r.get("ts") or 0)
        if ts > 0:
            day = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
            by_day[day] += cost
        by_model[str(r.get("model") or "?")] += cost
    return {
        "calls": len(rows),
        "total_cny": round(total, 4),
        "total_tokens": tokens,
        "by_day": dict(sorted(by_day.items())),
        "by_model": dict(sorted(by_model.items(), key=lambda kv: kv[1], reverse=True)),
    }


def analyze_memory_ops(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """记忆管线聚合：各步骤 ok/error 分布与失败率。"""
    by_op: dict[str, dict[str, int]] = defaultdict(lambda: {"ok": 0, "error": 0})
    for r in rows:
        op = str(r.get("op") or "?")
        by_op[op]["ok" if r.get("status") == "ok" else "error"] += 1
    steps = {
        op: {**counts, "fail_rate": counts["error"] / max(1, counts["ok"] + counts["error"])}
        for op, counts in sorted(by_op.items())
    }
    total = len(rows)
    errors = sum(c["error"] for c in by_op.values())
    return {"total": total, "errors": errors, "steps": steps}


def apply_rules(
    tool_section: dict[str, Any],
    memory_section: dict[str, Any],
    *,
    thresholds: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """四条检测规则 → issues（附规则式 recommendations）。"""
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    issues: list[dict[str, Any]] = []
    rate = tool_section.get("success_rate")
    if rate is not None and tool_section["total"] >= th["min_samples"]:
        if rate < th["success_rate"]:
            issues.append(
                {
                    "rule": "success_rate_drop",
                    "severity": "critical",
                    "evidence": f"工具成功率 {rate:.1%}（{tool_section['total']} 次调用）",
                    "recommendation": "检查 top_errors 归因与慢工具；核对工具超时配置与外部依赖。",
                }
            )
        if tool_section["avg_duration_ms"] > th["avg_duration_ms"]:
            issues.append(
                {
                    "rule": "latency_spike",
                    "severity": "high",
                    "evidence": f"平均工具耗时 {tool_section['avg_duration_ms'] / 1000:.1f}s",
                    "recommendation": "查看 slow_tools；调低 TOOL_TIMEOUT_SECONDS 或拆分慢工具。",
                }
            )
        for tool, count in (tool_section.get("tools_with_errors") or {}).items():
            if count > th["tool_errors"]:
                issues.append(
                    {
                        "rule": "tool_error_rate",
                        "severity": "medium",
                        "evidence": f"工具 {tool} 在窗口内错误 {count} 次",
                        "recommendation": "单独排查该工具的外部依赖与参数 schema。",
                    }
                )
    for op, step in (memory_section.get("steps") or {}).items():
        samples = step["ok"] + step["error"]
        if samples >= th["min_samples"] and step["fail_rate"] > th["pipeline_fail_rate"]:
            issues.append(
                {
                    "rule": "pipeline_fail_spike",
                    "severity": "high",
                    "evidence": f"记忆管线 {op} 失败率 {step['fail_rate']:.1%}（{samples} 次）",
                    "recommendation": "查 memory_ops.error_text 与 LLM_FAST_* 配额。",
                }
            )
    return issues


def build_report(
    tool_rows: list[dict[str, Any]],
    cost_rows: list[dict[str, Any]],
    memory_rows: list[dict[str, Any]],
    *,
    since_ts: float,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """组装完整报告（时间窗与阈值随报告落档，便于事后对比）。"""
    tools = analyze_tool_calls(tool_rows)
    memory = analyze_memory_ops(memory_rows)
    issues = apply_rules(tools, memory, thresholds=thresholds)
    return {
        "generated_at": time.time(),
        "since_ts": since_ts,
        "thresholds": {**DEFAULT_THRESHOLDS, **(thresholds or {})},
        "tools": tools,
        "costs": analyze_costs(cost_rows),
        "memory_pipeline": memory,
        "issues": issues,
        "summary": {
            "issue_count": len(issues),
            "worst_severity": max((i["severity"] for i in issues), default="none"),
        },
    }


# ─────────────────────────── 经协议取数（零裸 SQL） ───────────────────────────


async def collect_rows(since_ts: float, *, limit: int = 5000) -> dict[str, list[dict[str, Any]]]:
    """按 Settings 装配只读组件取三份数据；memory 包缺席时该节为空。

    toollog / costmeter 是基座组件（按 CHECKPOINTER_BACKEND 装配）；memory_ops
    来自记忆包 store——脚本（非基座源码）可以直接 import 记忆包（ADR-003
    只约束基座运行时不 import）。
    """
    from agent_base.core.config import Settings
    from agent_base.extensions.toollog import build_tool_call_recorder

    settings = Settings(_env_file=None)
    tool_rows: list[dict[str, Any]] = []
    recorder = build_tool_call_recorder(settings)
    if recorder is not None:
        tool_rows = await recorder.list_recent(since_ts=since_ts, limit=limit)
        closer = getattr(recorder, "aclose", None)
        if closer is not None:
            await closer()

    cost_rows: list[dict[str, Any]] = []
    backend = settings.checkpointer.backend
    if backend == "sqlite":
        from agent_base.extensions.costmeter import SqliteCostLedger

        ledger = await SqliteCostLedger.create(settings.checkpointer.sqlite_path)
    elif backend == "mysql":
        from agent_base.extensions.costmeter import MysqlCostLedger

        ledger = MysqlCostLedger(settings)
    else:
        from agent_base.extensions.costmeter import MemoryCostLedger

        ledger = MemoryCostLedger()
    try:
        cost_rows = [r.as_dict() for r in await ledger.list_recent(since_ts=since_ts, limit=limit)]
    finally:
        closer = getattr(ledger, "aclose", None)
        if closer is not None:
            await closer()

    memory_rows: list[dict[str, Any]] = []
    try:
        from agent_base_memory.store import build_memory_store

        store = await build_memory_store(settings)
        if store is not None:
            memory_rows = [
                {"op": op.op, "status": op.status} for op in await store.list_ops(None, limit=limit)
            ]
            store_closer = getattr(store, "aclose", None)
            if store_closer is not None:
                await store_closer()
    except ImportError:
        memory_rows = []  # 未安装记忆包：记忆节缺席（报告注明）。
    return {"tools": tool_rows, "costs": cost_rows, "memory": memory_rows}


def render_text(report: dict[str, Any]) -> str:
    """人读摘要（--format text）。"""
    lines = [
        "== agent-base 质量报告 ==",
        f"窗口起点：{time.strftime('%Y-%m-%d %H:%M', time.localtime(report['since_ts']))}",
        f"问题数：{report['summary']['issue_count']}"
        f"（最严重：{report['summary']['worst_severity']}）",
        "",
        "-- 工具 --",
        f"调用 {report['tools']['total']} 次，"
        f"成功率 {(report['tools']['success_rate'] or 0):.1%}，"
        f"平均耗时 {report['tools']['avg_duration_ms'] / 1000:.1f}s",
    ]
    for err in report["tools"]["top_errors"]:
        lines.append(f"  错误归因 {err['pattern']}: {err['count']}")
    for slow in report["tools"]["slow_tools"][:3]:
        lines.append(
            f"  慢工具 {slow['tool']}: p95 {slow['p95_ms'] / 1000:.1f}s（{slow['calls']} 次）"
        )
    costs = report["costs"]
    lines += [
        "",
        "-- 成本 --",
        f"LLM 调用 {costs['calls']} 次，{costs['total_cny']} 元，{costs['total_tokens']} tokens",
        f"  模型分布：{costs['by_model']}",
        "",
        "-- 记忆管线 --",
    ]
    if not report["memory_pipeline"]["steps"]:
        lines.append("  （无数据：未安装记忆包或窗口内无操作）")
    else:
        lines.append(
            f"步骤执行 {report['memory_pipeline']['total']} 次，"
            f"错误 {report['memory_pipeline']['errors']} 次"
        )
    for op, step in report["memory_pipeline"]["steps"].items():
        lines.append(
            f"  {op}: ok={step['ok']} error={step['error']} 失败率={step['fail_rate']:.1%}"
        )
    if report["issues"]:
        lines += ["", "-- 命中规则 --"]
        for issue in report["issues"]:
            lines.append(f"  [{issue['severity']}] {issue['rule']}：{issue['evidence']}")
            lines.append(f"    → {issue['recommendation']}")
    else:
        lines += ["", "未命中检测规则。"]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="运行质量分析报告（ADR-008）")
    parser.add_argument("--since-hours", type=float, default=24.0, help="聚合时间窗（小时）")
    parser.add_argument("--format", choices=["json", "text"], default="text")
    parser.add_argument("--limit", type=int, default=5000, help="每数据源最大行数")
    args = parser.parse_args()
    since_ts = time.time() - args.since_hours * 3600
    try:
        rows = asyncio.run(collect_rows(since_ts, limit=args.limit))
        report = build_report(rows["tools"], rows["costs"], rows["memory"], since_ts=since_ts)
    except Exception as exc:
        print(f"报告生成失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if args.format == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    else:
        print(render_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
