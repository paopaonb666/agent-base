"""工具调用评测（ADR-011，Loop 4 的评测半环）：有/无 few-shot 两臂对照。

回答"few-shot 样本到底有没有用、有多大用"。三组核心指标：

- **工具选择准确率**——首个调用命中（first）/ 期望工具集覆盖（set）两个口径；
- **参数准确率**——期望字段按 exact / contains / regex 匹配实际参数
  （只校验 expected_args 里列出的字段，其余字段不惩罚）；
- **任务完成率**——success_criteria 判定：contains（子串，默认，零成本）
  或 llm（``core.verification.LLMJudgeGrader``，需真实 LLM）。

跑批形态与 ``quality_report.py`` 同款：**纯函数核心 + 薄 CLI**，tests 直接
导入纯函数测。轨迹不回读审计表——从图执行的返回状态里取（含全部
AIMessage.tool_calls），一次运行一次落袋，不依赖存储后端。

两臂语义：baseline 关闭 few-shot；treatment 开启（``TOOL_FEWSHOT_*``）。
嵌入未配置时 treatment 与 baseline 等价（few-shot 降级为不注入），报告
会显式注明——这份报告不能当对照用，但跑通流程仍有价值。

用法::

    python scripts/tool_eval.py [--eval evals/tool_calls.jsonl] \\
        [--judge contains|llm] [--format json|text] [--arm baseline|treatment|both]

需要 ``LLM_API_KEY``（跑真实模型）；两臂对照另需 ``MEMORY_EMBEDDING_API_KEY``。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from typing import Any

_TOOL_ARGS_MATCH_MODES = ("exact", "contains", "regex")


# ── 评测集 ──────────────────────────────────────────────────────────────


def load_cases(path: str) -> list[dict[str, Any]]:
    """加载评测集（JSONL）；行缺失必需键即报错——评测集错了跑出来的
    数字都是假的，宁可启动失败。"""
    cases: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
                cases.append(
                    {
                        "id": str(row["id"]),
                        "query": str(row["query"]),
                        "expected_tools": [str(t) for t in row.get("expected_tools") or []],
                        "expected_args": dict(row.get("expected_args") or {}),
                        "arg_match": str(row.get("arg_match") or "exact"),
                        "success_criteria": str(row.get("success_criteria") or ""),
                    }
                )
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{lineno} 非法评测行：{exc}") from exc
            if cases[-1]["arg_match"] not in _TOOL_ARGS_MATCH_MODES:
                raise ValueError(
                    f"{path}:{lineno} 未知 arg_match {cases[-1]['arg_match']!r}；"
                    f"可选 {_TOOL_ARGS_MATCH_MODES}"
                )
    if not cases:
        raise ValueError(f"评测集为空：{path}")
    return cases


# ── 轨迹提取与指标（纯函数） ────────────────────────────────────────────


def extract_calls(messages: list[Any]) -> list[dict[str, Any]]:
    """从图返回状态提取全部 tool_calls（按发生顺序）。"""
    calls: list[dict[str, Any]] = []
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            calls.append(
                {
                    "name": str(call.get("name") or ""),
                    "args": dict(call.get("args") or {}),
                }
            )
    return calls


def final_answer(messages: list[Any]) -> str:
    """最后一条 AI 文本消息（工具调用后的总结）。"""
    for message in reversed(messages):
        content = str(getattr(message, "content", "") or "")
        if getattr(message, "type", "") == "ai" and content.strip():
            return content
    return ""


def first_tool_hit(calls: list[dict[str, Any]], expected: list[str]) -> bool:
    """首个工具调用命中期望的第一名（有明确期望时才有意义）。"""
    if not expected:
        return True
    return bool(calls) and calls[0]["name"] == expected[0]


def tool_set_hit(calls: list[dict[str, Any]], expected: list[str]) -> bool:
    """期望工具集被实际调用覆盖（允许额外调用，不惩罚多步）。"""
    if not expected:
        return True
    actual = {c["name"] for c in calls}
    return set(expected) <= actual


def args_match(actual: Any, expected: Any, mode: str) -> bool:
    """单字段匹配：exact 全等；contains 子串；regex 正则。"""
    actual_text = "" if actual is None else str(actual)
    expected_text = str(expected)
    if mode == "exact":
        return actual_text == expected_text
    if mode == "contains":
        return expected_text.casefold() in actual_text.casefold()
    try:
        return re.search(expected_text, actual_text) is not None
    except re.error:
        return False


def tool_args_match(
    calls: list[dict[str, Any]], tool: str, expected_args: dict[str, Any], mode: str
) -> bool | None:
    """该工具的期望参数是否在某次调用里全部满足。

    只校验 expected_args 列出的字段（子集语义）。None = 工具根本没被
    调用过（不计入参数准确率的分母，但选择指标已经惩罚过它）。
    """
    tool_calls = [c["args"] for c in calls if c["name"] == tool]
    if not tool_calls:
        return None
    if not expected_args:
        return True  # 调了就算对（completion-only 用例）
    for actual_args in tool_calls:
        if all(args_match(actual_args.get(k), v, mode) for k, v in expected_args.items()):
            return True
    return False


def contains_pass(answer: str, criteria: str) -> bool:
    """contains 判定：criteria（可 JSON 数组多值）全部为答案子串。"""
    values = [criteria]
    if criteria.startswith("["):
        try:
            values = [str(v) for v in json.loads(criteria)]
        except json.JSONDecodeError:
            values = [criteria]
    lowered = answer.casefold()
    return all(str(v).casefold() in lowered for v in values)


def score_case(case: dict[str, Any], calls: list[dict[str, Any]]) -> dict[str, Any]:
    """单用例的选择/参数指标（完成率另算——判定可能要调判官）。"""
    mode = case["arg_match"]
    params: dict[str, Any] = {}
    for tool in case["expected_tools"]:
        expected = case["expected_args"].get(tool, {})
        params[tool] = tool_args_match(calls, tool, expected, mode)
    return {
        "id": case["id"],
        "expected": list(case["expected_tools"]),
        "first_hit": first_tool_hit(calls, case["expected_tools"]),
        "set_hit": tool_set_hit(calls, case["expected_tools"]),
        "params": params,
        "calls": [c["name"] for c in calls],
    }


def aggregate(scores: list[dict[str, Any]], completions: dict[str, bool]) -> dict[str, Any]:
    """三指标聚合：分母只计入适用的用例（选择指标只看声明了期望工具的；
    参数指标只看"工具被调过"的字段对；完成率只看写了 success_criteria 的）。"""
    selection = [s for s in scores if s.get("expected")]
    param_pairs = [ok for s in scores for ok in s["params"].values() if ok is not None]
    completion_pairs = list(completions.values())

    def rate(values: list[bool]) -> float | None:
        return round(sum(1 for v in values if v) / len(values), 4) if values else None

    return {
        "cases": len(scores),
        "tool_selection_first": rate([s["first_hit"] for s in selection]),
        "tool_selection_set": rate([s["set_hit"] for s in selection]),
        "param_accuracy": rate(param_pairs),
        "task_completion": rate(completion_pairs),
        "errors": sum(1 for s in scores if s.get("error")),
    }


def build_report(
    baseline: dict[str, Any],
    treatment: dict[str, Any],
    *,
    meta: dict[str, Any],
) -> dict[str, Any]:
    """两臂对照报告：每指标 baseline/treatment/delta（沿 ADR-008 的 JSON
    稳定契约风格，便于留档与事后对比）。"""
    keys = ("tool_selection_first", "tool_selection_set", "param_accuracy", "task_completion")
    delta = {}
    for key in keys:
        b, t = baseline.get(key), treatment.get(key)
        delta[key] = None if b is None or t is None else round(t - b, 4)
    return {
        "generated_at": time.time(),
        **meta,
        "baseline": baseline,
        "treatment": treatment,
        "delta": delta,
    }


def render_text(report: dict[str, Any]) -> str:
    """人读摘要（--format text）。"""
    lines = [
        "== 工具调用评测（few-shot 两臂对照）== ",
        f"评测集：{report.get('eval_path')}（{report['baseline']['cases']} 条）",
        f"判定器：{report.get('judge')}；few-shot：{report.get('fewshot_note')}",
        "",
        f"{'指标':<24}{'baseline':>10}{'treatment':>12}{'delta':>10}",
    ]

    def _fmt(value: Any) -> str:
        return "-" if value is None else f"{value:.1%}"

    for key, label in (
        ("tool_selection_first", "工具选择准确率（首调）"),
        ("tool_selection_set", "工具选择准确率（集合）"),
        ("param_accuracy", "参数填写准确率"),
        ("task_completion", "任务完成率"),
    ):
        b, t = report["baseline"].get(key), report["treatment"].get(key)
        d = report["delta"].get(key)
        lines.append(
            f"{label:<24}{_fmt(b):>10}{_fmt(t):>12}"
            f"{('-' if d is None else f'{d:+.1%}'):>10}"
        )
    lines.append("")
    lines.append(
        f"错误用例：baseline {report['baseline']['errors']} / "
        f"treatment {report['treatment']['errors']}"
    )
    return "\n".join(lines)


# ── 跑批（需要真实 LLM；纯函数之上的薄执行层） ─────────────────────────


async def run_arm(
    cases: list[dict[str, Any]],
    *,
    fewshot: bool,
    judge: str,
    eval_path: str,
    limit: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """跑一臂：装配 → 逐用例执行 → 打分聚合。返回 (scores, completions, note)。"""
    from langchain_core.messages import HumanMessage

    from agent_base.core.config import Settings
    from agent_base.core.contracts import ModuleContext
    from agent_base.core.graphs import build_single_agent_graph
    from agent_base.core.llm import build_llm
    from agent_base.core.tool_fewshot import build_tool_fewshot
    from agent_base.core.tool_resilience import ToolResilience
    from agent_base.core.tool_routing import build_tool_embedder
    from agent_base.core.tools import build_tool_pool
    from agent_base.modules.chat.module import ChatModule
    from agent_base.tools.registry import build_toolkit_tools, toolkit_tiers, toolkit_timeouts

    settings = Settings(tool_fewshot_enabled=fewshot) if fewshot else Settings()
    llm = build_llm(settings)
    resilience = ToolResilience.from_settings(settings.toolkit)
    toolkit_tools = build_toolkit_tools(settings)
    tiers = toolkit_tiers(settings)
    pool = build_tool_pool(
        {"chat": ChatModule()},
        timeout=settings.tool_timeout_seconds,
        extra_tools=toolkit_tools,
        timeouts=toolkit_timeouts(settings),
        resilience=resilience,
        tiers=tiers,
    )
    note: dict[str, Any] = {}
    fewshot_store = None
    if fewshot:
        embedder = build_tool_embedder(settings)
        fewshot_store = build_tool_fewshot(settings, embedder=embedder)
        if fewshot_store is None:
            note["fewshot_degraded"] = "嵌入不可用：treatment 与 baseline 等价，非有效对照"
    ctx = ModuleContext(
        settings=settings,
        llm=llm,
        tools=pool,
        tool_resilience=resilience,
        tool_fewshot=fewshot_store,
    )
    graph = build_single_agent_graph(ctx, name="tool-eval")

    judge_llm = build_llm(settings) if judge == "llm" else None

    async def _grade(case: dict[str, Any], answer: str) -> bool:
        if not case["success_criteria"]:
            return True
        if judge == "llm":
            from agent_base.core.verification import LLMJudgeGrader

            grader = LLMJudgeGrader(
                judge_llm, rubric=f"任务：{case['query']}。合格标准：{case['success_criteria']}"
            )
            result = await grader.check(answer)
            return result.grade.value == "PASS"
        return contains_pass(answer, case["success_criteria"])

    scores: list[dict[str, Any]] = []
    completions: dict[str, bool] = {}
    for case in cases[: (limit or len(cases))]:
        try:
            result = await graph.ainvoke(
                {"messages": [HumanMessage(content=case["query"])]},
                config={"recursion_limit": settings.agent_recursion_limit},
            )
            messages = result["messages"]
            calls = extract_calls(messages)
            score = score_case(case, calls)
            if case["success_criteria"]:
                completions[case["id"]] = await _grade(case, final_answer(messages))
            else:
                score["criteria_skipped"] = True
            scores.append(score)
        except Exception as exc:  # 单用例失败不废整臂
            scores.append(
                {
                    "id": case["id"],
                    "expected": list(case["expected_tools"]),
                    "first_hit": False,
                    "set_hit": False,
                    "params": {t: False for t in case["expected_tools"]},
                    "calls": [],
                    "error": f"{type(exc).__name__}: {exc}"[:200],
                }
            )
            if case["success_criteria"]:
                completions[case["id"]] = False
    return scores, completions, note


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    cases = load_cases(args.eval)
    arms: dict[str, dict[str, Any]] = {}
    notes: dict[str, Any] = {}
    wanted = ("baseline", "treatment") if args.arm == "both" else (args.arm,)
    for arm in wanted:
        scores, completions, note = await run_arm(
            cases, fewshot=(arm == "treatment"), judge=args.judge, eval_path=args.eval
        )
        arms[arm] = aggregate(scores, completions)
        arms[arm]["case_details"] = [
            {k: v for k, v in s.items() if k != "error"} for s in scores
        ]
        if note:
            notes[arm] = note
    meta = {
        "eval_path": args.eval,
        "judge": args.judge,
        "fewshot_note": notes.get("treatment", {}).get(
            "fewshot_degraded", "嵌入可用，两臂为有效对照"
        )
        if "treatment" in arms
        else "仅运行 baseline 臂",
    }
    baseline = arms.get("baseline") or arms["treatment"]
    treatment = arms.get("treatment") or arms["baseline"]
    return build_report(baseline, treatment, meta=meta)


def main() -> int:
    parser = argparse.ArgumentParser(description="工具调用评测（ADR-011：few-shot 两臂对照）")
    parser.add_argument("--eval", default="evals/tool_calls.jsonl", help="评测集路径（JSONL）")
    parser.add_argument("--judge", choices=["contains", "llm"], default="contains")
    parser.add_argument("--format", choices=["json", "text"], default="text")
    parser.add_argument(
        "--arm", choices=["baseline", "treatment", "both"], default="both", help="跑哪几臂"
    )
    args = parser.parse_args()
    try:
        report = asyncio.run(_run(args))
    except Exception as exc:
        print(f"评测失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if args.format == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    else:
        print(render_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
