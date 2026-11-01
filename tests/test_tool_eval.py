"""工具调用评测（ADR-011）的测试：指标纯函数、聚合、报告渲染。

与 test_quality_report 同款：经 importlib 加载脚本模块，纯函数直测——
跑批路径需要真实 LLM，不在单测覆盖（冒烟由 --arm baseline 手动验证）。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "tool_eval.py"
_EVALSET = Path(__file__).resolve().parents[1] / "evals" / "tool_calls.jsonl"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("tool_eval", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ev = _load_module()


# ── 评测集加载 ──────────────────────────────────────────────────────────


def test_load_real_evalset_is_valid() -> None:
    cases = ev.load_cases(str(_EVALSET))
    assert len(cases) >= 10
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)), "用例 id 必须唯一"
    for case in cases:
        assert case["query"]
        assert case["arg_match"] in ("exact", "contains", "regex")
        # 期望参数只允许给期望工具的字段（写错工具名的用例是坏用例）
        assert set(case["expected_args"]) <= set(case["expected_tools"]) | {""}


def test_load_rejects_bad_lines(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="非法评测行"):
        ev.load_cases(str(path))


def test_load_rejects_empty_set(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="评测集为空"):
        ev.load_cases(str(path))


def test_load_rejects_unknown_match_mode(tmp_path: Path) -> None:
    path = tmp_path / "mode.jsonl"
    path.write_text(
        json.dumps(
            {"id": "x", "query": "q", "expected_tools": [], "arg_match": "fuzzy"}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="arg_match"):
        ev.load_cases(str(path))


# ── 轨迹提取 ────────────────────────────────────────────────────────────


def _trace() -> list[object]:
    return [
        HumanMessage(content="算一下"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "calculator", "args": {"expression": "2 ** 10"}, "id": "c1"}
            ],
        ),
        ToolMessage(content="1024", tool_call_id="c1", name="calculator"),
        AIMessage(content="结果是 1024。"),
    ]


def test_extract_calls_in_order() -> None:
    calls = ev.extract_calls(_trace())
    assert calls == [{"name": "calculator", "args": {"expression": "2 ** 10"}}]


def test_final_answer_takes_last_ai_text() -> None:
    assert ev.final_answer(_trace()) == "结果是 1024。"


# ── 匹配器 ──────────────────────────────────────────────────────────────


def test_args_match_modes() -> None:
    assert ev.args_match("Asia/Tokyo", "Asia/Tokyo", "exact")
    assert not ev.args_match("asia/tokyo", "Asia/Tokyo", "exact")
    assert ev.args_match("结果 = 42000 元", "42000", "contains")
    assert not ev.args_match("结果 = 42000 元", "4.2万", "contains")
    assert ev.args_match("3 ** 8", r"3\s*\*\*\s*8", "regex")
    assert not ev.args_match("3 * 8", r"3\s*\*\*\s*8", "regex")
    assert ev.args_match("x", "[invalid", "regex") is False  # 坏正则不炸


def test_tool_args_match_semantics() -> None:
    calls = [{"name": "json_query", "args": {"json_text": "{...}", "path": "[1]"}}]
    assert ev.tool_args_match(calls, "json_query", {"path": "[1]"}, "exact") is True
    # 子集语义：expected 没列的字段不校验
    assert ev.tool_args_match(calls, "json_query", {}, "exact") is True
    assert ev.tool_args_match(calls, "web_search", {"q": "x"}, "exact") is None


def test_contains_pass_single_and_list() -> None:
    assert ev.contains_pass("答案是 6561", "6561")
    assert not ev.contains_pass("答案是 6561", "6562")
    assert ev.contains_pass('春和夏都在', '["春", "夏"]')
    assert not ev.contains_pass("只有春", '["春", "夏"]')


# ── 打分与聚合 ──────────────────────────────────────────────────────────


def _case(case_id: str, tools: list[str], args: dict | None = None) -> dict:
    return {
        "id": case_id,
        "query": f"q-{case_id}",
        "expected_tools": tools,
        "expected_args": args or {},
        "arg_match": "exact",
        "success_criteria": "",
    }


def test_score_and_aggregate_denominators() -> None:
    case_a = _case("a", ["calculator"], {"calculator": {"expression": "3 ** 8"}})
    case_b = _case("b", ["calculator"])  # 无参数期望：调了就算对
    case_c = _case("c", [])  # 无工具期望：不进选择分母

    scores = [
        ev.score_case(
            case_a, [{"name": "calculator", "args": {"expression": "3 ** 8"}}]
        ),  # 全对
        ev.score_case(
            case_b, [{"name": "json_query", "args": {}}]
        ),  # 选错工具：选择 ✗；参数 None（calculator 没被调）
        ev.score_case(case_c, []),  # 无期望：全过
    ]
    completions = {"a": True, "b": False}
    metrics = ev.aggregate(scores, completions)
    assert metrics["tool_selection_first"] == 0.5  # a 命中 b 未中（c 不进分母）
    assert metrics["tool_selection_set"] == 0.5
    assert metrics["param_accuracy"] == 1.0  # 只计 a 的字段对（b 为 None）
    assert metrics["task_completion"] == 0.5
    assert metrics["errors"] == 0


def test_aggregate_error_cases_counted() -> None:
    scores = [
        {
            "id": "x",
            "expected": ["calculator"],
            "first_hit": False,
            "set_hit": False,
            "params": {"calculator": False},
            "calls": [],
            "error": "Boom",
        }
    ]
    metrics = ev.aggregate(scores, {"x": False})
    assert metrics["tool_selection_first"] == 0.0
    assert metrics["param_accuracy"] == 0.0
    assert metrics["errors"] == 1


# ── 报告 ────────────────────────────────────────────────────────────────


def test_build_report_computes_delta() -> None:
    baseline = {
        "cases": 3,
        "tool_selection_first": 0.5,
        "tool_selection_set": 0.6,
        "param_accuracy": 0.7,
        "task_completion": 0.8,
        "errors": 0,
    }
    treatment = {**baseline, "tool_selection_first": 0.7, "task_completion": None}
    report = ev.build_report(baseline, treatment, meta={"judge": "contains"})
    assert report["delta"]["tool_selection_first"] == pytest.approx(0.2)
    assert report["delta"]["task_completion"] is None  # 任一臂缺数据不算差
    text = ev.render_text(report)
    assert "工具选择准确率（首调）" in text
    assert "few-shot" in text
