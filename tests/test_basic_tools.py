"""零依赖基础工具的测试（M2）：current_time / calculator / json_query。"""

from __future__ import annotations

import pytest

from agent_base.tools.basic import calculator, current_time, json_query

# ── current_time ─────────────────────────────────────────────────────


def test_current_time_default_is_local_with_tz_name() -> None:
    rendered = current_time.invoke({})
    assert "T" in rendered  # ISO 8601
    assert "(" in rendered and ")" in rendered  # 时区名注记


def test_current_time_utc() -> None:
    rendered = current_time.invoke({"tz": "UTC"})
    assert rendered.endswith("(UTC)")


def test_current_time_iana_zone() -> None:
    rendered = current_time.invoke({"tz": "Asia/Shanghai"})
    assert "+08:00" in rendered
    assert "Asia/Shanghai" in rendered


def test_current_time_rejects_unknown_zone() -> None:
    with pytest.raises(ValueError, match="未知时区"):
        current_time.invoke({"tz": "Mars/Olympus_Mons"})


# ── calculator ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 ** 10", "1024"),
        ("144 / 12", "12"),
        ("(1 + 2) * 3", "9"),
        ("7 // 2", "3"),
        ("7 % 3", "1"),
        ("-(3 + 4)", "-7"),
        ("+5", "5"),
        ("0.5 + 0.25", "0.75"),
    ],
)
def test_calculator_evaluates(expression: str, expected: str) -> None:
    assert calculator.invoke({"expression": expression}) == expected


def test_calculator_rejects_division_by_zero() -> None:
    with pytest.raises(ValueError, match="除数为零"):
        calculator.invoke({"expression": "1 / 0"})


def test_calculator_rejects_syntax_error() -> None:
    with pytest.raises(ValueError, match="语法错误"):
        calculator.invoke({"expression": "1 +"})


def test_calculator_rejects_names_and_calls() -> None:
    # AST 白名单之外的节点（名字、调用）必须被拒绝——表达式来自模型输出，
    # 绝不能走到 eval 语义。
    with pytest.raises(ValueError, match="不支持的语法"):
        calculator.invoke({"expression": "__import__('os').system('true')"})


def test_calculator_rejects_non_numeric_literal() -> None:
    with pytest.raises(ValueError, match="只支持数字字面量"):
        calculator.invoke({"expression": "'hello' + 'world'"})


def test_calculator_rejects_bool_literal() -> None:
    # bool 是 int 的子类，但 True/False 不是算术，一律拒绝。
    with pytest.raises(ValueError, match="只支持数字字面量"):
        calculator.invoke({"expression": "True + 1"})


# ── json_query ───────────────────────────────────────────────────────

_DOCUMENT = '{"data": {"items": [{"name": "first"}, {"name": "second"}]}, "count": 2}'


def test_json_query_extracts_nested_path() -> None:
    result = json_query.invoke({"json_text": _DOCUMENT, "path": "data.items[1].name"})
    assert result == '"second"'


def test_json_query_handles_multiple_indices() -> None:
    result = json_query.invoke({"json_text": "[[1, 2], [3, 4]]", "path": "[1][0]"})
    assert result == "3"


def test_json_query_empty_path_summarizes_object() -> None:
    result = json_query.invoke({"json_text": _DOCUMENT})
    assert "JSON 对象" in result and "data" in result and "count" in result


def test_json_query_empty_path_summarizes_list_and_scalar() -> None:
    assert "JSON 数组" in json_query.invoke({"json_text": "[1, 2, 3]"})
    assert "JSON 标量" in json_query.invoke({"json_text": "42"})


def test_json_query_invalid_json() -> None:
    with pytest.raises(ValueError, match="JSON 解析失败"):
        json_query.invoke({"json_text": "{not json"})


def test_json_query_missing_key_lists_available() -> None:
    with pytest.raises(ValueError, match="可用键"):
        json_query.invoke({"json_text": _DOCUMENT, "path": "data.nope"})


def test_json_query_index_out_of_range() -> None:
    with pytest.raises(ValueError, match="越界"):
        json_query.invoke({"json_text": _DOCUMENT, "path": "data.items[9]"})


def test_json_query_index_on_non_list() -> None:
    with pytest.raises(ValueError, match="不可下标"):
        json_query.invoke({"json_text": _DOCUMENT, "path": "count[0]"})


def test_json_query_malformed_path() -> None:
    with pytest.raises(ValueError, match="缺少右括号"):
        json_query.invoke({"json_text": _DOCUMENT, "path": "data[0"})
    with pytest.raises(ValueError, match="不是非负整数"):
        json_query.invoke({"json_text": _DOCUMENT, "path": "data[x]"})
    with pytest.raises(ValueError, match="含空段"):
        json_query.invoke({"json_text": _DOCUMENT, "path": "data..items"})


def test_json_query_truncates_huge_result() -> None:
    result = json_query.invoke({"json_text": '{"blob": "' + "x" * 10_000 + '"}', "path": "blob"})
    assert "已截断" in result
    assert len(result) < 10_000
