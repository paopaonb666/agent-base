"""零依赖基础工具（M2）：current_time / calculator / json_query。

不依赖任何第三方库与网络，是 ``TOOLKIT_ENABLED`` 的默认集合——它们
同时是注册表治理语义的首批"用户"。工具的 docstring 是给模型看的
说明书（chat-agent 的经验：docstring 即提示词），描述如何以及何时
调用，而不只是复述实现。
"""

from __future__ import annotations

import ast
import json
import operator
from collections.abc import Callable
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import tool

from agent_base.tools.spec import ToolSpec

# json_query 返回体的长度封顶：查询结果可能是一个巨大的子树，超长
# 输出既浪费上下文也没有可操作性。
_JSON_RESULT_MAX_CHARS = 4000

# calculator 允许的 AST 节点白名单：算术二元/一元运算与数字字面量。
_BIN_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


@tool
def current_time(tz: str = "local") -> str:
    """获取当前日期时间。返回 ISO 8601 格式（含时区偏移），例如
    2026-09-12T14:30:00+08:00。

    Args:
        tz: 时区。"local"（服务器本地时区，默认）、"UTC" 或 IANA 名称
            （如 "Asia/Shanghai"、"America/New_York"）。
    """
    if tz.strip().lower() in ("local", ""):
        now = datetime.now().astimezone()
        return f"{now.isoformat()} ({now.tzname()})"
    if tz.strip().lower() == "utc":
        return f"{datetime.now().isoformat()} (UTC)"
    try:
        zone = ZoneInfo(tz.strip())
    except ZoneInfoNotFoundError as exc:
        raise ValueError(
            f"未知时区 {tz!r}：请使用 IANA 时区名（如 Asia/Shanghai）或 'local'/'UTC'"
        ) from exc
    now = datetime.now(zone)
    return f"{now.isoformat()} ({zone.key}，{now.tzname()})"


def _safe_eval(node: ast.expr) -> int | float:
    """对 AST 白名单求值；绝不使用 eval——表达式来自模型输出。"""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError(f"只支持数字字面量，收到了 {node.value!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Div) and right == 0:
            raise ValueError("除数为零")
        result = _BIN_OPS[type(node.op)](left, right)
        assert isinstance(result, (int, float))
        return result
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        result = _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
        assert isinstance(result, (int, float))
        return result
    raise ValueError("表达式包含不支持的语法；仅支持 + - * / // % ** 与括号")


def _format_number(value: int | float) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


@tool
def calculator(expression: str) -> str:
    """计算算术表达式的值。仅支持四则运算（+ - * / // %）、幂运算（**）
    与括号，例如 "(1 + 2) * 3 ** 2"。不支持变量、函数调用与单位换算。

    Args:
        expression: 算术表达式，如 "144 / 12" 或 "2 ** 10"。
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"表达式语法错误：{expression!r}（{exc.msg}）") from exc
    return _format_number(_safe_eval(tree.body))


def _resolve_path(document: object, path: str) -> object:
    """解析 ``a.b[0][1]`` 形式的点/下标路径；键名不支持含点号。"""
    current = document
    for segment in path.split("."):
        token = segment.strip()
        if not token:
            raise ValueError(f"路径 {path!r} 含空段（连续的点号？）")
        parts = token.split("[")
        key = parts[0]
        if key:
            if not isinstance(current, dict) or key not in current:
                available = (
                    ", ".join(map(str, list(current)[:20]))
                    if isinstance(current, dict)
                    else type(current).__name__
                )
                hint = f"，可用键：{available}" if isinstance(current, dict) else ""
                raise ValueError(
                    f"路径 {path!r} 在键 {key!r} 处失败：当前是 {type(current).__name__}{hint}"
                )
            current = current[key]
        for raw in parts[1:]:
            if not raw.endswith("]"):
                raise ValueError(f"路径 {path!r} 的下标 {raw!r} 缺少右括号")
            index_text = raw[:-1]
            if not index_text.isdigit():
                raise ValueError(f"路径 {path!r} 的下标 {index_text!r} 不是非负整数")
            index = int(index_text)
            if not isinstance(current, list):
                raise ValueError(
                    f"路径 {path!r} 在 [{index}] 处失败：当前是 {type(current).__name__}，不可下标"
                )
            if index >= len(current):
                raise ValueError(f"路径 {path!r} 的下标 [{index}] 越界：列表长度 {len(current)}")
            current = current[index]
    return current


@tool
def json_query(json_text: str, path: str = "") -> str:
    """查询 JSON 文本中的值。path 为空时返回结构概要（顶层键或元素数）；
    否则按 ``a.b[0].c`` 形式的点/下标路径提取子值。键名不支持含点号。

    Args:
        json_text: 要查询的 JSON 文本（对象或数组）。
        path: 可选的取值路径，如 "data.items[0].name"；留空返回概要。
    """
    try:
        document: object = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON 解析失败（第 {exc.lineno} 行）：{exc.msg}") from exc

    if not path.strip():
        if isinstance(document, dict):
            keys = ", ".join(map(str, list(document)[:50]))
            return f"JSON 对象，{len(document)} 个键：{keys}"
        if isinstance(document, list):
            return f"JSON 数组，{len(document)} 个元素；用下标路径（如 '[0]'）访问"
        return f"JSON 标量：{document!r}"

    value = _resolve_path(document, path.strip())
    rendered = json.dumps(value, ensure_ascii=False)
    if len(rendered) > _JSON_RESULT_MAX_CHARS:
        rendered = rendered[:_JSON_RESULT_MAX_CHARS] + "…（已截断）"
    return rendered


# 注册条目：零依赖、只读安全级别，因此是 TOOLKIT_ENABLED 的默认成员。
BASIC_TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="current_time",
        category="basic",
        safety="read",
        factory=lambda _settings: [current_time],
        available=lambda _settings: True,
        unavailable_reason="",
    ),
    ToolSpec(
        name="calculator",
        category="basic",
        safety="read",
        factory=lambda _settings: [calculator],
        available=lambda _settings: True,
        unavailable_reason="",
    ),
    ToolSpec(
        name="json_query",
        category="basic",
        safety="read",
        factory=lambda _settings: [json_query],
        available=lambda _settings: True,
        unavailable_reason="",
    ),
)

__all__ = ["BASIC_TOOL_SPECS", "calculator", "current_time", "json_query"]
