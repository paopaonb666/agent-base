"""工具调用韧性层（ADR-009）：异常分类、健康注册表、结构化参数错误反馈。

核查缺口（工具调用模块核查，2026-11）的补齐——此前工具执行路径一次
失败即终止、注册表启动后冻结、参数错误细节被压平成 300 字符单字符串：

- **异常分类** ``classify_tool_error``：transient（网络/限流）可重试；
  schema（参数校验失败）走结构化 per-field 反馈 + 修复预算；permanent
  （工具内部逻辑错误）不重试——重试也不会好，直接反馈给模型自救。
- **健康注册表** ``ToolHealthRegistry``：per-tool 连续失败计数，达到阈值
  进入短时冷却（指数翻倍，同 scheduler 的退避语义），冷却期内调用短路
  返回可读原因（黑名单）；全局滚动窗口失败预算耗尽 → 熔断态：暂停一切
  工具重试 + critical 日志（防批量雪崩：依赖塌方时不再叠加无效重试），
  但**不禁用工具**——可用性优先，雪崩靠少打无效请求抑制。
- **结构化反馈** ``ToolErrorReporter``：schema 错误按字段列出（langgraph
  ``ToolInvocationError`` 自带 ``filtered_errors``，此前被
  ``handle_tool_error`` 压平截断）；per-(thread, tool) 连续校验失败达到
  ``arg_repair_max_attempts`` 给出终止性反馈，让模型停止调用该工具
  （此前只有全局 recursion_limit 的间接边界，无法定向止损）。

边界与权衡（详见 docs/adr/009-tool-resilience.md）：

- **超时默认不重试**：同步路径超时后底层线程可能仍在执行，盲目重试会
  造成重复副作用（``TOOL_RETRY_ON_TIMEOUT`` 可显式打开）；
- 重试不重放审批语义：审批门在 ToolNode 之上拦截，短路/重试都在
  ``_TimeoutTool`` 内部，互不可见；
- 黑名单只统计**执行**失败（timeout/error）；blocked 短路与 schema
  校验失败（模型的问题，不是工具坏）不计入，否则会把"模型传错参"
  误判成"工具坏了"。
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from langgraph.prebuilt.tool_node import ToolInvocationError
from pydantic import ValidationError

logger = logging.getLogger(__name__)

# ── 错误文本卫生（从 core.tools 迁来，方向：tools → tool_resilience 单向依赖）──
# 工具错误文本会进入 ToolMessage（模型 + 会话历史回放都会看到）：
# URL 脱敏、长度封顶，避免把内部端点/巨型负载泄给前端。
TOOL_ERROR_URL_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+.-]*://\S+")
TOOL_ERROR_MAX_CHARS = 300

# schema 错误按字段列出，字段细节比堆栈更有价值：上限放宽到 1000，
# 避免多字段场景被 300 截断丢掉后半的关键字段。
SCHEMA_ERROR_MAX_CHARS = 1000

ErrorClass = str  # "transient" | "schema" | "permanent"

# 瞬时错误的类型基底：内置网络/超时族（asyncio.TimeoutError 在 3.11+
# 与 TimeoutError 同一类型，3.10 仍需单独列出）。
_TRANSIENT_BASE_TYPES: tuple[type[BaseException], ...] = (
    ConnectionError,
    TimeoutError,
    asyncio.TimeoutError,
)

# 不 import httpx/openai（它们不是基座的直接依赖），按 MRO 名字匹配
# 两家的网络错误族——依赖树保持轻薄，分类仍然稳定。
_TRANSIENT_MRO_NAMES = frozenset(
    {
        "httperror",
        "transporterror",
        "apiconnectionerror",
        "apitimeouterror",
        "connecterror",
        "connecttimeout",
        "readtimeout",
    }
)

_TRANSIENT_MESSAGE_PATTERNS = (
    "rate limit",
    "too many requests",
    "429",
    "temporarily unavailable",
    "service unavailable",
    "bad gateway",
    "overloaded",
    "connection reset",
    "connection refused",
    "connection aborted",
    "timed out",
)


def classify_tool_error(exc: Exception) -> ErrorClass:
    """把工具异常分成三档，重试与反馈策略据此分野。

    - ``schema``——参数校验失败（pydantic 直抛或 langgraph 包装版）：
      模型的错，可修复，走结构化反馈；
    - ``transient``——网络/限流类：基础设施的错，退避后重试可能成功；
    - ``permanent``——其余（工具内部逻辑错误等）：重试无益，直接反馈。
    """
    if isinstance(exc, (ValidationError, ToolInvocationError)):
        return "schema"
    if isinstance(exc, _TRANSIENT_BASE_TYPES):
        return "transient"
    mro_names = {t.__name__.lower() for t in type(exc).__mro__}
    if mro_names & _TRANSIENT_MRO_NAMES:
        return "transient"
    text = str(exc).lower()
    if any(pattern in text for pattern in _TRANSIENT_MESSAGE_PATTERNS):
        return "transient"
    return "permanent"


def _current_thread_id() -> str:
    """修复预算的线程键：复用 core.tools 的图上下文解析（延迟导入避免环）。"""
    from agent_base.core.tools import _execution_context

    return _execution_context()[0]


def _format_field_errors(exc: Exception) -> str:
    """把校验错误压成 per-field 单行文本（模型可据此精确定位修复）。"""
    errors: list[dict[str, Any]] = []
    if isinstance(exc, ToolInvocationError):
        # filtered_errors 已剔除非模型控制的注入参数；退回全量防御缺失。
        source = getattr(exc, "source", None)
        errors = list(getattr(exc, "filtered_errors", None) or getattr(source, "errors", lambda: [])())
    elif isinstance(exc, ValidationError):
        errors = list(exc.errors())
    if not errors:
        return str(exc)[:200]
    parts: list[str] = []
    for err in errors[:8]:  # 字段数也封顶，防超长 schema 一口气刷屏
        loc = err.get("loc") or ()
        name = ".".join(str(part) for part in loc if part != "__root__") or "(root)"
        msg = str(err.get("msg") or "invalid value")
        got = err.get("input")
        suffix = f" (got {got!r:.60})" if got is not None else ""
        parts.append(f'field "{name}": {msg}{suffix}')
    return "; ".join(parts)


@dataclass
class _ToolHealth:
    """单个工具的健康状态（注册表内部账本）。"""

    consecutive_failures: int = 0
    cooldown_until: float = 0.0
    cooldown_count: int = 0  # 冷却档位：指数翻倍的指数


class ToolHealthRegistry:
    """进程内工具健康注册表：连续失败 → 短时冷却；全局失败预算 → 熔断态。

    线程安全：同步路径可能经 langgraph 的执行器线程并发落账
    （``_TimeoutTool._track`` 是唯一写入方），一把轻量锁足够——账本
    操作是 O(1) dict/deque，不存在锁内 IO。
    """

    def __init__(
        self,
        *,
        blacklist_threshold: int = 3,
        blacklist_cooldown_seconds: float = 60.0,
        blacklist_max_cooldown_seconds: float = 3600.0,
        global_failure_window_seconds: float = 60.0,
        global_failure_budget: int = 10,
    ) -> None:
        self._blacklist_threshold = blacklist_threshold
        self._blacklist_cooldown = blacklist_cooldown_seconds
        self._blacklist_max_cooldown = blacklist_max_cooldown_seconds
        self._global_window = global_failure_window_seconds
        self._global_budget = global_failure_budget
        self._lock = threading.Lock()
        self._tools: dict[str, _ToolHealth] = {}
        self._global_failures: deque[float] = deque()
        self._global_shed_until = 0.0

    # -- 写入侧 ----------------------------------------------------------

    def record(self, name: str, outcome: str) -> None:
        """记录一次执行结局（ok/timeout/error）；blocked 不入账。

        成功即全量清零（连续失败与冷却档位）——间歇性抖动不该被历史
        惩罚；失败则连续计数 +1，达到阈值且当前不在冷却中就进入/加深
        冷却。阈值 ≤ 0 表示禁用对应机制（黑名单/全局预算各自独立关闭）。
        """
        with self._lock:
            now = time.monotonic()
            if outcome == "ok":
                self._tools.pop(name, None)
                return
            health = self._tools.setdefault(name, _ToolHealth())
            health.consecutive_failures += 1
            if (
                self._blacklist_threshold > 0
                and health.consecutive_failures >= self._blacklist_threshold
                and now >= health.cooldown_until
            ):
                health.cooldown_count += 1
                duration = min(
                    self._blacklist_cooldown * (2 ** (health.cooldown_count - 1)),
                    self._blacklist_max_cooldown,
                )
                health.cooldown_until = now + duration
                logger.warning(
                    "tool resilience: 工具 %s 连续失败 %d 次，进入 %.0fs 冷却（第 %d 档）",
                    name,
                    health.consecutive_failures,
                    duration,
                    health.cooldown_count,
                )
            if self._global_budget > 0:
                self._global_failures.append(now)
                self._prune_global(now)
                if (
                    len(self._global_failures) >= self._global_budget
                    and now >= self._global_shed_until
                ):
                    self._global_shed_until = now + self._global_window
                    logger.critical(
                        "tool resilience: 全局失败预算耗尽（%d 次/%.0fs），熔断 %.0fs：暂停全部工具重试",
                        len(self._global_failures),
                        self._global_window,
                        self._global_window,
                    )

    def _prune_global(self, now: float) -> None:
        horizon = now - self._global_window
        while self._global_failures and self._global_failures[0] < horizon:
            self._global_failures.popleft()

    # -- 读取侧 ----------------------------------------------------------

    def is_blocked(self, name: str) -> tuple[bool, str]:
        """工具是否处于冷却；True 时附带给模型/日志看的可读原因。"""
        with self._lock:
            health = self._tools.get(name)
            if health is None:
                return False, ""
            remaining = health.cooldown_until - time.monotonic()
            if remaining <= 0:
                return False, ""
            return True, (
                f"tool {name!r} is temporarily blocked after "
                f"{health.consecutive_failures} consecutive failures; "
                f"cooldown {remaining:.0f}s remaining — use another tool or answer directly"
            )

    @property
    def global_shed_active(self) -> bool:
        """全局熔断态：窗口滑过自动恢复（不需要显式复位）。"""
        with self._lock:
            return time.monotonic() < self._global_shed_until

    def retries_allowed(self) -> bool:
        """熔断态下暂停一切重试（防雪崩的核心开关：少打无效请求）。"""
        return not self.global_shed_active

    def health(self) -> str:
        """/health 组件口径：ok / degraded（有工具在冷却或全局熔断中）。"""
        if self.global_shed_active:
            return "degraded"
        with self._lock:
            now = time.monotonic()
            if any(h.cooldown_until > now for h in self._tools.values()):
                return "degraded"
        return "ok"

    def snapshot(self) -> dict[str, Any]:
        """冷却中工具清单（观测用；时间换算成剩余秒便于消费）。"""
        with self._lock:
            now = time.monotonic()
            blocked = {
                name: {
                    "consecutive_failures": h.consecutive_failures,
                    "cooldown_seconds_left": round(max(0.0, h.cooldown_until - now), 1),
                }
                for name, h in self._tools.items()
                if h.cooldown_until > now
            }
            return {
                "blocked_tools": blocked,
                "global_shed": self.global_shed_active,
            }


class ToolErrorReporter:
    """有状态版 ``handle_tool_error``：结构化参数反馈 + 修复预算。

    语义分档：

    - schema 错误 → per-field 结构化文本（带尝试计数与修复指引）；
      同一 (thread, tool) 连续失败达 ``max_repair_attempts`` 次给**终止性**
      反馈，让模型停止调用该工具（定向止损，替代全局 recursion_limit
      的粗边界）；
    - 其余错误 → 维持 URL 脱敏 + 300 截断（阶段 3 的既有语义）。
    """

    def __init__(
        self,
        *,
        registry: ToolHealthRegistry | None = None,
        max_repair_attempts: int = 3,
    ) -> None:
        self._registry = registry
        self._max_repair_attempts = max_repair_attempts
        self._lock = threading.Lock()
        self._repair_counts: dict[tuple[str, str], tuple[int, float]] = {}
        self._repair_ttl_seconds = 900.0  # 预算 15 分钟未续即失效（线程级自然回收）

    def handle(self, exc: Exception, tool: str = "") -> str:
        """错误归一入口：ToolNode 的 ``handle_tool_errors`` 回调与本库三个
        收口（ToolNode / 审批门 / planner 执行器）共用。

        ``tool`` 带默认值是刻意的：ToolNode 回调只传异常（schema 错误
        时从 ``ToolInvocationError.tool_name`` 自取），审批门与 planner
        的收口知道工具名、按位置第二参传入。
        """
        if classify_tool_error(exc) == "schema":
            return self._schema_feedback(exc, tool=tool)
        text = TOOL_ERROR_URL_RE.sub("<redacted-url>", str(exc))
        return f"tool execution failed: {text}"[:TOOL_ERROR_MAX_CHARS]

    def _schema_feedback(self, exc: Exception, *, tool: str) -> str:
        tool_name = tool or str(getattr(exc, "tool_name", "") or "unknown")
        key = (_current_thread_id(), tool_name)
        with self._lock:
            self._prune_expired()
            count, _ = self._repair_counts.get(key, (0, 0.0))
            count += 1
            self._repair_counts[key] = (count, time.monotonic())
        fields = _format_field_errors(exc)
        if count >= self._max_repair_attempts:
            return (
                f"argument validation failed {count} times in a row for tool {tool_name!r}; "
                "STOP calling this tool — answer without it or ask the user for clarification. "
                f"Last errors: {fields}"
            )[:SCHEMA_ERROR_MAX_CHARS]
        return (
            f"argument validation failed for tool {tool_name!r} "
            f"(attempt {count}/{self._max_repair_attempts}): {fields}. "
            "Fix the argument format and retry."
        )[:SCHEMA_ERROR_MAX_CHARS]

    def note_success(self, tool: str, thread_id: str = "") -> None:
        """成功调用清空该 (thread, tool) 的修复预算（连续计数中断即归零）。"""
        key = (thread_id or _current_thread_id(), tool)
        with self._lock:
            self._repair_counts.pop(key, None)

    def _prune_expired(self) -> None:
        if len(self._repair_counts) < 256:
            return  # 常规规模不扫描；超界才做过期清扫
        horizon = time.monotonic() - self._repair_ttl_seconds
        for key in [k for k, (_, ts) in self._repair_counts.items() if ts < horizon]:
            del self._repair_counts[key]


@dataclass
class ToolResilience:
    """韧性层的装配产物：池包装器与图工厂共用的配置 + 组件束。

    ``from_settings`` 是唯一构造路径（bootstrap 调用）；各阈值来自
    ``ToolkitSettings`` 的韧性节（``TOOL_RETRY_*`` / ``TOOL_BLACKLIST_*``
    / ``TOOL_GLOBAL_FAILURE_*`` / ``TOOL_ARG_REPAIR_*``）。
    """

    registry: ToolHealthRegistry = field(default_factory=ToolHealthRegistry)
    reporter: ToolErrorReporter = field(default_factory=ToolErrorReporter)
    retry_enabled: bool = True
    retry_max_attempts: int = 2
    retry_backoff_base_seconds: float = 0.5
    retry_backoff_cap_seconds: float = 8.0
    retry_on_timeout: bool = False

    @classmethod
    def from_settings(cls, toolkit_settings: Any) -> "ToolResilience":
        s = toolkit_settings
        registry = ToolHealthRegistry(
            blacklist_threshold=s.blacklist_threshold,
            blacklist_cooldown_seconds=s.blacklist_cooldown_seconds,
            blacklist_max_cooldown_seconds=s.blacklist_max_cooldown_seconds,
            global_failure_window_seconds=s.global_failure_window_seconds,
            global_failure_budget=s.global_failure_budget,
        )
        reporter = ToolErrorReporter(
            registry=registry,
            max_repair_attempts=s.arg_repair_max_attempts,
        )
        return cls(
            registry=registry,
            reporter=reporter,
            retry_enabled=s.retry_enabled,
            retry_max_attempts=s.retry_max_attempts,
            retry_backoff_base_seconds=s.retry_backoff_base_seconds,
            retry_backoff_cap_seconds=s.retry_backoff_cap_seconds,
            retry_on_timeout=s.retry_on_timeout,
        )

    def attempts_for(self) -> int:
        """本次调用的总尝试次数上限（重试关闭/熔断态 → 1 = 不重试）。"""
        if not self.retry_enabled or not self.registry.retries_allowed():
            return 1
        return max(1, self.retry_max_attempts)

    def backoff_delay(self, attempt: int) -> float:
        """指数退避 + 抖动：base * 2^attempt 封顶，再叠加 ≤10% 抖动。"""
        delay = min(
            self.retry_backoff_base_seconds * (2**attempt),
            self.retry_backoff_cap_seconds,
        )
        return delay * (1.0 + random.uniform(0.0, 0.1))


__all__ = [
    "SCHEMA_ERROR_MAX_CHARS",
    "TOOL_ERROR_MAX_CHARS",
    "TOOL_ERROR_URL_RE",
    "ErrorClass",
    "ToolErrorReporter",
    "ToolHealthRegistry",
    "ToolResilience",
    "classify_tool_error",
]
