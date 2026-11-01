"""共享工具池（阶段 3）。

每个模块通过 ``AgentModule.get_tools()`` 贡献工具；每个运行时装配一次
池，再通过 ``ModuleContext.tools`` 交还给模块。模块把池绑定到它们的
LLM，并通过 LangGraph 的 ``ToolNode`` 执行它，后者把工具失败归一成
``ToolMessage`` 反馈（模型看到错误并能恢复——工具异常绝不会让图崩溃）。

每个工具都用挂钟超时（``TOOL_TIMEOUT_SECONDS``）包装。在异步路径
（服务器和 CLI 使用的路径）中，超时会干净地取消 await；在同步路径中，
调用方在截止时刻立即收到 ``ToolTimeoutError``，但底层调用仍在其工作
线程中继续运行直至自然结束（有界泄漏，已记录在案的权衡——Python 线程
无法被杀死）。

ADR-009 起包装器可挂韧性层（``resilience`` 参数）：冷却中的工具短路
返回可读原因（黑名单），瞬时错误指数退避重试，成功/失败落健康注册表。
不挂时行为与阶段 3 完全一致。
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.runnables.config import ensure_config
from langchain_core.tools import BaseTool
from langgraph.config import get_config

from agent_base.core.contracts import AgentModule
from agent_base.core.threads import ThreadId
from agent_base.core.tool_resilience import (
    TOOL_ERROR_MAX_CHARS,
    TOOL_ERROR_URL_RE,
    ToolResilience,
    classify_tool_error,
)
from agent_base.extensions.metrics import TOOL_METRICS
from agent_base.extensions.observability import get_request_id
from agent_base.extensions.toollog import (
    ARGS_MAX_CHARS,
    ERROR_MAX_CHARS,
    RESULT_MAX_CHARS,
    ToolCallRecord,
)
from agent_base.tools.streaming import emit_tool_call

DEFAULT_TOOL_TIMEOUT_SECONDS = 30.0

logger = logging.getLogger(__name__)


class ToolTimeoutError(TimeoutError):
    """某个工具超出了其挂钟预算而被截断。"""


class ToolPoolError(ValueError):
    """无法装配工具池（工具名重复）时抛出。"""


def handle_tool_error(exc: Exception) -> str:
    """把任何工具失败归一成模型可读的反馈（无状态版，兼容收口）。

    LangGraph 1.2 默认的 ``handle_tool_errors`` 只会转换参数错误，
    其余的一律重新抛出；基座的契约更强——坏掉的工具绝不能打断对话，
    因此每个异常都被转换为模型能够回应的 ``ToolMessage``（阶段 3）。

    ADR-009 起装配了韧性层（``ToolResilience.reporter``）的运行时改用
    有状态版 ``ToolErrorReporter.handle``：schema 错误按字段结构化反馈
    并执行修复预算；本函数仍是未装配韧性层调用方（测试直调、旧接线）
    的语义基线，两者对非 schema 错误的输出逐字节一致。
    """
    text = TOOL_ERROR_URL_RE.sub("<redacted-url>", str(exc))
    return f"tool execution failed: {text}"[:TOOL_ERROR_MAX_CHARS]


def _json_safe(value: Any) -> dict[str, Any]:
    """把工具参数转成可 JSON 序列化的形式；不可序列化的值降级为 repr。"""
    try:
        loaded: Any = json.loads(json.dumps(value, ensure_ascii=False, default=str))
        return loaded if isinstance(loaded, dict) else {"_raw": str(value)}
    except (TypeError, ValueError):
        return {"_raw": str(value)}


def _execution_context() -> tuple[str, str]:
    """读取当前图执行的 (thread_id, module)；不在图上下文时留空。

    thread_id 命名空间是 ``{module}:{user_thread_id}``（server/CLI 统一
    规则，值对象见 ``core.threads.ThreadId``），module 前缀据此拆出；
    直调工具（单测、CLI ad-hoc）没有该上下文，记录照常产生，只是这
    两个字段为空。
    """
    try:
        configurable = get_config().get("configurable") or {}
        thread_id = str(configurable.get("thread_id") or "")
    except Exception:
        return "", ""
    parsed = ThreadId.try_parse(thread_id)
    module = parsed.module if parsed is not None else ""
    return thread_id, module


class _TimeoutTool(BaseTool):
    """包装 ``inner``，让同步和异步两种运行模式都遵守同一挂钟预算。

    本类也是**工具调用可观测的唯一收口**（M5）：执行前后发
    ``tool_call`` 流事件（前端渲染参数/结果面板），完结时把记录交给
    ``recorder`` 落库——成功、超时、异常三种结局都会被记录。
    """

    inner: BaseTool
    timeout: float
    recorder: Any = None  # ToolCallRecorder | None；协议类型交由运行时鸭子匹配
    resilience: Any = None  # ToolResilience | None；黑名单短路/重试/落账（ADR-009）
    tier: str = "basic"  # 分层（ADR-010）：basic 永远绑定，advanced 参与路由

    def _blocked_reason(self) -> str | None:
        """韧性黑名单检查：冷却中返回给模型看的短路原因，否则 None。"""
        if self.resilience is None:
            return None
        blocked, reason = self.resilience.registry.is_blocked(self.name)
        return reason if blocked else None

    def _should_retry(
        self,
        attempt: int,
        attempts: int,
        *,
        timeout: bool = False,
        exc: Exception | None = None,
    ) -> bool:
        """重试判据：最后一轮不重试；仅瞬时错误重试；超时须显式开启。

        熔断态下 ``attempts_for()`` 已返回 1（无下一轮），自然短路。
        """
        if attempt >= attempts - 1 or self.resilience is None:
            return False
        if timeout:
            # 同步路径超时后底层线程仍在执行（见 _run 注释），盲目重试
            # 会造成重复副作用——默认关闭，TOOL_RETRY_ON_TIMEOUT 显式打开。
            return bool(self.resilience.retry_on_timeout)
        assert exc is not None
        return classify_tool_error(exc) == "transient"

    def _log_retry(self, attempt: int, attempts: int, delay: float, cause: str) -> None:
        logger.warning(
            "tool %s: 第 %d/%d 次尝试失败，%.1fs 后重试：%s",
            self.name,
            attempt + 1,
            attempts,
            delay,
            cause[:120],
        )

    def _track(
        self,
        call_id: str,
        safe_args: dict[str, Any],
        started: float,
        outcome: str,
        result_text: str,
        error_text: str,
    ) -> None:
        """完结一支调用：发流事件 + 交审计记录（两者都绝不抛错打断对话）。"""
        duration_ms = int((time.perf_counter() - started) * 1000)
        thread_id, module = _execution_context()
        try:
            emit_tool_call(
                call_id,
                self.name,
                "end",
                result=result_text or None,
                status=outcome,
                duration_ms=duration_ms,
                error=error_text or None,
            )
        except Exception:
            logger.debug("tool %s: emit tool_call end failed", self.name, exc_info=True)
        if self.recorder is not None:
            try:
                self.recorder.record(
                    ToolCallRecord(
                        call_id=call_id,
                        tool=self.name,
                        status=outcome,
                        args_json=json.dumps(safe_args, ensure_ascii=False)[:ARGS_MAX_CHARS],
                        result_text=result_text[:RESULT_MAX_CHARS],
                        error_text=error_text[:ERROR_MAX_CHARS],
                        duration_ms=duration_ms,
                        thread_id=thread_id,
                        module=module,
                        request_id=get_request_id(),
                    )
                )
            except Exception:
                logger.exception("tool %s: recorder.record failed", self.name)
        # 韧性层落账（ADR-009）：blocked 短路不执行也不计失败；schema 校验
        # 失败到不了这里（在 BaseTool.invoke 就被拦下，由 reporter 计数）。
        if self.resilience is not None:
            if outcome == "ok":
                self.resilience.reporter.note_success(self.name, thread_id)
            if outcome != "blocked":
                self.resilience.registry.record(self.name, outcome)
        TOOL_METRICS.observe_tool(self.name, outcome, time.perf_counter() - started)

    def _run(self, **kwargs: Any) -> Any:
        # daemon 线程 + Event.wait：截止时间在调用方一侧得到遵守，超时后
        # 调用方立即返回；失控工具留在后台跑完（Python 线程无法被杀死）。
        # 刻意不用 ThreadPoolExecutor——它的 non-daemon 工作线程会被
        # 解释器退出的 atexit join 卡住，一个失控工具就能拖住整个进程。
        call_id = uuid.uuid4().hex[:12]
        safe_args = _json_safe(kwargs)
        emit_tool_call(call_id, self.name, "start", args=safe_args)
        started = time.perf_counter()
        outcome = "ok"
        result_text = ""
        error_text = ""
        try:
            # 短时黑名单（ADR-009）：冷却中的工具不执行；原因文本作为正常
            # 结果返回，模型据此改道而不是反复撞同一堵墙。
            blocked_reason = self._blocked_reason()
            if blocked_reason is not None:
                outcome = "blocked"
                error_text = blocked_reason
                return blocked_reason
            attempts = 1 if self.resilience is None else self.resilience.attempts_for()
            # config 在调用方上下文取出后随参数进工作线程——ContextVar
            # 不会跨线程传播，必须在 start 之前固化（与 _arun 同语义）。
            config = ensure_config()
            for attempt in range(attempts):
                result: list[Any] = []
                error: list[Exception] = []
                done = threading.Event()

                def _target() -> None:
                    try:
                        result.append(self.inner.invoke(kwargs, config))
                    except Exception as exc:
                        error.append(exc)
                    finally:
                        done.set()

                worker = threading.Thread(target=_target, name=f"tool-{self.name}", daemon=True)
                worker.start()
                if not done.wait(timeout=self.timeout):
                    outcome = "timeout"
                    error_text = f"tool {self.name!r} exceeded {self.timeout}s timeout"
                    if self._should_retry(attempt, attempts, timeout=True):
                        delay = self.resilience.backoff_delay(attempt)
                        self._log_retry(attempt, attempts, delay, error_text)
                        time.sleep(delay)
                        continue
                    raise ToolTimeoutError(error_text)
                if error:
                    exc = error[0]
                    outcome = "error"
                    error_text = handle_tool_error(exc)
                    if self._should_retry(attempt, attempts, exc=exc):
                        delay = self.resilience.backoff_delay(attempt)
                        self._log_retry(attempt, attempts, delay, error_text)
                        time.sleep(delay)
                        continue
                    raise exc
                result_text = str(result[0])
                outcome = "ok"
                error_text = ""
                return result[0]
            raise RuntimeError(  # pragma: no cover - _should_retry 保证最后一轮不 continue
                f"tool {self.name!r} retry loop exhausted without an outcome"
            )
        finally:
            self._track(call_id, safe_args, started, outcome, result_text, error_text)

    async def _arun(self, **kwargs: Any) -> Any:
        call_id = uuid.uuid4().hex[:12]
        safe_args = _json_safe(kwargs)
        emit_tool_call(call_id, self.name, "start", args=safe_args)
        started = time.perf_counter()
        outcome = "ok"
        result_text = ""
        error_text = ""
        try:
            # 短时黑名单（ADR-009）：冷却中的工具不执行；原因文本作为正常
            # 结果返回，模型据此改道而不是反复撞同一堵墙。
            blocked_reason = self._blocked_reason()
            if blocked_reason is not None:
                outcome = "blocked"
                error_text = blocked_reason
                return blocked_reason
            attempts = 1 if self.resilience is None else self.resilience.attempts_for()
            # RunnableConfig 透传（M4 修复）：ensure_config 从当前运行上下文
            # 取出调用方传入的 config——此前 _TimeoutTool 不转发 config，
            # callbacks/metadata/configurable 在包装层被静默丢弃。
            config = ensure_config()
            for attempt in range(attempts):
                try:
                    value = await asyncio.wait_for(
                        self.inner.ainvoke(kwargs, config), timeout=self.timeout
                    )
                    result_text = str(value)
                    outcome = "ok"
                    error_text = ""
                    return value
                # 3.10 上 asyncio.TimeoutError 与内置 TimeoutError 是不同类型，两者都要接住。
                except (TimeoutError, asyncio.TimeoutError) as exc:
                    outcome = "timeout"
                    error_text = f"tool {self.name!r} exceeded {self.timeout}s timeout"
                    if self._should_retry(attempt, attempts, timeout=True):
                        # 异步路径的 wait_for 会取消内层任务，重试无悬挂风险
                        # （与同步路径不同），仍默认关闭、由配置显式打开。
                        delay = self.resilience.backoff_delay(attempt)
                        self._log_retry(attempt, attempts, delay, error_text)
                        await asyncio.sleep(delay)
                        continue
                    raise ToolTimeoutError(error_text) from exc
                except Exception as exc:
                    outcome = "error"
                    error_text = handle_tool_error(exc)
                    if self._should_retry(attempt, attempts, exc=exc):
                        delay = self.resilience.backoff_delay(attempt)
                        self._log_retry(attempt, attempts, delay, error_text)
                        await asyncio.sleep(delay)
                        continue
                    raise
            raise RuntimeError(  # pragma: no cover - _should_retry 保证最后一轮不 continue
                f"tool {self.name!r} retry loop exhausted without an outcome"
            )
        finally:
            self._track(call_id, safe_args, started, outcome, result_text, error_text)


def build_tool_pool(
    modules: dict[str, AgentModule],
    *,
    timeout: float = DEFAULT_TOOL_TIMEOUT_SECONDS,
    extra_tools: Sequence[BaseTool] = (),
    timeouts: Mapping[str, float] | None = None,
    recorder: Any = None,
    resilience: ToolResilience | None = None,
    tiers: Mapping[str, str] | None = None,
) -> list[BaseTool]:
    """把每个模块的工具收集进一个带超时包装、无冲突的池。

    顺序遵循模块装配顺序（``AGENT_MODULES``），``extra_tools``（工具库
    注册表产出的基座内置工具）追加在末尾。重复的工具名会中止启动——
    两个同名工具会让工具调用路由变得模糊，因此这是一个快速失败的
    配置错误；该检查覆盖模块工具与工具库工具的整体。

    ``timeouts`` 支持按工具名覆盖全局 ``timeout``（如网络搜索要用显著
    短于全局的预算）；未列出的工具沿用全局值。``recorder`` 是工具调用
    审计记录器（extensions/toollog），挂到每个包装器上实现全量落库。
    ``resilience`` 是韧性层（ADR-009：黑名单短路/瞬时错误退避重试/健康
    落账），None 时包装器行为与阶段 3 完全一致。``tiers`` 是按工具名的
    分层声明（ADR-010：basic/advanced），未列出的工具默认 basic。
    """
    overrides = timeouts or {}
    tier_overrides = tiers or {}
    pool: list[BaseTool] = []
    seen: set[str] = set()
    candidates: list[tuple[str, AgentModule | None, BaseTool]] = []
    for module in modules.values():
        for tool in module.get_tools():
            candidates.append((tool.name, module, tool))
    candidates.extend((tool.name, None, tool) for tool in extra_tools)
    for name, owner, tool in candidates:
        if name in seen:
            source = f"module {owner.name!r}" if owner is not None else "toolkit"
            raise ToolPoolError(
                f"duplicate tool name {name!r} contributed by {source}; "
                "tool names must be unique across modules and toolkit"
            )
        seen.add(name)
        pool.append(
            _TimeoutTool(
                name=name,
                description=tool.description,
                args_schema=tool.args_schema,
                inner=tool,
                timeout=overrides.get(name, timeout),
                recorder=recorder,
                resilience=resilience,
                tier=tier_overrides.get(name, "basic"),
            )
        )
    return pool
