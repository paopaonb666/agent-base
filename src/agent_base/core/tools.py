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
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.runnables.config import ensure_config
from langchain_core.tools import BaseTool
from langgraph.config import get_config

from agent_base.core.contracts import AgentModule
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

# 工具错误文本会进入 ToolMessage（模型 + 会话历史回放都会看到）：
# URL 脱敏、长度封顶，避免把内部端点/巨型负载泄给前端。
_TOOL_ERROR_URL_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+.-]*://\S+")
_TOOL_ERROR_MAX_CHARS = 300


class ToolTimeoutError(TimeoutError):
    """某个工具超出了其挂钟预算而被截断。"""


class ToolPoolError(ValueError):
    """无法装配工具池（工具名重复）时抛出。"""


def handle_tool_error(exc: Exception) -> str:
    """把任何工具失败归一成模型可读的反馈。

    LangGraph 1.2 默认的 ``handle_tool_errors`` 只会转换参数错误，
    其余的一律重新抛出；基座的契约更强——坏掉的工具绝不能打断对话，
    因此每个异常都被转换为模型能够回应的 ``ToolMessage``（阶段 3）。
    """
    text = _TOOL_ERROR_URL_RE.sub("<redacted-url>", str(exc))
    return f"tool execution failed: {text}"[:_TOOL_ERROR_MAX_CHARS]


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
    规则），module 前缀据此拆出；直调工具（单测、CLI ad-hoc）没有该
    上下文，记录照常产生，只是这两个字段为空。
    """
    try:
        configurable = get_config().get("configurable") or {}
        thread_id = str(configurable.get("thread_id") or "")
    except Exception:
        return "", ""
    module = thread_id.split(":", 1)[0] if ":" in thread_id else ""
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
                thread_id, module = _execution_context()
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
            result: list[Any] = []
            error: list[Exception] = []
            done = threading.Event()
            # config 在调用方上下文取出后随参数进工作线程——ContextVar
            # 不会跨线程传播，必须在 start 之前固化（与 _arun 同语义）。
            config = ensure_config()

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
                raise ToolTimeoutError(error_text)
            if error:
                outcome = "error"
                error_text = handle_tool_error(error[0])
                raise error[0]
            result_text = str(result[0])
            return result[0]
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
            # RunnableConfig 透传（M4 修复）：ensure_config 从当前运行上下文
            # 取出调用方传入的 config——此前 _TimeoutTool 不转发 config，
            # callbacks/metadata/configurable 在包装层被静默丢弃。
            config = ensure_config()
            try:
                value = await asyncio.wait_for(
                    self.inner.ainvoke(kwargs, config), timeout=self.timeout
                )
                result_text = str(value)
                return value
            # 3.10 上 asyncio.TimeoutError 与内置 TimeoutError 是不同类型，两者都要接住。
            except (TimeoutError, asyncio.TimeoutError) as exc:
                outcome = "timeout"
                error_text = f"tool {self.name!r} exceeded {self.timeout}s timeout"
                raise ToolTimeoutError(error_text) from exc
            except Exception as exc:
                outcome = "error"
                error_text = handle_tool_error(exc)
                raise
        finally:
            self._track(call_id, safe_args, started, outcome, result_text, error_text)


def build_tool_pool(
    modules: dict[str, AgentModule],
    *,
    timeout: float = DEFAULT_TOOL_TIMEOUT_SECONDS,
    extra_tools: Sequence[BaseTool] = (),
    timeouts: Mapping[str, float] | None = None,
    recorder: Any = None,
) -> list[BaseTool]:
    """把每个模块的工具收集进一个带超时包装、无冲突的池。

    顺序遵循模块装配顺序（``AGENT_MODULES``），``extra_tools``（工具库
    注册表产出的基座内置工具）追加在末尾。重复的工具名会中止启动——
    两个同名工具会让工具调用路由变得模糊，因此这是一个快速失败的
    配置错误；该检查覆盖模块工具与工具库工具的整体。

    ``timeouts`` 支持按工具名覆盖全局 ``timeout``（如网络搜索要用显著
    短于全局的预算）；未列出的工具沿用全局值。``recorder`` 是工具调用
    审计记录器（extensions/toollog），挂到每个包装器上实现全量落库。
    """
    overrides = timeouts or {}
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
            )
        )
    return pool
