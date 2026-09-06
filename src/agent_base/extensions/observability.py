"""可观测性基础：请求 ID（request_id）传播 + 结构化日志。

第 2 阶段交付物。继承自 chat-agent 评审中的 A1：请求 ID 存放在一个
``contextvars.ContextVar`` 中，每一条日志记录都由一个 logging ``Filter`` 补充上
该 ID，格式化器则渲染为人类可读的行（开发环境）或 JSON（生产环境）。这里没有
任何内容与特定入口耦合——CLI 以及之后的 FastAPI 都会调用本模块。

追踪（LangSmith / Langfuse）保持由环境开关控制：本模块从不读取或在
环境所提供范围之外硬编码任何具体厂商设置。
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

# 环境中的请求标识符。模块 / 节点 / 入口通过 ``get_request_id()`` 读取它；
# 每次顶层调用都会生成一个新值。
_REQUEST_ID: ContextVar[str] = ContextVar("request_id", default="")

# 用于追踪配置消息的日志器名称。
_CONFIG_LOGGER = "agent_base.tracing"

# 本模块安装的流处理器名称，这样重复调用 setup（测试、重载）时是替换而非复制处理器。
_HANDLER_NAME = "agent_base.console"

_TRUTHY = {"1", "true", "yes", "on"}


def _inject_request_id(record: logging.LogRecord) -> None:
    record.request_id = get_request_id() or "-"


class RequestIdFilter(logging.Filter):
    """把环境中的 ``request_id`` 附着到每一条日志记录上。"""

    def filter(self, record: logging.LogRecord) -> bool:
        _inject_request_id(record)
        return True


class JsonFormatter(logging.Formatter):
    """每行一个 JSON 对象：机器可解析的结构化日志。"""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class HumanFormatter(logging.Formatter):
    """面向本地开发的可读控制台行。"""

    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s")


def get_request_id() -> str:
    """返回环境中的请求 ID（未设置时返回 ""）。"""
    return _REQUEST_ID.get()


@contextmanager
def request_id(rid: str | None = None) -> Iterator[str]:
    """在某个请求 ID 下运行一个代码块；未提供时自动生成一个。"""
    token = _REQUEST_ID.set(rid or new_request_id())
    try:
        yield _REQUEST_ID.get()
    finally:
        _REQUEST_ID.reset(token)


def new_request_id() -> str:
    """一个简短、唯一、可排序的请求 ID。"""
    return f"{uuid.uuid4().hex[:12]}"


def setup_logging(*, json_lines: bool = False, level: int = logging.INFO) -> None:
    """配置一次 root logger：request-id 过滤器 + 流处理器。

    ``json_lines`` 选择结构化的 JSON 输出；默认的人类可读格式面向本地
    开发。幂等：重复调用会复用具名处理器，因此测试可以重新配置而不会
    堆积处理器。

    过滤器附着在*处理器*上，而不是 root logger 上：``Logger.filter``
    只会查询 logger 自己的过滤器，绝不查询其祖先的过滤器，所以 root
    级别的过滤器会静默漏掉通过其他 logger 发出的记录（langchain 大量
    使用了 ``logging``）。
    """
    root = logging.getLogger()
    root.setLevel(level)
    handler = next((h for h in root.handlers if h.get_name() == _HANDLER_NAME), None)
    if handler is None:
        handler = logging.StreamHandler()
        handler.set_name(_HANDLER_NAME)
        root.addHandler(handler)
    handler.setFormatter(JsonFormatter() if json_lines else HumanFormatter())
    handler.filters = [RequestIdFilter()]  # 替换，保持 setup 幂等


def log_tracing_config() -> None:
    """报告环境启用了哪些追踪 provider（由环境开关控制）。

    LangSmith 由 langchain 通过标准环境变量（``LANGSMITH_TRACING`` /
    ``LANGSMITH_API_KEY``）原生消费；Langfuse 同理。基座不导入任何厂商
    SDK——它只把环境已经决定好的状态暴露出来。
    """
    logger = logging.getLogger(_CONFIG_LOGGER)
    env = os.environ
    langsmith = _env_flag(env.get("LANGSMITH_TRACING")) or bool(env.get("LANGSMITH_API_KEY"))
    langfuse = bool(env.get("LANGFUSE_PUBLIC_KEY") and env.get("LANGFUSE_SECRET_KEY"))
    logger.info(
        "tracing: langsmith=%s langfuse=%s （设置 LANGSMITH_TRACING=true / LANGFUSE_* 密钥以启用）",
        "on" if langsmith else "off",
        "on" if langfuse else "off",
    )


def _env_flag(value: str | None) -> bool:
    return (value or "").strip().lower() in _TRUTHY
