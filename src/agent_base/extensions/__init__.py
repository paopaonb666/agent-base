"""阶段 2 的扩展点（不属于 ``core`` 的额外模块）。

``observability`` —— request_id 传播 + 结构化日志，追踪由环境开关控制。
阶段 3 会在这里加入 ``memory``（checkpointer 接线）。
"""

from agent_base.extensions.observability import (
    HumanFormatter,
    JsonFormatter,
    RequestIdFilter,
    get_request_id,
    log_tracing_config,
    new_request_id,
    request_id,
    setup_logging,
)

__all__ = [
    "HumanFormatter",
    "JsonFormatter",
    "RequestIdFilter",
    "get_request_id",
    "log_tracing_config",
    "new_request_id",
    "request_id",
    "setup_logging",
]
