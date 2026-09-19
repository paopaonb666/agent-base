"""阶段 2 的扩展点（不属于 ``core`` 的额外模块）。

``observability`` —— request_id 传播 + 结构化日志，追踪由环境开关控制。
``memory``（checkpointer 接线）、``collab``（supervisor）、``events``（SSE
事件契约）、``metrics``、``toollog``（工具审计）、``filestore``（附件存储）、
``mysql57``（MySQL 5.7 兼容 saver）均已是本包成员。
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
