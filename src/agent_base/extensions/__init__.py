"""Stage 2 extension points (extra modules not part of ``core``).

``observability`` — request_id propagation + structured logging, env-gated
tracing. Stage 3 will add ``memory`` (checkpointer wiring) here.
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
