"""Observability foundation: request_id propagation + structured logging.

Stage 2 deliverable. Inherits A1 from the chat-agent review: the request ID
lives in a ``contextvars.ContextVar``, every log record is enriched with it
by a logging ``Filter``, and the formatter renders human-readable lines
(development) or JSON (production). Nothing here is coupled to a specific
entrypoint -- CLI, then FastAPI later, both call into this module.

Tracing (LangSmith / Langfuse) stays env-gated: this module never reads or
hardcodes a concrete vendor setting beyond what the environment provides.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

# The ambient request identifier. Modules / nodes / entrypoints read it via
# ``get_request_id()``; a fresh value is generated per top-level invocation.
_REQUEST_ID: ContextVar[str] = ContextVar("request_id", default="")

# Logger name used for trace configuration messages.
_CONFIG_LOGGER = "agent_base.tracing"

# Name of the stream handler this module installs, so repeated setup calls
# (tests, reloads) replace rather than duplicate handlers.
_HANDLER_NAME = "agent_base.console"

_TRUTHY = {"1", "true", "yes", "on"}


def _inject_request_id(record: logging.LogRecord) -> None:
    record.request_id = get_request_id() or "-"


class RequestIdFilter(logging.Filter):
    """Attach the ambient ``request_id`` to every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        _inject_request_id(record)
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line: machine-parseable structured logs."""

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
    """Readable console line for local development."""

    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s")


def get_request_id() -> str:
    """Return the ambient request id ("" when none is set)."""
    return _REQUEST_ID.get()


@contextmanager
def request_id(rid: str | None = None) -> Iterator[str]:
    """Run a block under a request id, generating one when not provided."""
    token = _REQUEST_ID.set(rid or new_request_id())
    try:
        yield _REQUEST_ID.get()
    finally:
        _REQUEST_ID.reset(token)


def new_request_id() -> str:
    """A short, unique, sortable request id."""
    return f"{uuid.uuid4().hex[:12]}"


def setup_logging(*, json_lines: bool = False, level: int = logging.INFO) -> None:
    """Configure the root logger once: request-id filter + stream handler.

    ``json_lines`` selects structured JSON output; the default human format
    is for local development. Idempotent: repeated calls reuse the named
    handler, so tests can re-configure without stacking handlers.

    The filter is attached to the *handler*, not the root logger:
    ``Logger.filter`` consults only the logger's OWN filters, never its
    ancestors', so a root-level filter would silently miss records emitted
    through other loggers (langchain uses ``logging`` heavily).
    """
    root = logging.getLogger()
    root.setLevel(level)
    handler = next((h for h in root.handlers if h.get_name() == _HANDLER_NAME), None)
    if handler is None:
        handler = logging.StreamHandler()
        handler.set_name(_HANDLER_NAME)
        root.addHandler(handler)
    handler.setFormatter(JsonFormatter() if json_lines else HumanFormatter())
    handler.filters = [RequestIdFilter()]  # replace, keep setup idempotent


def log_tracing_config() -> None:
    """Report which tracing providers the environment enables (env-gated).

    LangSmith is consumed natively by langchain through standard env vars
    (``LANGSMITH_TRACING`` / ``LANGSMITH_API_KEY``); Langfuse the same way.
    The base does not import vendor SDKs -- it only surfaces what the
    environment already decided.
    """
    logger = logging.getLogger(_CONFIG_LOGGER)
    env = os.environ
    langsmith = _env_flag(env.get("LANGSMITH_TRACING")) or bool(env.get("LANGSMITH_API_KEY"))
    langfuse = bool(env.get("LANGFUSE_PUBLIC_KEY") and env.get("LANGFUSE_SECRET_KEY"))
    logger.info(
        "tracing: langsmith=%s langfuse=%s "
        "(set LANGSMITH_TRACING=true / LANGFUSE_* keys to enable)",
        "on" if langsmith else "off",
        "on" if langfuse else "off",
    )


def _env_flag(value: str | None) -> bool:
    return (value or "").strip().lower() in _TRUTHY
