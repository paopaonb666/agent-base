"""The SSE event contract (Stage 4).

Inherits A2 from the chat-agent review: the agent's progress is explicit —
step transitions, token deltas, sources, termination — as a closed set of
Pydantic models, so the wire format is validated and versionable instead
of ad-hoc strings. ``SourcesEvent`` is deliberately part of the contract
even though the base never emits it: future modules (e.g. RAG) publish
citations through it without touching the encoder.

Wire format: Server-Sent Events, one event per model::

    event: delta
    data: {"type":"delta","content":"he"}
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class StepEvent(BaseModel):
    """A named stage of the agent's run (node running / completed / error)."""

    type: Literal["step"] = "step"
    name: str
    status: Literal["running", "completed", "error"]
    detail: str | None = None


class DeltaEvent(BaseModel):
    """A token-level piece of the assistant's reply (streamed)."""

    type: Literal["delta"] = "delta"
    content: str


class Source(BaseModel):
    """One cited origin (future RAG modules publish these)."""

    title: str
    url: str | None = None


class SourcesEvent(BaseModel):
    """Citations attached to a reply (base never emits; contract reserved)."""

    type: Literal["sources"] = "sources"
    sources: list[Source] = Field(default_factory=list)


class DoneEvent(BaseModel):
    """Terminal success marker; carries the thread id for resumption."""

    type: Literal["done"] = "done"
    thread_id: str | None = None


class ErrorEvent(BaseModel):
    """Terminal failure marker with a human-readable message."""

    type: Literal["error"] = "error"
    message: str


class PingEvent(BaseModel):
    """Heartbeat keeping intermediaries from closing an idle stream."""

    type: Literal["ping"] = "ping"


AgentEvent = StepEvent | DeltaEvent | SourcesEvent | DoneEvent | ErrorEvent | PingEvent


def encode_sse(event: AgentEvent) -> str:
    """Render one contract event as a Server-Sent Events frame."""
    return f"event: {event.type}\ndata: {event.model_dump_json()}\n\n"
