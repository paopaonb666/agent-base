"""Conversation-state assembly: the checkpointer (Stage 3).

SHORT-TERM dialogue state only — thread-level persistence so a conversation
survives across turns (and, with the sqlite backend, across process
restarts). Long-term memory (cross-conversation, vector stores, extraction)
is out of scope for the base by design (scope red line) and will arrive as
a future module.

Backends (``CHECKPOINTER_BACKEND``):
- ``memory``  — ``InMemorySaver``, zero dependencies, process-lifetime only.
- ``sqlite``  — ``AsyncSqliteSaver`` over ``CHECKPOINTER_SQLITE_PATH``.

The whole runtime is async-first (CLI wraps its session in ``asyncio.run``,
the server runs under uvicorn's loop), so the sqlite backend uses the async
saver: the sync ``SqliteSaver`` raises ``NotImplementedError`` under
``astream``. ``InMemorySaver`` supports both modes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

if TYPE_CHECKING:  # pragma: no cover - import avoided at runtime
    from agent_base.core.config import Settings


async def build_checkpointer(settings: Settings) -> BaseCheckpointSaver[Any]:
    """Assemble the checkpointer selected by ``settings.checkpointer_backend``.

    Must be awaited inside the event loop that will run the graphs — the
    sqlite saver binds an aiosqlite connection and an ``asyncio.Lock`` to
    the calling loop, so constructing it in a throwaway loop would break
    later use. ``create_runtime`` is async for exactly this reason.
    """
    if settings.checkpointer_backend == "memory":
        return InMemorySaver()
    # Deferred import: aiosqlite is only needed when sqlite is selected.
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    # NOTE: deliberately NOT AsyncSqliteSaver.from_conn_string — its context
    # manager closes the connection as soon as it is garbage collected,
    # which would silently kill the saver mid-session. A directly awaited
    # connection lives until close_checkpointer() (or process exit).
    conn = await aiosqlite.connect(settings.checkpointer_sqlite_path)
    saver = AsyncSqliteSaver(conn)
    await saver.setup()
    return saver


async def close_checkpointer(checkpointer: BaseCheckpointSaver[Any]) -> None:
    """Best-effort teardown (tests create many savers; processes can skip)."""
    conn = getattr(checkpointer, "conn", None)
    closer = getattr(conn, "close", None)
    if closer is not None:
        result = closer()
        if hasattr(result, "__await__"):
            await result
