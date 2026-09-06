"""Interactive CLI entrypoint.

Usage::

    python -m agent_base --message "你好"           # one-shot
    python -m agent_base --module chat             # interactive loop
    python -m agent_base --module supervisor       # multi-agent session (Stage 4)
    python -m agent_base --thread-id <id>          # resume a conversation
    python -m agent_base --version

Conversation state (Stage 3): every session runs under a ``thread_id`` and
the checkpointer (memory by default, sqlite via CHECKPOINTER_BACKEND)
persists history per thread. Pass ``--thread-id`` to resume a previous
conversation — with the sqlite backend this survives process restarts.
Each turn sends only the NEW message; history is replayed from the
checkpointer, never re-sent by the CLI.

Observability (Stage 2): each session runs under a fresh ``request_id``;
all logs emitted during the session carry it.

The session runs inside ``asyncio.run`` because the runtime is async-first
(the sqlite checkpointer and true LLM-call cancellation both require it).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import uuid
from collections.abc import Sequence
from typing import Any, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from agent_base import __version__
from agent_base.core.bootstrap import SUPERVISOR_MODULE, AgentRuntime, create_runtime
from agent_base.core.config import SettingsError
from agent_base.extensions.observability import (
    get_request_id,
    log_tracing_config,
    request_id,
    setup_logging,
)

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-base",
        description="Run an agent module on the agent-base runtime (LangGraph).",
    )
    parser.add_argument(
        "--module",
        default="chat",
        help="module to converse with; must be listed in AGENT_MODULES "
        f"(or '{SUPERVISOR_MODULE}' for the multi-agent session)",
    )
    parser.add_argument(
        "--message",
        "-m",
        dest="message",
        help="send a single message and exit (non-interactive)",
    )
    parser.add_argument(
        "--thread-id",
        dest="thread_id",
        help="resume a previous conversation (persisted by the checkpointer; "
        "survives process restarts with CHECKPOINTER_BACKEND=sqlite)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


async def _invoke(
    runtime: AgentRuntime, module_name: str, thread_id: str, text: str
) -> list[BaseMessage]:
    """Send one user turn; the checkpointer replays the rest of the history."""
    graph = runtime.graph(module_name)
    # The thread id is namespaced per module: different graphs share one
    # checkpointer, and an un-namespaced id would mix their states.
    config: RunnableConfig = {"configurable": {"thread_id": f"{module_name}:{thread_id}"}}
    result: Any = await graph.ainvoke({"messages": [HumanMessage(content=text)]}, config)
    return cast(list[BaseMessage], result["messages"])


def _reply(messages: list[BaseMessage]) -> str:
    """The human-facing reply: the last assistant message, if any."""
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.content:
            return str(message.content)
    return ""


async def _one_shot(runtime: AgentRuntime, module_name: str, thread_id: str, text: str) -> None:
    updated = await _invoke(runtime, module_name, thread_id, text)
    reply = _reply(updated)
    if reply:
        print(reply)
    print(f"[thread_id: {thread_id}]", file=sys.stderr)


async def _interactive(runtime: AgentRuntime, module_name: str, thread_id: str) -> None:
    print(f"agent-base [{module_name}] — type a message; Ctrl+C or 'quit' to exit")
    print(f"thread_id: {thread_id} (pass --thread-id {thread_id} to resume)")
    while True:
        try:
            line = input("you> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        text = line.strip()
        if not text:
            continue
        if text.lower() in {"quit", "exit"}:
            return
        messages = await _invoke(runtime, module_name, thread_id, text)
        reply = _reply(messages)
        if reply:
            print(f"agent> {reply}")


async def _run(args: argparse.Namespace) -> int:
    try:
        runtime = await create_runtime()
    except SettingsError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    setup_logging(json_lines=runtime.settings.log_json)
    log_tracing_config()
    thread_id = args.thread_id or uuid.uuid4().hex[:12]
    with request_id():
        logger.info("cli: start module=%s thread_id=%s", args.module, thread_id)
        try:
            if args.message is not None:
                await _one_shot(runtime, args.module, thread_id, args.message)
            else:
                await _interactive(runtime, args.module, thread_id)
        except KeyError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        finally:
            # Always close the trace — even when the LLM call blew up — so a
            # session has a definitive end marker under one request_id.
            logger.info("cli: done request_id=%s", get_request_id())
    await runtime.close()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        # Entry points fail with a message, not a stack dump (the session
        # log already carries the request_id for debugging).
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
