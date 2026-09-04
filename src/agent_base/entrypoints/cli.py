"""Interactive CLI entrypoint.

Usage::

    python -m agent_base --message "你好"     # one-shot
    python -m agent_base --module chat       # interactive loop
    python -m agent_base --version

Multi-turn state is held by the CLI (accumulating the message list and
passing full history each turn). Persisting it across process restarts is
the checkpointer's job (Stage 3), not this entrypoint's.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import Any, cast

from langchain_core.messages import BaseMessage, HumanMessage

from agent_base import __version__
from agent_base.core.bootstrap import AgentRuntime, create_runtime
from agent_base.core.config import SettingsError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-base",
        description="Run an agent module on the agent-base runtime (LangGraph).",
    )
    parser.add_argument(
        "--module",
        default="chat",
        help="module to converse with; must be listed in AGENT_MODULES",
    )
    parser.add_argument(
        "--message",
        "-m",
        dest="message",
        help="send a single message and exit (non-interactive)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def _invoke(
    runtime: AgentRuntime, module_name: str, messages: list[BaseMessage]
) -> list[BaseMessage]:
    graph = runtime.graph(module_name)
    result: Any = graph.invoke({"messages": messages})
    return cast(list[BaseMessage], result["messages"])


def _one_shot(runtime: AgentRuntime, module_name: str, text: str) -> None:
    updated = _invoke(runtime, module_name, [HumanMessage(content=text)])
    if updated:
        print(updated[-1].content)


def _interactive(runtime: AgentRuntime, module_name: str) -> None:
    print(f"agent-base [{module_name}] — type a message; Ctrl+C or 'quit' to exit")
    messages: list[BaseMessage] = []
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
        messages.append(HumanMessage(content=text))
        messages = _invoke(runtime, module_name, messages)
        if messages:
            print(f"agent> {messages[-1].content}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        runtime = create_runtime()
    except SettingsError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    try:
        if args.message is not None:
            _one_shot(runtime, args.module, args.message)
        else:
            _interactive(runtime, args.module)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
