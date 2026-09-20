"""交互式 CLI 入口。

用法::

    python -m agent_base --message "你好"           # 单轮
    python -m agent_base --module chat             # 交互式循环
    python -m agent_base --module supervisor       # 多 Agent 会话（阶段 4）
    python -m agent_base --thread-id <id>          # 恢复一次对话
    python -m agent_base --version

对话状态（阶段 3）：每个会话都在一个 ``thread_id`` 下运行，checkpointer
（默认 memory，通过 CHECKPOINTER_BACKEND 用 sqlite）按 thread 持久化
历史。传入 ``--thread-id`` 可恢复之前的对话——使用 sqlite 后端时，
进程重启后依然存在。每一轮只发送**新**的消息；历史从 checkpointer
回放，CLI 从不重发。

可观测性（阶段 2）：每个会话都在一个全新的 ``request_id`` 下运行；
会话期间发出的所有日志都会带上它。

会话运行在 ``asyncio.run`` 中，因为运行时是异步优先的（sqlite
checkpointer 和真正的 LLM 调用取消都需要它）。
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import uuid
from collections.abc import Sequence
from contextlib import suppress
from typing import Any, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from agent_base import __version__
from agent_base.core.bootstrap import (
    SUPERVISOR_MODULE,
    AgentRuntime,
    UnknownModuleError,
    create_runtime,
)
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
    """发送一轮用户消息；其余历史由 checkpointer 回放。"""
    graph = runtime.graph(module_name)
    # thread id 按模块划分命名空间：不同的图共用一个 checkpointer，
    # 未划分命名空间的 id 会把它们的状态混在一起。
    config: RunnableConfig = {
        "recursion_limit": runtime.settings.agent_recursion_limit,
        "configurable": {"thread_id": f"{module_name}:{thread_id}"},
    }
    result: Any = await graph.ainvoke({"messages": [HumanMessage(content=text)]}, config)
    return cast(list[BaseMessage], result["messages"])


def _reply(messages: list[BaseMessage]) -> str:
    """面向人类的回复：最后一条 assistant 消息（如果有的话）。"""
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

    setup_logging(json_lines=runtime.settings.observability.log_json)
    log_tracing_config()
    thread_id = args.thread_id or uuid.uuid4().hex[:12]
    try:
        with request_id():
            logger.info("cli: start module=%s thread_id=%s", args.module, thread_id)
            try:
                if args.message is not None:
                    await _one_shot(runtime, args.module, thread_id, args.message)
                else:
                    await _interactive(runtime, args.module, thread_id)
            except UnknownModuleError as exc:
                # 只接住"模块不存在"；图内部真正的 KeyError 不该被吞成这条文案。
                print(f"error: {exc}", file=sys.stderr)
                return 2
            finally:
                # 始终关闭 trace——即便 LLM 调用爆炸了——这样一次会话在同一个
                # request_id 下就有一个确定的结束标记。
                logger.info("cli: done request_id=%s", get_request_id())
    finally:
        # 任何退出路径（含 UnknownModuleError 的 return 2 与 KeyboardInterrupt
        # 中断）都要释放 checkpointer 连接。
        with suppress(Exception):
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
        # 入口以一条消息失败，而不是堆栈转储（会话日志已经携带 request_id
        # 可供调试）。
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
