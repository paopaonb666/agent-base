"""向 MySQL checkpointer 插入演示对话数据（独立种子脚本，与基座代码解耦）。

用法::

    .venv/Scripts/python scripts/seed_mysql_demo.py

解耦原则：
- 不手写 SQL、不触碰 checkpointer 表结构——数据由 LangGraph 的
  ``AIOMySQLSaver`` 以 100% 原生格式写入（含工具调用轨迹）；
- 不改基座任何一行代码——只用基座的公开装配接口（Settings /
  build_checkpointer / build_tool_pool / AgentRuntime）；
- 不花真实 token——用 tests/fakes.py 的 ``ScriptedChatModel`` 驱动
  真实的 chat 图（这正是测试所用的替身，支持工具调用）。

写入的会话随后可通过以下方式消费（与线上数据无任何区别）：
- ``GET /v1/agents/chat/threads/{thread_id}`` 历史回放端点；
- 前端 ``http://localhost:3000/?module=chat&threadId={thread_id}``
  （本地无 transcript 时自动走 checkpointer 恢复路径）。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# 允许从仓库任意位置运行：src 已 pip install -e，tests 需要手动加入。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from langchain_core.messages import AIMessage, HumanMessage

from agent_base.core.bootstrap import AgentRuntime
from agent_base.core.config import Settings
from agent_base.core.tools import build_tool_pool
from agent_base.extensions.memory import (
    build_checkpointer,
    close_checkpointer,
)
from agent_base.modules.chat.module import ChatModule
from fakes import ScriptedChatModel

# (thread_id, [(用户消息, [该轮的 scripted 模型响应序列]), ...])
# 带工具调用的轮次消耗两个响应：先是 tool_calls，工具执行后再回 agent
# 生成最终文本——这正是 LangGraph 的 agent -> tools -> agent 往返。
CONVERSATIONS: list[tuple[str, list[tuple[str, list[AIMessage]]]]] = [
    (
        "seed-demo-weather",
        [
            (
                "帮我测试一下 echo 工具，输入用 ping",
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": "echo", "args": {"text": "ping"}, "id": "call_seed_1"}
                        ],
                    ),
                    AIMessage(content="echo 工具工作正常：它返回了 'echo: ping'。"),
                ],
            ),
            (
                "再调用一次，输入用 demo",
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": "echo", "args": {"text": "demo"}, "id": "call_seed_2"}
                        ],
                    ),
                    AIMessage(content="第二次调用也成功了：'echo: demo'。工具池状态健康。"),
                ],
            ),
        ],
    ),
    (
        "seed-demo-intro",
        [
            (
                "你是谁？",
                [
                    AIMessage(
                        content=(
                            "我是 agent-base 基座里的 chat 演示助手，基于 LangGraph 运行时构建。"
                        )
                    )
                ],
            ),
            (
                "这条对话数据是怎么来的？",
                [
                    AIMessage(
                        content="这条会话由种子脚本写入 MySQL checkpointer，"
                        "重启进程也不会丢失——你现在看到的就是持久化恢复的效果。"
                    )
                ],
            ),
        ],
    ),
]


async def seed() -> int:
    # 读 .env 的 MySQL 连接参数，仅覆盖后端选择（.env 默认是 memory）。
    settings = Settings(checkpointer_backend="mysql")
    print(
        f"connecting to MySQL checkpointer at {settings.checkpointer_mysql_host}:"
        f"{settings.checkpointer_mysql_port}, db={settings.checkpointer_mysql_database!r}"
    )
    saver = await build_checkpointer(settings)  # setup() 幂等建表

    model = ScriptedChatModel(
        [resp for _, turns in CONVERSATIONS for _, resps in turns for resp in resps]
    )
    runtime = AgentRuntime(
        settings=settings,
        llm=model,
        modules={"chat": ChatModule()},
        tools=build_tool_pool({"chat": ChatModule()}),
        checkpointer=saver,
    )
    graph = runtime.graph("chat")

    try:
        for thread_id, turns in CONVERSATIONS:
            for user_text, _ in turns:
                result = await graph.ainvoke(
                    {"messages": [HumanMessage(content=user_text)]},
                    {"configurable": {"thread_id": f"chat:{thread_id}"}},
                )
                reply = next(
                    (m.content for m in reversed(result["messages"]) if getattr(m, "content", "")),
                    "",
                )
                print(f"  [{thread_id}] {user_text!r} -> {str(reply)[:60]!r}")
        print("done: 2 conversations seeded (chat:seed-demo-weather / chat:seed-demo-intro)")
        return 0
    finally:
        await close_checkpointer(saver)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(seed()))
