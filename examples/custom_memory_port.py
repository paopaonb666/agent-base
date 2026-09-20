"""演示：在基座包外实现一个自定义 MemoryPort，并整体替换记忆实现。

演示什么
    记忆子系统对模块与基座只暴露 `MemoryPort` 协议（core/contracts.py）。
    本脚本在基座之外写一个最小实现（进程内 dict 存储），构造 AgentRuntime
    时注入，并按 server 路由的同样方式消费它——证明"换记忆后端"不需要改
    基座代码，也不需要改任何模块（依赖倒置的端口在组合根替换即可）。

怎么跑
    python examples/custom_memory_port.py
    无需 LLM API key——对话模型用 langchain-core 内置的 FakeListChatModel
    顶替，本 demo 演示的是记忆端口，不是模型调用。

预期输出
    ctx.memory 类型 : DictMemoryPort      ← 模块 build_graph 拿到的就是自定义实现
    capture_turn    : 记下 1 条事实
    compose_context : [自定义记忆-注入块] 用户提到过：我对海鲜过敏
    search(...)     : 命中 1 条
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage

from agent_base.core.bootstrap import AgentRuntime
from agent_base.core.config import Settings
from agent_base.core.registry import load_modules


def _bigrams(text: str) -> list[str]:
    """中文二元组——与基座 BM25 检索的记号化口径一致。"""
    return [text[i : i + 2] for i in range(len(text) - 1)]


class DictMemoryPort:
    """`MemoryPort` 协议的最小自定义实现：进程内 dict 存储。"""

    def __init__(self) -> None:
        self._facts: dict[str, list[str]] = {}

    async def search(
        self,
        *,
        user_id: str,
        agent_id: str | None,
        query: str,
        top_k: int | None = None,
    ) -> Any:
        grams = _bigrams(query)
        facts = self._facts.get(agent_id or "", [])
        hits = [f for f in facts if any(g in f for g in grams)][: top_k or 3]
        return {"hits": [{"text": f} for f in hits]}

    async def compose_context(
        self, *, user_id: str, agent_id: str, thread_id: str, query: str
    ) -> str | None:
        facts = self._facts.get(agent_id, [])
        if not facts:
            return None
        return "[自定义记忆-注入块] " + "；".join(facts)

    async def capture_turn(
        self, *, user_id: str, agent_id: str, thread_id: str, messages: list[Any]
    ) -> Any:
        text = str(getattr(messages[-1], "content", "")) if messages else ""
        self._facts.setdefault(agent_id, []).append(f"用户提到过：{text}")
        return {"captured": 1}

    async def aclose(self) -> None:
        return None


async def main() -> None:
    port = DictMemoryPort()
    runtime = AgentRuntime(
        settings=Settings(),
        llm=FakeListChatModel(responses=["ok"]),
        modules=load_modules(["chat"]),
        memory=port,  # ← 组合根注入；生产环境默认装配自研 MemoryService
    )
    ctx = runtime.context()
    print(f"ctx.memory 类型 : {type(ctx.memory).__name__}")

    agent_id = "chat"
    await port.capture_turn(
        user_id="demo",
        agent_id=agent_id,
        thread_id="t1",
        messages=[HumanMessage(content="我对海鲜过敏")],
    )
    block = await port.compose_context(
        user_id="demo", agent_id=agent_id, thread_id="t1", query="海鲜过敏"
    )
    hits = await port.search(user_id="demo", agent_id=agent_id, query="能吃海鲜吗")
    print("capture_turn    : 记下 1 条事实")
    print(f"compose_context : {block}")
    print(f"search(...)     : 命中 {len(hits['hits'])} 条")

    await runtime.close()


if __name__ == "__main__":
    asyncio.run(main())
