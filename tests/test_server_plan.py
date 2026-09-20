"""plan 只读查询端点（B7）的测试。"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent_base.core.bootstrap import AgentRuntime
from agent_base.core.config import Settings
from agent_base.entrypoints.server import create_app
from agent_base.modules.chat.module import ChatModule
from agent_base.modules.planner.module import PlannerModule
from agent_base.modules.writer.module import WriterModule
from fakes import ScriptedChatModel


def _runtime(model: Any = None, *, with_planner: bool = True) -> AgentRuntime:
    modules: dict[str, Any] = {"chat": ChatModule(), "writer": WriterModule()}
    if with_planner:
        modules["planner"] = PlannerModule()
    return AgentRuntime(
        settings=Settings(_env_file=None, llm_api_key="sk-test"),
        llm=model if model is not None else ScriptedChatModel([]),
        modules=modules,
        tools=[],
        checkpointer=InMemorySaver(),
    )


def _client(model: Any = None, *, with_planner: bool = True) -> TestClient:
    return TestClient(create_app(runtime=_runtime(model, with_planner=with_planner)))


def _seed_plan(rt: AgentRuntime, thread_ns: str) -> None:
    """直接经 planner 图在指定线程命名空间写一份计划状态。"""

    async def seed() -> None:
        graph = rt.graph("planner")
        config = {"configurable": {"thread_id": thread_ns}}
        await graph.ainvoke(
            {"messages": [("human", "做 A")]},
            {**config, "recursion_limit": 50},
        )

    import asyncio

    asyncio.run(seed())


def test_plan_endpoint_returns_seeded_state() -> None:
    rt = _runtime(
        ScriptedChatModel(
            [
                AIMessage(content='[{"id": 1, "goal": "A"}]'),
                AIMessage(content="A 完成"),
                AIMessage(content="综合完毕"),
            ]
        )
    )
    _seed_plan(rt, "chat:t1")
    with TestClient(create_app(runtime=rt)) as client:
        response = client.get("/v1/agents/chat/threads/t1/plan")
    assert response.status_code == 200
    body = response.json()
    assert body["module"] == "chat" and body["thread_id"] == "t1"
    assert body["tasks"][0]["goal"] == "A"
    assert body["cursor"] >= 1 and body["replans"] == 0


def test_plan_endpoint_503_without_planner_module() -> None:
    with _client(with_planner=False) as client:
        response = client.get("/v1/agents/chat/threads/t1/plan")
    assert response.status_code == 503


def test_plan_endpoint_404_unknown_module() -> None:
    with _client() as client:
        response = client.get("/v1/agents/nope/threads/t1/plan")
    assert response.status_code == 404


def test_plan_endpoint_empty_defaults_for_unplanned_thread() -> None:
    # 线程从未跑过 plan（chat 写的旧线程）→ 空默认值（决策 6 读取侧）。
    with _client() as client:
        response = client.get("/v1/agents/chat/threads/fresh/plan")
    assert response.status_code == 200
    assert response.json() == {
        "thread_id": "fresh",
        "module": "chat",
        "tasks": [],
        "cursor": 0,
        "replans": 0,
    }
