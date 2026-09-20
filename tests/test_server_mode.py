"""invoke mode 路由（C1）的测试：chat | plan | auto。

决策 5 核心断言：mode=plan 时图选择 planner，但 thread 仍是
``{module}:{thread_id}``——计划状态落在 chat:t1 命名空间，thread_index
与记忆 agent_id 仍是 chat。
"""

from __future__ import annotations

from typing import Any

import pytest
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


def _planner_model() -> ScriptedChatModel:
    return ScriptedChatModel(
        [
            AIMessage(content='[{"id": 1, "goal": "A"}]'),
            AIMessage(content="A 完成"),
            AIMessage(content="综合完毕"),
        ]
    )


def _invoke(client: TestClient, module: str, body: dict[str, Any]) -> str:
    with client.stream(
        "POST", f"/v1/agents/{module}/invoke", json=body, headers={"X-User-Id": "u1"}
    ) as response:
        assert response.status_code == 200
        return "".join(response.iter_text())


def test_mode_plan_routes_to_planner_but_keeps_chat_namespace() -> None:
    rt = _runtime(_planner_model())
    with TestClient(create_app(runtime=rt)) as client:
        raw = _invoke(client, "chat", {"message": "做 A", "thread_id": "t1", "mode": "plan"})
    assert '"goal": "A"' in raw.replace(" ", "") or "A" in raw  # 流里有计划内容
    # 计划状态落在 chat:t1（决策 5），可由 B7 端点读出。
    import asyncio

    async def read() -> dict[str, Any]:
        graph = rt.graph("planner")
        snap = await graph.aget_state({"configurable": {"thread_id": "chat:t1"}})
        return dict(snap.values or {})

    values = asyncio.run(read())
    assert values.get("tasks") and values["tasks"][0]["goal"] == "A"


def test_mode_plan_without_planner_module_503() -> None:
    with TestClient(create_app(runtime=_runtime(with_planner=False))) as client:
        response = client.post(
            "/v1/agents/chat/invoke",
            json={"message": "hi", "mode": "plan"},
            headers={"X-User-Id": "u1"},
        )
    assert response.status_code == 503


def test_mode_auto_heuristics() -> None:
    # 短消息无附件 → chat；长消息 → plan。
    rt = _runtime(_planner_model())
    with TestClient(create_app(runtime=rt)) as client:
        short = client.post(
            "/v1/agents/chat/invoke",
            json={"message": "hi", "mode": "auto"},
            headers={"X-User-Id": "u1"},
        )
        assert short.status_code == 200
        long_message = "这是一条非常长的复杂任务描述。" * 30  # >200 字
        long = client.post(
            "/v1/agents/chat/invoke",
            json={"message": long_message, "thread_id": "big", "mode": "auto"},
            headers={"X-User-Id": "u1"},
        )
        assert long.status_code == 200

    # 长消息走的是 planner 图：chat:big 下有计划状态。
    import asyncio

    async def read() -> dict[str, Any]:
        graph = rt.graph("planner")
        snap = await graph.aget_state({"configurable": {"thread_id": "chat:big"}})
        return dict(snap.values or {})

    values = asyncio.run(read())
    assert values.get("tasks")


def test_mode_auto_without_planner_silently_degrades_to_chat() -> None:
    model = ScriptedChatModel([AIMessage(content="普通回复")])
    with TestClient(create_app(runtime=_runtime(model, with_planner=False))) as client:
        long_message = "这是一条非常长的复杂任务描述。" * 30
        response = client.post(
            "/v1/agents/chat/invoke",
            json={"message": long_message, "mode": "auto"},
            headers={"X-User-Id": "u1"},
        )
        assert response.status_code == 200
        assert "普通回复" in "".join(response.iter_text())


def test_default_mode_chat_unchanged() -> None:
    model = ScriptedChatModel([AIMessage(content="普通回复")])
    with TestClient(create_app(runtime=_runtime(model))) as client:
        response = client.post(
            "/v1/agents/chat/invoke",
            json={"message": "你好"},
            headers={"X-User-Id": "u1"},
        )
        assert response.status_code == 200
        assert "普通回复" in "".join(response.iter_text())


# ─────────────────── profile 档位分流（T1.2：Tier 1/2） ───────────────────


def test_profile_fast_routes_to_fast_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """profile="fast"：请求由快档模型应答，主力模型零消耗。"""
    from agent_base.core import bootstrap as bootstrap_module

    main_model = ScriptedChatModel([AIMessage(content="主力回复")])
    fast_model = ScriptedChatModel([AIMessage(content="快速回复")])
    settings = Settings(
        _env_file=None,
        llm_api_key="sk-test",
        llm_fast_base_url="https://fast.example.com/v1",
        llm_fast_model="glm-4.5-flash",
    )
    rt = AgentRuntime(
        settings=settings,
        llm=main_model,
        modules={"chat": ChatModule()},
        tools=[],
        checkpointer=InMemorySaver(),
    )
    monkeypatch.setattr(
        bootstrap_module,
        "build_llm",
        lambda s, profile="main": fast_model if profile == "fast" else main_model,
    )
    with TestClient(create_app(runtime=rt)) as client:
        raw = _invoke(client, "chat", {"message": "hi", "thread_id": "t1", "profile": "fast"})
    assert "快速回复" in raw
    assert main_model.received == []  # 主力模型零消耗


def test_profile_fast_unconfigured_400() -> None:
    """LLM_FAST_* 未配置时 profile="fast" 必须显式 400，不静默降级。"""
    rt = _runtime(with_planner=False)
    with TestClient(create_app(runtime=rt)) as client:
        response = client.post(
            "/v1/agents/chat/invoke",
            json={"message": "hi", "profile": "fast"},
            headers={"X-User-Id": "u1"},
        )
    assert response.status_code == 400
    assert "LLM_FAST" in response.json()["detail"]


def test_profile_default_main_unchanged() -> None:
    """profile 缺省 = main：行为与历史完全一致。"""
    model = ScriptedChatModel([AIMessage(content="主力回复")])
    with TestClient(create_app(runtime=_runtime(model, with_planner=False))) as client:
        raw = _invoke(client, "chat", {"message": "你好"})
    assert "主力回复" in raw


def test_profile_views_share_checkpointer(monkeypatch: pytest.MonkeyPatch) -> None:
    """快档视图与主力共享 checkpointer：线程历史跨档连续。"""
    from agent_base.core import bootstrap as bootstrap_module

    main_model = ScriptedChatModel([AIMessage(content="主力回复"), AIMessage(content="主力第二轮")])
    fast_model = ScriptedChatModel([AIMessage(content="快速回复")])
    settings = Settings(
        _env_file=None,
        llm_api_key="sk-test",
        llm_fast_base_url="https://fast.example.com/v1",
        llm_fast_model="glm-4.5-flash",
    )
    rt = AgentRuntime(
        settings=settings,
        llm=main_model,
        modules={"chat": ChatModule()},
        tools=[],
        checkpointer=InMemorySaver(),
    )
    monkeypatch.setattr(
        bootstrap_module,
        "build_llm",
        lambda s, profile="main": fast_model if profile == "fast" else main_model,
    )
    with TestClient(create_app(runtime=rt)) as client:
        assert "主力回复" in _invoke(client, "chat", {"message": "q1", "thread_id": "t1"})
        assert "快速回复" in _invoke(
            client, "chat", {"message": "q2", "thread_id": "t1", "profile": "fast"}
        )

    import asyncio

    async def read() -> list[str]:
        graph = rt.graph("chat")
        snap = await graph.aget_state({"configurable": {"thread_id": "chat:t1"}})
        return [str(getattr(m, "content", "")) for m in (snap.values or {}).get("messages", [])]

    contents = asyncio.run(read())
    assert any("q1" in c for c in contents) and any("q2" in c for c in contents)
