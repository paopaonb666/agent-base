"""敏感工具审批门（ADR-005）的测试：名单解析、interrupt 暂停、三分支恢复、
HTTP resume 端点、未启用零行为变化。"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agent_base.core.bootstrap import AgentRuntime
from agent_base.core.config import Settings
from agent_base.core.contracts import ModuleContext
from agent_base.core.graphs import build_single_agent_graph
from agent_base.entrypoints.server import create_app
from agent_base.modules.chat.module import ChatModule
from agent_base.tools.registry import resolve_approval_names
from fakes import ScriptedChatModel


def _settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {"llm_api_key": "sk-test"}
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


@tool
def risky_tool(x: int) -> str:
    """测试用的可审批工具。"""
    return f"executed:{x}"


# ─────────────────────────── 名单解析 ───────────────────────────


def test_resolve_approval_names_disabled() -> None:
    assert resolve_approval_names(_settings()) == frozenset()
    assert (
        resolve_approval_names(_settings(tool_approval_tools="risky_tool")) == frozenset()
    )  # 未启用时名单不生效


def test_resolve_approval_names_tools_and_safety() -> None:
    names = resolve_approval_names(
        _settings(
            tool_approval_enabled=True,
            tool_approval_tools="risky_tool,mcp_github_create_pr",
        )
    )
    assert names == frozenset({"risky_tool", "mcp_github_create_pr"})
    # safety 级放大：exec 级的内置工具（python_repl）整级进名单。
    with_exec = resolve_approval_names(
        _settings(tool_approval_enabled=True, tool_approval_safety_levels="exec")
    )
    assert "python_repl" in with_exec


def test_safety_levels_validation() -> None:
    with pytest.raises(ValueError, match="unknown levels"):
        _settings(tool_approval_enabled=True, tool_approval_safety_levels="nuclear")


# ─────────────────────────── 图级行为 ───────────────────────────


def _ctx(model: Any, settings: Settings, tools: list[Any]) -> ModuleContext:
    return ModuleContext(settings=settings, llm=model, checkpointer=InMemorySaver(), tools=tools)


def _tool_call_msg() -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "risky_tool", "args": {"x": 1}, "id": "call_1", "type": "tool_call"}],
    )


async def _collect(stream: Any) -> list[Any]:
    return [payload async for _, payload in stream]


async def test_gate_interrupts_and_resumes_approve() -> None:
    model = ScriptedChatModel([_tool_call_msg(), AIMessage(content="已执行")])
    settings = _settings(tool_approval_enabled=True, tool_approval_tools="risky_tool")
    graph = build_single_agent_graph(_ctx(model, settings, [risky_tool]), name="t")
    cfg: dict[str, Any] = {"configurable": {"thread_id": "t1"}}
    updates = await _collect(
        graph.astream({"messages": [HumanMessage(content="go")]}, cfg, stream_mode=["updates"])
    )
    assert any("__interrupt__" in u for u in updates)
    # 暂停态在快照里可见（resume 端点据此 409 判定）。
    snapshot = await graph.aget_state(cfg)
    assert any(getattr(task, "interrupts", None) for task in snapshot.tasks)
    # approve 恢复：工具执行，模型看到结果后正常收尾。
    updates = await _collect(
        graph.astream(
            Command(resume={"call_1": {"action": "approve"}}), cfg, stream_mode=["updates"]
        )
    )
    assert any("tools" in u for u in updates)
    snapshot = await graph.aget_state(cfg)
    messages = snapshot.values["messages"]
    assert any(
        getattr(m, "type", "") == "tool" and "executed:1" in str(m.content) for m in messages
    )
    assert messages[-1].content == "已执行"


async def test_gate_reject_injects_denial() -> None:
    model = ScriptedChatModel([_tool_call_msg(), AIMessage(content="好的，不执行")])
    settings = _settings(tool_approval_enabled=True, tool_approval_tools="risky_tool")
    graph = build_single_agent_graph(_ctx(model, settings, [risky_tool]), name="t")
    cfg: dict[str, Any] = {"configurable": {"thread_id": "t2"}}
    await _collect(
        graph.astream({"messages": [HumanMessage(content="go")]}, cfg, stream_mode=["updates"])
    )
    await _collect(
        graph.astream(
            Command(resume={"call_1": {"action": "reject"}}), cfg, stream_mode=["updates"]
        )
    )
    snapshot = await graph.aget_state(cfg)
    tool_msgs = [m for m in snapshot.values["messages"] if getattr(m, "type", "") == "tool"]
    assert len(tool_msgs) == 1
    assert "拒绝" in str(tool_msgs[0].content)


async def test_gate_modify_replaces_args() -> None:
    model = ScriptedChatModel([_tool_call_msg(), AIMessage(content="收到")])
    settings = _settings(tool_approval_enabled=True, tool_approval_tools="risky_tool")
    graph = build_single_agent_graph(_ctx(model, settings, [risky_tool]), name="t")
    cfg: dict[str, Any] = {"configurable": {"thread_id": "t3"}}
    await _collect(
        graph.astream({"messages": [HumanMessage(content="go")]}, cfg, stream_mode=["updates"])
    )
    await _collect(
        graph.astream(
            Command(resume={"call_1": {"action": "modify", "args": {"x": 99}}}),
            cfg,
            stream_mode=["updates"],
        )
    )
    snapshot = await graph.aget_state(cfg)
    tool_msgs = [m for m in snapshot.values["messages"] if getattr(m, "type", "") == "tool"]
    assert "executed:99" in str(tool_msgs[0].content)


async def test_gate_passes_through_unlisted_tools() -> None:
    """名单外的工具调用不 interrupt，直接执行（ToolNode 同语义）。"""
    model = ScriptedChatModel([_tool_call_msg(), AIMessage(content="完成")])
    settings = _settings(tool_approval_enabled=True, tool_approval_tools="other_tool")
    graph = build_single_agent_graph(_ctx(model, settings, [risky_tool]), name="t")
    cfg: dict[str, Any] = {"configurable": {"thread_id": "t4"}}
    updates = await _collect(
        graph.astream({"messages": [HumanMessage(content="go")]}, cfg, stream_mode=["updates"])
    )
    assert not any("__interrupt__" in u for u in updates)
    snapshot = await graph.aget_state(cfg)
    assert any(
        getattr(m, "type", "") == "tool" and "executed:1" in str(m.content)
        for m in snapshot.values["messages"]
    )


async def test_gate_disabled_uses_tool_node() -> None:
    """未启用（默认）：图保持 ToolNode 原状，零行为变化。"""
    model = ScriptedChatModel([_tool_call_msg(), AIMessage(content="完成")])
    graph = build_single_agent_graph(_ctx(model, _settings(), [risky_tool]), name="t")
    cfg: dict[str, Any] = {"configurable": {"thread_id": "t5"}}
    updates = await _collect(
        graph.astream({"messages": [HumanMessage(content="go")]}, cfg, stream_mode=["updates"])
    )
    assert not any("__interrupt__" in u for u in updates)
    snapshot = await graph.aget_state(cfg)
    assert any(
        getattr(m, "type", "") == "tool" and "executed:1" in str(m.content)
        for m in snapshot.values["messages"]
    )


# ─────────────────────────── HTTP 端到端 ───────────────────────────


def _frames(raw: str) -> list[dict[str, Any]]:
    events = []
    for block in raw.strip().split("\n\n"):
        lines = block.split("\n")
        etype = lines[0].removeprefix("event: ")
        events.append({"event": etype, **json.loads(lines[1].removeprefix("data: "))})
    return events


def _approval_client(model: Any) -> TestClient:
    runtime = AgentRuntime(
        settings=_settings(tool_approval_enabled=True, tool_approval_tools="risky_tool"),
        llm=model,
        modules={"chat": ChatModule()},
        tools=[risky_tool],
        checkpointer=InMemorySaver(),
    )
    return TestClient(create_app(runtime=runtime))


def test_http_approval_flow_approve() -> None:
    model = ScriptedChatModel([_tool_call_msg(), AIMessage(content="已执行")])
    with _approval_client(model) as client:
        # 第一段：invoke → approval_request + done（流暂停待审批）。
        first = client.post("/v1/agents/chat/invoke", json={"message": "go"})
        assert first.status_code == 200
        events = _frames(first.text)
        approvals = [e for e in events if e["event"] == "approval_request"]
        assert len(approvals) == 1
        assert approvals[0]["thread_id"]
        assert approvals[0]["requests"][0]["tool"] == "risky_tool"
        assert approvals[0]["requests"][0]["tool_call_id"] == "call_1"
        # 第二段：resume(approve) → 工具执行 → 模型收尾 delta + done。
        thread_id = approvals[0]["thread_id"]
        second = client.post(
            f"/v1/agents/chat/threads/{thread_id}/resume",
            json={"decisions": {"call_1": {"action": "approve"}}},
        )
        assert second.status_code == 200
        events2 = _frames(second.text)
        assert not [e for e in events2 if e["event"] == "approval_request"]
        assert any(e["event"] == "delta" and "已执行" in e["content"] for e in events2)
        assert events2[-1]["event"] == "done"


def test_http_resume_without_pending_409() -> None:
    model = ScriptedChatModel([AIMessage(content="没有工具调用")])
    with _approval_client(model) as client:
        first = client.post("/v1/agents/chat/invoke", json={"message": "hi", "thread_id": "s1"})
        assert first.status_code == 200
        second = client.post(
            "/v1/agents/chat/threads/s1/resume",
            json={"decisions": {"call_1": {"action": "approve"}}},
        )
        assert second.status_code == 409


def test_http_invoke_without_approval_config_unchanged() -> None:
    """默认配置（未启用审批）下 invoke 行为不变：无 approval_request 事件。"""
    model = ScriptedChatModel([_tool_call_msg(), AIMessage(content="完成")])
    runtime = AgentRuntime(
        settings=_settings(),
        llm=model,
        modules={"chat": ChatModule()},
        tools=[risky_tool],
        checkpointer=InMemorySaver(),
    )
    with TestClient(create_app(runtime=runtime)) as client:
        first = client.post("/v1/agents/chat/invoke", json={"message": "go"})
        events = _frames(first.text)
        assert not [e for e in events if e["event"] == "approval_request"]
        assert any(e["event"] == "delta" and "完成" in e["content"] for e in events)
