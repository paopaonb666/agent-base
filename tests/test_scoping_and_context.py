"""S1（线程/附件作用域）与 H3（上下文引擎全模块生效）的回归测试。

- S1：thread_index 在位时，用户 B 无法通过任何线程端点读到/删除用户 A
  的会话；附件不能引用他人上传的 file_id（IDOR 组合链入口）。
- H3：writer/hello 经共享图工厂获得与 chat 相同的注入去重——历史里的
  注入型系统消息不再随轮次累积。
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent_base.core.bootstrap import AgentRuntime
from agent_base.core.config import Settings
from agent_base.core.context import ATTACHMENT_CONTEXT_PREFIX, MEMORY_CONTEXT_PREFIX
from agent_base.core.contracts import ModuleContext
from agent_base.entrypoints.server import create_app
from agent_base.extensions.filestore import MemoryUploadedFileStore, UploadedFileInfo
from agent_base.memory.store import MemoryMemoryStore
from agent_base.modules.chat.module import ChatModule
from agent_base.modules.writer.module import WriterModule
from fakes import ScriptedChatModel

ALICE = {"X-User-Id": "alice"}
BOB = {"X-User-Id": "bob"}


def _scoped_runtime(model: Any = None) -> AgentRuntime:
    """带 thread_index 的运行时（等价 create_runtime 的生产装配形态）。"""
    return AgentRuntime(
        settings=Settings(_env_file=None, llm_api_key="sk-test"),
        llm=model if model is not None else ScriptedChatModel([AIMessage(content="ok")]),
        modules={"chat": ChatModule()},
        tools=[],
        checkpointer=InMemorySaver(),
        thread_index=MemoryMemoryStore(),
    )


# ─────────────────────────── S1：线程作用域 ───────────────────────────


def _invoke(client: TestClient, headers: dict[str, str]) -> None:
    response = client.post("/v1/agents/chat/invoke", json={"message": "hi"}, headers=headers)
    assert response.status_code == 200, response.text


def test_threads_are_scoped_to_owner() -> None:
    runtime = _scoped_runtime()
    with TestClient(create_app(runtime=runtime)) as client:
        _invoke(client, ALICE)
        # 列表：Bob 看不到 Alice 的线程，Alice 看得到自己的。
        assert client.get("/v1/agents/chat/threads", headers=BOB).json()["threads"] == []
        alice_threads = client.get("/v1/agents/chat/threads", headers=ALICE).json()["threads"]
        assert len(alice_threads) == 1
        thread_id = alice_threads[0]["thread_id"]
        assert alice_threads[0]["title"] == "hi"
        # 历史：404 掩蔽（不泄露线程存在性）。
        assert client.get(f"/v1/agents/chat/threads/{thread_id}", headers=BOB).status_code == 404
        assert client.get(f"/v1/agents/chat/threads/{thread_id}", headers=ALICE).status_code == 200
        # 工具审计：同样仅属主。
        url = f"/v1/agents/chat/threads/{thread_id}/tool-calls"
        assert client.get(url, headers=BOB).status_code == 404
        assert client.get(url, headers=ALICE).status_code == 200
        # 删除：Bob 删不掉，Alice 能删；删除后索引行也消失。
        assert client.delete(f"/v1/agents/chat/threads/{thread_id}", headers=BOB).status_code == 404
        assert client.delete(f"/v1/agents/chat/threads/{thread_id}", headers=ALICE).json() == {
            "deleted": True
        }
        assert client.get("/v1/agents/chat/threads", headers=ALICE).json()["threads"] == []


def test_invoke_upserts_thread_index_with_owner() -> None:
    runtime = _scoped_runtime()
    with TestClient(create_app(runtime=runtime)) as client:
        _invoke(client, ALICE)
        # 直接查存储：属主与模块都落在索引上。
        rows = asyncio.run(runtime.thread_index.list_thread_index("alice", module="chat"))
        assert len(rows) == 1
        assert rows[0].user_id == "alice"
        assert rows[0].module == "chat"
        assert rows[0].thread_id.startswith("chat:")


# ─────────────────────────── S1：附件属主 ───────────────────────────


def _file_for(runtime: AgentRuntime, user_id: str, file_id: str) -> None:
    runtime.file_store = MemoryUploadedFileStore()
    info = UploadedFileInfo(
        file_id=file_id,
        filename="note.txt",
        format="txt",
        extracted_text="secret content",
        text_len=14,
        user_id=user_id,
    )
    asyncio.run(runtime.file_store.save(info))


def test_attachments_reject_other_users_files() -> None:
    runtime = _scoped_runtime()
    _file_for(runtime, "alice", "alicefile")
    with TestClient(create_app(runtime=runtime)) as client:
        # Bob 引用 Alice 的 file_id：400 拒绝（缺失与越权同文案，不给
        # 第三方探测 file_id 存在性的通道）。
        response = client.post(
            "/v1/agents/chat/invoke",
            json={"message": "hi", "thread_id": "t1", "attachments": ["alicefile"]},
            headers=BOB,
        )
        assert response.status_code == 400
        assert "附件不存在或无权访问" in response.json()["detail"]
        # 缺失的 id 同样 400。
        response = client.post(
            "/v1/agents/chat/invoke",
            json={"message": "hi", "thread_id": "t1", "attachments": ["ghost"]},
            headers=ALICE,
        )
        assert response.status_code == 400
        # 属主本人可以引用。
        response = client.post(
            "/v1/agents/chat/invoke",
            json={"message": "hi", "thread_id": "t2", "attachments": ["alicefile"]},
            headers=ALICE,
        )
        assert response.status_code == 200


# ─────────────────── H3：上下文引擎对 writer 生效 ───────────────────


def test_writer_history_does_not_accumulate_injections() -> None:
    model = ScriptedChatModel([AIMessage(content="a"), AIMessage(content="b")])
    settings = Settings(_env_file=None, llm_api_key="sk-test")
    ctx = ModuleContext(settings=settings, llm=model, checkpointer=InMemorySaver())
    graph = WriterModule().build_graph(ctx)
    config = {"configurable": {"thread_id": "chat:ctx-test"}}
    # 第一轮：注入块 + 消息（server 真实注入形态）。
    first = [SystemMessage(content=f"{MEMORY_CONTEXT_PREFIX}注入"), HumanMessage(content="one")]
    asyncio.run(graph.ainvoke({"messages": first}, config))
    # 第二轮：checkpointer 历史已含上一轮的注入块；新的模型输入必须
    # 剔除历史注入块，只保留模块提示词（不落库）。
    asyncio.run(graph.ainvoke({"messages": [HumanMessage(content="two")]}, config))
    received = model.received[-1]
    systems = [m for m in received if getattr(m, "type", "") == "system"]
    assert len(systems) == 1
    assert systems[0].content.startswith("You are a professional writing agent")
    assert all(MEMORY_CONTEXT_PREFIX not in str(m.content) for m in received)


def test_chat_trims_attachment_injections_for_all_modules() -> None:
    """附件标记（core.context）与记忆标记一样驱动去重：历史附件块不累积。"""
    model = ScriptedChatModel([AIMessage(content="a"), AIMessage(content="b")])
    settings = Settings(_env_file=None, llm_api_key="sk-test")
    ctx = ModuleContext(settings=settings, llm=model, checkpointer=InMemorySaver())
    graph = ChatModule().build_graph(ctx)
    config = {"configurable": {"thread_id": "chat:att-test"}}
    first = [
        SystemMessage(content=f"{ATTACHMENT_CONTEXT_PREFIX}：全文"),
        HumanMessage(content="one"),
    ]
    asyncio.run(graph.ainvoke({"messages": first}, config))
    asyncio.run(graph.ainvoke({"messages": [HumanMessage(content="two")]}, config))
    received = model.received[-1]
    assert all(ATTACHMENT_CONTEXT_PREFIX not in str(m.content) for m in received)
