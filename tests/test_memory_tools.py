"""Agent 记忆工具与文档知识库（M6e）的测试。

覆盖：chunk_text 边界、ingest/search_knowledge、四个记忆工具（含
ContextVar 作用域与 memory_update_block 的 append/replace/截断）、
bootstrap 装配、上传端点的知识库摄取。
"""

from __future__ import annotations

import asyncio
import time

import pytest
from langchain_core.tools import BaseTool

from agent_base.core.bootstrap import create_runtime
from agent_base.core.config import Settings
from agent_base.memory.service import MemoryService, chunk_text
from agent_base.memory.store import MemoryMemoryStore
from agent_base.memory.tools import build_memory_tools, set_memory_scope
from fakes import HashEmbedding


def _settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {"llm_api_key": "sk-test"}
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


def _service(**overrides: object) -> MemoryService:
    return MemoryService(
        store=MemoryMemoryStore(), embedder=HashEmbedding(), settings=_settings(**overrides)
    )


# ─────────────────────────── chunk_text ───────────────────────────


def test_chunk_text_boundaries() -> None:
    assert chunk_text("", chunk_chars=10, overlap=2) == []
    assert chunk_text("短文本", chunk_chars=10, overlap=2) == ["短文本"]
    text = "abcdefghij" * 3  # 30 字符
    parts = chunk_text(text, chunk_chars=10, overlap=2)
    # 末块可以更短，其余块满窗口。
    assert all(len(p) <= 10 for p in parts)
    assert len(parts[0]) == 10
    # 相邻块重叠 2 字符。
    assert parts[0][-2:] == parts[1][:2]
    # 覆盖完整：拼接首块 + 各块非重叠尾巴能还原原文。
    rebuilt = parts[0] + "".join(p[2:] for p in parts[1:])
    assert rebuilt == text


def test_chunk_text_overlap_must_be_smaller_than_chunk() -> None:
    # overlap == chunk_chars 会导致 step=0 死循环——Settings 校验器在
    # 配置层拦截；函数自身用 max(1, ...) 兜底不死循环。
    parts = chunk_text("x" * 25, chunk_chars=10, overlap=10)
    assert parts


# ─────────────────────────── 摄取与检索 ───────────────────────────


async def test_ingest_and_search_knowledge() -> None:
    service = _service(memory_doc_chunk_chars=50, memory_doc_chunk_overlap=10)
    text = "LangGraph 是一个用于构建有状态 Agent 的框架。" * 5
    count = await service.ingest_document(
        file_id="f1", user_id="alice", agent_id="chat", text=text
    )
    assert count > 1
    assert await service.store.count_chunks_for_file("f1") == count
    hits = await service.search_knowledge(user_id="alice", agent_id="chat", query="LangGraph 框架")
    assert hits
    assert hits[0].score > 0
    # user 隔离。
    assert await service.search_knowledge(user_id="bob", agent_id="chat", query="LangGraph") == []
    # 空文本不摄取。
    assert (
        await service.ingest_document(file_id="f2", user_id="alice", agent_id="chat", text="  ")
        == 0
    )
    ops = await service.store.list_ops()
    assert any(op.op == "ingest" and op.status == "ok" for op in ops)


async def test_ingest_failure_records_error_op(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _service()

    async def broken_embed(texts):  # type: ignore[no-untyped-def]
        raise RuntimeError("embed down")

    monkeypatch.setattr(service.embedder, "embed", broken_embed)
    count = await service.ingest_document(
        file_id="f1", user_id="u", agent_id="chat", text="一些内容"
    )
    assert count == 0
    ops = await service.store.list_ops()
    assert any(op.op == "ingest" and op.status == "error" for op in ops)


# ─────────────────────────── 记忆工具 ───────────────────────────


def _tools(**overrides: object) -> list[BaseTool]:
    return build_memory_tools(_service(**overrides))


def test_build_memory_tools_names() -> None:
    assert [t.name for t in _tools()] == [
        "memory_search",
        "memory_save",
        "memory_update_block",
        "knowledge_search",
    ]


async def test_memory_save_and_search_with_scope() -> None:
    service = _service()
    (search, save, _block, _knowledge) = build_memory_tools(service)
    set_memory_scope("alice", "chat:t1")
    saved = await save.ainvoke({"content": "用户的部署环境是 Ubuntu 22.04"})
    assert "已记住" in saved
    found = await search.ainvoke({"query": "Ubuntu 部署环境"})
    assert "Ubuntu 22.04" in found
    # 作用域切换后不可见（ContextVar 随任务传播，这里同步任务内切换）。
    set_memory_scope("bob", "chat:t2")
    missing = await search.ainvoke({"query": "Ubuntu 部署环境"})
    assert "没有找到相关记忆" in missing


async def test_memory_update_block_append_replace_and_truncation() -> None:
    service = _service()
    (_search, _save, block, _knowledge) = build_memory_tools(service)
    set_memory_scope("alice", "chat:t1")
    # append 到不存在的块 → 创建。
    out = await block.ainvoke({"label": "human", "content": "用户是工程师", "operation": "append"})
    assert "v1" in out
    # 再次 append。
    await block.ainvoke({"label": "human", "content": "偏好 Python", "operation": "append"})
    stored = await service.store.get_block("alice", "chat", "human")
    assert stored is not None
    assert stored.version == 2
    assert "用户是工程师" in stored.content and "偏好 Python" in stored.content
    # replace。
    await block.ainvoke({"label": "human", "content": "全新画像", "operation": "replace"})
    replaced = await service.store.get_block("alice", "chat", "human")
    assert replaced is not None and replaced.content == "全新画像"
    # 非法 operation。
    bad = await block.ainvoke({"label": "human", "content": "x", "operation": "delete"})
    assert "未知 operation" in bad
    # 超长截断：保尾部。
    await block.ainvoke(
        {
            "label": "persona",
            "content": "头" + "身" * 3000,
            "operation": "replace",
        }
    )
    persona = await service.store.get_block("alice", "chat", "persona")
    assert persona is not None
    assert len(persona.content) <= persona.char_limit
    assert persona.content.endswith("身")  # 保尾部


async def test_knowledge_search_tool() -> None:
    service = _service()
    await service.ingest_document(
        file_id="f1",
        user_id="alice",
        agent_id="chat",
        text="项目采用 PostgreSQL 15 存储业务数据，Redis 做缓存。",
    )
    (_search, _save, _block, knowledge) = build_memory_tools(service)
    set_memory_scope("alice", "chat:t1")
    out = await knowledge.ainvoke({"query": "PostgreSQL"})
    assert "PostgreSQL 15" in out
    set_memory_scope("bob", "chat:t2")
    out = await knowledge.ainvoke({"query": "PostgreSQL"})
    assert "没有相关内容" in out


# ─────────────────────── bootstrap 装配与上传摄取 ───────────────────────


async def test_create_runtime_includes_memory_tools(tmp_path) -> None:
    settings = _settings(
        checkpointer_backend="sqlite",
        checkpointer_sqlite_path=str(tmp_path / "state.db"),
        memory_doc_chunk_chars=50,
        memory_doc_chunk_overlap=10,
    )
    runtime = await create_runtime(settings)
    try:
        assert runtime.memory is not None
        names = {t.name for t in runtime.tools}
        assert {"memory_search", "memory_save", "memory_update_block", "knowledge_search"} <= names
        assert runtime.context().memory is runtime.memory
    finally:
        await runtime.close()


async def test_upload_endpoint_ingests_document(tmp_path) -> None:
    """上传文档后，知识库应出现分块（fire-and-forget，轮询等待）。"""
    from fastapi.testclient import TestClient
    from langgraph.checkpoint.memory import InMemorySaver

    from agent_base.core.bootstrap import AgentRuntime
    from agent_base.entrypoints.server import create_app
    from agent_base.modules.chat.module import ChatModule
    from fakes import ScriptedChatModel

    settings = _settings()
    runtime = AgentRuntime(
        settings=settings,
        llm=ScriptedChatModel([]),
        modules={"chat": ChatModule()},
        tools=[],
        checkpointer=InMemorySaver(),
    )
    runtime.memory = MemoryService(
        store=MemoryMemoryStore(), embedder=HashEmbedding(), settings=settings
    )
    from agent_base.extensions.filestore import MemoryUploadedFileStore

    runtime.file_store = MemoryUploadedFileStore()
    content = "项目采用 PostgreSQL 15 存储业务数据。" * 20
    with TestClient(create_app(runtime=runtime)) as client:
        response = client.post(
            "/v1/agents/chat/files",
            files={"file": ("notes.txt", content.encode("utf-8"), "text/plain")},
        )
        assert response.status_code == 200
        file_id = response.json()["file_id"]
        service = client.app.state.runtime.memory
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if await service.store.count_chunks_for_file(file_id) > 0:
                break
            await asyncio.sleep(0.05)
        assert await service.store.count_chunks_for_file(file_id) > 0, "知识库摄取未生效"
        await service.aclose()
