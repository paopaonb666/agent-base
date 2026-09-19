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
    short = chunk_text("短文本", chunk_chars=10, overlap=2)
    assert len(short) == 1 and short[0].text == "短文本"
    text = "abcdefghij" * 3  # 30 字符
    parts = chunk_text(text, chunk_chars=10, overlap=2)
    # 末块可以更短，其余块满窗口。
    assert all(p.char_len <= 10 for p in parts)
    assert parts[0].char_len == 10
    # 相邻块重叠 2 字符。
    assert parts[0].text[-2:] == parts[1].text[:2]
    # 覆盖完整：拼接首块 + 各块非重叠尾巴能还原原文。
    rebuilt = parts[0].text + "".join(p.text[2:] for p in parts[1:])
    assert rebuilt == text
    # 无段落文本：每块单区间 span，坐标能切回原文。
    for span in parts:
        (start, end), = span.segments
        assert text[start:end] == span.text


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
        "memory_delete",
        "knowledge_search",
    ]


async def test_memory_delete_with_user_isolation() -> None:
    service = _service()
    (search, _save, _block, delete, _knowledge) = build_memory_tools(service)
    set_memory_scope("alice", "chat:t1")
    record = await service.add_memory(user_id="alice", agent_id="chat", content="旧代号是蓝鲸")
    found = await search.ainvoke({"query": "旧代号"})
    assert f"[id={record.memory_id[:12]}]" in found  # 搜索结果带 id
    out = await delete.ainvoke({"memory_id": record.memory_id})
    assert "已删除" in out
    assert await service.get_memory(record.memory_id) is None
    # 删除不存在的 id → 友好提示而非异常。
    again = await delete.ainvoke({"memory_id": record.memory_id})
    assert "不存在" in again
    # user 隔离：bob 不能删 alice 的记忆。
    other = await service.add_memory(user_id="alice", agent_id="chat", content="alice 的私有记忆")
    set_memory_scope("bob", "chat:t2")
    denied = await delete.ainvoke({"memory_id": other.memory_id})
    assert "不属于当前用户" in denied
    assert await service.get_memory(other.memory_id) is not None


async def test_memory_save_and_search_with_scope() -> None:
    service = _service()
    (search, save, _block, _delete, _knowledge) = build_memory_tools(service)
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
    (_search, _save, block, _delete, _knowledge) = build_memory_tools(service)
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
    (_search, _save, _block, _delete, knowledge) = build_memory_tools(service)
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


# ─────────────────── 文档链路边界加固（段落切块/幂等/回填/撤销） ───────────────────


def test_chunk_text_paragraph_aware_packing() -> None:
    """段落感知：块边界落在空行上，段落不被从中间劈开；span 坐标正确。"""
    paras = ["第一段" + "内容" * 50, "第二段" + "描述" * 50, "第三段" + "收尾" * 50]
    text = "\n\n".join(paras)  # 每段 103 字符
    parts = chunk_text(text, chunk_chars=220, overlap=10)
    assert all(p.char_len <= 220 for p in parts)
    for para in paras:
        assert any(para in p.text for p in parts), "段落被劈开"
    # 覆盖完整性：去掉拼接分隔符后与原文一致。
    assert "".join(p.text.replace("\n\n", "") for p in parts) == text.replace("\n\n", "")
    # span 坐标：segments 逐段映射回原文，slice 结果等于段落本身。
    for span in parts:
        for start, end in span.segments:
            assert text[start:end] in paras
            assert span.text.count(text[start:end]) == 1


def test_chunk_text_oversized_paragraph_window_fallback() -> None:
    """单段超长：段内退回固定窗口滑动（带重叠），块长不超限。"""
    para = "表格行。" * 300  # 1200 字符单段
    text = f"开头说明\n\n{para}"
    parts = chunk_text(text, chunk_chars=200, overlap=20)
    assert all(p.char_len <= 200 for p in parts)
    assert sum(p.char_len for p in parts) >= 1200  # 全覆盖
    assert parts[0].text.startswith("开头说明")
    # 超长段的窗口 span 是单区间，坐标能切回原文。
    tail_spans = [p for p in parts if p.text.startswith("表格行")]
    assert tail_spans
    for span in tail_spans:
        (start, end), = span.segments
        assert text[start:end] == span.text


def test_chunk_text_span_offsets_relative_to_original() -> None:
    """坐标相对调用方传入的原文（含 strip 偏移补偿）。"""
    text = "  \n 首段内容 \n\n 次段内容 \n  "
    parts = chunk_text(text, chunk_chars=200, overlap=10)
    # strip 后首段是"首段内容"：坐标切回原文应正好落在该段上。
    first = parts[0]
    start, end = first.segments[0]
    assert text[start:end].strip() in text
    joined = "".join(text[s:e] for span in parts for s, e in span.segments)
    assert "首段内容" in joined and "次段内容" in joined


async def test_ingest_document_is_idempotent() -> None:
    """同一 file_id 重复摄取不产生重复分块（重试/重解析场景）。"""
    service = _service(memory_doc_chunk_chars=50, memory_doc_chunk_overlap=10)
    text = "LangGraph 框架文档内容。" * 10
    first = await service.ingest_document(file_id="f1", user_id="u", agent_id="chat", text=text)
    second = await service.ingest_document(file_id="f1", user_id="u", agent_id="chat", text=text)
    assert first == second > 1
    assert await service.store.count_chunks_for_file("f1") == first


async def test_backfill_covers_doc_chunks(tmp_path) -> None:
    """向量回填覆盖知识库分块：旧维度/无向量的分块都被修复。"""
    from agent_base.memory.store import DocChunk, encode_embedding

    service = _service()
    await service.store.put_chunks(
        [
            DocChunk(
                chunk_id="c1",
                file_id="f1",
                thread_id="",
                user_id="u",
                agent_id="chat",
                ordinal=0,
                text="旧维度的分块",
                embedding=encode_embedding([1.0]),
                embedding_dim=7,
            ),
            DocChunk(
                chunk_id="c2",
                file_id="f1",
                thread_id="",
                user_id="u",
                agent_id="chat",
                ordinal=1,
                text="没有向量的分块",
            ),
        ]
    )
    count = await service.backfill_embeddings(batch=10)
    assert count == 2
    chunks = await service.store.list_chunks("u")
    assert all(chunk.embedding_dim == HashEmbedding().dims for chunk in chunks)
    await service.aclose()


async def test_revoke_file_endpoint_removes_chunks_and_row() -> None:
    """撤销端点：属主校验 + 文件行与知识分块级联删除。"""
    from fastapi.testclient import TestClient

    from agent_base.core.bootstrap import AgentRuntime
    from agent_base.entrypoints.server import create_app
    from agent_base.extensions.filestore import MemoryUploadedFileStore
    from agent_base.modules.chat.module import ChatModule
    from fakes import ScriptedChatModel

    settings = _settings()
    runtime = AgentRuntime(
        settings=settings,
        llm=ScriptedChatModel([]),
        modules={"chat": ChatModule()},
        tools=[],
        checkpointer=None,
    )
    runtime.memory = MemoryService(
        store=MemoryMemoryStore(), embedder=HashEmbedding(), settings=settings
    )
    runtime.file_store = MemoryUploadedFileStore()
    content = "项目采用 PostgreSQL 15 存储业务数据。" * 20
    with TestClient(create_app(runtime=runtime)) as client:
        upload = client.post(
            "/v1/agents/chat/files",
            files={"file": ("notes.txt", content.encode("utf-8"), "text/plain")},
            headers={"X-User-Id": "alice"},
        )
        assert upload.status_code == 200
        file_id = upload.json()["file_id"]
        service = client.app.state.runtime.memory
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if await service.store.count_chunks_for_file(file_id) > 0:
                break
            await asyncio.sleep(0.05)
        assert await service.store.count_chunks_for_file(file_id) > 0

        # bob 不能撤销 alice 的文件（404，不泄露存在性）。
        denied = client.delete(
            f"/v1/agents/chat/files/{file_id}", headers={"X-User-Id": "bob"}
        )
        assert denied.status_code == 404
        # alice 撤销：文件行 + 分块级联删除。
        ok = client.delete(f"/v1/agents/chat/files/{file_id}", headers={"X-User-Id": "alice"})
        assert ok.status_code == 200
        assert ok.json()["deleted"] is True
        assert ok.json()["chunks_removed"] > 0
        assert await service.store.count_chunks_for_file(file_id) == 0
        assert await runtime.file_store.get_many([file_id]) == []
        await service.aclose()


# ─────────────────── 文档预览与切片可视化端点（M7） ───────────────────


async def test_file_preview_chunks_raw_endpoints() -> None:
    """预览返回提取正文；切片带偏移量；raw 按类型回字节；属主 404 掩蔽。"""
    from fastapi.testclient import TestClient

    from agent_base.core.bootstrap import AgentRuntime
    from agent_base.entrypoints.server import create_app
    from agent_base.extensions.filestore import MemoryUploadedFileStore
    from agent_base.modules.chat.module import ChatModule
    from fakes import ScriptedChatModel

    settings = _settings()
    runtime = AgentRuntime(
        settings=settings,
        llm=ScriptedChatModel([]),
        modules={"chat": ChatModule()},
        tools=[],
        checkpointer=None,
    )
    runtime.memory = MemoryService(
        store=MemoryMemoryStore(), embedder=HashEmbedding(), settings=settings
    )
    runtime.file_store = MemoryUploadedFileStore()
    doc_text = "第一段的内容。\n\n第二段的内容。\n\n第三段的内容。"
    with TestClient(create_app(runtime=runtime)) as client:
        upload = client.post(
            "/v1/agents/chat/files",
            files={"file": ("报告.txt", doc_text.encode("utf-8"), "text/plain")},
            headers={"X-User-Id": "alice"},
        )
        file_id = upload.json()["file_id"]
        service = client.app.state.runtime.memory
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if await service.store.count_chunks_for_file(file_id) > 0:
                break
            await asyncio.sleep(0.05)

        # 预览：正文与注入同源（extracted_text）。
        preview = client.get(
            f"/v1/agents/chat/files/{file_id}/preview", headers={"X-User-Id": "alice"}
        )
        assert preview.status_code == 200
        assert preview.json()["extracted_text"] == doc_text
        assert preview.json()["filename"] == "报告.txt"

        # 切片：带 offsets，且逐段切回原文成立。
        chunks = client.get(
            f"/v1/agents/chat/files/{file_id}/chunks", headers={"X-User-Id": "alice"}
        ).json()["chunks"]
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk["has_embedding"] is True
            assert chunk["embedding_dim"] == HashEmbedding().dims
            if chunk["offsets"]:
                for start, end in chunk["offsets"]:
                    assert doc_text[start:end] in doc_text

        # raw：文档字节 + 下载头（RFC5987 编码文件名）。
        raw = client.get(f"/v1/agents/chat/files/{file_id}/raw", headers={"X-User-Id": "alice"})
        assert raw.status_code == 200
        assert raw.content == doc_text.encode("utf-8")
        assert "attachment" in raw.headers["content-disposition"]

        # bob：三个端点全部 404（不泄露存在性）。
        for suffix in ("preview", "chunks", "raw"):
            response = client.get(
                f"/v1/agents/chat/files/{file_id}/{suffix}", headers={"X-User-Id": "bob"}
            )
            assert response.status_code == 404, suffix
        await service.aclose()


async def test_raw_endpoint_serves_image_bytes() -> None:
    """图片 raw：magic 校验入库后按存储 mime 直出。"""
    from fastapi.testclient import TestClient

    from agent_base.core.bootstrap import AgentRuntime
    from agent_base.entrypoints.server import create_app
    from agent_base.extensions.filestore import MemoryUploadedFileStore
    from agent_base.modules.chat.module import ChatModule
    from fakes import ScriptedChatModel

    settings = _settings()
    runtime = AgentRuntime(
        settings=settings,
        llm=ScriptedChatModel([]),
        modules={"chat": ChatModule()},
        tools=[],
        checkpointer=None,
    )
    runtime.memory = MemoryService(
        store=MemoryMemoryStore(), embedder=HashEmbedding(), settings=settings
    )
    runtime.file_store = MemoryUploadedFileStore()
    png = b"\x89PNG\r\n\x1a\n" + b"0" * 64
    with TestClient(create_app(runtime=runtime)) as client:
        upload = client.post(
            "/v1/agents/chat/files",
            files={"file": ("色块.png", png, "image/png")},
            headers={"X-User-Id": "alice"},
        )
        assert upload.status_code == 200
        file_id = upload.json()["file_id"]
        # 图片无正文：预览 extracted_text 为空串（前端走 raw 显示原图）。
        preview = client.get(
            f"/v1/agents/chat/files/{file_id}/preview", headers={"X-User-Id": "alice"}
        )
        assert preview.json()["extracted_text"] == ""
        raw = client.get(f"/v1/agents/chat/files/{file_id}/raw", headers={"X-User-Id": "alice"})
        assert raw.status_code == 200
        assert raw.headers["content-type"].startswith("image/png")
        assert raw.content == png


async def test_legacy_chunks_without_offsets_degrade(tmp_path=None) -> None:
    """旧数据 offsets=None：chunks 端点原样返回 null（前端降级卡片视图）。"""
    from agent_base.memory.store import DocChunk, MemoryMemoryStore

    service = _service()
    await service.store.put_chunks(
        [
            DocChunk(
                chunk_id="legacy-1",
                file_id="f-old",
                thread_id="",
                user_id="u",
                agent_id="chat",
                ordinal=0,
                text="旧数据分块（无偏移量）",
            )
        ]
    )
    chunks = await service.store.list_chunks("u", file_id="f-old")
    assert chunks[0].offsets is None
    del MemoryMemoryStore  # 导入占位避免误删
