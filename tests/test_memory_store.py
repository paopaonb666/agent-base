"""记忆存储层（M6a）的测试：三种后端的数据语义一致性。

核心断言集中在 ``_check_store_semantics``——in-memory 与 sqlite（以及
可用的 MySQL）跑同一组用例，保证跨后端行为一致；MySQL 测试在缺库时
自动跳过（与 test_memory.py 的既有约定一致）。
"""

from __future__ import annotations

import os
import uuid
from dataclasses import replace
from pathlib import Path

import pytest

from agent_base.core.config import Settings
from agent_base.memory import (
    DocChunk,
    MemoryBlock,
    MemoryMemoryStore,
    MemoryOp,
    MemoryRecord,
    MemoryStoreError,
    SessionSummary,
    SqliteMemoryStore,
    build_memory_store,
    decode_embedding,
    encode_embedding,
)
from agent_base.memory.store import MysqlMemoryStore


def _settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {"llm_api_key": "sk-test"}
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


def _record(**overrides: object) -> MemoryRecord:
    defaults: dict[str, object] = {
        "memory_id": uuid.uuid4().hex[:16],
        "user_id": "alice",
        "agent_id": "chat",
        "kind": "semantic",
        "content": "用户偏好简洁的中文回答",
        "tags": ["偏好"],
        "embedding": encode_embedding([0.1, 0.2, 0.3]),
        "embedding_dim": 3,
        "salience": 0.8,
        "source_thread_id": "chat:t1",
        "source_refs": ["msg-1"],
        # 显式错开时间戳：list_memories 按 updated_at DESC 排序。
        "created_at": 1_000_000.0,
        "updated_at": 1_000_000.0,
    }
    defaults.update(overrides)
    return MemoryRecord(**defaults)  # type: ignore[arg-type]


async def test_encode_decode_embedding_roundtrip() -> None:
    vector = [0.25, -1.5, 3.75, 0.0, 1024.5]
    decoded = decode_embedding(encode_embedding(vector))
    assert len(decoded) == len(vector)
    for a, b in zip(vector, decoded, strict=True):
        assert a == pytest.approx(b, abs=1e-6)
    assert decode_embedding(b"") == []


async def test_in_memory_store_rejects_unknown_kind_and_status() -> None:
    store = MemoryMemoryStore()
    with pytest.raises(MemoryStoreError):
        await store.upsert_memory(_record(kind="gossip"))
    with pytest.raises(MemoryStoreError):
        await store.upsert_memory(_record(status="pinned"))


async def test_build_memory_store_backend_selection(tmp_path: Path) -> None:
    assert isinstance(
        await build_memory_store(_settings(checkpointer_backend="memory")), MemoryMemoryStore
    )
    sqlite_store = await build_memory_store(
        _settings(checkpointer_backend="sqlite", checkpointer_sqlite_path=str(tmp_path / "m.db"))
    )
    assert isinstance(sqlite_store, SqliteMemoryStore)
    await sqlite_store.aclose()  # type: ignore[attr-defined]
    # Settings 层会拒绝未知后端（快速失败）；这里用桩对象直测装配函数
    # 的防御分支——未知后端返回 None 而不是崩溃。
    from types import SimpleNamespace

    stub = SimpleNamespace(checkpointer_backend="nonsense")
    assert await build_memory_store(stub) is None  # type: ignore[arg-type]


async def _check_store_semantics(store: MemoryMemoryStore | SqliteMemoryStore) -> None:
    """跨后端的语义一致性：CRUD、过滤、级联与审计。"""
    # ---- memories：写入与读回 --------------------------------------------
    keep = _record(updated_at=1_000_003.0)
    newer = _record(content="用户在做 LangGraph 项目", updated_at=1_000_004.0, kind="episodic")
    superseded = _record(status="superseded", updated_at=1_000_002.0)
    other_agent = _record(agent_id="writer", updated_at=1_000_001.0)
    global_one = _record(agent_id="*", content="全模块共享：用户时区是 UTC+8")
    other_user = _record(user_id="bob")
    for record in (keep, newer, superseded, other_agent, global_one, other_user):
        await store.upsert_memory(record)

    fetched = await store.get_memory(keep.memory_id)
    assert fetched is not None
    assert fetched.content == keep.content
    assert fetched.tags == ["偏好"]
    assert fetched.source_refs == ["msg-1"]
    assert decode_embedding(fetched.embedding or b"") == pytest.approx([0.1, 0.2, 0.3], abs=1e-6)
    assert fetched.embedding_dim == 3
    assert fetched.salience == pytest.approx(0.8)
    assert fetched.last_accessed_at is None

    # 过滤：用户隔离 + agent 作用域（含 "*" 全局）+ 状态 + 类别。
    for_alice_chat = await store.list_memories("alice", agent_id="chat")
    assert {r.memory_id for r in for_alice_chat} == {
        keep.memory_id,
        newer.memory_id,
        global_one.memory_id,
    }
    only_chat = await store.list_memories("alice", agent_id="chat", include_global=False)
    assert {r.memory_id for r in only_chat} == {keep.memory_id, newer.memory_id}
    assert [r.memory_id for r in only_chat] == [newer.memory_id, keep.memory_id]  # updated_at DESC
    episodic = await store.list_memories("alice", kinds=["episodic"])
    assert {r.memory_id for r in episodic} == {newer.memory_id}
    active_only = await store.list_memories("alice", statuses=("active",))
    assert superseded.memory_id not in {r.memory_id for r in active_only}
    assert {r.memory_id for r in await store.list_memories("bob")} == {other_user.memory_id}

    # touch：访问计数 + 最近访问时间。
    await store.touch_memories([keep.memory_id])
    touched = await store.get_memory(keep.memory_id)
    assert touched is not None
    assert touched.access_count == 1
    assert touched.last_accessed_at is not None

    # 更新：整行覆盖（整合管线的 UPDATE 语义）。
    await store.upsert_memory(replace(keep, content="更新后的偏好", status="superseded"))
    updated = await store.get_memory(keep.memory_id)
    assert updated is not None
    assert updated.content == "更新后的偏好"
    assert updated.status == "superseded"

    # 删除。
    assert await store.delete_memory(other_user.memory_id) is True
    assert await store.delete_memory(other_user.memory_id) is False
    assert await store.get_memory(other_user.memory_id) is None

    # ---- blocks：常驻记忆块 -------------------------------------------------
    await store.upsert_block(
        MemoryBlock(
            user_id="alice", agent_id="chat", label="persona", content="务实的工程师", version=0
        )
    )
    await store.upsert_block(
        MemoryBlock(user_id="alice", agent_id="chat", label="human", content="喜欢简洁", version=2)
    )
    persona = await store.get_block("alice", "chat", "persona")
    assert persona is not None
    assert persona.content == "务实的工程师"
    labels = [b.label for b in await store.list_blocks("alice", "chat")]
    assert labels == ["human", "persona"]  # label ASC
    assert await store.get_block("bob", "chat", "persona") is None
    # 覆盖写。
    await store.upsert_block(
        MemoryBlock(
            user_id="alice",
            agent_id="chat",
            label="persona",
            content="务实的工程师，偏好 Python",
            version=1,
        )
    )
    bumped = await store.get_block("alice", "chat", "persona")
    assert bumped is not None and bumped.version == 1
    assert await store.delete_block("alice", "chat", "persona") is True
    assert await store.get_block("alice", "chat", "persona") is None

    # ---- summaries + chunks：线程级联删除 ------------------------------------
    await store.upsert_summary(
        SessionSummary(
            user_id="alice",
            thread_id="chat:t1",
            agent_id="chat",
            summary="用户在调试记忆系统",
            covered_message_count=8,
        )
    )
    await store.upsert_summary(
        SessionSummary(user_id="alice", thread_id="chat:t2", summary="另一个会话")
    )
    got = await store.get_summary("alice", "chat:t1")
    assert got is not None
    assert got.summary == "用户在调试记忆系统"
    assert got.agent_id == "chat"
    assert await store.get_summary("bob", "chat:t1") is None

    t1_chunk_base = uuid.uuid4().hex[:8]
    chunks = [
        DocChunk(
            chunk_id=f"{t1_chunk_base}-{i}",
            file_id="file-a",
            thread_id="chat:t1",
            user_id="alice",
            agent_id="chat",
            ordinal=i,
            text=f"分块 {i}",
            embedding=encode_embedding([float(i)]),
            embedding_dim=1,
        )
        for i in range(2)
    ]
    chunks.append(
        DocChunk(
            chunk_id=uuid.uuid4().hex[:16],
            file_id="file-b",
            thread_id="chat:t2",
            user_id="alice",
            agent_id="chat",
            ordinal=0,
            text="另一个文件的分块",
        )
    )
    await store.put_chunks(chunks)
    assert await store.count_chunks_for_file("file-a") == 2
    assert await store.count_chunks_for_file("file-c") == 0
    file_a_chunks = await store.list_chunks("alice", file_id="file-a")
    assert [c.ordinal for c in file_a_chunks] == [0, 1]
    assert decode_embedding(file_a_chunks[0].embedding or b"") == pytest.approx([0.0], abs=1e-6)
    assert await store.delete_chunks_for_file("file-a") == 2

    # 级联：删线程清摘要 + 分块，但 memories 刻意保留（跨会话语义）。
    await store.put_chunks(
        [
            DocChunk(
                chunk_id=uuid.uuid4().hex[:16],
                file_id="file-a",
                thread_id="chat:t1",
                user_id="alice",
                agent_id="chat",
                ordinal=0,
                text="t1 分块",
            )
        ]
    )
    await store.delete_for_thread("chat:t1")
    assert await store.get_summary("alice", "chat:t1") is None
    assert await store.get_summary("alice", "chat:t2") is not None
    remaining_files = {c.file_id for c in await store.list_chunks("alice")}
    assert "file-b" in remaining_files
    assert all(c.thread_id != "chat:t1" for c in await store.list_chunks("alice"))
    assert await store.get_memory(keep.memory_id) is not None

    # ---- ops：审计记录 -------------------------------------------------------
    for i in range(3):
        await store.record_op(
            MemoryOp(
                op_id=uuid.uuid4().hex[:16],
                op="manual",
                user_id="alice" if i < 2 else "bob",
                detail={"index": i},
            )
        )
    alice_ops = await store.list_ops("alice")
    assert len(alice_ops) == 2
    assert all(op.user_id == "alice" for op in alice_ops)
    assert {op.detail["index"] for op in alice_ops} == {0, 1}
    all_ops = await store.list_ops(limit=10)
    assert len(all_ops) == 3
    errored = MemoryOp(op_id=uuid.uuid4().hex[:16], op="extract", status="error", error_text="boom")
    await store.record_op(errored)
    latest = (await store.list_ops(limit=1))[0]
    assert latest.op_id == errored.op_id and latest.status == "error"


async def test_in_memory_store_semantics() -> None:
    await _check_store_semantics(MemoryMemoryStore())


async def test_sqlite_store_semantics(tmp_path: Path) -> None:
    store = await SqliteMemoryStore.create(str(tmp_path / "memory.db"))
    try:
        await _check_store_semantics(store)
    finally:
        await store.aclose()


async def test_sqlite_store_create_is_idempotent(tmp_path: Path) -> None:
    db = str(tmp_path / "memory.db")
    store1 = await SqliteMemoryStore.create(db)
    await store1.upsert_memory(_record(memory_id="fixed-id"))
    store2 = await SqliteMemoryStore.create(db)  # 重复建表 + 复用同一文件
    try:
        fetched = await store2.get_memory("fixed-id")
        assert fetched is not None
    finally:
        await store1.aclose()
        await store2.aclose()


async def test_mysql_store_semantics() -> None:
    """MySQL 后端语义一致性（本机无 MySQL 时自动跳过）。"""
    pytest.importorskip("aiomysql")
    import aiomysql

    host = os.environ.get("MYSQL_HOST", "127.0.0.1")
    port = int(os.environ.get("MYSQL_PORT", "3306"))
    user = os.environ.get("MYSQL_USER", "root")
    password = os.environ.get("MYSQL_PASSWORD", "")

    try:
        conn = await aiomysql.connect(
            host=host, port=port, user=user, password=password, autocommit=True
        )
    except Exception as exc:
        pytest.skip(f"MySQL 不可用：{exc}")

    db = f"agent_base_mem_test_{uuid.uuid4().hex[:8]}"
    async with conn.cursor() as cur:
        await cur.execute(f"CREATE DATABASE `{db}` CHARACTER SET utf8mb4")
    conn.close()

    try:
        store = MysqlMemoryStore(
            _settings(
                checkpointer_backend="mysql",
                checkpointer_mysql_host=host,
                checkpointer_mysql_port=port,
                checkpointer_mysql_user=user,
                checkpointer_mysql_password=password,
                checkpointer_mysql_database=db,
            )
        )
        await _check_store_semantics(store)
    finally:
        cleanup = await aiomysql.connect(
            host=host, port=port, user=user, password=password, autocommit=True
        )
        async with cleanup.cursor() as cur:
            await cur.execute(f"DROP DATABASE IF EXISTS `{db}`")
        cleanup.close()


async def _check_needing_embedding(store: MemoryMemoryStore | SqliteMemoryStore) -> None:
    """回填查询只返回 active、非画像、向量缺失或维度不符的记录。"""
    stale = _record(
        memory_id="stale-dim", embedding=encode_embedding([1.0]), embedding_dim=1
    )
    missing = _record(memory_id="no-vec", embedding=None, embedding_dim=None)
    fresh = _record(
        memory_id="ok-vec",
        embedding=encode_embedding([0.1, 0.2, 0.3]),
        embedding_dim=3,
    )
    archived_no_vec = _record(memory_id="archived", status="archived", embedding=None)
    await store.upsert_memory(stale)
    await store.upsert_memory(missing)
    await store.upsert_memory(fresh)
    await store.upsert_memory(archived_no_vec)
    from agent_base.memory.store import MemoryRecord, profile_memory_id

    await store.upsert_memory(
        MemoryRecord(
            memory_id=profile_memory_id("alice"),
            user_id="alice",
            agent_id="*",
            kind="semantic",
            content="画像",
            tags=["profile"],
        )
    )
    needed = await store.list_memories_needing_embedding(3)
    assert {r.memory_id for r in needed} == {"stale-dim", "no-vec"}


async def test_list_memories_needing_embedding_filter_in_memory() -> None:
    await _check_needing_embedding(MemoryMemoryStore())


async def test_list_memories_needing_embedding_filter_sqlite(tmp_path: Path) -> None:
    store = await SqliteMemoryStore.create(str(tmp_path / "bf.db"))
    try:
        await _check_needing_embedding(store)
    finally:
        await store.aclose()


async def _check_versions_semantics(store: MemoryMemoryStore | SqliteMemoryStore) -> None:
    """版本史跨后端语义：写入、按记忆过滤、最新在前、按 id 取回。"""
    from agent_base.memory.store import MemoryVersion

    for i, op in enumerate(("create", "update", "delete")):
        await store.record_version(
            MemoryVersion(
                version_id=f"v{i}",
                memory_id="m1",
                user_id="alice",
                op=op,
                content="" if op == "delete" else f"第{i}版内容",
                previous_content=f"第{max(0, i - 1)}版内容" if i > 0 else "",
                created_at=1_000_000.0 + i,
            )
        )
    await store.record_version(
        MemoryVersion(
            version_id="v-other",
            memory_id="m2",
            user_id="alice",
            op="create",
            content="别的记忆",
            created_at=1_000_000.0,
        )
    )
    versions = await store.list_versions("m1")
    assert [v.version_id for v in versions] == ["v2", "v1", "v0"]  # 最新在前
    assert versions[0].op == "delete" and versions[0].content == ""
    assert versions[0].previous_content == "第1版内容"
    tombstone = await store.get_version("v2")
    assert tombstone is not None and tombstone.previous_content == "第1版内容"
    assert await store.get_version("ghost") is None
    # user 隔离不做在 store 层（属主校验在 service/server 边界）。
    assert len(await store.list_versions("m2")) == 1


async def test_versions_semantics_in_memory() -> None:
    await _check_versions_semantics(MemoryMemoryStore())


async def test_versions_semantics_sqlite(tmp_path: Path) -> None:
    store = await SqliteMemoryStore.create(str(tmp_path / "ver.db"))
    try:
        await _check_versions_semantics(store)
    finally:
        await store.aclose()
