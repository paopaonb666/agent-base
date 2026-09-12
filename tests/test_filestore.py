"""extensions/filestore 的测试（M4b）：字节入库、绑定、级联、孤儿清理。"""

from __future__ import annotations

import sys
import time
import types

import pytest

from agent_base.core.config import Settings
from agent_base.extensions.filestore import (
    MemoryUploadedFileStore,
    MysqlUploadedFileStore,
    SqliteUploadedFileStore,
    UploadedFileInfo,
    build_uploaded_file_store,
)


def _info(file_id: str = "f1", thread_id: str = "", **overrides: object) -> UploadedFileInfo:
    fields: dict[str, object] = {
        "file_id": file_id,
        "filename": "简历.pdf",
        "format": "pdf",
        "pages": 2,
        "paragraphs": None,
        "truncated": False,
        "text_len": 1771,
        "extracted_text": "刘骁铖 简历正文…",
        "content": b"%PDF-1.7 fake bytes",
        "thread_id": thread_id,
        "module": "chat",
    }
    fields.update(overrides)
    return UploadedFileInfo(**fields)  # type: ignore[arg-type]


async def test_memory_roundtrip_order_and_bind() -> None:
    store = MemoryUploadedFileStore()
    await store.save(_info("f1"))
    await store.save(_info("f2"))
    # get_many 按请求顺序返回（注入顺序与前端附件列表一致）
    rows = await store.get_many(["f2", "f1"])
    assert [r.file_id for r in rows] == ["f2", "f1"]
    assert rows[0].content == b"%PDF-1.7 fake bytes"
    assert rows[0].extracted_text.startswith("刘骁铖")
    await store.bind_thread(["f1", "f2"], "chat:t1")
    await store.delete_for_thread("chat:t1")
    assert await store.get_many(["f1", "f2"]) == []


async def test_memory_purge_orphans() -> None:
    store = MemoryUploadedFileStore()
    old = _info("old", created_at=time.time() - 25 * 3600)
    fresh = _info("fresh", created_at=time.time())
    bound = _info("bound", thread_id="chat:t1", created_at=time.time() - 25 * 3600)
    for info in (old, fresh, bound):
        await store.save(info)
    removed = await store.purge_orphans(max_age_hours=24)
    assert removed == 1
    remaining = await store.get_many(["old", "fresh", "bound"])
    assert [r.file_id for r in remaining] == ["fresh", "bound"]


async def test_sqlite_roundtrip_with_bytes(tmp_path: object) -> None:
    store = await SqliteUploadedFileStore.create(str(tmp_path / "state.db"))  # type: ignore[arg-type]
    await store.save(_info("f1"))
    rows = await store.get_many(["f1"])
    assert len(rows) == 1
    assert rows[0].content == b"%PDF-1.7 fake bytes"
    assert rows[0].pages == 2
    assert rows[0].truncated is False
    await store.bind_thread(["f1"], "chat:t1")
    rows = await store.get_many(["f1"])
    assert rows[0].thread_id == "chat:t1"
    await store.delete_for_thread("chat:t1")
    assert await store.get_many(["f1"]) == []
    await store.aclose()


async def test_sqlite_purge_orphans(tmp_path: object) -> None:
    store = await SqliteUploadedFileStore.create(str(tmp_path / "s2.db"))  # type: ignore[arg-type]
    await store.save(_info("old", created_at=time.time() - 25 * 3600))
    await store.save(_info("fresh", created_at=time.time()))
    removed = await store.purge_orphans(max_age_hours=24)
    assert removed == 1
    assert await store.get_many(["old"]) == []
    await store.aclose()


# ── mysql（桥接路径，aiomysql 用替身） ──────────────────────────────


class _FakeCursor:
    def __init__(self, sink: list[tuple[str, object]]) -> None:
        self._sink = sink

    async def __aenter__(self) -> _FakeCursor:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def execute(self, sql: str, params: object = None) -> None:
        self._sink.append((sql, params))

    async def fetchall(self) -> list[object]:
        return []


class _FakeConnection:
    def __init__(self, sink: list[tuple[str, object]]) -> None:
        self._sink = sink
        self.closed = False

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self._sink)

    def close(self) -> None:
        self.closed = True


async def test_mysql_save_and_bind(monkeypatch: pytest.MonkeyPatch) -> None:
    executed: list[tuple[str, object]] = []

    async def fake_connect(**kwargs: object) -> _FakeConnection:
        return _FakeConnection(executed)

    monkeypatch.setitem(sys.modules, "aiomysql", types.SimpleNamespace(connect=fake_connect))
    settings = Settings(_env_file=None, checkpointer_backend="mysql")
    store = MysqlUploadedFileStore(settings)
    await store.save(_info("f1"))
    await store.bind_thread(["f1"], "chat:t1")
    await store.delete_for_thread("chat:t1")
    sqls = [sql for sql, _ in executed]
    assert any("CREATE TABLE IF NOT EXISTS uploaded_files" in s for s in sqls)
    assert any("INSERT INTO uploaded_files" in s and "LONGBLOB" not in s for s in sqls)
    assert any(s.startswith("UPDATE uploaded_files") for s in sqls)
    assert any(s.startswith("DELETE FROM uploaded_files") for s in sqls)


# ── 装配 ────────────────────────────────────────────────────────────


async def test_build_store_follows_backend(tmp_path: object) -> None:
    memory_store = await build_uploaded_file_store(Settings(_env_file=None))
    assert isinstance(memory_store, MemoryUploadedFileStore)
    sqlite_store = await build_uploaded_file_store(
        Settings(
            _env_file=None,
            checkpointer_backend="sqlite",
            checkpointer_sqlite_path=str(tmp_path / "s.db"),  # type: ignore[arg-type]
        )
    )
    assert isinstance(sqlite_store, SqliteUploadedFileStore)
    await sqlite_store.aclose()
