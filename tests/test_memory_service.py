"""记忆服务门面（M6b）与管理 REST 端点的测试。"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from agent_base.core.bootstrap import AgentRuntime
from agent_base.core.config import Settings
from agent_base.entrypoints.server import create_app
from agent_base.memory.service import MemoryService, build_memory_service
from agent_base.memory.store import MemoryStoreError, SqliteMemoryStore
from fakes import HashEmbedding, ScriptedChatModel


def _settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {"llm_api_key": "sk-test"}
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


async def _sqlite_service(tmp_path: Path, **overrides: object) -> MemoryService:
    store = await SqliteMemoryStore.create(str(tmp_path / "memory.db"))
    settings = _settings(**overrides)
    return MemoryService(store=store, embedder=HashEmbedding(), settings=settings)


async def test_add_memory_embeds_content(tmp_path: Path) -> None:
    service = await _sqlite_service(tmp_path)
    record = await service.add_memory(
        user_id="alice", agent_id="chat", content="用户偏好简洁回答"
    )
    assert record.embedding is not None
    assert record.embedding_dim == HashEmbedding().dims
    fetched = await service.get_memory(record.memory_id)
    assert fetched is not None and fetched.content == "用户偏好简洁回答"
    await service.aclose()


async def test_add_memory_rejects_unknown_kind(tmp_path: Path) -> None:
    service = await _sqlite_service(tmp_path)
    with pytest.raises(MemoryStoreError):
        await service.add_memory(user_id="u", agent_id="chat", content="x", kind="gossip")
    await service.aclose()


async def test_update_memory_reembeds_on_content_change(tmp_path: Path) -> None:
    service = await _sqlite_service(tmp_path)
    record = await service.add_memory(user_id="u", agent_id="chat", content="旧内容")
    updated = await service.update_memory(record.memory_id, content="全新的内容")
    assert updated is not None
    assert updated.content == "全新的内容"
    assert updated.embedding is not None
    # 内容未变时保留原向量（避免无谓的 embedding 调用）。
    same = await service.update_memory(record.memory_id, salience=0.9)
    assert same is not None and same.salience == pytest.approx(0.9)
    assert same.embedding == updated.embedding
    assert await service.update_memory("不存在") is None
    await service.aclose()


async def test_search_and_user_isolation(tmp_path: Path) -> None:
    service = await _sqlite_service(tmp_path)
    await service.add_memory(user_id="alice", agent_id="chat", content="用户使用 LangGraph")
    await service.add_memory(user_id="bob", agent_id="chat", content="Bob 使用 TensorFlow")
    results = await service.search(user_id="alice", agent_id="chat", query="LangGraph")
    assert results
    assert all(item.record.user_id == "alice" for item in results)
    assert results[0].record.content == "用户使用 LangGraph"
    await service.aclose()


async def test_build_memory_service_disabled() -> None:
    assert await build_memory_service(_settings(memory_enabled=False)) is None


# ─────────────────────────── 管理 REST 端点 ───────────────────────────


def _runtime(**overrides: object) -> AgentRuntime:
    """带内存记忆服务的 runtime（HashEmbedding，无外部依赖）。"""
    rt = AgentRuntime(
        settings=_settings(**overrides),
        llm=ScriptedChatModel([]),
        modules={},
        tools=[],
        checkpointer=InMemorySaver(),
    )
    if rt.memory is not None or rt.settings.memory_enabled:
        from agent_base.memory.store import MemoryMemoryStore

        rt.memory = MemoryService(
            store=MemoryMemoryStore(), embedder=HashEmbedding(), settings=rt.settings
        )
    return rt


def _client(**overrides: object) -> TestClient:
    return TestClient(create_app(runtime=_runtime(**overrides)))


async def test_memory_endpoints_crud_flow() -> None:
    client = _client()
    with client as c:
        # 未带 X-User-Id → 缺省作用域。
        created = c.post(
            "/v1/memory",
            json={"content": "用户偏好深色主题", "tags": ["UI"]},
        )
        assert created.status_code == 200
        body = created.json()
        assert body["agent_id"] == "*"
        assert body["has_embedding"] is True
        memory_id = body["memory_id"]

        # 搜索命中。
        found = c.get("/v1/memory", params={"q": "深色主题"})
        assert found.status_code == 200
        hits = found.json()["memories"]
        assert hits and hits[0]["memory_id"] == memory_id and hits[0]["score"] > 0

        # 浏览 + 过滤。
        listed = c.get("/v1/memory", params={"kind": "semantic"})
        assert listed.status_code == 200
        assert any(item["memory_id"] == memory_id for item in listed.json()["memories"])

        # user 作用域隔离。
        scoped = c.get("/v1/memory", headers={"X-User-Id": "someone-else"})
        assert scoped.json()["memories"] == []

        # 部分更新。
        patched = c.patch(
            f"/v1/memory/{memory_id}", json={"content": "用户偏好浅色主题", "status": "archived"}
        )
        assert patched.status_code == 200
        assert patched.json()["content"] == "用户偏好浅色主题"
        assert patched.json()["status"] == "archived"

        # 删除。
        deleted = c.delete(f"/v1/memory/{memory_id}")
        assert deleted.json() == {"deleted": True}
        assert c.get("/v1/memory").json()["memories"] == []


async def test_memory_endpoints_validation() -> None:
    client = _client()
    with client as c:
        assert c.post("/v1/memory", json={"content": "x", "kind": "gossip"}).status_code == 400
        assert c.post("/v1/memory", json={"content": ""}).status_code == 422
        assert (
            c.get("/v1/memory", params={"kind": "gossip"}).status_code == 400
        )
        bad_header = c.get("/v1/memory", headers={"X-User-Id": "bad user id!"})
        assert bad_header.status_code == 400
        assert c.patch("/v1/memory/nope", json={"salience": 0.1}).status_code == 404
        assert c.patch("/v1/memory/nope", json={"status": "bogus"}).status_code == 400
        assert c.delete("/v1/memory/nope").json() == {"deleted": False}


async def test_memory_endpoints_disabled_503() -> None:
    rt = _runtime(memory_enabled=False)
    with TestClient(create_app(runtime=rt)) as c:
        assert c.get("/v1/memory").status_code == 503
        assert c.post("/v1/memory", json={"content": "x"}).status_code == 503
        # 未启用时 /health 不出现 memory 组件（缺席 ≠ 降级）。
        health = c.get("/health").json()
        assert "memory" not in health["components"]
        assert health["status"] == "ok"


async def test_memory_health_component() -> None:
    with _client() as c:
        health = c.get("/health").json()
        assert health["components"]["memory"] == "ok"


# ─────────────────────── 管理收尾端点（M6f） ───────────────────────


async def test_memory_blocks_profile_audit_endpoints() -> None:
    client = _client()
    with client as c:
        # blocks：PUT 创建（缺省全局作用域）→ GET 列出 → 覆盖写版本递增
        # → DELETE。
        created = c.put(
            "/v1/memory/blocks/persona",
            json={"content": "务实的工程助手", "char_limit": 1500},
        )
        assert created.status_code == 200
        assert created.json()["agent_id"] == "*"
        assert created.json()["version"] == 1
        listed = c.get("/v1/memory/blocks")
        assert listed.json()["blocks"][0]["label"] == "persona"
        bumped = c.put("/v1/memory/blocks/persona", json={"content": "更务实的助手"})
        assert bumped.json()["version"] == 2
        scoped = c.get("/v1/memory/blocks", params={"module": "chat"})
        assert scoped.json()["blocks"] == []  # 创建时是全局块
        assert c.delete("/v1/memory/blocks/persona").json() == {"deleted": True}
        assert c.get("/v1/memory/blocks").json()["blocks"] == []

        # profile：先造画像（服务层）→ GET 读取。
        rt = client.app.state.runtime
        await rt.memory.save_profile("default", {"偏好": ["简洁回答"]})
        profile = c.get("/v1/memory/profile").json()
        assert profile["profile"] == {"偏好": ["简洁回答"]}

        # audit：手动落一条 op 后校验结构与 user 作用域过滤。
        await rt.memory.record_op(op="manual", user_id="default", detail={"note": "x"})
        audit = c.get("/v1/memory/audit").json()
        assert audit["ops"] and audit["ops"][0]["op"] == "manual"
        assert audit["ops"][0]["detail"] == {"note": "x"}
        empty = c.get("/v1/memory/audit", headers={"X-User-Id": "someone-else"})
        assert empty.json()["ops"] == []

    with _client(memory_enabled=False) as c:
        assert c.get("/v1/memory/blocks").status_code == 503
        assert c.get("/v1/memory/profile").status_code == 503
        assert c.get("/v1/memory/audit").status_code == 503
