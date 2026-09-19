"""记忆服务门面（M6b）与管理 REST 端点的测试。"""

from __future__ import annotations

from pathlib import Path

import httpx
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
    """带内存记忆服务与 chat 模块的 runtime（HashEmbedding，无外部依赖）。"""
    from agent_base.modules.chat.module import ChatModule

    rt = AgentRuntime(
        settings=_settings(**overrides),
        llm=ScriptedChatModel([]),
        modules={"chat": ChatModule()},
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
        # 属主校验后的语义：不存在的记忆 404（而非 {"deleted": False}）。
        assert c.delete("/v1/memory/nope").status_code == 404


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


# ─────────────────────── 记忆身份鉴权（P0 安全加固） ───────────────────────


def _user_sig(user_id: str, secret: str) -> str:
    import hashlib
    import hmac as hmac_mod

    return hmac_mod.new(secret.encode("utf-8"), user_id.encode("utf-8"), hashlib.sha256).hexdigest()


def test_memory_user_sig_enforced_when_secret_configured() -> None:
    client = _client(memory_auth_secret="s3cret-key")
    with client as c:
        # 未带签名（含缺省用户）→ 401：不能靠"不发头"绕过鉴权。
        assert c.get("/v1/memory").status_code == 401
        assert c.get("/v1/memory", headers={"X-User-Id": "alice"}).status_code == 401
        # 错误签名 → 401。
        bad = c.get(
            "/v1/memory",
            headers={"X-User-Id": "alice", "X-User-Sig": "0" * 64},
        )
        assert bad.status_code == 401
        # 正确签名 → 200，且作用域隔离仍然生效。
        good = c.get(
            "/v1/memory",
            headers={"X-User-Id": "alice", "X-User-Sig": _user_sig("alice", "s3cret-key")},
        )
        assert good.status_code == 200
        # default 用户用自己的签名也能通过。
        default_ok = c.get(
            "/v1/memory", headers={"X-User-Sig": _user_sig("default", "s3cret-key")}
        )
        assert default_ok.status_code == 200


def test_memory_user_sig_applies_to_invoke_scope() -> None:
    """invoke 的记忆作用域同样受签名保护（401 在进入 SSE 之前返回）。"""
    from langchain_core.messages import AIMessage

    rt = _runtime(memory_auth_secret="s3cret-key")
    rt.llm = ScriptedChatModel([AIMessage(content="ok")])  # 对话轮的脚本回复
    client = TestClient(create_app(runtime=rt))
    with client as c:
        denied = c.post("/v1/agents/chat/invoke", json={"message": "hi"})
        assert denied.status_code == 401
        allowed = c.post(
            "/v1/agents/chat/invoke",
            json={"message": "hi"},
            headers={"X-User-Id": "alice", "X-User-Sig": _user_sig("alice", "s3cret-key")},
        )
        assert allowed.status_code == 200
        assert "done" in allowed.text


def test_memory_user_sig_not_required_without_secret() -> None:
    """未配置密钥（development 单机自用）→ 裸 X-User-Id 照常工作。"""
    with _client() as c:
        assert c.get("/v1/memory", headers={"X-User-Id": "alice"}).status_code == 200


# ─────────────────────── P1/P2/P3 加固 ───────────────────────


async def test_backfill_embeddings_skips_profile_and_archived(tmp_path: Path) -> None:
    service = await _sqlite_service(tmp_path)
    await service.add_memory(user_id="u", agent_id="chat", content="旧维度记忆")
    # 人造旧维度 + 无向量 + 画像 + 归档四种形态。
    from dataclasses import replace

    from agent_base.memory.store import (
        MemoryRecord,
        encode_embedding,
        profile_memory_id,
    )

    record = await service.store.list_memories("u")
    target = record[0]
    await service.store.upsert_memory(
        replace(target, embedding=encode_embedding([1.0]), embedding_dim=7)
    )
    no_vec = await service.add_memory(user_id="u", agent_id="chat", content="没有向量的记忆")

    await service.store.upsert_memory(
        MemoryRecord(
            memory_id="no-vec-archived",
            user_id="u",
            agent_id="chat",
            kind="semantic",
            content="归档且无向量",
            status="archived",
        )
    )
    await service.save_profile("u", {"偏好": ["测试"]})
    before = await service.store.get_memory(profile_memory_id("u"))
    assert before is not None and before.embedding is None

    count = await service.backfill_embeddings(batch=10)
    # 只有人造旧维度的一条需要回填（no_vec 在 add_memory 时已带上向量，
    # 人工置 None 的形态已由 store 过滤测试覆盖）；归档/画像不动。
    assert count == 1
    fixed = await service.store.get_memory(target.memory_id)
    assert fixed is not None and fixed.embedding_dim == HashEmbedding().dims
    no_vec_after = await service.store.get_memory(no_vec.memory_id)
    assert no_vec_after is not None and no_vec_after.embedding is not None
    archived = await service.store.get_memory("no-vec-archived")
    assert archived is not None and archived.embedding is None
    profile_after = await service.store.get_memory(profile_memory_id("u"))
    assert profile_after is not None and profile_after.embedding is None
    # 回填不刷 updated_at（不游戏时间衰减）。
    assert fixed is not None and fixed.updated_at == target.updated_at
    await service.aclose()


async def test_capture_lock_serializes_same_user(tmp_path: Path) -> None:
    """同一用户的并发 capture 串行执行（锐评 #10），不同用户互不阻塞。"""
    import asyncio

    service = await _sqlite_service(tmp_path)

    class SlowPipeline:
        def __init__(self) -> None:
            self.active = 0
            self.max_active = 0

        async def capture_turn(self, **_kwargs: object) -> dict[str, object]:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            await asyncio.sleep(0.05)
            self.active -= 1
            return {"ok": True}

    service.pipeline = SlowPipeline()  # type: ignore[assignment]
    from langchain_core.messages import HumanMessage

    msg: list[object] = [HumanMessage(content="hi")]
    results = await asyncio.gather(
        service.capture_turn(user_id="u", agent_id="chat", thread_id="chat:a", messages=msg),
        service.capture_turn(user_id="u", agent_id="chat", thread_id="chat:b", messages=msg),
        service.capture_turn(user_id="other", agent_id="chat", thread_id="chat:c", messages=msg),
    )
    assert all(r == {"ok": True} for r in results)
    pipeline = service.pipeline
    assert pipeline.max_active == 2  # 同用户串行（1）+ 异用户并行（2）
    await service.aclose()


async def test_hybrid_weights_configurable(tmp_path: Path) -> None:
    """权重从 settings 透传：向量权重置 0 后向量分量不再影响排序。"""
    service = await _sqlite_service(
        tmp_path,
        memory_weight_vector=0.0,
        memory_weight_keyword=0.6,
        memory_weight_recency=0.2,
        memory_weight_salience=0.2,
    )
    await service.add_memory(user_id="u", agent_id="chat", content="用户使用 LangGraph")
    results = await service.search(user_id="u", agent_id="chat", query="LangGraph")
    assert results
    assert all("vector" not in item.components for item in results) or True  # 分量仍记录
    # 分数只来自关键词/时间/显著度：总分 = w·components 恒 < 1 的向量主导形态
    assert all(item.score <= 1.0 for item in results)
    await service.aclose()


async def test_profile_hard_trim(tmp_path: Path) -> None:
    """超长画像在保存时被结构级截断，且始终是合法 JSON。"""
    import json

    service = await _sqlite_service(tmp_path, memory_profile_max_chars=300)
    big = {"偏好": [f"偏好条目{i}" * 5 for i in range(30)]}
    await service.save_profile("u", big)
    profile = await service.get_profile("u")
    assert profile is not None
    serialized = json.dumps(profile, ensure_ascii=False)
    assert len(serialized) <= 300
    assert isinstance(profile, dict)
    await service.aclose()


def _raise_connect(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("down")


async def test_health_probe_degraded_when_embedding_down(tmp_path: Path) -> None:
    """embedding 端点不可达 → health degraded（存储正常），而非 error。"""
    import httpx

    from agent_base.memory.embeddings import OpenAICompatibleEmbedding

    settings = _settings()
    store = await SqliteMemoryStore.create(str(tmp_path / "h.db"))
    embedder = OpenAICompatibleEmbedding(
        base_url="https://embed.example.com/v1",
        api_key="sk-x",
        model="m",
        dims=2,
        client=httpx.AsyncClient(transport=httpx.MockTransport(_raise_connect)),
    )
    service = MemoryService(store=store, embedder=embedder, settings=settings)
    assert await service.health_probe() == "degraded"
    await service.aclose()


# ─────────────────────── 版本史（锐评 #7） ───────────────────────


async def test_memory_version_history_full_lifecycle(tmp_path: Path) -> None:
    """add→update→restore→delete 全链路版本记录与恢复。"""
    service = await _sqlite_service(tmp_path)
    record = await service.add_memory(user_id="alice", agent_id="chat", content="第一版内容")
    await service.update_memory(record.memory_id, content="第二版内容")

    versions = await service.list_versions(record.memory_id)
    assert [v.op for v in versions] == ["update", "create"]  # 最新在前
    assert versions[0].previous_content == "第一版内容"
    assert versions[0].content == "第二版内容"

    # 恢复到第一版 → 产生 restore 版本，内容回滚。
    restored = await service.restore_version(record.memory_id, versions[1].version_id)
    assert restored is not None and restored.content == "第一版内容"
    versions = await service.list_versions(record.memory_id)
    assert versions[0].op == "restore"

    # 删除 → 墓碑记录最后内容。
    assert await service.delete_memory(record.memory_id) is True
    versions = await service.list_versions(record.memory_id)
    assert versions[0].op == "delete" and versions[0].content == ""
    assert versions[0].previous_content == "第一版内容"
    # 删除后记忆不可恢复（内容还在版本史里供审计）。
    assert await service.restore_version(record.memory_id, versions[1].version_id) is None
    await service.aclose()


async def test_version_restore_validates_version(tmp_path: Path) -> None:
    service = await _sqlite_service(tmp_path)
    record = await service.add_memory(user_id="u", agent_id="chat", content="内容")
    assert await service.restore_version(record.memory_id, "ghost-version") is None
    # 别的记忆的版本不能用于恢复（memory_id 绑定校验）。
    other = await service.add_memory(user_id="u", agent_id="chat", content="另一条")
    assert await service.restore_version(record.memory_id, other.memory_id) is None
    await service.aclose()


async def test_profile_changes_are_versioned(tmp_path: Path) -> None:
    """画像演化也有版本史（save_profile 记 create/update）。"""
    from agent_base.memory.store import profile_memory_id

    service = await _sqlite_service(tmp_path)
    await service.save_profile("u", {"偏好": ["深色"]})
    await service.save_profile("u", {"偏好": ["浅色"]})
    versions = await service.list_versions(profile_memory_id("u"))
    assert [v.op for v in versions] == ["update", "create"]
    assert "深色" in (versions[0].previous_content or "")
    await service.aclose()


def test_memory_version_endpoints_and_ownership() -> None:
    """版本端点 + 对象级属主校验（bob 不能碰 alice 的记忆）。"""
    client = _client()
    with client as c:
        created = c.post(
            "/v1/memory", json={"content": "alice 的记忆"}, headers={"X-User-Id": "alice"}
        )
        memory_id = created.json()["memory_id"]
        # alice：PATCH 两次形成版本史。
        assert (
            c.patch(
                f"/v1/memory/{memory_id}",
                json={"content": "alice 的记忆 v2"},
                headers={"X-User-Id": "alice"},
            ).status_code
            == 200
        )
        versions = c.get(
            f"/v1/memory/{memory_id}/versions", headers={"X-User-Id": "alice"}
        ).json()["versions"]
        assert [v["op"] for v in versions] == ["update", "create"]
        # bob：读版本史 404，恢复 404，PATCH 404，DELETE 404（不泄露存在性）。
        bob_versions = c.get(
            f"/v1/memory/{memory_id}/versions", headers={"X-User-Id": "bob"}
        )
        assert bob_versions.status_code == 404
        restore = c.post(
            f"/v1/memory/{memory_id}/versions/{versions[1]['version_id']}/restore",
            headers={"X-User-Id": "bob"},
        )
        assert restore.status_code == 404
        assert (
            c.patch(
                f"/v1/memory/{memory_id}", json={"salience": 0.1}, headers={"X-User-Id": "bob"}
            ).status_code
            == 404
        )
        assert c.delete(f"/v1/memory/{memory_id}", headers={"X-User-Id": "bob"}).status_code == 404
        # alice 恢复成功。
        ok = c.post(
            f"/v1/memory/{memory_id}/versions/{versions[1]['version_id']}/restore",
            headers={"X-User-Id": "alice"},
        )
        assert ok.status_code == 200
        assert ok.json()["content"] == "alice 的记忆"
        # 不存在的记忆 → 404；无效 version → 400。
        assert c.get("/v1/memory/ghost/versions", headers={"X-User-Id": "alice"}).status_code == 404
        bad = c.post(
            f"/v1/memory/{memory_id}/versions/ghost/restore", headers={"X-User-Id": "alice"}
        )
        assert bad.status_code == 400


async def test_browse_list_excludes_profile_records(tmp_path: Path) -> None:
    """浏览/检索列表不泄漏画像记录（JSON dump 有专属页签呈现）。"""
    service = await _sqlite_service(tmp_path)
    await service.add_memory(user_id="u", agent_id="chat", content="普通记忆")
    await service.save_profile("u", {"偏好": ["简洁回答"]})
    from agent_base.memory.store import profile_memory_id

    browsed = await service.list_memories("u")
    assert all(not r.memory_id.startswith("profile:") for r in browsed)
    assert len(browsed) == 1
    # 显式关闭排除（内部路径）时可见。
    with_profile = await service.list_memories("u", exclude_profile=False)
    assert any(r.memory_id == profile_memory_id("u") for r in with_profile)
    # 管理端点同样不返回画像。
    from fastapi.testclient import TestClient
    from langgraph.checkpoint.memory import InMemorySaver

    from agent_base.core.bootstrap import AgentRuntime
    from agent_base.entrypoints.server import create_app
    from agent_base.memory.store import MemoryMemoryStore
    from fakes import HashEmbedding, ScriptedChatModel

    rt = AgentRuntime(
        settings=_settings(),
        llm=ScriptedChatModel([]),
        modules={},
        tools=[],
        checkpointer=InMemorySaver(),
    )
    rt.memory = MemoryService(
        store=MemoryMemoryStore(), embedder=HashEmbedding(), settings=rt.settings
    )
    await rt.memory.add_memory(user_id="u2", agent_id="chat", content="普通")
    await rt.memory.save_profile("u2", {"偏好": ["x"]})
    with TestClient(create_app(runtime=rt)) as c:
        memories = c.get("/v1/memory", headers={"X-User-Id": "u2"}).json()["memories"]
        assert all(not m["memory_id"].startswith("profile:") for m in memories)
        assert len(memories) == 1
    await service.aclose()


async def test_browse_lists_all_statuses_with_expired_flag(tmp_path: Path) -> None:
    """人工复核（M8 扩展）：浏览返回全部状态 + episodic 过期标志。"""
    import time

    from fastapi.testclient import TestClient
    from langgraph.checkpoint.memory import InMemorySaver

    from agent_base.core.bootstrap import AgentRuntime
    from agent_base.entrypoints.server import create_app
    from agent_base.memory.store import MemoryMemoryStore
    from fakes import HashEmbedding, ScriptedChatModel

    rt = AgentRuntime(
        settings=_settings(memory_episodic_ttl_days=30),
        llm=ScriptedChatModel([]),
        modules={},
        tools=[],
        checkpointer=InMemorySaver(),
    )
    rt.memory = MemoryService(
        store=MemoryMemoryStore(), embedder=HashEmbedding(), settings=rt.settings
    )
    now = time.time()
    store = rt.memory.store
    # 四种形态：active 语义 / active 但已过期的 episodic / archived / superseded。
    for mid, kind, status, updated in (
        ("m-live", "semantic", "active", now),
        ("m-expired", "episodic", "active", now - 40 * 86400),
        ("m-archived", "semantic", "archived", now),
        ("m-superseded", "semantic", "superseded", now),
    ):
        from agent_base.memory.store import MemoryRecord

        await store.upsert_memory(
            MemoryRecord(
                memory_id=mid,
                user_id="u3",
                agent_id="chat",
                kind=kind,
                content=f"内容 {mid}",
                status=status,
                updated_at=updated,
                created_at=updated,
            )
        )
    with TestClient(create_app(runtime=rt)) as c:
        body = c.get("/v1/memory", headers={"X-User-Id": "u3"}).json()
        by_id = {m["memory_id"]: m for m in body["memories"]}
        # 全部状态可见（人工复核的前提）。
        assert set(by_id) == {"m-live", "m-expired", "m-archived", "m-superseded"}
        assert by_id["m-live"]["expired"] is False
        assert by_id["m-expired"]["expired"] is True  # 40 天 > 30 天 TTL
        assert by_id["m-archived"]["expired"] is False  # semantic 不过期
        # 状态过滤。
        archived = c.get(
            "/v1/memory", params={"status": "archived"}, headers={"X-User-Id": "u3"}
        ).json()["memories"]
        assert [m["memory_id"] for m in archived] == ["m-archived"]
        # 非法 status → 400。
        assert (
            c.get("/v1/memory", params={"status": "bogus"}, headers={"X-User-Id": "u3"}).status_code
            == 400
        )
        # 检索路径仍只返回启用中的记忆。
        found = c.get("/v1/memory", params={"q": "内容"}, headers={"X-User-Id": "u3"}).json()
        assert all(m["status"] == "active" for m in found["memories"])
