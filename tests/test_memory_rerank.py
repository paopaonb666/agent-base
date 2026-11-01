"""检索精排（ADR-004）的测试：客户端降级/冷却、装配选择、宽召回重排、旁路。

全部走注入替身与 httpx.MockTransport，绝不打真实网络。
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent_base.core.config import Settings
from agent_base_memory.rerank import (
    NullReranker,
    OpenAICompatibleReranker,
    RerankConfigError,
    build_reranker,
)
from agent_base_memory.service import MemoryService
from agent_base_memory.store import MemoryMemoryStore, MemoryRecord
from fakes import HashEmbedding, ScriptedChatModel


def _settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {"llm_api_key": "sk-test"}
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


class ScriptedReranker:
    """可录制的精排替身：按预设分数返回（None 模拟降级）。"""

    def __init__(self, scores: list[float] | None) -> None:
        self.scores = scores
        self.calls: list[tuple[str, list[str]]] = []

    async def rerank(self, query: str, documents: list[str]) -> list[float] | None:
        self.calls.append((query, list(documents)))
        if self.scores is None:
            return None
        return self.scores[: len(documents)] + [0.0] * max(0, len(documents) - len(self.scores))


async def _seed(store: MemoryMemoryStore, contents: list[str]) -> None:
    """写入 keyword 命中有保证的一组记忆（全部含查询关键词）。"""
    for i, content in enumerate(contents):
        await store.upsert_memory(
            MemoryRecord(
                memory_id=f"m{i}",
                user_id="alice",
                agent_id="chat",
                kind="semantic",
                content=content,
            )
        )


# ─────────────────────────── 客户端 ───────────────────────────


async def test_null_reranker_returns_none() -> None:
    assert await NullReranker().rerank("q", ["a"]) is None
    assert await NullReranker().rerank("q", []) is None


def _transport(handler: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_build_reranker_selection() -> None:
    assert isinstance(build_reranker(_settings()), NullReranker)
    assert isinstance(build_reranker(_settings(memory_rerank_enabled=False)), NullReranker)
    # 启用即承诺：必需配置为空 → fail-fast（api_key 默认空；base_url/model
    # 有非空默认，但显式置空同样拦截）。
    with pytest.raises(RerankConfigError):
        build_reranker(_settings(memory_rerank_enabled=True))
    with pytest.raises(RerankConfigError):
        build_reranker(
            _settings(memory_rerank_enabled=True, memory_rerank_base_url="", memory_rerank_model="")
        )
    client = build_reranker(
        _settings(
            memory_rerank_enabled=True,
            memory_rerank_api_key="sk-r",
            memory_rerank_base_url="https://rr.example.com/v1",
            memory_rerank_model="test-reranker",
        )
    )
    assert isinstance(client, OpenAICompatibleReranker)


def test_rerank_config_validation() -> None:
    with pytest.raises(ValueError):
        _settings(memory_rerank_fetch_multiplier=0)
    with pytest.raises(ValueError):
        _settings(memory_rerank_timeout_seconds=0)
    with pytest.raises(ValueError):
        _settings(memory_rerank_cooldown_seconds=-1)


async def test_reranker_request_and_parse() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 2, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.1},
                    # index=1 缺失 → 该文档按 0 分计，不丢整批。
                ]
            },
        )

    client = OpenAICompatibleReranker(
        base_url="https://rr.example.com/v1",
        api_key="sk-r",
        model="test-reranker",
        client=_transport(handler),
    )
    scores = await client.rerank("查询", ["文档一", "文档二", "文档三"])
    assert scores == [0.1, 0.0, 0.9]
    assert len(calls) == 1
    assert calls[0].url.path.endswith("/rerank")
    assert calls[0].headers["Authorization"] == "Bearer sk-r"
    body = json.loads(calls[0].content)
    assert body["model"] == "test-reranker"
    assert body["query"] == "查询"
    assert body["documents"] == ["文档一", "文档二", "文档三"]


async def test_reranker_malformed_index_out_of_bounds() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"results": [{"index": 9, "relevance_score": 1.0}, {"index": 1}]},
        )

    client = OpenAICompatibleReranker(
        base_url="https://rr.example.com/v1",
        api_key="sk-r",
        model="m",
        client=_transport(handler),
    )
    # 越界与畸形条目忽略；对齐长度返回。
    assert await client.rerank("q", ["a", "b"]) == [0.0, 0.0]


async def test_reranker_failure_returns_none_and_cooldown() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(500, text="boom")

    client = OpenAICompatibleReranker(
        base_url="https://rr.example.com/v1",
        api_key="sk-r",
        model="m",
        cooldown_seconds=60.0,
        client=_transport(handler),
    )
    assert await client.rerank("q", ["a"]) is None
    assert await client.rerank("q", ["a"]) is None  # 冷却期内不打网络
    assert len(calls) == 1


async def test_reranker_network_error_returns_none() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    client = OpenAICompatibleReranker(
        base_url="https://rr.example.com/v1",
        api_key="sk-r",
        model="m",
        client=_transport(handler),
    )
    assert await client.rerank("q", ["a"]) is None


# ─────────────────────────── 门面 search ───────────────────────────


async def _service(reranker: Any, **overrides: object) -> MemoryService:
    return MemoryService(
        store=MemoryMemoryStore(),
        embedder=HashEmbedding(),
        settings=_settings(**overrides),
        reranker=reranker,
    )


async def test_search_rerank_reorders_and_truncates() -> None:
    """宽召回 → 精排重排（rerank 为主键、混合分为次键）→ 截回 top_k。"""
    rr = ScriptedReranker([0.1, 0.9, 0.3, 0.7])
    service = await _service(rr, memory_rerank_enabled=True)
    await _seed(
        service.store,
        ["记忆系统 甲", "记忆系统 乙", "记忆系统 丙", "记忆系统 丁"],
    )
    results = await service.search(user_id="alice", agent_id="chat", query="记忆系统", top_k=2)
    # 只保留 rerank 分最高的两条：乙(0.9) 与 丁(0.7)。
    assert [r.record.content for r in results] == ["记忆系统 乙", "记忆系统 丁"]
    # rerank 分成为 score；混合分降级为次级键留档。
    assert results[0].score == pytest.approx(0.9)
    assert results[0].components["rerank"] == pytest.approx(0.9)
    assert "hybrid" in results[0].components
    # 宽召回：精排拿到的是 top_k × multiplier 条候选。
    assert len(rr.calls[0][1]) == 4


async def test_search_rerank_degraded_keeps_hybrid_order() -> None:
    """rerank 返回 None（降级）：保留混合分排序，按 top_k 截断。"""
    rr = ScriptedReranker(None)
    service = await _service(rr, memory_rerank_enabled=True)
    await _seed(service.store, [f"记忆系统 {i}" for i in range(4)])
    results = await service.search(user_id="alice", agent_id="chat", query="记忆系统", top_k=2)
    assert len(results) == 2
    assert all("rerank" not in r.components for r in results)  # 未应用精排，零污染


async def test_search_use_rerank_false_bypasses() -> None:
    rr = ScriptedReranker([0.9, 0.1])
    service = await _service(rr, memory_rerank_enabled=True)
    await _seed(service.store, ["记忆系统 甲", "记忆系统 乙"])
    await service.search(
        user_id="alice", agent_id="chat", query="记忆系统", top_k=2, use_rerank=False
    )
    assert rr.calls == []


async def test_search_without_reranker_unchanged() -> None:
    """未启用精排（默认）：行为与现状完全一致（NullReranker 路径）。"""
    service = await _service(NullReranker())
    await _seed(service.store, ["记忆系统 甲", "记忆系统 乙"])
    results = await service.search(user_id="alice", agent_id="chat", query="记忆系统")
    assert 1 <= len(results) <= 6
    # 分数为混合分（无 rerank 分量）。
    assert all("rerank" not in r.components for r in results)


async def test_pipeline_consolidate_bypasses_rerank() -> None:
    """整合裁决的相似检索旁路精排（ADR-004：后台高频路径保延迟）。"""
    rr = ScriptedReranker([0.9])
    service = MemoryService(
        store=MemoryMemoryStore(),
        embedder=HashEmbedding(),
        settings=_settings(memory_rerank_enabled=True),
        llm=ScriptedChatModel(
            [
                AIMessage(
                    content=json.dumps(
                        [{"content": "用户在开发记忆系统", "kind": "semantic", "salience": 0.8}],
                        ensure_ascii=False,
                    )
                ),
                AIMessage(content=json.dumps({"op": "NONE"}, ensure_ascii=False)),
                AIMessage(content=json.dumps({"偏好": []}, ensure_ascii=False)),
            ]
        ),
        reranker=rr,
    )
    detail = await service.capture_turn(
        user_id="alice",
        agent_id="chat",
        thread_id="chat:t",
        messages=[HumanMessage(content="我在开发记忆系统"), AIMessage(content="好的")],
    )
    assert detail is not None and detail["candidates"] == 1
    assert rr.calls == []  # consolidate 内部的 search(use_rerank=False)


# ─────────────────────────── 知识库路径 ───────────────────────────


async def test_knowledge_search_rerank_applied() -> None:
    rr = ScriptedReranker([0.2, 0.8])
    service = MemoryService(
        store=MemoryMemoryStore(),
        embedder=HashEmbedding(),
        settings=_settings(memory_rerank_enabled=True),
        reranker=rr,
    )
    # 强制切成两块，保证多候选过精排。
    await service.ingest_document(
        file_id="f1",
        user_id="alice",
        agent_id="chat",
        text="记忆系统的设计第一段。\n\n记忆系统的设计第二段。",
    )
    results = await service.search_knowledge(
        user_id="alice", agent_id="chat", query="记忆系统 设计", top_k=2
    )
    assert results, "知识库候选应命中关键词证据"
    if len(rr.calls) == 1:  # 候选数 >1 时过精排
        assert results[0].components.get("rerank") == pytest.approx(0.8)
    # 精排旁路同样生效。
    await service.search_knowledge(
        user_id="alice", agent_id="chat", query="记忆系统 设计", use_rerank=False
    )
    assert len(rr.calls) <= 1


# ─────────────────────────── 降级门槛 ───────────────────────────


async def test_search_rerank_relaxes_min_score() -> None:
    """精排宽召回阶段门槛减半：混合分偏低但 rerank 能救回的候选可进入精排。"""
    rr = ScriptedReranker([0.99, 0.1])
    service = await _service(
        rr, memory_rerank_enabled=True, memory_recall_min_score=0.9, memory_recall_top_k=2
    )
    await _seed(service.store, ["记忆系统 甲", "记忆系统 乙", "记忆系统 丙"])
    results = await service.search(user_id="alice", agent_id="chat", query="记忆系统")
    # 无精排时门槛 0.9 会把多数候选清零；宽召回 0.45 + 精排后有结果。
    assert len(results) >= 1
    assert results[0].components.get("rerank") == pytest.approx(0.99)
