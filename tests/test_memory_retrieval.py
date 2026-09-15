"""混合检索（M6b）的测试：分词、BM25、衰减、混合打分与完整召回。"""

from __future__ import annotations

import pytest

from agent_base.memory.retrieval import (
    cosine_similarity,
    filter_expired,
    recall_memories,
    recency_factor,
    score_memories,
    tokenize,
)
from agent_base.memory.store import MemoryMemoryStore, MemoryRecord, encode_embedding
from fakes import HashEmbedding


def test_tokenize_mixed_language() -> None:
    tokens = tokenize("用户偏好 Python 3")
    assert "python" in tokens
    assert "用户" in tokens and "偏好" in tokens
    # 中文按二元组展开。
    assert "户偏" in tokens


def test_cosine_similarity() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([1.0], [1.0, 0.0]) is None  # 维度不一致
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) is None  # 零向量
    assert cosine_similarity([], []) is None


def test_recency_factor_halving() -> None:
    now = 1_000_000.0
    assert recency_factor(now, now=now, half_life_days=10) == 1.0
    # 半衰期后权重恰好减半。
    assert recency_factor(now - 10 * 86400, now=now, half_life_days=10) == 0.5


def test_filter_expired_only_affects_episodic() -> None:
    now = 100.0 * 86400.0  # 100 天后：updated_at=0 的记录已超过 30 天 TTL
    old_episodic = MemoryRecord(
        memory_id="a", user_id="u", agent_id="chat", kind="episodic", content="旧", updated_at=0.0
    )
    old_semantic = MemoryRecord(
        memory_id="b",
        user_id="u",
        agent_id="chat",
        kind="semantic",
        content="老但长期",
        updated_at=0.0,
    )
    fresh = MemoryRecord(
        memory_id="c", user_id="u", agent_id="chat", kind="episodic", content="新", updated_at=now
    )
    kept = filter_expired([old_episodic, old_semantic, fresh], 30, now=now)
    assert [r.memory_id for r in kept] == ["b", "c"]
    # TTL=0 表示不过期。
    assert len(filter_expired([old_episodic], 0, now=now)) == 1


def test_score_memories_relevant_keyword_ranks_first() -> None:
    now = 1_000_000.0
    hit = MemoryRecord(
        memory_id="hit",
        user_id="u",
        agent_id="chat",
        kind="semantic",
        content="用户的部署环境是 Ubuntu 22.04 与 docker compose",
        updated_at=now,
        salience=0.5,
    )
    miss = MemoryRecord(
        memory_id="miss",
        user_id="u",
        agent_id="chat",
        kind="semantic",
        content="用户喜欢喝美式咖啡",
        updated_at=now,
        salience=0.5,
    )
    scored = score_memories(
        [hit, miss], "docker compose 怎么部署", None, now=now, half_life_days=14
    )
    by_id = {item.record.memory_id: item for item in scored}
    assert by_id["hit"].score > by_id["miss"].score
    assert by_id["hit"].components["keyword"] > 0.0


def test_score_memories_vector_component_and_renormalization() -> None:
    now = 1_000_000.0
    with_vec = MemoryRecord(
        memory_id="v",
        user_id="u",
        agent_id="chat",
        kind="semantic",
        content="用户在做记忆系统",
        embedding=encode_embedding([1.0, 0.0]),
        updated_at=now,
    )
    without_vec = MemoryRecord(
        memory_id="n",
        user_id="u",
        agent_id="chat",
        kind="semantic",
        content="用户在做记忆系统",
        updated_at=now,
    )
    # 查询有向量：有向量的候选吃满向量分量。
    scored = score_memories(
        [with_vec, without_vec], "记忆系统", [1.0, 0.0], now=now, half_life_days=14
    )
    by_id = {item.record.memory_id: item for item in scored}
    assert by_id["v"].components["vector"] == pytest.approx(1.0)
    assert "vector" not in by_id["n"].components
    assert by_id["v"].score > by_id["n"].score
    # 查询无向量：全部候选走关键词+时间+显著度，分数仍归一到 0..1。
    scored = score_memories([with_vec, without_vec], "记忆系统", None, now=now, half_life_days=14)
    assert all(item.score <= 1.0 for item in scored)


async def test_recall_memories_end_to_end() -> None:
    store = MemoryMemoryStore()
    now = 1_000_000.0
    embedder = HashEmbedding()
    for i, content in enumerate(
        [
            "用户的项目使用 LangGraph 框架",
            "用户喜欢深夜写代码",
            "用户的猫叫团子",
        ]
    ):
        vectors = await embedder.embed([content])
        await store.upsert_memory(
            MemoryRecord(
                memory_id=f"m{i}",
                user_id="alice",
                agent_id="chat",
                kind="semantic",
                content=content,
                embedding=encode_embedding(vectors[0]),
                updated_at=now - i * 100.0,
            )
        )
    top = await recall_memories(
        store,
        user_id="alice",
        agent_id="chat",
        query="LangGraph 项目情况",
        embedder=embedder,
        top_k=2,
        episodic_ttl_days=30,
        half_life_days=14,
        now=now,
    )
    assert len(top) == 2
    assert top[0].record.content == "用户的项目使用 LangGraph 框架"
    # 访问记账被触发。
    touched = await store.get_memory(top[0].record.memory_id)
    assert touched is not None and touched.access_count == 1
    # user 隔离：别人的记忆不可见。
    empty = await recall_memories(
        store,
        user_id="bob",
        agent_id="chat",
        query="LangGraph",
        embedder=None,
        top_k=5,
        episodic_ttl_days=30,
        half_life_days=14,
        now=now,
    )
    assert empty == []
    # 最低分阈值：毫无关联的提问（纯靠 recency/salience 凑数）不召回。
    noisy = await recall_memories(
        store,
        user_id="alice",
        agent_id="chat",
        query="量子纠缠的实验装置",
        embedder=None,
        top_k=5,
        episodic_ttl_days=30,
        half_life_days=14,
        now=now,
        min_score=0.3,
    )
    assert noisy == []


# ─────────────────────── BM25 缓存（锐评 #2） ───────────────────────


def test_bm25_cache_hits_and_rebuilds() -> None:
    from agent_base.memory.retrieval import Bm25Cache

    cache = Bm25Cache(maxsize=4)
    docs = ["用户使用 LangGraph", "用户喜欢猫"]
    query = tokenize("LangGraph")
    first = cache.scores(docs, query)
    # 命中缓存：结果一致。
    second = cache.scores(docs, query)
    assert first == second
    # 内容变化 → 不同键 → 重建（行为与直接构建一致）。
    changed = cache.scores([*docs, "新增一条记忆"], query)
    direct = score_memories.__wrapped__ if hasattr(score_memories, "__wrapped__") else None
    assert len(changed) == 3
    del direct
    # 未出现过的键不污染既有条目。
    assert cache.scores(docs, query) == first


def test_bm25_cache_lru_eviction_and_disable() -> None:
    from agent_base.memory.retrieval import Bm25Cache

    cache = Bm25Cache(maxsize=2)
    cache.scores(["a 文档"], tokenize("a"))
    cache.scores(["b 文档"], tokenize("b"))
    cache.scores(["c 文档"], tokenize("c"))  # 淘汰最久的 a
    cache.scores(["a 文档"], tokenize("a"))  # 重建仍正确
    assert len(cache._entries) <= 2
    # maxsize=0 显式禁用：不驻留任何条目，行为与直接构建一致。
    from agent_base.memory.retrieval import _Bm25

    disabled = Bm25Cache(maxsize=0)
    direct = _Bm25([tokenize("x")]).scores(tokenize("x"))
    assert disabled.scores(["x"], tokenize("x")) == direct
    assert len(disabled._entries) == 0
