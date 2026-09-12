"""混合检索（M6b）：向量 + BM25 + 时间衰减 + 显著度。

四个信号（可缺失分量自动重归一，总分保持 0..1 可比）：

- **向量余弦**——语义召回（同义改写也能命中）；只在查询与候选双方
  都有**同维**向量时参与；
- **BM25**——关键词精确召回（型号、人名、代码标识符等语义模型容易
  漏的东西）；中文用二元组（bigram）、英文用词元；
- **时间衰减**——指数半衰期（默认 14 天），久未访问的记忆权重走低；
- **显著度**——写入时由管线/人工标定的 0..1 权重。

遗忘策略：episodic 记忆按 ``MEMORY_EPISODIC_TTL_DAYS`` 过滤（0 表示
不过期）；semantic/procedural 不过期——它们的价值恰恰在长期。
"""

from __future__ import annotations

import logging
import math
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from agent_base.memory.store import MemoryRecord, decode_embedding

logger = logging.getLogger(__name__)

# 混合权重（和为 1）。向量是语义召回的主力，BM25 兜关键词，时间与
# 显著度做轻量排序先验——刻意没有做成配置：调权是检索质量工程，参数
# 面暴露得越多越难推理，先以代码常量锁死（改动走评审）。
W_VECTOR = 0.40
W_KEYWORD = 0.30
W_RECENCY = 0.15
W_SALIENCE = 0.15
_WEIGHTS: dict[str, float] = {
    "vector": W_VECTOR,
    "keyword": W_KEYWORD,
    "recency": W_RECENCY,
    "salience": W_SALIENCE,
}

# BM25 经典参数。
_BM25_K1 = 1.5
_BM25_B = 0.75

_ASCII_WORD_RE = re.compile(r"[a-zA-Z0-9_]+")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    """检索分词：ASCII 词（小写）+ 中文二元组。

    中文无空格分词，二元组是无词典检索的实用折中——"用户偏好"产出
    ``用户``/``户偏``/``偏好``，足以命中改写查询。
    """
    tokens: list[str] = []
    tokens.extend(word.lower() for word in _ASCII_WORD_RE.findall(text))
    for run in _CJK_RUN_RE.findall(text):
        if len(run) == 1:
            tokens.append(run)
            continue
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float | None:
    """余弦相似度；维度不一致或零向量返回 None（不可比即不计分）。"""
    if len(a) != len(b) or not a:
        return None
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a <= 0.0 or norm_b <= 0.0:
        return None
    return dot / math.sqrt(norm_a * norm_b)


def recency_factor(updated_at: float, *, now: float, half_life_days: float) -> float:
    """时间衰减因子：``0.5 ** (age_days / half_life)``，永不归零。"""
    age_days = max(0.0, now - updated_at) / 86400.0
    return float(0.5 ** (age_days / max(half_life_days, 1e-9)))


def filter_expired(
    records: Sequence[MemoryRecord], episodic_ttl_days: int, *, now: float
) -> list[MemoryRecord]:
    """按类别应用 TTL：episodic 过期剔除，其余类别不过期。"""
    if episodic_ttl_days <= 0:
        return list(records)
    cutoff = now - episodic_ttl_days * 86400.0
    return [
        record
        for record in records
        if record.kind != "episodic" or record.updated_at >= cutoff
    ]


class _Bm25:
    """极简 BM25（纯 Python）：候选集量级是单用户的记忆条数（千级），

    全量内存打分完全够用，不引入索引基础设施。
    """

    def __init__(self, docs_tokens: list[list[str]]) -> None:
        self._docs = docs_tokens
        self._avgdl = (sum(len(d) for d in docs_tokens) / len(docs_tokens)) if docs_tokens else 0.0
        self._df: dict[str, int] = {}
        for doc in docs_tokens:
            for term in set(doc):
                self._df[term] = self._df.get(term, 0) + 1

    def scores(self, query_tokens: Sequence[str]) -> list[float]:
        n = len(self._docs)
        if n == 0:
            return []
        raw: list[float] = []
        for doc in self._docs:
            tf: dict[str, int] = {}
            for term in doc:
                tf[term] = tf.get(term, 0) + 1
            score = 0.0
            for term in query_tokens:
                count = tf.get(term)
                if not count:
                    continue
                df = self._df.get(term, 0)
                idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
                length_ratio = len(doc) / (self._avgdl or 1.0)
                denom = count + _BM25_K1 * (1.0 - _BM25_B + _BM25_B * length_ratio)
                score += idf * count * (_BM25_K1 + 1.0) / denom
            raw.append(score)
        return raw


@dataclass(frozen=True)
class ScoredMemory:
    """一条带混合分的记忆；``components`` 保留各分量的原始值供调试。"""

    record: MemoryRecord
    score: float
    components: dict[str, float] = field(default_factory=dict)


def score_memories(
    records: Sequence[MemoryRecord],
    query: str,
    query_embedding: Sequence[float] | None,
    *,
    now: float,
    half_life_days: float,
) -> list[ScoredMemory]:
    """对候选集统一打分；分量缺失（无向量/无关键词命中）自动重归一。"""
    if not records:
        return []
    docs_tokens = [tokenize(record.content) for record in records]
    bm25_raw = _Bm25(docs_tokens).scores(tokenize(query))
    bm25_max = max(bm25_raw, default=0.0)

    scored: list[ScoredMemory] = []
    for record, keyword_raw in zip(records, bm25_raw, strict=True):
        components: dict[str, float] = {
            "keyword": (keyword_raw / bm25_max) if bm25_max > 0 else 0.0,
            "recency": recency_factor(record.updated_at, now=now, half_life_days=half_life_days),
            "salience": min(max(record.salience, 0.0), 1.0),
        }
        # 向量分量：查询与候选必须同维可比。
        if query_embedding is not None and record.embedding is not None:
            sim = cosine_similarity(query_embedding, decode_embedding(record.embedding))
            if sim is not None:
                # 余弦可为负（反向语义）：截到 0..1 只保留正向证据。
                components["vector"] = min(max(sim, 0.0), 1.0)
        total_weight = sum(_WEIGHTS[name] for name in components)
        score = sum(_WEIGHTS[name] * value for name, value in components.items())
        if total_weight > 0:
            score /= total_weight
        scored.append(ScoredMemory(record=record, score=score, components=components))
    return scored


@runtime_checkable
class _Embedder(Protocol):
    """retrieval 对 embedding 客户端的窄依赖（便于测试替身）。"""

    async def embed(self, texts: Sequence[str]) -> list[list[float]] | None: ...


async def recall_memories(
    store: Any,
    *,
    user_id: str,
    agent_id: str | None,
    query: str,
    embedder: _Embedder | None,
    top_k: int,
    episodic_ttl_days: int,
    half_life_days: float,
    now: float | None = None,
) -> list[ScoredMemory]:
    """一次完整召回：取候选 → TTL 过滤 → 混合打分 → top-k + 访问记账。"""
    records = await store.list_memories(user_id, agent_id=agent_id)
    timestamp = time.time() if now is None else now
    records = filter_expired(records, episodic_ttl_days, now=timestamp)
    if not records:
        return []
    query_embedding: list[float] | None = None
    if embedder is not None:
        vectors = await embedder.embed([query])
        if vectors:
            query_embedding = vectors[0]
    scored = score_memories(
        records, query, query_embedding, now=timestamp, half_life_days=half_life_days
    )
    scored.sort(key=lambda item: item.score, reverse=True)
    top = scored[: max(1, top_k)]
    # 访问记账（被召回本身是记忆"还活着"的证据）：失败只记日志。
    try:
        await store.touch_memories([item.record.memory_id for item in top])
    except Exception:
        logger.warning("memory: 召回记账失败", exc_info=True)
    return top


__all__ = [
    "W_KEYWORD",
    "W_RECENCY",
    "W_SALIENCE",
    "W_VECTOR",
    "ScoredMemory",
    "cosine_similarity",
    "filter_expired",
    "recall_memories",
    "recency_factor",
    "score_memories",
    "tokenize",
]
