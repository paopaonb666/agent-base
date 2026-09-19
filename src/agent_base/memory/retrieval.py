"""混合检索（M6b）：向量 + BM25 + 时间衰减 + 显著度。

四个信号（缺失分量按 0 计，分数为绝对相关度）：

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
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from agent_base.memory.store import PROFILE_ID_PREFIX, DocChunk, MemoryRecord, decode_embedding

# 召回候选集上限（H2 容量契约）：与 store.list_memories 的默认上限一致。
_CANDIDATE_LIMIT = 2000

logger = logging.getLogger(__name__)

# 混合权重默认值（和为 1）。向量是语义召回的主力，BM25 兜关键词，
# 时间与显著度做轻量排序先验。M6 加固后可经 MEMORY_WEIGHT_* 配置覆盖
# （Settings 校验和与范围）；调权是检索质量工程，改动应有评测依据。
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
        record for record in records if record.kind != "episodic" or record.updated_at >= cutoff
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


class Bm25Cache:
    """BM25 索引的进程内 LRU（锐评 #2 的修正实现）。

    与评审建议的"按用户缓存 + 写入失效"不同，这里键是**候选文本元组
    本身**（内容即身份）：候选集中任何一条内容增删改都会产生不同的
    键，旧键随之自然淘汰——零失效钩子，不存在忘记失效导致的脏索引。

    命中缓存跳过的是两段 Python 级 O(N) 循环：逐文档分词 + df 统计；
    键指纹计算是 C 级字符串哈希（~GB/s），量级差两个数量级以上。
    容量按条目数限制（一条 ≈ 一个作用域的候选集文本 + 分词结果，
    与 recall 本就要全量加载的记录同量级），默认 64 个作用域。
    """

    def __init__(self, maxsize: int = 64) -> None:
        self._maxsize = max(0, maxsize)
        self._entries: OrderedDict[tuple[str, ...], _Bm25] = OrderedDict()

    def scores(self, texts: Sequence[str], query_tokens: Sequence[str]) -> list[float]:
        """在 texts 候选集上对 query 打 BM25 分（命中则复用已建索引）。"""
        if self._maxsize == 0:  # 显式禁用：退回每次构建
            return _Bm25([tokenize(text) for text in texts]).scores(query_tokens)
        key = tuple(texts)
        index = self._entries.get(key)
        if index is None:
            index = _Bm25([tokenize(text) for text in texts])
            self._entries[key] = index
            while len(self._entries) > self._maxsize:
                self._entries.popitem(last=False)
        else:
            self._entries.move_to_end(key)
        return index.scores(query_tokens)


# 进程级共享缓存：单事件循环内无 await 的纯 dict 操作，无需加锁。
_BM25_CACHE = Bm25Cache()


@dataclass(frozen=True)
class ScoredChunk:
    """一个带混合分的知识库分块（分量语义同 ScoredMemory，无显著度）。"""

    chunk: DocChunk
    score: float
    components: dict[str, float] = field(default_factory=dict)


def _hybrid_scores(
    texts: Sequence[str],
    embeddings: Sequence[bytes | None],
    timestamps: Sequence[float],
    saliences: Sequence[float | None],
    query: str,
    query_embedding: Sequence[float] | None,
    *,
    now: float,
    half_life_days: float,
    weights: Mapping[str, float] | None = None,
) -> list[tuple[float, dict[str, float]]]:
    """混合打分的公共核心（memories 与 doc_chunks 共用）。

    缺失分量按 0 计、总分恒以全部权重为分母——分数是**绝对相关度**：
    既无关键词命中也无语义证据的候选，天花板只有 recency+salience
    （≤0.3），检索侧的最低分阈值才能可靠地把纯噪音挡在门外。
    """
    w = weights or _WEIGHTS
    bm25_raw = _BM25_CACHE.scores(texts, tokenize(query))
    bm25_max = max(bm25_raw, default=0.0)
    results: list[tuple[float, dict[str, float]]] = []
    for i, keyword_raw in enumerate(bm25_raw):
        components: dict[str, float] = {
            "keyword": (keyword_raw / bm25_max) if bm25_max > 0 else 0.0,
            "recency": recency_factor(timestamps[i], now=now, half_life_days=half_life_days),
        }
        salience = saliences[i]
        if salience is not None:
            components["salience"] = min(max(salience, 0.0), 1.0)
        # 向量分量：查询与候选必须同维可比。
        embedding = embeddings[i]
        if query_embedding is not None and embedding is not None:
            sim = cosine_similarity(query_embedding, decode_embedding(embedding))
            if sim is not None:
                # 余弦可为负（反向语义）：截到 0..1 只保留正向证据。
                components["vector"] = min(max(sim, 0.0), 1.0)
        score = sum(w[name] * value for name, value in components.items())
        results.append((score, components))
    return results


def score_memories(
    records: Sequence[MemoryRecord],
    query: str,
    query_embedding: Sequence[float] | None,
    *,
    now: float,
    half_life_days: float,
    weights: Mapping[str, float] | None = None,
) -> list[ScoredMemory]:
    """对记忆候选集统一打分。"""
    if not records:
        return []
    scored = _hybrid_scores(
        [record.content for record in records],
        [record.embedding for record in records],
        [record.updated_at for record in records],
        [record.salience for record in records],
        query,
        query_embedding,
        now=now,
        half_life_days=half_life_days,
        weights=weights,
    )
    return [
        ScoredMemory(record=record, score=score, components=components)
        for record, (score, components) in zip(records, scored, strict=True)
    ]


def score_chunks(
    chunks: Sequence[DocChunk],
    query: str,
    query_embedding: Sequence[float] | None,
    *,
    now: float,
    half_life_days: float,
    weights: Mapping[str, float] | None = None,
) -> list[ScoredChunk]:
    """对知识库分块统一打分（分块没有显著度，其余分量一致）。"""
    if not chunks:
        return []
    scored = _hybrid_scores(
        [chunk.text for chunk in chunks],
        [chunk.embedding for chunk in chunks],
        [chunk.created_at for chunk in chunks],
        [None] * len(chunks),
        query,
        query_embedding,
        now=now,
        half_life_days=half_life_days,
        weights=weights,
    )
    return [
        ScoredChunk(chunk=chunk, score=score, components=components)
        for chunk, (score, components) in zip(chunks, scored, strict=True)
    ]


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
    min_score: float = 0.0,
    weights: Mapping[str, float] | None = None,
) -> list[ScoredMemory]:
    """一次完整召回：取候选 → TTL 过滤 → 混合打分 → top-k + 访问记账。"""
    # 容量契约（H2）：候选集上限与 store 的列表上限一致。达到上限说明
    # 用户记忆量已超出当前"全量拉取 + 内存打分"实现的设计容量——更早
    # 的记忆本轮永远召不回，必须显式告警而不是静默截断。
    records = await store.list_memories(user_id, agent_id=agent_id, limit=_CANDIDATE_LIMIT)
    if len(records) >= _CANDIDATE_LIMIT:
        logger.warning(
            "memory: 召回候选达到上限 %d（user=%r, agent=%r）——更早的记忆不参与本轮检索；"
            "请考虑分片或引入候选下推索引（见 README 容量契约）",
            _CANDIDATE_LIMIT,
            user_id,
            agent_id,
        )
    timestamp = time.time() if now is None else now
    records = filter_expired(records, episodic_ttl_days, now=timestamp)
    # 用户画像不是可召回记忆：它由上下文组装单独、整体注入（M6d）。
    records = [record for record in records if not record.memory_id.startswith(PROFILE_ID_PREFIX)]
    if not records:
        return []
    query_embedding: list[float] | None = None
    if embedder is not None:
        vectors = await embedder.embed([query])
        if vectors:
            query_embedding = vectors[0]
    scored = score_memories(
        records,
        query,
        query_embedding,
        now=timestamp,
        half_life_days=half_life_days,
        weights=weights,
    )
    scored.sort(key=lambda item: item.score, reverse=True)
    # 最低分门槛：双条件——总分达标，且至少有一个**相关性证据**
    # （关键词命中或语义相似）。纯 recency+salience 凑数（天花板 0.3）
    # 的噪音记忆，宁可空手也不召回（UI 实测发现 F3）。
    if min_score > 0:
        scored = [
            item
            for item in scored
            if item.score >= min_score
            and (item.components.get("keyword", 0.0) > 0 or item.components.get("vector", 0.0) > 0)
        ]
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
    "ScoredChunk",
    "ScoredMemory",
    "cosine_similarity",
    "filter_expired",
    "recall_memories",
    "recency_factor",
    "score_chunks",
    "score_memories",
    "tokenize",
]
