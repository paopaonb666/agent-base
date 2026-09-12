"""搜索结果管线（M3，移植自 chat-agent）：缓存 → 质量过滤 → 相关性评分。

低质结果模式（URL/标题黑名单）刻意留空作为扩展点：chat-agent 里的
"字典站/拼音站"名单是它自己的业务垃圾源，不属于基座。部署方按需
覆写 ``LOW_QUALITY_URL_PATTERNS`` / ``LOW_QUALITY_TITLE_PATTERNS``。
"""

from __future__ import annotations

import time
from collections import OrderedDict

from agent_base.tools.search.engines import SearchResult

LOW_QUALITY_URL_PATTERNS: tuple[str, ...] = ()
LOW_QUALITY_TITLE_PATTERNS: tuple[str, ...] = ()

# 缓存条目上限：防止长驻进程在大量不同 query 下内存无界增长；
# 淘汰最旧的条目（搜索缓存被丢弃是安全的，只是多一次真实搜索）。
CACHE_MAX_ENTRIES = 256


class SearchCache:
    """进程内 TTL 缓存：命中返回缓存结果，过期/未命中返回 None。

    刻意不进 MySQL——搜索缓存丢了无碍正确性，不值得加存储依赖。
    """

    def __init__(self, ttl_seconds: float) -> None:
        self._ttl = ttl_seconds
        self._entries: OrderedDict[str, tuple[float, list[SearchResult]]] = OrderedDict()

    def get(self, query: str) -> list[SearchResult] | None:
        entry = self._entries.get(query)
        if entry is None:
            return None
        at, results = entry
        if time.monotonic() - at >= self._ttl:
            self._entries.pop(query, None)
            return None
        self._entries.move_to_end(query)
        return results

    def put(self, query: str, results: list[SearchResult]) -> None:
        self._entries[query] = (time.monotonic(), results)
        self._entries.move_to_end(query)
        while len(self._entries) > CACHE_MAX_ENTRIES:
            self._entries.popitem(last=False)


def is_low_quality(result: SearchResult) -> bool:
    """按 URL/标题模式过滤低质结果；默认无模式（扩展点）。"""
    url = result.url.lower()
    title = result.title.lower()
    return any(p in url for p in LOW_QUALITY_URL_PATTERNS) or any(
        p in title for p in LOW_QUALITY_TITLE_PATTERNS
    )


def score_relevance(result: SearchResult, query: str) -> float:
    """query 词项在 title/snippet 中的匹配度；标题匹配权重更高。"""
    terms = [t for t in query.lower().split() if len(t) > 1]
    if not terms:
        return 0.0
    title = result.title.lower()
    snippet = result.snippet.lower()
    matches = sum(1 for t in terms if t in title or t in snippet)
    title_matches = sum(1 for t in terms if t in title)
    return matches + title_matches * 2.0


def filter_and_rank(
    results: list[SearchResult], query: str, max_results: int
) -> list[SearchResult]:
    """过滤低质结果、按相关性降序排序并重新编号。"""
    scored = [(score_relevance(r, query), r) for r in results if not is_low_quality(r)]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    ranked = [r for _, r in scored[:max_results]]
    return [
        SearchResult(
            title=r.title,
            url=r.url,
            snippet=r.snippet,
            position=i + 1,
            engine=r.engine,
        )
        for i, r in enumerate(ranked)
    ]


def format_web_context(results: list[SearchResult]) -> str:
    """把结果渲染为带编号的引用上下文（编号供模型在回复中标注来源）。"""
    if not results:
        return ""
    lines = ["以下是从互联网搜索到的相关信息：\n"]
    for r in results:
        lines.append(f"[{r.position}] {r.title}")
        lines.append(f"    URL: {r.url}")
        lines.append(f"    摘要: {r.snippet} (来源: {r.engine})")
        lines.append("")
    return "\n".join(lines)
