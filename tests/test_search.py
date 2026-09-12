"""web_search 与搜索管线的测试（M3）：引擎降级、缓存、评分、工具行为。

全部走替身引擎 / MockTransport，不出真实网络。
"""

from __future__ import annotations

import httpx
import pytest

from agent_base.core.config import Settings
from agent_base.tools.search import pipeline
from agent_base.tools.search.engines import (
    DuckDuckGoEngine,
    FallbackSearchManager,
    SearchEngine,
    SearchResult,
    TavilyEngine,
    build_engines,
)
from agent_base.tools.search.pipeline import SearchCache, filter_and_rank, format_web_context
from agent_base.tools.search.tool import _web_search_available, build_web_search_tool


def _result(title: str, url: str = "https://example.com/a", snippet: str = "……") -> SearchResult:
    return SearchResult(title=title, url=url, snippet=snippet, position=1, engine="fake")


class _StubDDGS:
    """``ddgs.DDGS`` 的替身：ddgs 9.x 是动态代理，无法 monkeypatch。"""

    def __init__(self, rows: list[dict[str, str]]) -> None:
        self._rows = rows
        self.seen_backends: list[str] = []

    def __enter__(self) -> _StubDDGS:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def text(self, query: str, max_results: int, **kwargs: object) -> list[dict[str, str]]:
        assert max_results == 5
        if "backend" in kwargs:
            self.seen_backends.append(str(kwargs["backend"]))
        return self._rows


class _FakeEngine(SearchEngine):
    """脚本化引擎：按序返回预设结果或抛出异常。"""

    name = "fake"

    def __init__(self, outcomes: list[list[SearchResult] | Exception]) -> None:
        self._outcomes = outcomes
        self.calls: list[str] = []

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        self.calls.append(query)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, llm_api_key="sk-test", **overrides)  # type: ignore[arg-type]


def _make_tool(engine: SearchEngine, settings: Settings | None = None) -> object:
    """用指定替身引擎构建 web_search 工具（注入 engine_factory）。"""
    return build_web_search_tool(settings or _settings(), engine_factory=lambda p, **kw: [engine])


def _mute_events(monkeypatch: pytest.MonkeyPatch) -> None:
    """让工具的事件发射在测试中变成空操作（事件断言见专门用例）。"""
    monkeypatch.setattr(
        "agent_base.tools.search.tool.emit_step", lambda *a, **kw: None, raising=False
    )
    monkeypatch.setattr(
        "agent_base.tools.search.tool.emit_sources", lambda *a, **kw: None, raising=False
    )


# ── FallbackSearchManager ────────────────────────────────────────────


async def test_fallback_first_engine_wins() -> None:
    a = _FakeEngine([[_result("from a")]])
    b = _FakeEngine([[_result("from b")]])
    results = await FallbackSearchManager([a, b]).search("q", 5)
    assert [r.title for r in results] == ["from a"]
    assert a.calls == ["q"] and b.calls == []


async def test_fallback_skips_failing_and_empty_engines() -> None:
    a = _FakeEngine([RuntimeError("boom")])
    b = _FakeEngine([[]])  # 返回空视为未命中，继续降级
    c = _FakeEngine([[_result("from c")]])
    results = await FallbackSearchManager([a, b, c]).search("q", 5)
    assert [r.title for r in results] == ["from c"]


async def test_fallback_all_fail_returns_empty() -> None:
    a = _FakeEngine([RuntimeError("boom")])
    assert await FallbackSearchManager([a]).search("q", 5) == []


# ── 引擎实现 ─────────────────────────────────────────────────────────


async def test_tavily_engine_maps_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer tvly-key"
        return httpx.Response(
            200,
            json={"results": [{"title": "T", "url": "https://t.co/1", "content": "body"}]},
        )

    engine = TavilyEngine(api_key="tvly-key", transport=httpx.MockTransport(handler))
    results = await engine.search("query", 5)
    assert results[0] == SearchResult(
        title="T", url="https://t.co/1", snippet="body", position=1, engine="tavily"
    )


async def test_tavily_engine_propagates_http_errors() -> None:
    engine = TavilyEngine(
        api_key="k", transport=httpx.MockTransport(lambda request: httpx.Response(500))
    )
    with pytest.raises(httpx.HTTPStatusError):
        await engine.search("query", 5)


async def test_duckduckgo_engine_maps_response() -> None:
    rows = [{"title": "T", "href": "https://ddg.co/1", "body": "snippet"}]
    stub = _StubDDGS(rows)
    engine = DuckDuckGoEngine(ddgs_factory=lambda: stub)
    results = await engine.search("query", 5)
    assert results[0] == SearchResult(
        title="T", url="https://ddg.co/1", snippet="snippet", position=1, engine="duckduckgo"
    )
    # 默认钉选 duckduckgo 上游：避免 auto 模式扇出不可达引擎拖死聚合。
    assert stub.seen_backends == ["duckduckgo"]


async def test_duckduckgo_engine_honors_backend_override() -> None:
    stub = _StubDDGS([{"title": "T", "href": "https://bing.co/1", "body": "s"}])
    engine = DuckDuckGoEngine(ddgs_factory=lambda: stub, backend="bing")
    await engine.search("query", 5)
    assert stub.seen_backends == ["bing"]


def test_build_engines_skips_unusable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent_base.tools.search.engines.ddgs_available", lambda: False)
    assert build_engines(["tavily", "duckduckgo"], tavily_api_key="") == []
    assert [e.name for e in build_engines(["tavily"], tavily_api_key="tvly-key")] == ["tavily"]
    assert build_engines(["bogus"], tavily_api_key="") == []
    monkeypatch.setattr("agent_base.tools.search.engines.ddgs_available", lambda: True)
    assert [e.name for e in build_engines(["duckduckgo"], tavily_api_key="")] == ["duckduckgo"]


# ── 管线 ─────────────────────────────────────────────────────────────


def test_cache_hit_and_ttl_expiry(monkeypatch: pytest.MonkeyPatch) -> None:
    cache = SearchCache(ttl_seconds=0.05)
    cache.put("q", [_result("cached")])
    assert cache.get("q") is not None
    # 时钟前移 1 秒 → TTL 过期（先捕获真实 monotonic，避免递归）。
    real = pipeline.time.monotonic
    monkeypatch.setattr(pipeline.time, "monotonic", lambda: real() + 1.0)
    assert cache.get("q") is None


def test_cache_evicts_oldest_beyond_cap() -> None:
    cache = SearchCache(ttl_seconds=300.0)
    for i in range(pipeline.CACHE_MAX_ENTRIES + 10):
        cache.put(f"q{i}", [_result(f"r{i}")])
    assert len(cache._entries) == pipeline.CACHE_MAX_ENTRIES
    assert cache.get("q0") is None  # 最旧的被淘汰


def test_filter_and_rank_filters_scores_and_renumbers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline, "LOW_QUALITY_URL_PATTERNS", ("spam.site",))
    results = [
        _result("拉康", url="https://spam.site/x"),  # URL 命中低质模式 → 过滤
        _result("无关标题", url="https://x.com/1", snippet="拉康 出现在摘要"),  # 仅摘要命中
        _result("拉康 镜像阶段", url="https://x.com/2", snippet="别的"),  # 标题命中
    ]
    ranked = filter_and_rank(results, "拉康", max_results=5)
    # 标题匹配权重（2x）更高，因此排在仅摘要命中之前；spam 被过滤。
    assert [r.title for r in ranked] == ["拉康 镜像阶段", "无关标题"]
    # 过滤后剩余结果重新连续编号。
    assert [r.position for r in ranked] == [1, 2]


def test_filter_and_rank_caps_max_results() -> None:
    results = [_result(f"拉康 结果{i}", url=f"https://x.com/{i}") for i in range(10)]
    assert len(filter_and_rank(results, "拉康", max_results=3)) == 3


def test_format_web_context_numbered_lines() -> None:
    rendered = format_web_context(
        [
            SearchResult(
                title="标题一", url="https://x.com/1", snippet="s", position=1, engine="e"
            ),
            SearchResult(
                title="标题二", url="https://x.com/2", snippet="s", position=2, engine="e"
            ),
        ]
    )
    assert "[1] 标题一" in rendered and "https://x.com/1" in rendered
    assert "[2] 标题二" in rendered
    assert format_web_context([]) == ""


# ── web_search 工具 ──────────────────────────────────────────────────


async def test_web_search_returns_context_and_emits_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _FakeEngine(
        [
            [
                SearchResult(
                    title="拉康 镜像阶段",
                    url="https://x.com/1",
                    snippet="镜像阶段 是……",
                    position=1,
                    engine="fake",
                )
            ]
        ]
    )
    events: list[dict[str, object]] = []
    monkeypatch.setattr(
        "agent_base.tools.search.tool.emit_step",
        lambda name, status, detail=None: events.append(
            {"type": "step", "name": name, "status": status, "detail": detail}
        ),
    )
    monkeypatch.setattr(
        "agent_base.tools.search.tool.emit_sources",
        lambda sources: events.append({"type": "sources", "sources": sources}),
    )
    tool = _make_tool(engine)

    rendered = await tool.ainvoke({"query": "拉康 镜像阶段"})  # type: ignore[attr-defined]

    assert "[1] 拉康 镜像阶段" in rendered
    steps = [e for e in events if e["type"] == "step"]
    assert [s["status"] for s in steps] == ["running", "completed"]
    sources_event = next(e for e in events if e["type"] == "sources")
    assert sources_event["sources"] == [{"title": "拉康 镜像阶段", "url": "https://x.com/1"}]


async def test_web_search_empty_result_returns_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mute_events(monkeypatch)
    tool = _make_tool(_FakeEngine([[]]))
    rendered = await tool.ainvoke({"query": "冷门问题"})  # type: ignore[attr-defined]
    assert "未找到相关搜索结果" in rendered
    assert "重试" in rendered


async def test_web_search_rejects_out_of_bounds_max_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mute_events(monkeypatch)
    tool = _make_tool(_FakeEngine([[]]))
    with pytest.raises(ValueError, match="max_results"):
        await tool.ainvoke({"query": "q", "max_results": 0})  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="max_results"):
        await tool.ainvoke({"query": "q", "max_results": 99})  # type: ignore[attr-defined]


async def test_web_search_caches_by_query(monkeypatch: pytest.MonkeyPatch) -> None:
    _mute_events(monkeypatch)
    engine = _FakeEngine([[_result("拉康 结果", url="https://x.com/1", snippet="拉康")]])
    tool = _make_tool(engine)
    await tool.ainvoke({"query": "拉康"})  # type: ignore[attr-defined]
    await tool.ainvoke({"query": "拉康"})  # type: ignore[attr-defined]
    assert engine.calls.count("拉康") == 1  # 第二次命中缓存


def test_web_search_availability_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent_base.tools.search.engines.ddgs_available", lambda: False)
    assert not _web_search_available(_settings())  # duckduckgo 缺 ddgs、无 tavily
    assert not _web_search_available(
        _settings(search_engine_priority="tavily")
    )  # 指定了 tavily 但没有 key
    assert _web_search_available(
        _settings(search_engine_priority="tavily", tavily_api_key="tvly-key")
    )  # key 兜底
    monkeypatch.setattr("agent_base.tools.search.engines.ddgs_available", lambda: True)
    assert _web_search_available(_settings())  # ddgs 兜底
