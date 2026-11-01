"""向量工具路由（ADR-010）的测试：top-k 召回、分层、近轮回捞、降级链。"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import PrivateAttr

from agent_base.core.config import Settings
from agent_base.core.contracts import ModuleContext
from agent_base.core.graphs import build_single_agent_graph
from agent_base.core.tool_resilience import ToolHealthRegistry
from agent_base.core.tool_routing import (
    NullToolEmbedder,
    ToolRouter,
    build_tool_router,
    cosine_similarity,
    recent_tool_names,
)
from agent_base.core.tools import build_tool_pool
from tests.fakes import HashEmbedding, ScriptedChatModel


# 工具描述用可区分的英文关键词：HashEmbedding 按 token 重叠算相似度，
# 让"相关者得分更高"可以被确定性断言。
_ADVANCED = [
    ("wiki_search", "search wikipedia encyclopedia articles knowledge"),
    ("code_run", "execute python code snippets programming"),
    ("weather_lookup", "weather forecast temperature rain climate"),
    ("db_query", "database sql query tables records"),
    ("news_feed", "latest news headlines breaking events"),
    ("translate_text", "translate language chinese english words"),
]
_BASIC = [
    ("current_time", "current date and time clock"),
    ("calculator", "arithmetic expression calculator math"),
]


def _mk(name: str, description: str) -> BaseTool:
    def _run() -> str:
        return "ok"

    return StructuredTool.from_function(func=_run, name=name, description=description)


def _pool() -> list[BaseTool]:
    tools = [_mk(n, d) for n, d in _BASIC + _ADVANCED]
    advanced_names = {name for name, _ in _ADVANCED}
    return build_tool_pool({}, extra_tools=tools, tiers={n: "advanced" for n in advanced_names})


def _router(pool: list[BaseTool], **kwargs: Any) -> ToolRouter:
    return ToolRouter(pool, embedder=HashEmbedding(), **kwargs)


async def test_select_returns_basic_union_topk() -> None:
    pool = _pool()
    router = _router(pool, top_k=2)
    selected = await router.select("search encyclopedia articles")
    names = {t.name for t in selected}
    # basic 全集永远在场
    assert {name for name, _ in _BASIC} <= names
    # 语义相关的 advanced 排进 top-k
    assert "wiki_search" in names
    # 结果规模受控：basic(2) + top_k(2)
    assert len(selected) <= 2 + 2
    # 池序稳定（确定性）
    assert [t.name for t in selected] == [t.name for t in pool if t.name in names]


async def test_select_preserves_recent_tools() -> None:
    """近轮已用工具强制保留——防任务中途工具消失（幽灵调用）。"""
    pool = _pool()
    router = _router(pool, top_k=1)
    selected = await router.select(
        "search encyclopedia articles", recent_tools=["db_query", "current_time"]
    )
    names = {t.name for t in selected}
    assert "db_query" in names  # 不在 top-k 也保留
    assert "wiki_search" in names  # 语义 top-1


async def test_select_excludes_blocked_tools() -> None:
    pool = _pool()
    registry = ToolHealthRegistry(blacklist_threshold=1, blacklist_cooldown_seconds=60.0)
    registry.record("calculator", "error")
    registry.record("calculator", "error")  # 进入冷却
    router = _router(pool, top_k=2, registry=registry)
    selected = await router.select("arithmetic math calculator expression")
    names = {t.name for t in selected}
    assert "calculator" not in names  # 冷却工具不进绑定集


async def test_select_degrades_to_full_pool_on_unavailable_embedder() -> None:
    class _ColdEmbedder:
        """永远返回 None 的 embedder（冷却/降级语义）。"""

        dims: int | None = 32

        async def embed(self, texts: Any) -> list[list[float]] | None:
            return None

    pool = _pool()
    router = ToolRouter(pool, embedder=_ColdEmbedder(), top_k=2)
    assert router.active is True  # 非 Null：走运行时降级路径
    selected = await router.select("encyclopedia")
    assert [t.name for t in selected] == [t.name for t in pool]  # 回全量池


async def test_small_advanced_pool_short_circuits() -> None:
    """advanced 本来就不超 top_k：路由无收益，全量即最优。"""
    tools = [_mk("a", "alpha"), _mk("b", "beta"), _mk("c", "gamma")]
    pool = build_tool_pool({}, extra_tools=tools, tiers={"a": "advanced"})
    router = _router(pool, top_k=5)
    selected = await router.select("alpha")
    assert len(selected) == 3


def test_null_embedder_marks_router_inactive() -> None:
    router = ToolRouter(_pool(), embedder=NullToolEmbedder(), top_k=2)
    assert router.active is False


def test_cosine_similarity_edge_cases() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 3.0]) == pytest.approx(0.0)
    assert cosine_similarity([1.0], [1.0, 2.0]) == 0.0  # 维度不齐
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0  # 零向量


def test_recent_tool_names_scans_recent_rounds() -> None:
    from langchain_core.messages import ToolMessage

    messages = [
        HumanMessage(content="hi"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "old_tool", "args": {}, "id": "c0"},
                {"name": "kept_tool", "args": {}, "id": "c1"},
            ],
        ),
        ToolMessage(content="r", tool_call_id="c0", name="old_tool"),
        ToolMessage(content="r", tool_call_id="c1", name="kept_tool"),
        AIMessage(content="", tool_calls=[{"name": "new_tool", "args": {}, "id": "c2"}]),
    ]
    assert set(recent_tool_names(messages, rounds=3)) == {
        "old_tool",
        "kept_tool",
        "new_tool",
    }
    assert set(recent_tool_names(messages, rounds=1)) == {"new_tool"}


def test_build_tool_router_disabled_returns_none() -> None:
    settings = Settings(_env_file=None)
    assert build_tool_router(_pool(), settings=settings) is None


def test_build_tool_router_without_key_is_inactive() -> None:
    """启用但未配 embedding key：装配成功但 active=False（安静降级）。"""
    settings = Settings(_env_file=None, tool_routing_enabled=True)
    router = build_tool_router(_pool(), settings=settings)
    assert router is not None
    assert router.active is False


# ── 图级集成：每轮动态绑定 ──────────────────────────────────────────────


class _RecordingModel(ScriptedChatModel):
    """记录最近一次 bind_tools 的工具名（断言每轮动态绑定）。"""

    _bound: list[str] = PrivateAttr(default_factory=list)

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        self._bound = [t.name for t in tools]
        return self


async def test_graph_binds_selected_subset_per_turn() -> None:
    pool = _pool()
    settings = Settings(_env_file=None)
    model = _RecordingModel([AIMessage(content="done")])
    ctx = ModuleContext(
        settings=settings,
        llm=model,
        tools=pool,
        tool_router=_router(pool, top_k=1),
    )
    graph = build_single_agent_graph(ctx, name="routing-demo")
    await graph.ainvoke(
        {"messages": [HumanMessage(content="search encyclopedia articles")]},
        config={"recursion_limit": 6},
    )
    bound = set(model._bound)
    assert "wiki_search" in bound  # 语义 top-1 进入绑定集
    assert {name for name, _ in _BASIC} <= bound  # basic 全保留
    assert len(bound) <= 2 + 1  # 规模受控（top_k=1）


async def test_graph_keeps_full_binding_when_router_inactive() -> None:
    """未启用路由：建图时全量绑定，行为零变化（快路径）。"""
    pool = _pool()
    settings = Settings(_env_file=None)
    model = _RecordingModel([AIMessage(content="done")])
    ctx = ModuleContext(settings=settings, llm=model, tools=pool)
    graph = build_single_agent_graph(ctx, name="plain-demo")
    await graph.ainvoke(
        {"messages": [HumanMessage(content="hello")]},
        config={"recursion_limit": 6},
    )
    assert set(model._bound) == {t.name for t in pool}
