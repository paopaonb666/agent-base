"""工具 few-shot（ADR-011）的测试：选样、阈值剪枝、可用性过滤、注入。"""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool

from agent_base.core.config import Settings
from agent_base.core.contracts import ModuleContext
from agent_base.core.graphs import build_single_agent_graph
from agent_base.core.tool_fewshot import (
    DEFAULT_FEWSHOT_PATH,
    FewShotExample,
    ToolFewShot,
    build_tool_fewshot,
    load_fewshot_examples,
)
from tests.fakes import HashEmbedding, ScriptedChatModel


def _examples() -> list[FewShotExample]:
    return [
        FewShotExample(
            query="search encyclopedia articles about quantum physics",
            tool="wiki_search",
            args={"query": "quantum physics"},
            note="keep proper nouns intact",
        ),
        FewShotExample(
            query="compute arithmetic expression with numbers",
            tool="calculator",
            args={"expression": "2 ** 10"},
        ),
        FewShotExample(
            query="weather forecast temperature tomorrow rain",
            tool="weather_lookup",
            args={"city": "Beijing"},
        ),
    ]


class _VectorEmbedder:
    """按文本查表的 embedder：样本与查询各配正交单位向量，相似度精确可控。

    HashEmbedding 的 32 维哈希会让无关文本偶发高相似度（维度碰撞），
    阈值/过滤类断言需要确定性的向量几何。
    """

    dims: int | None = 3

    def __init__(self) -> None:
        self.table: dict[str, list[float]] = {
            "search encyclopedia articles about quantum physics": [1.0, 0.0, 0.0],
            "compute arithmetic expression with numbers": [0.0, 1.0, 0.0],
            "weather forecast temperature tomorrow rain": [0.0, 0.0, 1.0],
            "Q": [1.0, 0.0, 0.0],  # 与 ex1 完全同向
        }

    async def embed(self, texts: object) -> list[list[float]] | None:
        return [self.table.get(t, [0.0, 0.0, 0.0]) for t in texts]  # type: ignore[union-attr]


def _store(embedder: object | None = None, **kwargs: object) -> ToolFewShot:
    return ToolFewShot(
        _examples(),
        embedder=embedder or HashEmbedding(),
        **kwargs,  # type: ignore[arg-type]
    )


def test_seed_file_loads_and_is_wellformed() -> None:
    examples = load_fewshot_examples(DEFAULT_FEWSHOT_PATH)
    assert len(examples) >= 10
    # 每条样本都必须可 JSON 序列化且工具名非空（渲染格式依赖）
    for example in examples:
        assert example.tool and example.query
        json.dumps(example.args, ensure_ascii=False)


def test_load_skips_bad_lines(tmp_path: object) -> None:
    path = tmp_path / "bad.jsonl"  # type: ignore[operator]
    path.write_text(
        '{"query": "q", "tool": "t", "args": {}}\n'
        "not json at all\n"
        '{"tool": "missing-query", "args": {}}\n',
        encoding="utf-8",
    )
    examples = load_fewshot_examples(path)
    assert len(examples) == 1


async def test_select_orders_by_similarity() -> None:
    store = _store()
    scored = await store.select("search encyclopedia articles quantum")
    assert scored, "应有选样"
    assert scored[0][1].tool == "wiki_search"  # 最相似者排第一
    scores = [score for score, _ in scored]
    assert scores == sorted(scores, reverse=True)  # 降序


async def test_render_segment_formats_examples() -> None:
    store = _store(top_n=1)
    segment = await store.render_segment("search encyclopedia articles quantum")
    assert "工具调用示例" in segment
    assert "调用：wiki_search(" in segment
    assert '"query": "quantum physics"' in segment
    assert "要点：keep proper nouns intact" in segment


async def test_render_segment_skips_below_min_score() -> None:
    """低于阈值不注入：零样本优于错样本（规避不匹配样本误导）。"""
    store = _store(_VectorEmbedder(), min_score=0.9)
    segment = await store.render_segment("Q")  # 与 ex1 同向：相似度 1.0
    assert "wiki_search" in segment
    far = await store.render_segment("UNKNOWN QUERY")  # 零向量：全部 0 分
    assert far == ""  # 无关查询宁可零注入


async def test_render_segment_filters_unavailable_tools() -> None:
    """未绑定工具的示例滤掉：教模型调不存在的工具等于教它幻觉。"""
    store = _store(_VectorEmbedder(), min_score=0.5, top_n=1)
    segment = await store.render_segment("Q", available={"calculator"})
    # top-1 是 wiki_search 但不在绑定集；calculator 相似度为 0 过不了阈值
    assert segment == ""


def test_build_tool_fewshot_disabled_returns_none() -> None:
    assert build_tool_fewshot(Settings(_env_file=None)) is None


def test_build_tool_fewshot_without_embedder_returns_none() -> None:
    """启用但无嵌入（未配 key）：安静退出，注入侧零变更。"""
    settings = Settings(_env_file=None, tool_fewshot_enabled=True)
    assert build_tool_fewshot(settings) is None


def test_build_tool_fewshot_with_embedder() -> None:
    settings = Settings(_env_file=None, tool_fewshot_enabled=True)
    store = build_tool_fewshot(settings, embedder=HashEmbedding())
    assert store is not None
    assert len(store.examples) >= 10  # 默认种子库


async def test_graph_injects_fewshot_segment() -> None:
    def _run() -> str:
        return "ok"

    pool_tool = StructuredTool.from_function(
        func=_run, name="wiki_search", description="search encyclopedia"
    )
    settings = Settings(_env_file=None)
    model = ScriptedChatModel([AIMessage(content="done")])
    ctx = ModuleContext(
        settings=settings,
        llm=model,
        tools=[pool_tool],
        tool_fewshot=_store(top_n=1),
    )
    graph = build_single_agent_graph(ctx, name="fewshot-demo")
    await graph.ainvoke(
        {"messages": [HumanMessage(content="search encyclopedia articles quantum")]},
        config={"recursion_limit": 6},
    )
    received = model.received[-1]
    injected = [m for m in received if getattr(m, "type", "") == "system"]
    assert any("工具调用示例" in str(m.content) for m in injected)


async def test_graph_without_fewshot_keeps_prompt_unchanged() -> None:
    def _run() -> str:
        return "ok"

    pool_tool = StructuredTool.from_function(
        func=_run, name="wiki_search", description="search encyclopedia"
    )
    settings = Settings(_env_file=None)
    model = ScriptedChatModel([AIMessage(content="done")])
    ctx = ModuleContext(settings=settings, llm=model, tools=[pool_tool])
    graph = build_single_agent_graph(ctx, name="no-fewshot-demo")
    await graph.ainvoke(
        {"messages": [HumanMessage(content="search encyclopedia articles quantum")]},
        config={"recursion_limit": 6},
    )
    received = model.received[-1]
    assert not any(
        "工具调用示例" in str(m.content)
        for m in received
        if getattr(m, "type", "") == "system"
    )
