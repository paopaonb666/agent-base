"""web_search 工具（M3）：把搜索管线包装成共享池里的 ``BaseTool``。

工具 docstring 沿用 chat-agent 验证过的 Query 生成规范——它就是提示词
的一部分，直接决定搜索召回质量。错误分工比 chat-agent 更干净：

- **预期的空结果** → 返回带重试指引的字符串（模型能据此换词重试）；
- **意外异常** → 直接抛出，由共享池的 ``handle_tool_error`` 归一成
  ToolMessage（基座已有的机制，工具层不重复兜底）。

执行过程中通过 ``emit_step`` / ``emit_sources`` 发 UI 进度事件
（M3.5）：前端显示"联网搜索 running → completed"与可点击的来源列表。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

from langchain_core.tools import BaseTool, tool

from agent_base.core.config import Settings
from agent_base.tools.search.engines import (
    FallbackSearchManager,
    SearchEngine,
    SearchResult,
    build_engines,
)
from agent_base.tools.search.pipeline import (
    SearchCache,
    filter_and_rank,
    format_web_context,
)
from agent_base.tools.spec import ToolSpec
from agent_base.tools.streaming import emit_sources, emit_step

logger = logging.getLogger(__name__)

# 引擎调用的硬超时来自 settings.search.timeout_seconds；这里是工具层
# 的 max_results 上下界——模型传 0 或负数应得到可读的错误。
MAX_RESULTS_BOUNDS = (1, 20)

_EMPTY_RESULT_GUIDANCE = (
    "未找到相关搜索结果。请检查搜索关键词：专有名词必须保持完整"
    "（不要拆成单字），并用空格组合多个关键词后重试。"
)


async def _run_search(
    manager: FallbackSearchManager,
    cache: SearchCache,
    query: str,
    max_results: int,
    timeout: float,
) -> list[SearchResult]:
    """缓存优先的搜索执行；引擎全失败时返回空列表（管理器已记日志）。"""
    cached = cache.get(query)
    if cached is not None:
        return cached
    try:
        raw = await asyncio.wait_for(manager.search(query, max_results * 2), timeout=timeout)
    except (TimeoutError, asyncio.TimeoutError):
        logger.error("web_search timed out after %.1fs", timeout)
        return []
    ranked = filter_and_rank(raw, query, max_results)
    if ranked:
        cache.put(query, ranked)
    return ranked


def build_web_search_tool(
    settings: Settings,
    engine_factory: Callable[..., list[SearchEngine]] = build_engines,
) -> BaseTool:
    """按 settings 装配 web_search 工具实例（注册表工厂调用）。

    ``engine_factory`` 供测试注入替身引擎（生产路径用 ``build_engines``）。
    """
    manager = FallbackSearchManager(
        engine_factory(
            [e.strip().lower() for e in settings.search.engine_priority.split(",") if e.strip()],
            tavily_api_key=settings.search.tavily_api_key.get_secret_value(),
            ddgs_backend=settings.search.ddgs_backend,
        )
    )
    cache = SearchCache(ttl_seconds=settings.search.cache_ttl_seconds)

    @tool
    async def web_search(query: str, max_results: int = 8) -> str:
        """搜索互联网获取最新信息。当需要了解实时新闻、当前事件、
        最新数据或不确定的事实时调用此工具。

        Query 生成规范：
        1. 专有名词保护 —— 人名、地名、术语必须保持完整，禁止拆成单字。
           错误示例："拉的理论"（"拉康"被拆分）
           正确示例："拉康 精神分析 理论"
        2. 多关键词组合 —— 用空格分隔多个关键词，提高搜索精度。
           错误示例："拉康"（太宽泛）
           正确示例："拉康 精神分析 镜像阶段"
        3. 保留原文语言 —— 中文问题用中文 query，英文术语保留英文。

        Args:
            query: 搜索关键词或问题，尽量精简准确。
            max_results: 返回结果条数上限（1-20，默认 8）。
        """
        if not (MAX_RESULTS_BOUNDS[0] <= max_results <= MAX_RESULTS_BOUNDS[1]):
            raise ValueError(f"max_results 必须在 {MAX_RESULTS_BOUNDS} 内，收到了 {max_results}")

        emit_step("web_search", "running", f"搜索：{query[:40]}")
        results = await _run_search(
            manager, cache, query, max_results, settings.search.timeout_seconds
        )
        emit_step("web_search", "completed", f"找到 {len(results)} 条结果")
        emit_sources([{"title": r.title, "url": r.url} for r in results])

        if not results:
            return _EMPTY_RESULT_GUIDANCE
        return format_web_context(results)

    return web_search


def _web_search_available(settings: Settings) -> bool:
    """至少一个配置的引擎可用（ddgs 已安装 / tavily 有 key）。"""
    from agent_base.tools.search.engines import ddgs_available

    priority = [e.strip().lower() for e in settings.search.engine_priority.split(",") if e.strip()]
    if "duckduckgo" in priority and ddgs_available():
        return True
    return "tavily" in priority and bool(settings.search.tavily_api_key.get_secret_value())


_UNAVAILABLE_REASON = (
    "没有任何可用的搜索引擎：duckduckgo 需要安装可选依赖"
    "（pip install 'agent-base[search]'），tavily 需要配置 TAVILY_API_KEY"
)


WEB_SEARCH_SPEC = ToolSpec(
    name="web_search",
    category="network",
    safety="network",
    # 包装一层以匹配 ToolSpec.factory 的签名（engine_factory 仅测试注入）。
    factory=lambda settings: [build_web_search_tool(settings)],
    available=_web_search_available,
    unavailable_reason=_UNAVAILABLE_REASON,
    # 引擎调用自身有 search_timeout_seconds（默认 15s）；池的预算只做
    # 兜底，因此设得略高于它即可。
    timeout=25.0,
)

__all__ = ["WEB_SEARCH_SPEC", "build_web_search_tool"]
