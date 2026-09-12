"""搜索工具包（M3）：引擎 / 管线 / 工具入口。"""

from agent_base.tools.search.engines import (
    DuckDuckGoEngine,
    FallbackSearchManager,
    SearchEngine,
    SearchResult,
    TavilyEngine,
    build_engines,
)
from agent_base.tools.search.pipeline import (
    SearchCache,
    filter_and_rank,
    format_web_context,
    score_relevance,
)

__all__ = [
    "DuckDuckGoEngine",
    "FallbackSearchManager",
    "SearchCache",
    "SearchEngine",
    "SearchResult",
    "TavilyEngine",
    "build_engines",
    "filter_and_rank",
    "format_web_context",
    "score_relevance",
]
