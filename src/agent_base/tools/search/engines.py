"""搜索引擎抽象层与降级管理器（M3，移植自 chat-agent 的已验证设计）。

``SearchEngine`` 把各家引擎归一成异步接口 + 统一的结果结构；
``FallbackSearchManager`` 按优先级依次尝试，任一成功即返回——单个
引擎的抖动不构成对话失败。DuckDuckGo（``ddgs``，免 key）是零成本
默认引擎，Tavily（需 key）作为高配选项。
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchResult:
    """归一化的一条搜索结果。"""

    title: str
    url: str
    snippet: str
    position: int
    engine: str


def ddgs_available() -> bool:
    """``ddgs`` 包是否已安装（``[search]`` extras 提供它）。"""
    return importlib.util.find_spec("ddgs") is not None


class SearchEngine(ABC):
    """搜索引擎抽象基类：异步接口，同步库在实现内用 to_thread 包装。"""

    name: str = ""

    @abstractmethod
    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        """执行搜索，返回统一格式的结果；失败抛异常由管理器降级。"""


class TavilyEngine(SearchEngine):
    """通过 Tavily REST API 执行搜索（需要 API key）。"""

    name = "tavily"
    _ENDPOINT = "https://api.tavily.com/search"

    def __init__(self, api_key: str, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._api_key = api_key
        # transport 供测试注入 MockTransport，生产路径恒为 None。
        self._transport = transport

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        async with httpx.AsyncClient(timeout=10.0, transport=self._transport) as client:
            resp = await client.post(
                self._ENDPOINT,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "query": query,
                    "search_depth": "basic",
                    "max_results": max_results,
                    "include_answer": False,
                },
            )
            resp.raise_for_status()
            data = resp.json()
        return [
            SearchResult(
                title=str(r.get("title", "")),
                url=str(r.get("url", "")),
                snippet=str(r.get("content", "")),
                position=i + 1,
                engine=self.name,
            )
            for i, r in enumerate(data.get("results", []))
        ]


class DuckDuckGoEngine(SearchEngine):
    """通过 DuckDuckGo 免费搜索 API 执行搜索（``ddgs`` 是同步库）。

    ``backend`` 钉选 ddgs 的上游引擎：默认 ``auto`` 会同时扇出十几个
    上游（google/yahoo/brave/startpage 等），受限网络下大量超时把聚合
    拖死；单钉可达引擎（duckduckgo / bing）实测快一个数量级。
    """

    name = "duckduckgo"

    def __init__(
        self,
        ddgs_factory: Callable[[], Any] | None = None,
        backend: str = "duckduckgo",
    ) -> None:
        # ddgs 9.x 的 DDGS 是动态代理（运行时转发属性），monkeypatch
        # 类属性会被绕过；测试通过注入工厂替身来隔离真实网络。
        self._ddgs_factory = ddgs_factory
        self._backend = backend

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        # 延迟导入：ddgs 是 [search] extras 的可选依赖，未安装时本引擎
        # 不会进入引擎列表（见 build_engines），运行到此处即代表已安装。
        from ddgs import DDGS

        factory = self._ddgs_factory or DDGS

        def _run() -> list[dict[str, Any]]:
            with factory() as ddgs:
                return list(ddgs.text(query, max_results=max_results, backend=self._backend))

        raw = await asyncio.to_thread(_run)
        return [
            SearchResult(
                title=str(r.get("title", "")),
                url=str(r.get("href", "")),
                snippet=str(r.get("body", "")),
                position=i + 1,
                engine=self.name,
            )
            for i, r in enumerate(raw)
        ]


class FallbackSearchManager:
    """按优先级依次尝试多个搜索引擎；任一成功即返回，全失败返回空。"""

    def __init__(self, engines: list[SearchEngine]) -> None:
        self.engines = engines

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        for engine in self.engines:
            try:
                results = await engine.search(query, max_results)
                if results:
                    logger.info("Search succeeded via %s (%d results)", engine.name, len(results))
                    return results
                logger.warning("Engine %s returned no results; trying next", engine.name)
            except Exception as exc:
                logger.warning("Engine %s failed: %s", engine.name, exc)
                continue
        logger.error("All search engines failed for query: %s", query)
        return []


def build_engines(
    priority: list[str],
    *,
    tavily_api_key: str = "",
    ddgs_backend: str = "duckduckgo",
) -> list[SearchEngine]:
    """按优先级配置组装引擎列表；不可用的引擎被跳过（记录原因）。"""
    engines: list[SearchEngine] = []
    for name in priority:
        if name == "tavily":
            if tavily_api_key:
                engines.append(TavilyEngine(api_key=tavily_api_key))
            else:
                logger.warning("TAVILY_API_KEY is empty; tavily engine skipped")
        elif name == "duckduckgo":
            if ddgs_available():
                engines.append(DuckDuckGoEngine(backend=ddgs_backend))
            else:
                logger.warning(
                    "ddgs package is not installed (pip install 'agent-base[search]'); "
                    "duckduckgo engine skipped"
                )
    return engines
