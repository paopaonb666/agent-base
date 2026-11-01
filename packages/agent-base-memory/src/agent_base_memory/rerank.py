"""检索精排客户端（ADR-004）：openai 兼容 /rerank + 失败降级。

混合检索（retrieval.py）的线性加权融合对"语义相似但表述不同"的候选
区分度有限；精排层在宽召回（top_k × 倍数）之后做一次交叉编码重排，
把真正相关的候选救回 top_k。降级链与 embedding 同款语义：

- ``MEMORY_RERANK_ENABLED=false``（默认）或启用但配置缺失（fail-fast，
  见 ``build_reranker``）→ ``NullReranker``；
- 调用失败（网络/HTTP/响应异常）→ 本次返回 ``None`` 并进入短暂冷却
  （默认 60s 内不再打真实网络）——精排端点故障绝不能拖垮检索主路径。

``rerank`` 因此**不抛异常**：返回 ``None`` 即"当前不可用"，调用方据此
保留混合分排序。分数与候选一一对应（按请求时的文档顺序对齐）。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import httpx

if TYPE_CHECKING:  # pragma: no cover - import avoided at runtime
    from agent_base.core.config import Settings

logger = logging.getLogger(__name__)

# 单条文档的字符封顶：精排模型的窗口有限（bge-reranker 系列为 8k token），
# 记忆与文档分块的正常长度远小于此；截断是防御而非常规路径。
MAX_DOC_CHARS = 6000

# 请求体/响应体的日志截断长度：绝不让原文或整包响应进日志。
_LOG_BODY_LIMIT = 200


class RerankError(RuntimeError):
    """rerank 调用失败（网络/HTTP/响应异常）；由客户端内部捕获并降级。"""


class RerankConfigError(ValueError):
    """rerank 配置不完整（启用即承诺：MEMORY_RERANK_ENABLED=true 但缺配置）。"""


@runtime_checkable
class Reranker(Protocol):
    """精排客户端的最小接口。"""

    async def rerank(self, query: str, documents: Sequence[str]) -> list[float] | None: ...


class NullReranker:
    """不精排：未启用时的常驻实现，检索保留混合分排序。"""

    async def rerank(self, query: str, documents: Sequence[str]) -> list[float] | None:
        return None


class OpenAICompatibleReranker:
    """openai 兼容 /rerank 端点客户端（硅基流动/Jina 风格契约均可）。

    请求体 ``{"model", "query", "documents"}``，响应体
    ``{"results": [{"index", "relevance_score"}, ...]}``（index 对应请求
    时的文档下标）。失败冷却：一次失败后 ``cooldown_seconds`` 内的调用
    直接返回 None，不打真实网络——避免半死不活的端点把每轮检索都拖慢
    一个超时周期。
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 5.0,
        cooldown_seconds: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._cooldown_seconds = cooldown_seconds
        self._client = client
        self._owns_client = client is None
        self._blocked_until = 0.0

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout_seconds)
        return self._client

    async def rerank(self, query: str, documents: Sequence[str]) -> list[float] | None:
        """给一组文档打相关度分（与输入顺序对齐）；失败返回 None。"""
        if not documents:
            return []
        if time.monotonic() < self._blocked_until:
            return None
        try:
            return await self._rerank_request(query, documents)
        except RerankError as exc:
            self._blocked_until = time.monotonic() + self._cooldown_seconds
            logger.warning(
                "memory: rerank 不可用，%ss 内保留混合分排序：%s", self._cooldown_seconds, exc
            )
            return None

    async def _rerank_request(self, query: str, documents: Sequence[str]) -> list[float]:
        client = await self._http()
        try:
            response = await client.post(
                f"{self._base_url}/rerank",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": self._model,
                    "query": query,
                    "documents": [doc[:MAX_DOC_CHARS] for doc in documents],
                },
            )
        except httpx.HTTPError as exc:
            raise RerankError(f"网络错误：{type(exc).__name__}") from exc
        if response.status_code != 200:
            raise RerankError(f"HTTP {response.status_code}：{response.text[:_LOG_BODY_LIMIT]}")
        try:
            payload: dict[str, Any] = response.json()
            results = payload["results"]
        except (ValueError, KeyError) as exc:
            raise RerankError(f"响应格式异常：{response.text[:_LOG_BODY_LIMIT]}") from exc
        if not isinstance(results, list):
            raise RerankError("响应 results 不是列表")
        # index 缺失或越界的条目按 0 分计（不因单条畸形丢整批结果）。
        scores = [0.0] * len(documents)
        for item in results:
            try:
                index = int(item["index"])
                score = float(item["relevance_score"])
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= index < len(documents):
                scores[index] = score
        return scores

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None


def build_reranker(settings: Settings) -> Reranker:
    """按 settings 装配精排客户端：未启用 → NullReranker；启用但缺配置
    → fail-fast（``RerankConfigError``，启用即承诺——与 embedding 的
    静默降级不同，rerank 是显式打开的质量开关，配置不齐说明装配错误，
    静默降级会把"以为有精排、其实没有"藏进检索质量里）。"""
    if not settings.memory.rerank_enabled:
        return NullReranker()
    api_key = settings.memory.rerank_api_key.get_secret_value().strip()
    missing = [
        name
        for name, value in (
            ("MEMORY_RERANK_API_KEY", api_key),
            ("MEMORY_RERANK_BASE_URL", settings.memory.rerank_base_url.strip()),
            ("MEMORY_RERANK_MODEL", settings.memory.rerank_model.strip()),
        )
        if not value
    ]
    if missing:
        raise RerankConfigError(
            f"MEMORY_RERANK_ENABLED=true 但缺少配置：{', '.join(missing)}"
            "——请补齐配置或显式关闭 MEMORY_RERANK_ENABLED"
        )
    return OpenAICompatibleReranker(
        base_url=settings.memory.rerank_base_url,
        api_key=api_key,
        model=settings.memory.rerank_model,
        timeout_seconds=settings.memory.rerank_timeout_seconds,
        cooldown_seconds=settings.memory.rerank_cooldown_seconds,
    )


__all__ = [
    "MAX_DOC_CHARS",
    "NullReranker",
    "OpenAICompatibleReranker",
    "RerankConfigError",
    "RerankError",
    "Reranker",
    "build_reranker",
]
