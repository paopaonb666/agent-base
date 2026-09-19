"""语义向量客户端（M6b）：openai 兼容 /embeddings + 关键词降级。

对话 LLM（如 DeepSeek）与 embedding 服务是两套独立配置：默认指向
硅基流动（``https://api.siliconflow.cn/v1``，BAAI/bge-m3，1024 维，
中英双语）。降级链（enterprise 语义）：

- ``MEMORY_EMBEDDING_API_KEY`` 未配置 → ``NullEmbedding``，检索退化为
  BM25 关键词 + 时间衰减 + 显著度，其余功能不受影响；
- 调用失败（网络/HTTP/响应异常）→ 本次返回 ``None`` 并进入短暂冷却
  （默认 60s 内不再打真实网络），之后自动恢复——一个挂掉的 embedding
  端点绝不能拖垮对话主流程，也不能被每个请求反复重试打爆。

``embed`` 因此**不抛异常**：返回 ``None`` 即"当前不可用"，调用方据此
走无向量路径。
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

# 单条输入的字符封顶：bge-m3 的窗口是 8192 token，超长文本截断即可
# （记忆与文档分块的正常长度远小于此；截断是防御而非常规路径）。
MAX_INPUT_CHARS = 6000

# 请求体/响应体的日志截断长度：绝不让 embedding 原文或整包响应进日志。
_LOG_BODY_LIMIT = 200


class EmbeddingError(RuntimeError):
    """embedding 调用失败（网络/HTTP/响应异常）；由客户端内部捕获并降级。"""


@runtime_checkable
class EmbeddingClient(Protocol):
    """embedding 客户端的最小接口。"""

    #: 已配置/已知的向量维度；None 表示不向量化（NullEmbedding）。
    dims: int | None

    async def embed(self, texts: Sequence[str]) -> list[list[float]] | None: ...


class NullEmbedding:
    """不向量化：embedding 未配置时的常驻实现，检索走关键词路径。"""

    dims: int | None = None

    async def embed(self, texts: Sequence[str]) -> list[list[float]] | None:
        return None


class OpenAICompatibleEmbedding:
    """openai 兼容 /embeddings 端点客户端（硅基流动/智谱/自建均可）。

    失败冷却：一次失败后 ``cooldown_seconds`` 内的调用直接返回 None，
    不打真实网络——避免半死不活的端点把每轮对话都拖慢一个超时周期。
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        dims: int,
        batch_size: int = 16,
        timeout_seconds: float = 10.0,
        cooldown_seconds: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self.dims: int | None = dims
        self._batch_size = max(1, batch_size)
        self._timeout_seconds = timeout_seconds
        self._cooldown_seconds = cooldown_seconds
        self._client = client
        self._owns_client = client is None
        self._blocked_until = 0.0
        self._dims_warned = False
        self._probe_result: tuple[float, str] | None = None

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout_seconds)
        return self._client

    async def embed(self, texts: Sequence[str]) -> list[list[float]] | None:
        """向量化一组文本；失败返回 None（含冷却期内的快速返回）。"""
        if not texts:
            return []
        if time.monotonic() < self._blocked_until:
            return None
        vectors: list[list[float]] = []
        try:
            for start in range(0, len(texts), self._batch_size):
                batch = [t[:MAX_INPUT_CHARS] for t in texts[start : start + self._batch_size]]
                vectors.extend(await self._embed_batch(batch))
        except EmbeddingError as exc:
            self._blocked_until = time.monotonic() + self._cooldown_seconds
            logger.warning(
                "memory: embedding 不可用，%ss 内走关键词降级：%s", self._cooldown_seconds, exc
            )
            return None
        return vectors

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        client = await self._http()
        try:
            response = await client.post(
                f"{self._base_url}/embeddings",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"model": self._model, "input": batch, "encoding_format": "float"},
            )
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"网络错误：{type(exc).__name__}") from exc
        if response.status_code != 200:
            raise EmbeddingError(f"HTTP {response.status_code}：{response.text[:_LOG_BODY_LIMIT]}")
        try:
            payload: dict[str, Any] = response.json()
            data = payload["data"]
        except (ValueError, KeyError) as exc:
            raise EmbeddingError(f"响应格式异常：{response.text[:_LOG_BODY_LIMIT]}") from exc
        if not isinstance(data, list) or len(data) != len(batch):
            raise EmbeddingError(
                f"响应条数不符：期望 {len(batch)}，实际"
                f" {len(data) if isinstance(data, list) else '非列表'}"
            )
        ordered = sorted(data, key=lambda item: int(item.get("index", 0)))
        vectors: list[list[float]] = []
        for item in ordered:
            vector = item.get("embedding")
            if not isinstance(vector, list):
                raise EmbeddingError("响应缺少 embedding 数组")
            self._check_dims(len(vector))
            vectors.append([float(x) for x in vector])
        return vectors

    def _check_dims(self, actual: int) -> None:
        """配置维度与实际不符只告警一次：以实际维度为准（逐行记录）。"""
        if self.dims is not None and actual != self.dims and not self._dims_warned:
            self._dims_warned = True
            logger.warning(
                "memory: embedding 维度与配置不符（配置 %s，实际 %s）——"
                "历史向量如与新维度混存将无法做余弦相似，建议清库重建",
                self.dims,
                actual,
            )

    async def probe(self) -> str:
        """轻量可达性探测（/health 用，锐评 #19）。

        与 LLM 探活同一语义：任何 HTTP 应答都算 ok（网络可达即证明），
        配额/鉴权不属于健康判定。结果缓存 60s——/health 会被负载均衡
        轮询，不能每次都打真实网络。
        """
        now = time.monotonic()
        if self._probe_result is not None and now - self._probe_result[0] < 60.0:
            return self._probe_result[1]
        status = "error"
        try:
            client = await self._http()
            await client.get(
                f"{self._base_url}/models",
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=self._timeout_seconds,
            )
            status = "ok"
        except httpx.HTTPError:
            logger.warning("memory: embedding 端点探活失败 %s", self._base_url)
        self._probe_result = (now, status)
        return status

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None


def build_embedding_client(settings: Settings) -> EmbeddingClient:
    """按 settings 装配 embedding 客户端：未启用/未配 key 一律降级。"""
    if not settings.memory.embedding_enabled:
        logger.info("memory: 语义检索已禁用（MEMORY_EMBEDDING_ENABLED=false）")
        return NullEmbedding()
    api_key = settings.memory.embedding_api_key.get_secret_value().strip()
    if not api_key:
        logger.info("memory: 未配置 MEMORY_EMBEDDING_API_KEY，检索走关键词降级")
        return NullEmbedding()
    return OpenAICompatibleEmbedding(
        base_url=settings.memory.embedding_base_url,
        api_key=api_key,
        model=settings.memory.embedding_model,
        dims=settings.memory.embedding_dims,
        batch_size=settings.memory.embedding_batch_size,
        timeout_seconds=settings.memory.embedding_timeout_seconds,
    )


__all__ = [
    "MAX_INPUT_CHARS",
    "EmbeddingClient",
    "EmbeddingError",
    "NullEmbedding",
    "OpenAICompatibleEmbedding",
    "build_embedding_client",
]
