"""向量工具路由（ADR-010）：语义 top-k 工具召回 + 基础/高级分层。

核查缺口：``bind_tools`` 永远拿到全量池——工具规模一大（MCP 接入后
动辄上百），模型的选择偏差与幻觉随选项数放大。路由按查询语义召回
top-k 个 ``advanced`` 工具，与 ``basic`` 全集、近轮已用工具取并集后
**每轮动态绑定**；不是把全量清单塞给模型再指望它选对。

分层语义（``ToolSpec.tier`` / 池包装器的 ``tier`` 属性）：

- ``basic``——永远绑定：零依赖、高确定性（current_time/calculator/
  json_query）。数量少，占用的 prompt 预算可忽略；
- ``advanced``——参与路由：语义型/重火力工具（web_search、
  python_repl、全部 MCP 工具）。

降级链（**路由是优化不是依赖**，任何一环失败都回退全量池，行为回到
今天）：embedder 未配置/记忆包缺席 → 首次索引或当轮查询嵌入返回 None
（冷却中）→ advanced 数量本来就不超过 top_k。冷却中的工具（ADR-009
黑名单）从绑定集里剔除——模型看不到就不会再选，这是"规避重复选择
失效工具"的提示词侧闭环。

core 不 import 记忆包（ADR-003 的约束方向，由测试静态把守）：嵌入
客户端经 ``agent_base.memory`` 组的 ``embedder`` 入口点发现（记忆包
注册），余弦相似度是 core 自带的十几行实现。
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Sequence
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from langchain_core.tools import BaseTool

if TYPE_CHECKING:  # pragma: no cover - import avoided at runtime
    from agent_base.core.config import Settings

logger = logging.getLogger(__name__)


@runtime_checkable
class ToolEmbedder(Protocol):
    """路由所需的嵌入端口：与记忆包 ``EmbeddingClient`` 同形（鸭子匹配）。

    返回 ``None`` = 本次不可用（冷却/未配置）——路由据此降级；契约上
    ``embed`` 永不抛错（记忆包客户端的降级语义），防御性 try 除外。
    """

    dims: int | None

    async def embed(self, texts: Sequence[str]) -> list[list[float]] | None: ...


class NullToolEmbedder:
    """永久空实现：未配置/记忆包缺席时的占位，路由据此走降级路径。"""

    dims: int | None = None

    async def embed(self, texts: Sequence[str]) -> list[list[float]] | None:
        return None


def build_tool_embedder(settings: Settings) -> ToolEmbedder:
    """经 entry point 发现嵌入客户端（基座源码不 import 记忆包）。

    记忆包在 ``agent_base.memory`` 组注册 ``embedder`` 入口；共享
    ``MEMORY_EMBEDDING_*`` 提供方（一进程一个嵌入源，路由不为再配一套
    key）。未启用/未配 key/记忆包缺席 → NullToolEmbedder，安静降级。
    """
    memory = settings.memory
    if not (
        memory.embedding_enabled and memory.embedding_api_key.get_secret_value().strip()
    ):
        return NullToolEmbedder()
    matches = [
        ep for ep in entry_points(group="agent_base.memory") if ep.name == "embedder"
    ]
    if not matches:
        logger.info("tool routing: 无 embedder 入口点（记忆包未安装），降级为全量池")
        return NullToolEmbedder()
    factory = matches[0].load()
    embedder = factory(settings)
    # 工厂自带降级（无 key → NullEmbedding）；返回值满足 ToolEmbedder 协议。
    return embedder if embedder is not None else NullToolEmbedder()


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """core 自带的余弦相似度（不依赖记忆包）：维度不齐/零向量 → 0。"""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _tool_tier(tool: BaseTool) -> str:
    """读工具分层；池包装器携带 tier，外来工具默认 basic。"""
    return str(getattr(tool, "tier", "basic") or "basic")


def _tool_text(tool: BaseTool) -> str:
    """嵌入用的工具描述文本：名字 + 说明 + 类别线索。"""
    parts = [tool.name, tool.description or ""]
    category = getattr(tool, "category", "")
    if category:
        parts.append(str(category))
    return " ".join(parts)[:2000]


class ToolRouter:
    """每轮按查询语义选择绑定子集；降级链完整，任何失败都回全量池。

    索引懒初始化：首次 ``select`` 时把全池描述批量嵌入一次（一趟网络
    往返）并缓存；此后每轮只嵌一条查询文本。工具池在进程内固定，向量
    无需失效——这是装配期静态池的自然推论。
    """

    def __init__(
        self,
        pool: Sequence[BaseTool],
        *,
        embedder: ToolEmbedder,
        top_k: int = 5,
        registry: Any = None,  # ToolHealthRegistry | None：冷却工具剔除
    ) -> None:
        self._pool = list(pool)
        self._embedder = embedder
        self._top_k = top_k
        self._registry = registry
        self._active = not isinstance(embedder, NullToolEmbedder)
        self._vectors: dict[str, list[float]] | None = None

    @property
    def active(self) -> bool:
        """False = 路由不生效，图工厂走建图时全量绑定的快路径。"""
        return self._active

    async def _ensure_index(self) -> bool:
        """首次调用时建立工具向量索引；失败返回 False（本轮降级，下轮重试）。"""
        if self._vectors is not None:
            return True
        if not self._pool:
            self._vectors = {}
            return True
        try:
            vectors = await self._embedder.embed([_tool_text(t) for t in self._pool])
        except Exception:  # 防御：协议承诺不抛，第三方实现未必
            logger.warning("tool routing: 工具索引嵌入失败，本轮降级", exc_info=True)
            return False
        if vectors is None or len(vectors) != len(self._pool):
            logger.info("tool routing: 嵌入不可用（冷却/未配置），本轮降级为全量池")
            return False
        self._vectors = {t.name: list(v) for t, v in zip(self._pool, vectors)}
        return True

    def _blocked(self, name: str) -> bool:
        if self._registry is None:
            return False
        blocked, _reason = self._registry.is_blocked(name)
        return blocked

    async def select(
        self, query: str, *, recent_tools: Iterable[str] = ()
    ) -> list[BaseTool]:
        """选择本轮绑定集：basic 全集 ∪ top-k advanced ∪ 近轮已用。

        任何降级条件命中都返回全量池（含 blocked 剔除——剔除黑名单是
        行为增强，与降级正交）；结果保持池序稳定（确定性）。
        """
        advanced = [t for t in self._pool if _tool_tier(t) == "advanced"]
        # advanced 本来就不超 top_k：路由无收益，全量即最优解。
        if not self._active or len(advanced) <= self._top_k:
            return self._excluding_blocked(self._pool)

        indexed = await self._ensure_index()
        if not indexed or self._vectors is None:
            return self._excluding_blocked(self._pool)
        try:
            query_vectors = await self._embedder.embed([query or " "])
        except Exception:
            logger.warning("tool routing: 查询嵌入失败，本轮降级", exc_info=True)
            return self._excluding_blocked(self._pool)
        if not query_vectors:
            return self._excluding_blocked(self._pool)
        query_vector = query_vectors[0]

        scored = sorted(
            (
                (cosine_similarity(query_vector, self._vectors.get(t.name, ())), t.name)
                for t in advanced
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        chosen = {name for _score, name in scored[: self._top_k]}
        chosen.update(recent_tools)
        selected = [t for t in self._pool if _tool_tier(t) == "basic" or t.name in chosen]
        return self._excluding_blocked(selected)

    def _excluding_blocked(self, tools: Sequence[BaseTool]) -> list[BaseTool]:
        """剔除冷却中的工具；剔空时退回原集（执行侧仍有短路兜底）。"""
        kept = [t for t in tools if not self._blocked(t.name)]
        return kept or list(tools)


def recent_tool_names(messages: Sequence[Any], *, rounds: int = 3) -> list[str]:
    """近 N 轮 AI 消息里出现过的工具名（含正在往返中的这一轮）。

    任务中途工具不能凭空消失——模型上一轮刚调过的工具如果被路由裁掉，
    历史里的 ToolMessage 会变成"幽灵调用"，下一轮即 unknown tool。近轮
    工具强制保留是路由正确性的护栏，不是可调风格。
    """
    names: list[str] = []
    rounds_seen = 0
    for message in reversed(messages):
        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            continue
        rounds_seen += 1
        if rounds_seen > rounds:
            break
        for call in tool_calls:
            name = str(call.get("name") or "")
            if name:
                names.append(name)
    return names


def build_tool_router(
    pool: Sequence[BaseTool],
    *,
    settings: Settings,
    registry: Any = None,
    embedder: ToolEmbedder | None = None,
) -> ToolRouter | None:
    """按配置装配路由器：未启用返回 None（图工厂走快路径，零变化）。

    ``embedder`` 可由调用方传入与 few-shot（ADR-011）共享的实例——同一
    个进程只有一个嵌入提供方，冷却态互通；缺省自建。
    """
    if not settings.toolkit.routing_enabled:
        return None
    return ToolRouter(
        pool,
        embedder=embedder if embedder is not None else build_tool_embedder(settings),
        top_k=settings.toolkit.routing_top_k,
        registry=registry,
    )


__all__ = [
    "NullToolEmbedder",
    "ToolEmbedder",
    "ToolRouter",
    "build_tool_embedder",
    "build_tool_router",
    "cosine_similarity",
    "recent_tool_names",
]
