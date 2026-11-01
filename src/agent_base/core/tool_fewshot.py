"""工具调用 few-shot（ADR-011）：按查询相似度动态选样注入提示词。

核查缺口：工具调用的准确性提升没有样本与度量——docstring 示例是静态
的，且没有评测证明它有用。本模块给出注入侧的实现，效果度量由
``scripts/tool_eval.py``（有/无样本两臂对照）承担。

设计要点：

- **动态选样**——样本库批量嵌入一次（懒初始化），每轮按当前查询的
  余弦相似度取 top-n。相似样本才注入：不匹配的示例会把模型带偏
  （比如给"算个数"的查询注入搜索示例）；
- **min_score 阈值**——最高分都低于阈值的查询宁可不注入（零样本优于
  错样本）；
- **可用性过滤**——渲染时可传当前绑定集，未绑定的工具示例直接滤掉
  （给模型看一个调不存在的工具的示例，等于教它 hallucinate）；
- **文本段注入**——样本渲染成 system prompt 的追加段（临时拼装、不
  落库、不参与修剪去重），不伪造 user/assistant 消息历史。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Sequence

from agent_base.core.tool_routing import (
    NullToolEmbedder,
    ToolEmbedder,
    build_tool_embedder,
    cosine_similarity,
)

if TYPE_CHECKING:  # pragma: no cover - import avoided at runtime
    from agent_base.core.config import Settings

logger = logging.getLogger(__name__)

# 打包内种子库：随 wheel 分发（evals/ 在仓库根、不进包）；用户样本库
# 经 TOOL_FEWSHOT_PATH 指向任意 JSONL 覆盖。
DEFAULT_FEWSHOT_PATH = Path(__file__).resolve().parent.parent / "tools" / "fewshot_seed.jsonl"


@dataclass(frozen=True)
class FewShotExample:
    """一条工具调用样本：用户会怎么说 → 该调什么 → 参数长什么样。"""

    query: str
    tool: str
    args: dict[str, Any]
    note: str = ""


def load_fewshot_examples(path: str | Path) -> list[FewShotExample]:
    """从 JSONL 加载样本；坏行跳过并告警（不因个别行废掉整个注入）。"""
    examples: list[FewShotExample] = []
    bad = 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            row = json.loads(text)
            examples.append(
                FewShotExample(
                    query=str(row["query"]),
                    tool=str(row["tool"]),
                    args=dict(row.get("args") or {}),
                    note=str(row.get("note") or ""),
                )
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            bad += 1
    if bad:
        logger.warning("tool fewshot: %d 行样本解析失败被跳过（%s）", bad, path)
    return examples


class ToolFewShot:
    """按查询相似度动态选择 few-shot 样本，渲染为提示词片段。

    样本向量懒初始化（首次调用批量嵌入一次并缓存）；此后每轮只嵌一条
    查询。嵌入不可用（冷却/降级）时 ``select`` 返回空——本轮零注入，
    下一轮自动重试。
    """

    def __init__(
        self,
        examples: Sequence[FewShotExample],
        *,
        embedder: ToolEmbedder,
        top_n: int = 2,
        min_score: float = 0.30,
    ) -> None:
        self._examples = list(examples)
        self._embedder = embedder
        self._top_n = top_n
        self._min_score = min_score
        self._vectors: list[list[float]] | None = None

    @property
    def examples(self) -> list[FewShotExample]:
        return list(self._examples)

    async def _ensure_vectors(self) -> bool:
        if self._vectors is not None:
            return True
        if not self._examples:
            self._vectors = []
            return True
        try:
            vectors = await self._embedder.embed([ex.query for ex in self._examples])
        except Exception:  # 防御：协议承诺不抛，第三方实现未必
            logger.warning("tool fewshot: 样本嵌入失败，本轮零注入", exc_info=True)
            return False
        if vectors is None or len(vectors) != len(self._examples):
            logger.info("tool fewshot: 嵌入不可用（冷却/未配置），本轮零注入")
            return False
        self._vectors = [list(v) for v in vectors]
        return True

    async def select(self, query: str) -> list[tuple[float, FewShotExample]]:
        """相似度选样：返回 (得分, 样本) 的 top-n（不过滤阈值）。"""
        if not self._examples or not await self._ensure_vectors():
            return []
        try:
            query_vectors = await self._embedder.embed([query or " "])
        except Exception:
            logger.warning("tool fewshot: 查询嵌入失败，本轮零注入", exc_info=True)
            return []
        if not query_vectors:
            return []
        scored = sorted(
            (
                (cosine_similarity(query_vectors[0], vector), example)
                for vector, example in zip(self._vectors or [], self._examples)
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        return scored[: self._top_n]

    async def render_segment(
        self, query: str, *, available: Iterable[str] | None = None
    ) -> str:
        """渲染注入段；空串 = 本轮不注入（调用方据此零变更）。

        ``available`` 是当前绑定给模型的工具名集合：未绑定工具的样本被
        滤掉（教模型调不存在的工具等于教它幻觉）。
        """
        allowed = set(available) if available is not None else None
        picked = [
            (score, example)
            for score, example in await self.select(query)
            if score >= self._min_score and (allowed is None or example.tool in allowed)
        ]
        if not picked:
            return ""
        lines = ["工具调用示例（按与当前请求的相似度挑选，仅作参考）："]
        for _score, example in picked:
            entry = (
                f"- 用户问：{example.query}\n"
                f"  调用：{example.tool}({json.dumps(example.args, ensure_ascii=False)})"
            )
            if example.note:
                entry += f"\n  要点：{example.note}"
            lines.append(entry)
        return "\n".join(lines)


def build_tool_fewshot(
    settings: Settings, *, embedder: ToolEmbedder | None = None
) -> ToolFewShot | None:
    """按配置装配 few-shot 注入器：任何一环不满足都返回 None（零注入）。

    未启用 / 样本库为空 / 嵌入不可用（未配 key、记忆包缺席）——三种
    降级都安静退出，图工厂对 None 走零变更路径。
    """
    toolkit = settings.toolkit
    if not toolkit.fewshot_enabled:
        return None
    path = toolkit.fewshot_path or str(DEFAULT_FEWSHOT_PATH)
    examples = load_fewshot_examples(path)
    if not examples:
        logger.warning("tool fewshot: 样本库为空（%s），few-shot 注入关闭", path)
        return None
    resolved = embedder if embedder is not None else build_tool_embedder(settings)
    if isinstance(resolved, NullToolEmbedder):
        logger.info("tool fewshot: 嵌入不可用，few-shot 注入关闭")
        return None
    return ToolFewShot(
        examples,
        embedder=resolved,
        top_n=toolkit.fewshot_top_n,
        min_score=toolkit.fewshot_min_score,
    )


__all__ = [
    "DEFAULT_FEWSHOT_PATH",
    "FewShotExample",
    "ToolFewShot",
    "build_tool_fewshot",
    "load_fewshot_examples",
]
