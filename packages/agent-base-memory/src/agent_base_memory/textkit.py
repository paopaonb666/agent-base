"""记忆侧的纯文本处理工具（P0-2 拆分）：分块与画像截断。

这些是无状态的纯函数，与"记忆服务"对象没有依赖关系——拆出来供
知识库摄取、切片可视化与画像保存复用，也便于独立单测。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


def _trim_profile_to_chars(profile: dict[str, Any], max_chars: int) -> dict[str, Any]:
    """画像结构级硬截断：从最后一个分节倒序逐条丢弃条目直到序列化
    长度达标；仍超长（存在巨型单条）则截断字符串值。输出永远是合法
    JSON——绝不让画像把注入预算缓慢撑爆。"""
    trimmed: dict[str, Any] = {}
    for key, value in profile.items():
        if isinstance(value, list):
            trimmed[key] = list(value)
        elif isinstance(value, dict):
            trimmed[key] = dict(value)
        else:
            trimmed[key] = value

    def _length() -> int:
        return len(json.dumps(trimmed, ensure_ascii=False))

    while _length() > max_chars:
        last_key = next(
            (
                key
                for key in reversed(list(trimmed))
                if isinstance(trimmed[key], list) and trimmed[key]
            ),
            None,
        )
        if last_key is None:
            break
        trimmed[last_key].pop()
        if not trimmed[last_key]:
            del trimmed[last_key]
    while _length() > max_chars:
        str_key: str | None = next(
            (k for k, v in trimmed.items() if isinstance(v, str) and v),
            None,
        )
        if str_key is None:
            # 只剩数字/空结构的极端兜底：几乎不可达，但绝不抛错。
            return {"truncated": True}
        trimmed[str_key] = trimmed[str_key][: max(1, max_chars // 2)]
    return trimmed


@dataclass(frozen=True)
class ChunkSpan:
    """一个切片：内容文本 + 在原文中的字符区间（前端高亮边界的依据）。

    ``segments`` 是 ``((start, end), …)`` 元组，坐标相对**调用方传入的
    原文**（含 strip 偏移补偿）。段落打包块的文本由多段拼成，单一
    (start, end) 不成立，因此是区间列表；固定窗口切片是单区间。
    """

    text: str
    segments: tuple[tuple[int, int], ...]

    @property
    def char_len(self) -> int:
        return len(self.text)


def chunk_text(text: str, *, chunk_chars: int, overlap: int) -> list[ChunkSpan]:
    """把文档文本切块（边界加固：段落感知优先，固定窗口兜底）。

    策略：
    - 优先按空行分段、贪心把**完整段落**打包进 ≤ chunk_chars 的块——
      切块边界落在段落上，表格/列表/代码块不再被从中间劈开；
    - 单段超过 chunk_chars 时段内退回固定窗口滑动（带 overlap）；
    - 全文没有段落分隔（或只有一段）时行为与旧版固定窗口完全一致
      （向后兼容：纯文本长文切块结果不变）。

    返回 span：除块文本外还带原文坐标，供前端把切片边界高亮回原文
    （切片可视化）；坐标相对传入的 text 原始串（strip 前的完整范围）。

    保证：任何块不超过 chunk_chars；非空内容全部覆盖；空文本返回空列表。
    """
    leading = len(text) - len(text.lstrip())
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_chars:
        return [ChunkSpan(text, ((leading, leading + len(text)),))]

    def _window(block: str, base: int) -> list[ChunkSpan]:
        step = max(1, chunk_chars - overlap)
        return [
            ChunkSpan(
                block[start : start + chunk_chars],
                ((base + start, base + min(start + chunk_chars, len(block))),),
            )
            for start in range(0, len(block), step)
        ]

    # 段落切分带坐标：finditer 拿每段在 stripped 文本中的位置。
    paragraphs: list[tuple[str, int]] = []
    last = 0
    for match in re.finditer(r"\n\s*\n", text):
        segment = text[last : match.start()]
        if segment.strip():
            paragraphs.append((segment.strip(), last + (len(segment) - len(segment.lstrip()))))
        last = match.end()
    tail = text[last:]
    if tail.strip():
        paragraphs.append((tail.strip(), last + (len(tail) - len(tail.lstrip()))))

    if len(paragraphs) <= 1:
        return _window(text, leading)

    chunks: list[ChunkSpan] = []
    current: list[tuple[str, int]] = []
    current_len = 0  # 已装段落总长（含段间空行分隔符）
    for para, pos in paragraphs:
        if len(para) > chunk_chars:
            # 超长段落：先结算已打包内容，段内固定窗口切。
            if current:
                chunks.append(_join_spans(current, leading))
                current, current_len = [], 0
            chunks.extend(_window(para, leading + pos))
            continue
        extra = len(para) + (2 if current else 0)
        if current_len + extra <= chunk_chars:
            current.append((para, pos))
            current_len += extra
        else:
            chunks.append(_join_spans(current, leading))
            current, current_len = [(para, pos)], len(para)
    if current:
        chunks.append(_join_spans(current, leading))
    return chunks


def _join_spans(paragraphs: list[tuple[str, int]], leading: int) -> ChunkSpan:
    """把打包进同一块的若干段落合成一个 span：文本以 \\n\\n 相接，
    segments 保留各段在原文中的真实区间（高亮时按段渲染）。"""
    joined = "\n\n".join(para for para, _ in paragraphs)
    segments = tuple((leading + pos, leading + pos + len(para)) for para, pos in paragraphs)
    return ChunkSpan(joined, segments)
