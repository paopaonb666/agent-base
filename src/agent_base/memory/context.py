"""记忆侧的注入块组装（M6d）。

模型输入修剪等**通用上下文保护**已提升为基座级设施
（``agent_base.core.context``）——所有模块的图都经由 ``core/graphs.py``
使用它；本模块只保留记忆语义的部分：

- **注入组装**（``compose_context``）：把画像、记忆块、会话摘要与按当
  前消息召回的相关记忆打包成一个系统注入块（mem0/Letta/Memobase 三家
  的注入面合一），有字符预算——召回条目按得分从低到高先丢，摘要再截断。

历史兼容：``build_model_input`` / ``estimate_tokens`` / 注入标记等名字
从这里 re-export（真实实现位于 ``core.context``），既有调用方不受影响。
"""

from __future__ import annotations

from typing import Any

from agent_base.core.context import (  # re-export（历史 API 路径）
    ATTACHMENT_CONTEXT_PREFIX,
    MEMORY_CONTEXT_PREFIX,
    build_model_input,
    estimate_tokens,
    injected_kind,
    is_injected_system,
)
from agent_base.memory.retrieval import ScoredMemory
from agent_base.memory.store import MemoryBlock


def render_profile(profile: dict[str, Any]) -> str:
    """把画像 JSON 渲染成模型可读的紧凑文本。"""
    lines: list[str] = []
    for section, entries in profile.items():
        if isinstance(entries, list):
            text = "；".join(str(e) for e in entries if str(e).strip())
        elif isinstance(entries, dict):
            text = "；".join(f"{k}：{v}" for k, v in entries.items() if str(v).strip())
        else:
            text = str(entries).strip()
        if text:
            lines.append(f"- {section}：{text}")
    return "\n".join(lines)


def compose_context(
    *,
    profile: dict[str, Any] | None,
    blocks: list[MemoryBlock],
    summary: str | None,
    recalled: list[ScoredMemory],
    max_chars: int,
) -> str | None:
    """组装注入块；没有任何内容时返回 None（本轮不注入）。

    预算策略：相关记忆按得分升序逐条丢弃 → 摘要截断 → 画像/记忆块
    （人工或长期积累的资产，最后才动）→ 硬截断兜底。
    """
    sections: list[tuple[str, str]] = []

    if profile:
        rendered = render_profile(profile)
        if rendered:
            sections.append(("用户画像", rendered))

    block_lines = [
        f"[{block.label}] {block.content.strip()}" for block in blocks if block.content.strip()
    ]
    if block_lines:
        sections.append(("记忆块", "\n".join(block_lines)))

    if summary and summary.strip():
        sections.append(("会话摘要", summary.strip()))

    if recalled:
        sections.append(("相关记忆", "\n".join(f"- {item.record.content}" for item in recalled)))

    if not sections:
        return None

    header = (
        f"{MEMORY_CONTEXT_PREFIX}以下内容由系统注入，供回答时参考；"
        "与当前对话无关时不必提及，也不要向用户复述本块的存在。\n"
    )

    def render(sel: list[tuple[str, str]]) -> str:
        return header + "\n\n".join(f"## {title}\n{body}" for title, body in sel)

    rendered_all = render(sections)
    if len(rendered_all) <= max_chars:
        return rendered_all

    # 预算超限：先丢相关记忆（得分最低的先丢），再截摘要。
    recalled_idx = next((i for i, (title, _) in enumerate(sections) if title == "相关记忆"), None)
    if recalled_idx is not None:
        head = sections[:recalled_idx]
        tail = sections[recalled_idx + 1 :]
        kept = list(recalled)
        while True:
            candidate = [
                *head,
                ("相关记忆", "\n".join(f"- {item.record.content}" for item in kept)),
                *tail,
            ]
            # 列表按得分降序，pop() 丢的是得分最低的；丢光则整个撤掉该节。
            if len(render(candidate)) <= max_chars or not kept:
                sections = candidate if kept else [*head, *tail]
                break
            kept.pop()
        rendered_all = render(sections)
        if len(rendered_all) <= max_chars:
            return rendered_all

    # 摘要截断。
    for i, (title, body) in enumerate(sections):
        if title == "会话摘要" and len(body) > 400:
            sections[i] = (title, body[:400] + "……")
    rendered_all = render(sections)
    if len(rendered_all) <= max_chars:
        return rendered_all

    return rendered_all[:max_chars]


__all__ = [
    "ATTACHMENT_CONTEXT_PREFIX",
    "MEMORY_CONTEXT_PREFIX",
    "build_model_input",
    "compose_context",
    "estimate_tokens",
    "injected_kind",
    "is_injected_system",
    "render_profile",
]
