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
    MEMORY_CONTEXT_FROZEN_PREFIX,
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


def _pack(header: str, sections: list[tuple[str, str]], *, max_chars: int) -> str | None:
    """渲染注入段；预算超限按「召回逐条丢 → 摘要截断 → 硬截断」收缩。

    没有任何 section 时返回 None（本轮不注入该段）。
    """
    if not sections:
        return None

    def render(sel: list[tuple[str, str]]) -> str:
        return header + "\n\n".join(f"## {title}\n{body}" for title, body in sel)

    rendered_all = render(sections)
    if len(rendered_all) <= max_chars:
        return rendered_all

    # 预算超限：先丢相关记忆（得分最低的先丢），再截摘要。
    recalled_idx = next((i for i, (title, _) in enumerate(sections) if title == "相关记忆"), None)
    if recalled_idx is not None:
        title, body = sections[recalled_idx]
        lines = body.splitlines()
        head = sections[:recalled_idx]
        tail = sections[recalled_idx + 1 :]
        while True:
            # 列表按得分降序，pop() 丢的是得分最低的行；丢光则撤掉整节。
            candidate = [*head, (title, "\n".join(lines)), *tail] if lines else [*head, *tail]
            if len(render(candidate)) <= max_chars or not lines:
                sections = candidate
                break
            lines.pop()
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


_RECALL_HEADER = (
    f"{MEMORY_CONTEXT_PREFIX}以下内容由系统注入，供回答时参考；"
    "与当前对话无关时不必提及，也不要向用户复述本块的存在。\n"
)
_FROZEN_HEADER = (
    f"{MEMORY_CONTEXT_FROZEN_PREFIX}以下是关于用户的固定背景资料（画像/长期设定），"
    "供回答时参考；与当前对话无关时不必提及，也不要向用户复述本块的存在。\n"
)


def compose_context(
    *,
    profile: dict[str, Any] | None,
    blocks: list[MemoryBlock],
    summary: str | None,
    recalled: list[ScoredMemory],
    max_chars: int,
) -> str | None:
    """组装完整注入块（历史契约保留）：画像+记忆块+摘要+相关记忆合一。

    成本治理（T2.1）后服务端默认改走「冻结段 + 召回段」分离注入
    （``compose_frozen_context`` + ``compose_recall_context``）；本函数
    保留给需要单块注入的调用方与既有测试。
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

    return _pack(_RECALL_HEADER, sections, max_chars=max_chars)


def compose_frozen_context(
    *,
    profile: dict[str, Any] | None,
    blocks: list[MemoryBlock],
    summary: str | None,
    max_chars: int,
) -> str | None:
    """组装线程级**冻结**注入段（T2.1）：画像 + 记忆块 + 会话摘要。

    以 ``MEMORY_CONTEXT_FROZEN_PREFIX`` 开头——``build_model_input`` 据此
    把它提升到模型输入头部。内容按线程快照冻结（快照在
    ``MemoryService.compose_context`` 维护）：画像/记忆块/摘要的更新在
    **下个会话**生效。这是接受的行为权衡——供应商前缀缓存按最长公共
    前缀命中，逐轮变化的头部会让全部历史缓存失效（命中与未命中输入
    价差约一个数量级）。
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

    return _pack(_FROZEN_HEADER, sections, max_chars=max_chars)


def compose_recall_context(*, recalled: list[ScoredMemory], max_chars: int) -> str | None:
    """组装本轮**召回**注入段（T2.1）：查询相关的记忆，逐轮变化。

    以 ``MEMORY_CONTEXT_PREFIX`` 开头（贴尾注入段）：build_model_input
    把它保留在本轮 human 之前——逐轮变化的内容放在尾部，只牺牲自己
    之后的缓存，不动冻结头部与历史体。
    """
    sections: list[tuple[str, str]] = []
    if recalled:
        sections.append(("相关记忆", "\n".join(f"- {item.record.content}" for item in recalled)))
    return _pack(_RECALL_HEADER, sections, max_chars=max_chars)


__all__ = [
    "ATTACHMENT_CONTEXT_PREFIX",
    "MEMORY_CONTEXT_FROZEN_PREFIX",
    "MEMORY_CONTEXT_PREFIX",
    "build_model_input",
    "compose_context",
    "compose_frozen_context",
    "compose_recall_context",
    "estimate_tokens",
    "injected_kind",
    "is_injected_system",
    "render_profile",
]
