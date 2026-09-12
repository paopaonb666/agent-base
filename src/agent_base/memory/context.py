"""上下文组装与短期压缩（M6d）。

两个职责：

- **注入组装**（``compose_context``）：把画像、记忆块、会话摘要与按当
  前消息召回的相关记忆打包成一个系统注入块（mem0/Letta/Memobase 三家
  的注入面合一），有字符预算——召回条目按得分从低到高先丢，摘要再截断。
- **模型输入修剪**（``build_model_input``）：chat 图发给模型的输入 =
  全量历史去"注入型系统消息"（只保留本轮最新一条）+ token 预算修剪
  （保尾部、工具调用序列不拆散）。checkpointer 里的全量历史刻意不动
  ——它是 recall 语义的载体，修剪只影响"本轮给模型看什么"。

注入标记（``MEMORY_CONTEXT_PREFIX`` / ``ATTACHMENT_CONTEXT_PREFIX``）
同时是识别"注入型系统消息"的依据：没有标记的 system 消息（如 writer
模板的临时拼装不落库，普通历史里根本没有 system）不受去重影响。
"""

from __future__ import annotations

from typing import Any

from agent_base.memory.retrieval import ScoredMemory
from agent_base.memory.store import MemoryBlock

# 注入型系统消息的标记前缀。server 层注入时必须以它开头；修剪逻辑靠
# 它区分"轮轮累积的注入块"与真实历史。
MEMORY_CONTEXT_PREFIX = "【长期记忆上下文】"
ATTACHMENT_CONTEXT_PREFIX = "以下是用户上传的附件内容"

# token 近似估算系数（DeepSeek 系 tokenizer 的经验值：中文约 0.6 token/字，
# ASCII 约 0.25 token/字符，即 ~4 字符/token）。估算宁大勿小——宁可早
# 修剪也不要静默撑爆上下文。
_CJK_TOKEN_PER_CHAR = 0.6
_ASCII_TOKEN_PER_CHAR = 0.25


def estimate_tokens(text: str) -> int:
    """CJK 感知的粗略 token 估算（不引入分词器依赖）。"""
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return int(cjk * _CJK_TOKEN_PER_CHAR + (len(text) - cjk) * _ASCII_TOKEN_PER_CHAR)


def is_injected_system(message: Any) -> bool:
    """判断一条消息是否是"注入型"系统消息（记忆上下文 / 附件全文）。"""
    if getattr(message, "type", "") != "system":
        return False
    content = message.content
    if isinstance(content, list):  # 多段 content：取 text 段
        content = "".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part)
            for part in content
        )
    if not isinstance(content, str):
        return False
    return content.startswith(MEMORY_CONTEXT_PREFIX) or content.startswith(
        ATTACHMENT_CONTEXT_PREFIX
    )


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
        sections.append(("相关记忆", "\n".join(
            f"- {item.record.content}" for item in recalled
        )))

    if not sections:
        return None

    header = (
        f"{MEMORY_CONTEXT_PREFIX}以下内容由系统注入，供回答时参考；"
        "与当前对话无关时不必提及，也不要向用户复述本块的存在。\n"
    )

    def render(sel: list[tuple[str, str]]) -> str:
        return header + "\n\n".join(
            f"## {title}\n{body}" for title, body in sel
        )

    rendered_all = render(sections)
    if len(rendered_all) <= max_chars:
        return rendered_all

    # 预算超限：先丢相关记忆（得分最低的先丢），再截摘要。
    recalled_idx = next(
        (i for i, (title, _) in enumerate(sections) if title == "相关记忆"), None
    )
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


def build_model_input(messages: list[Any], *, max_tokens: int) -> list[Any]:
    """构造发给模型的输入：注入去重 + 预算修剪（不修改原列表）。

    - 注入型系统消息只保留**最后一条**（本轮注入的），历史里的全部剔除
      ——否则附件全文/记忆块会随 checkpoint 轮轮累积，把上下文撑爆；
    - 从尾部按 token 预算保留近期消息；修剪边界不得落在工具调用序列
      中间（ToolMessage 的开头），保证 AIMessage(tool_calls) 与其
      ToolMessage 成对出现；
    - 至少保留最后一条消息（单条超预算的极端情况原样保留，交给模型
      的上下文上限兜底）。
    """
    if not messages:
        return []

    # 1) 注入去重：保留最后一次出现的注入块。
    last_injected = max(
        (i for i, message in enumerate(messages) if is_injected_system(message)),
        default=None,
    )
    kept_indices = [
        i
        for i, message in enumerate(messages)
        if not is_injected_system(message) or i == last_injected
    ]

    # 2) token 预算：从尾部向前累积。
    def token_of(index: int) -> int:
        message = messages[index]
        content = message.content
        if isinstance(content, list):
            content = "".join(
                str(part.get("text", "")) if isinstance(part, dict) else str(part)
                for part in content
            )
        return estimate_tokens(str(content)) + 4  # 每条消息的固定开销

    total = 0
    cut = 0  # kept_indices 里保留的起点
    for run in range(len(kept_indices) - 1, -1, -1):
        index = kept_indices[run]
        total += token_of(index)
        if total > max_tokens and run < len(kept_indices) - 1:
            # 超预算且至少保住最后一条；从这个更早的消息开始丢弃。
            cut = run + 1
            break

    kept = kept_indices[cut:]

    # 3) 修剪边界安全：起点不得是 ToolMessage（会拆散工具配对）。
    while (
        len(kept) > 1
        and cut > 0  # 只在真的发生修剪时才牺牲边界消息
        and getattr(messages[kept[0]], "type", "") == "tool"
    ):
        kept.pop(0)

    return [messages[i] for i in kept]


__all__ = [
    "ATTACHMENT_CONTEXT_PREFIX",
    "MEMORY_CONTEXT_PREFIX",
    "build_model_input",
    "compose_context",
    "estimate_tokens",
    "is_injected_system",
    "render_profile",
]
