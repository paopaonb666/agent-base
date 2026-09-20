"""基座级上下文引擎（H3 / P1-4）。

这是与记忆无关的**通用上下文保护**，从 ``memory/context.py`` 提升而来：
任何模块的图在把持久化历史发给模型之前都应当经过它，而不是只有 chat
模块独享。两个职责：

- **注入型系统消息识别**（``injected_kind`` / ``is_injected_system``）：
  server 层注入的记忆上下文与附件全文以约定的标记前缀开头；修剪逻辑靠
  前缀区分"轮轮累积的注入块"与真实历史。没有标记的 system 消息（如
  writer 模板的临时拼装，不落库）不受去重影响。
- **模型输入修剪**（``build_model_input``）：全量历史去"注入型系统消息"
  （只保留本轮最新一条）+ token 预算修剪（保尾部、工具调用序列不拆散）。
  checkpointer 里的全量历史刻意不动——它是 recall 语义的载体，修剪只
  影响"本轮给模型看什么"。

记忆语义的注入块**组装**仍在 ``memory/context.py``（画像/记忆块/摘要/
召回是记忆子系统的领域知识）；本模块只负责"给模型看什么"的通用规则。
"""

from __future__ import annotations

from typing import Any

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


def _content_text(content: Any) -> str:
    """多段 content（部分 provider 的多模态结构）拼接出文本段。"""
    if isinstance(content, list):
        return "".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content
        )
    return str(content)


def injected_kind(message: Any) -> str | None:
    """返回消息的注入类型（匹配的标记前缀）；非注入消息返回 None。"""
    if getattr(message, "type", "") != "system":
        return None
    content = _content_text(message.content)
    if content.startswith(MEMORY_CONTEXT_PREFIX):
        return MEMORY_CONTEXT_PREFIX
    if content.startswith(ATTACHMENT_CONTEXT_PREFIX):
        return ATTACHMENT_CONTEXT_PREFIX
    return None


def is_injected_system(message: Any) -> bool:
    """判断一条消息是否是"注入型"系统消息（记忆上下文 / 附件全文）。"""
    return injected_kind(message) is not None


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

    # 1) 注入去重：只保留"本轮"的注入块——最后一条 human 消息之前的
    #    连续注入段（记忆块 + 附件全文都可能在本轮注入，必须同时保留；
    #    UI 实测缺陷 F1）。历史轮次的注入块一律剔除：其内容已随当轮
    #    消费，留着只会让模型输入轮轮膨胀。
    last_human = max(
        (i for i, message in enumerate(messages) if getattr(message, "type", "") == "human"),
        default=None,
    )
    current_injections: set[int] = set()
    if last_human is not None:
        i = last_human - 1
        while i >= 0 and injected_kind(messages[i]) is not None:
            current_injections.add(i)
            i -= 1
    kept_indices = [
        i
        for i, message in enumerate(messages)
        if injected_kind(message) is None or i in current_injections
    ]

    # 2) token 预算：从尾部向前累积。
    def token_of(index: int) -> int:
        return estimate_tokens(_content_text(messages[index].content)) + 4  # 每条消息的固定开销

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


# 多段 content 的文本提取的公开别名（planner 等模块复用）。
content_text = _content_text


__all__ = [
    "ATTACHMENT_CONTEXT_PREFIX",
    "MEMORY_CONTEXT_PREFIX",
    "build_model_input",
    "content_text",
    "estimate_tokens",
    "injected_kind",
    "is_injected_system",
]
