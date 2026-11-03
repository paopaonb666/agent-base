"""基座存储的数据模型。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ThreadIndex:
    """一行会话索引（S1 线程作用域的数据模型基础）。

    ``thread_id`` 是带模块前缀的完整 id（``module:user_thread_id``）；
    属主（user_id）在 invoke 时写入，线程端点的列表/属主校验据此执行。
    """

    thread_id: str
    user_id: str
    module: str = ""
    title: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


# 消息节点的角色集合（ADR-013）：树只存"真实对话消息"——注入型系统
# 消息（记忆上下文/附件全文）按 core.context 的标记前缀识别，不入树。
NODE_ROLES = ("human", "ai", "tool")


@dataclass(frozen=True)
class MessageNode:
    """会话消息树的一个节点（ADR-013）：append-only，软删除。

    ``content`` 存文本；多段 content（多模态 image_url 等）以 JSON 形态
    存进 ``content`` 且 ``content_json=1``——物化回 LangChain 消息时
    需要完整保真，前端展示仍用提取文本 + ``attachments_json``。
    ``tool_calls_json`` 是 AIMessage 的调用清单（ToolMessage 靠
    ``tool_call_id`` 配对）；``seq`` 是线程内单调序号（同线程唯一，
    前端稳定排序的依据）。活跃叶子指针存在独立的 ``tree_pointers``
    表（不动 thread_index 的 schema——活跃分支指针必须持久化，这是
    HF chat-ui 的教训）。
    """

    node_id: str
    thread_id: str
    parent_id: str | None
    role: str
    content: str = ""
    content_json: int = 0
    tool_calls_json: str = "[]"
    tool_call_id: str = ""
    name: str = ""
    status: str = ""
    attachments_json: str = "[]"
    label: str = ""
    starred: bool = False
    archived: bool = False
    created_at: float = field(default_factory=time.time)
    seq: int = 0
    deleted_at: float | None = None
    # 来源 LangChain 消息 id（langgraph 进 state 时分配）：物化重建消息时
    # 原样带回（id 稳定 → checkpointer 的消息身份稳定），尾部追加与状态
    # 比对全部按 id 精确匹配，不依赖内容（"重新生成出两条相同回复"的
    # 内容歧义不存在）。
    msg_id: str = ""

    @property
    def deleted(self) -> bool:
        return self.deleted_at is not None
