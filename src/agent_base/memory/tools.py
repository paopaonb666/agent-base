"""Agent 记忆工具（M6e）：agent 自服务的记忆读写与知识库检索。

五个工具（Letta 式"agent 管理自己的记忆"）：

- ``memory_search``       —— 检索当前用户的长期记忆（结果带 memory_id）
- ``memory_save``         —— 把重要信息显式写入长期记忆
- ``memory_update_block`` —— 编辑常驻上下文的记忆块（persona/human/自定义）
- ``memory_delete``       —— 删除过时/错误的记忆（用户更正事实时配合 save）
- ``knowledge_search``    —— 检索用户上传文档构建的知识库

**作用域传递**：user_id / thread_id 不走 RunnableConfig 注入——工具池的
``_TimeoutTool`` 包装会丢掉 config（``inner.ainvoke(kwargs)``）——而是
用 ContextVar：server 的 ``_produce`` 在拉起图之前设置，随 asyncio 任务
上下文自然传播到每个工具调用；CLI 等未设置的场景优雅回退到
``default`` / 全局作用域。
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from agent_base.memory.service import MemoryService
from agent_base.memory.store import MemoryBlock

logger = logging.getLogger(__name__)

# 每轮对话的 user/thread 作用域（server._produce 设置；键固定）。
_memory_scope: ContextVar[dict[str, str] | None] = ContextVar("memory_scope", default=None)

DEFAULT_BLOCK_CHAR_LIMIT = 2000
# knowledge_search 单条结果回给模型的文本截断（防止单块把上下文撑爆）。
KNOWLEDGE_SNIPPET_CHARS = 400


def set_memory_scope(user_id: str, thread_id: str) -> None:
    """设置当前任务上下文里的记忆作用域（每轮对话开始时调用一次）。"""
    _memory_scope.set({"user_id": user_id, "thread_id": thread_id})


def _scope() -> tuple[str, str, str]:
    """读取作用域 → (user_id, agent_id, full_thread_id)。

    agent_id 从带模块前缀的 thread id（``chat:xxx``）解析；拿不到时
    回退 ``default`` / ``*``（CLI、手工 invoke 工具等场景）。
    """
    scope = _memory_scope.get() or {}
    user_id = scope.get("user_id") or "default"
    thread_id = scope.get("thread_id") or ""
    agent_id = thread_id.split(":", 1)[0] if ":" in thread_id else "*"
    return user_id, agent_id, thread_id


class _MemorySearchArgs(BaseModel):
    query: str = Field(
        description="要检索的记忆内容：关键词、人名、型号或语义描述", min_length=1, max_length=2000
    )
    limit: int = Field(default=5, ge=1, le=20, description="最多返回的条数")


class _MemorySaveArgs(BaseModel):
    content: str = Field(
        description="要记住的信息，第三人称、自包含的一句话（如'用户的部署环境是 Ubuntu 22.04'）",
        min_length=1,
        max_length=2000,
    )
    kind: str = Field(
        default="semantic",
        description="记忆类别：semantic（事实/偏好）、episodic（事件）、procedural（规则/流程）",
    )
    tags: list[str] = Field(default_factory=list, max_length=10, description="检索用标签")


class _MemoryBlockArgs(BaseModel):
    label: str = Field(
        description="记忆块标签：persona（助手人设）、human（用户画像摘要）或自定义标签",
        min_length=1,
        max_length=64,
        pattern=r"^[\w.-]+$",
    )
    operation: str = Field(
        default="append", description="append（追加，默认）或 replace（整块替换）"
    )
    content: str = Field(description="块内容", min_length=1, max_length=4000)


class _MemoryDeleteArgs(BaseModel):
    memory_id: str = Field(
        description="要删除的记忆 id（从 memory_search 结果的 [id=…] 中获取）",
        min_length=4,
        max_length=64,
        pattern=r"^[\w.-]+$",
    )


class _KnowledgeSearchArgs(BaseModel):
    query: str = Field(description="要在用户上传的文档里检索的内容", min_length=1, max_length=2000)
    limit: int = Field(default=5, ge=1, le=20, description="最多返回的分块数")


def build_memory_tools(memory: MemoryService) -> list[BaseTool]:
    """构建记忆工具集（绑定到运行时的 MemoryService 实例）。"""

    async def memory_search(query: str, limit: int = 5, **_kwargs: Any) -> str:
        user_id, agent_id, _ = _scope()
        agent_scope = agent_id if agent_id != "*" else None
        scored = await memory.search(
            user_id=user_id, agent_id=agent_scope, query=query, top_k=limit
        )
        if not scored:
            return "（没有找到相关记忆）"
        # 带 memory_id：agent 可据此调用 memory_delete 更正过时记忆。
        lines = [
            f"-（相关度 {item.score:.2f}）[id={item.record.memory_id[:12]}] {item.record.content}"
            for item in scored
        ]
        return "找到的相关记忆：\n" + "\n".join(lines)

    async def memory_save(
        content: str, kind: str = "semantic", tags: list[str] | None = None, **_kwargs: Any
    ) -> str:
        user_id, agent_id, thread_id = _scope()
        record = await memory.add_memory(
            user_id=user_id,
            agent_id=agent_id,
            content=content,
            kind=kind,
            tags=tags or [],
            source_thread_id=thread_id,
        )
        return f"已记住（memory_id={record.memory_id[:12]}…，kind={record.kind}）"

    async def memory_update_block(
        label: str, content: str, operation: str = "append", **_kwargs: Any
    ) -> str:
        user_id, agent_id, _ = _scope()
        agent = agent_id if agent_id != "*" else "*"
        if operation not in ("append", "replace"):
            return f"未知 operation {operation!r}（只支持 append / replace）"
        existing = await memory.store.get_block(user_id, agent, label)
        char_limit = existing.char_limit if existing else DEFAULT_BLOCK_CHAR_LIMIT
        if operation == "append" and existing is not None:
            merged = (existing.content.rstrip() + "\n" + content).strip()
        else:
            merged = content
        if len(merged) > char_limit:
            # 保尾部：最近写入的内容通常更重要（Letta 同语义）。前缀
            # 计入预算，保证结果不超过 char_limit。
            note = "……（较早内容已截断）\n"
            merged = note + merged[-(char_limit - len(note)) :]
        version = (existing.version if existing else 0) + 1
        await memory.store.upsert_block(
            MemoryBlock(
                user_id=user_id,
                agent_id=agent,
                label=label,
                content=merged,
                char_limit=char_limit,
                version=version,
            )
        )
        return f"记忆块 {label!r} 已更新（v{version}，{len(merged)}/{char_limit} 字符）"

    async def memory_delete(memory_id: str, **_kwargs: Any) -> str:
        """删除一条指定 id 的记忆（agent 更正/清理过时记忆时使用）。"""
        user_id, _, _ = _scope()
        record = await memory.get_memory(memory_id)
        if record is None:
            return f"记忆 {memory_id} 不存在（可能已删除，id 以 memory_search 结果为准）"
        # user 隔离：只能删自己的记忆。
        if record.user_id != user_id:
            return "无法删除：该记忆不属于当前用户。"
        deleted = await memory.delete_memory(memory_id)
        return f"已删除：{record.content[:60]}" if deleted else "删除失败（记录已不存在）"

    async def knowledge_search(query: str, limit: int = 5, **_kwargs: Any) -> str:
        user_id, agent_id, _ = _scope()
        agent_scope = agent_id if agent_id != "*" else None
        scored = await memory.search_knowledge(
            user_id=user_id, agent_id=agent_scope, query=query, top_k=limit
        )
        if not scored:
            return "（知识库中没有相关内容）"
        lines = [
            f"-（相关度 {item.score:.2f}）{item.chunk.text[:KNOWLEDGE_SNIPPET_CHARS]}"
            for item in scored
        ]
        return "知识库检索结果：\n" + "\n".join(lines)

    return [
        StructuredTool(
            name="memory_search",
            description=(
                "检索关于当前用户的长期记忆（偏好、背景、过往约定等）。"
                "当需要回忆用户提过的事情或偏好时调用。"
            ),
            args_schema=_MemorySearchArgs,
            coroutine=memory_search,
        ),
        StructuredTool(
            name="memory_save",
            description=(
                "把值得跨会话记住的信息写入长期记忆。用户说'记住……'、"
                "或对话中出现持久的偏好/约束/事实时调用。宁缺毋滥。"
            ),
            args_schema=_MemorySaveArgs,
            coroutine=memory_save,
        ),
        StructuredTool(
            name="memory_update_block",
            description=(
                "更新你自己的常驻记忆块：persona（你的人设与工作方式）、"
                "human（对当前用户的画像摘要）或自定义标签。这些块每轮"
                "对话都会出现在你的上下文里。"
            ),
            args_schema=_MemoryBlockArgs,
            coroutine=memory_update_block,
        ),
        StructuredTool(
            name="memory_delete",
            description=(
                "删除一条已过时或错误的长期记忆（先用 memory_search 找到"
                " id）。当用户更正了此前的事实（如换了项目代号、搬了城市）"
                "时，配合 memory_save 使用：删旧、存新。"
            ),
            args_schema=_MemoryDeleteArgs,
            coroutine=memory_delete,
        ),
        StructuredTool(
            name="knowledge_search",
            description=(
                "在用户上传过的文档构建的知识库里检索（PDF/DOCX/TXT 等）。"
                "当用户的问题可能在他们上传的文档中有答案时调用。"
            ),
            args_schema=_KnowledgeSearchArgs,
            coroutine=knowledge_search,
        ),
    ]


__all__ = [
    "DEFAULT_BLOCK_CHAR_LIMIT",
    "build_memory_tools",
    "set_memory_scope",
]
