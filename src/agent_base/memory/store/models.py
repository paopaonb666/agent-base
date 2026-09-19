"""记忆存储的数据模型（P0-3 拆分）：记录/块/摘要/分块/审计/版本 + 编解码。

时间戳在数据类里统一为 epoch 秒（``float``），落库形式由后端决定
（sqlite 存字符串，MySQL 存 datetime）。向量以 float32 小端 BLOB 编码
（``array`` 模块，无 numpy 依赖），``encode_embedding`` / ``decode_embedding``
负责编解码。
"""

from __future__ import annotations

import sys
import time
from array import array
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

# 记忆的合法类别（LangMem 分类法）与状态（整合管线的生命周期）。
KNOWN_MEMORY_KINDS: tuple[str, ...] = ("semantic", "episodic", "procedural")
KNOWN_MEMORY_STATUSES: tuple[str, ...] = ("active", "superseded", "archived")

# 用户画像的确定性 memory_id 前缀：画像复用 memories 表（agent_id="*"、
# tags=["profile"]），以 id 前缀与普通记忆区分——检索时被排除（画像由
# 上下文组装单独注入），管理端点里可见可删。
PROFILE_ID_PREFIX = "profile:"


def profile_memory_id(user_id: str) -> str:
    """某用户画像记录的确定性 id。"""
    return f"{PROFILE_ID_PREFIX}{user_id}"


class MemoryStoreError(ValueError):
    """存储层的快速失败：非法类别/状态等数据错误在写入边界被拒绝。"""


def encode_embedding(vector: Sequence[float]) -> bytes:
    """把向量编码为 float32 小端 BLOB（紧凑、跨后端一致的二进制形态）。"""
    buf = array("f", vector)
    if sys.byteorder == "big":  # pragma: no cover - 大端平台罕见，防御性分支
        buf.byteswap()
    return buf.tobytes()


def decode_embedding(blob: bytes) -> list[float]:
    """把 float32 小端 BLOB 解码回向量；空输入返回空表。"""
    if not blob:
        return []
    buf = array("f")
    buf.frombytes(bytes(blob))
    if sys.byteorder == "big":  # pragma: no cover - 大端平台罕见，防御性分支
        buf.byteswap()
    return list(buf)


@dataclass(frozen=True)
class MemoryRecord:
    """一条跨会话长期记忆。"""

    memory_id: str
    user_id: str
    agent_id: str  # 模块名；"*" 表示全模块共享
    kind: str  # semantic | episodic | procedural
    content: str
    tags: list[str] = field(default_factory=list)
    # float32 小端 BLOB（encode_embedding 的产物）；None = 未向量化。
    embedding: bytes | None = None
    embedding_dim: int | None = None
    salience: float = 0.5
    status: str = "active"  # active | superseded | archived
    source_thread_id: str = ""
    source_refs: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    last_accessed_at: float | None = None
    access_count: int = 0

    def meta(self) -> dict[str, Any]:
        """对外元信息（不含 embedding 字节与审计字段）。"""
        return {
            "memory_id": self.memory_id,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "kind": self.kind,
            "content": self.content,
            "tags": list(self.tags),
            "salience": self.salience,
            "status": self.status,
            "source_thread_id": self.source_thread_id,
            "has_embedding": self.embedding is not None,
            "access_count": self.access_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class MemoryBlock:
    """一个常驻上下文的记忆块（Letta 式，agent 可用工具自我编辑）。"""

    user_id: str
    agent_id: str
    label: str  # persona | human | 自定义标签
    content: str
    char_limit: int = 2000
    version: int = 0
    updated_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class SessionSummary:
    """一个线程的滚动摘要（短期压缩的载体，覆盖已修剪的旧历史）。"""

    user_id: str
    thread_id: str  # 带模块前缀的完整 thread id（模块名已含于其中）
    summary: str
    agent_id: str = ""
    covered_message_count: int = 0
    updated_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class DocChunk:
    """文档知识库的一个分块（上传文件的文本切片 + 可选 embedding）。

    ``offsets`` 是 ``((start, end), …)``——各段在原文（extracted_text）
    中的字符区间，切片可视化高亮边界用；段落打包块是多区间，固定窗口
    切片是单区间。旧数据为 None（前端降级为纯卡片视图）。
    """

    chunk_id: str
    file_id: str
    thread_id: str
    user_id: str
    agent_id: str
    ordinal: int
    text: str
    offsets: tuple[tuple[int, int], ...] | None = None
    embedding: bytes | None = None
    embedding_dim: int | None = None
    created_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class MemoryOp:
    """一次记忆系统操作的审计记录（成功与失败都记录）。"""

    op_id: str
    op: str  # extract | consolidate | summary | profile | ingest | manual | ...
    user_id: str = ""
    agent_id: str = ""
    thread_id: str = ""
    detail: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"  # ok | error
    error_text: str = ""
    duration_ms: int = 0
    created_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class MemoryVersion:
    """一条记忆的内容版本快照（版本史，锐评 #7 后半句）。

    op 语义：create（首版）/ update（内容或属性变更，content 为新值、
    previous_content 为旧值）/ delete（墓碑：content 为空串，
    previous_content 保留最后内容）/ restore（从历史版本恢复）。
    """

    version_id: str
    memory_id: str
    user_id: str
    op: str  # create | update | delete | restore
    content: str
    previous_content: str = ""
    status: str = "active"
    created_at: float = field(default_factory=time.time)


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
