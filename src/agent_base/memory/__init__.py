"""企业级记忆系统（M6）：跨会话长期记忆、用户画像与文档知识库。

README"范围红线"里"长期记忆以模块的形式到来"的兑现。本包刻意只依赖
``core``（config）与自身——它像 filestore/toollog 一样是基座的基础设施，
而不是业务模块；存储后端跟随 ``CHECKPOINTER_BACKEND``（memory/sqlite/
mysql），同一套表结构在三条后端上语义一致。

分层：

- ``store``      —— 数据模型 + 三后端存储（M6a）
- ``embeddings`` —— 语义向量客户端（硅基流动/openai 兼容 + 关键词降级）（M6b）
- ``retrieval``  —— 混合检索：向量 + BM25 + 时间衰减 + 显著度（M6b）
- ``pipeline``   —— 记忆形成：抽取/整合/画像/滚动摘要（M6c）
- ``context``    —— 上下文组装与短期压缩（M6d）
- ``tools``      —— agent 可调用的记忆工具（M6e）
- ``service``    —— 对入口层与模块暴露的门面（M6c 起）
"""

from agent_base.memory.embeddings import (
    EmbeddingClient,
    EmbeddingError,
    NullEmbedding,
    OpenAICompatibleEmbedding,
    build_embedding_client,
)
from agent_base.memory.retrieval import (
    ScoredMemory,
    cosine_similarity,
    recall_memories,
    score_memories,
    tokenize,
)
from agent_base.memory.service import MemoryService, build_memory_service, new_memory_id
from agent_base.memory.store import (
    KNOWN_MEMORY_KINDS,
    KNOWN_MEMORY_STATUSES,
    DocChunk,
    MemoryBlock,
    MemoryMemoryStore,
    MemoryOp,
    MemoryRecord,
    MemoryStore,
    MemoryStoreError,
    MysqlMemoryStore,
    SessionSummary,
    SqliteMemoryStore,
    build_memory_store,
    decode_embedding,
    encode_embedding,
)

__all__ = [
    "KNOWN_MEMORY_KINDS",
    "KNOWN_MEMORY_STATUSES",
    "DocChunk",
    "EmbeddingClient",
    "EmbeddingError",
    "MemoryBlock",
    "MemoryMemoryStore",
    "MemoryOp",
    "MemoryRecord",
    "MemoryService",
    "MemoryStore",
    "MemoryStoreError",
    "MysqlMemoryStore",
    "NullEmbedding",
    "OpenAICompatibleEmbedding",
    "ScoredMemory",
    "SessionSummary",
    "SqliteMemoryStore",
    "build_embedding_client",
    "build_memory_service",
    "build_memory_store",
    "cosine_similarity",
    "decode_embedding",
    "encode_embedding",
    "new_memory_id",
    "recall_memories",
    "score_memories",
    "tokenize",
]
