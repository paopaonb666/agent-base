"""为历史记忆回填 embedding（更换 embedding 模型/维度后运行一次）。

用法::

    python scripts/memory_backfill_embeddings.py

读取 .env 中的 MEMORY_EMBEDDING_* 与 CHECKPOINTER_BACKEND，把 active 且
向量缺失/维度不符的记忆（不含画像）批量重新向量化。保持原 updated_at，
不影响检索的时间衰减排序。
"""

from __future__ import annotations

import asyncio
import sys

from agent_base.core.config import Settings
from agent_base.memory.service import build_memory_service


async def main() -> int:
    settings = Settings()
    service = await build_memory_service(settings)
    if service is None:
        print("记忆系统未启用（MEMORY_ENABLED=false），无事可做")
        return 1
    try:
        if service.embedder.dims is None:
            print("未配置 embedding（MEMORY_EMBEDDING_API_KEY 为空），无事可做")
            return 0
        count = await service.backfill_embeddings()
        print(f"回填完成：{count} 条记忆的向量已更新到维度 {service.embedder.dims}")
        return 0
    finally:
        await service.aclose()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
