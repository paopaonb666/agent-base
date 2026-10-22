"""用户画像子域（M6c，P0-2 拆分自 MemoryService）。

Memobase 式结构化 JSON 画像：随对话分节合并演化，存为 memories 表里的
确定性记录（``profile:<user_id>``，agent_id="*"、tags=["profile"]），
整体注入上下文而不参与相似度召回。每次演化写一条版本快照（与记忆
内容变更同规）。
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

from agent_base.memory.audit import record_version_safe
from agent_base.memory.store import MemoryRecord, MemoryStore, MemoryVersion
from agent_base.memory.store.models import new_memory_id, profile_memory_id
from agent_base.memory.textkit import _trim_profile_to_chars

if TYPE_CHECKING:  # pragma: no cover - import avoided at runtime
    from agent_base.core.config import MemorySettings


class ProfileService:
    """用户画像读写：依赖收敛到 store + MemorySettings 节。"""

    def __init__(self, store: MemoryStore, settings: MemorySettings) -> None:
        self._store = store
        self._settings = settings

    async def get_profile(self, user_id: str) -> dict[str, Any] | None:
        """读取结构化用户画像（JSON dict）；不存在或畸形返回 None。"""
        record = await self._store.get_memory(profile_memory_id(user_id))
        if record is None:
            return None
        try:
            parsed = json.loads(record.content)
        except ValueError:
            return None
        return parsed if isinstance(parsed, dict) else None

    async def save_profile(self, user_id: str, profile: dict[str, Any]) -> None:
        """整包保存用户画像（管线合并后的完整 JSON）。

        画像不需要向量——它由上下文组装整体注入而不是按相似度召回，
        跳过 embedding 省一次 API 调用；检索侧已按 id 前缀排除画像。
        保存前做结构级硬截断（锐评漏项）：prompt 里的"600 字以内"只是
        软约束，这里保证序列化长度不超过 MEMORY_PROFILE_MAX_CHARS，
        防止画像随对话缓慢膨胀注入预算。
        """
        memory_id = profile_memory_id(user_id)
        existing = await self._store.get_memory(memory_id)
        now = time.time()
        profile = _trim_profile_to_chars(profile, self._settings.profile_max_chars)
        content = json.dumps(profile, ensure_ascii=False)
        await self._store.upsert_memory(
            MemoryRecord(
                memory_id=memory_id,
                user_id=user_id,
                agent_id="*",
                kind="semantic",
                content=content,
                tags=["profile"],
                salience=1.0,
                source_refs=["profile"],
                created_at=existing.created_at if existing else now,
                updated_at=now,
                last_accessed_at=existing.last_accessed_at if existing else None,
                access_count=existing.access_count if existing else 0,
            )
        )
        await record_version_safe(
            self._store,
            MemoryVersion(
                version_id=new_memory_id(),
                memory_id=memory_id,
                user_id=user_id,
                op="update" if existing else "create",
                content=content,
                previous_content=existing.content if existing else "",
            ),
        )
