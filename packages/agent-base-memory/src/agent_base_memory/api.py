"""长期记忆管理端点（H1 拆分）：CRUD / blocks / profile / audit / versions。

所有端点经 ``get_current_user`` 依赖取得作用域身份（S1 统一依赖）；
对象级操作再做属主比对——身份验通过不代表能碰别人的数据，404 而非
403 掩蔽存在性。
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from agent_base.entrypoints.server.deps import get_current_user, get_runtime, require_memory
from agent_base.memory.store import KNOWN_MEMORY_KINDS, KNOWN_MEMORY_STATUSES, MemoryBlock

router = APIRouter(prefix="/v1/memory", tags=["memory"])


class MemoryCreateRequest(BaseModel):
    """手工写入一条长期记忆（M6b 管理端点）。"""

    content: str = Field(min_length=1, max_length=20_000)
    kind: str = "semantic"
    tags: list[str] = Field(default_factory=list, max_length=10)
    # None → "*"（全模块共享）；模块名不校验（模块清单可动态变化）。
    agent_id: str | None = Field(default=None, max_length=64)
    salience: float = Field(default=0.5, ge=0.0, le=1.0)


class MemoryPatchRequest(BaseModel):
    """部分更新一条长期记忆；content 变化会触发重新向量化。"""

    content: str | None = Field(default=None, min_length=1, max_length=20_000)
    tags: list[str] | None = Field(default=None, max_length=10)
    salience: float | None = Field(default=None, ge=0.0, le=1.0)
    status: str | None = None


class MemoryBlockPutRequest(BaseModel):
    """手工写一个常驻记忆块（M6f）。"""

    content: str = Field(min_length=1, max_length=8000)
    # None → "*"（全模块共享的块，如全局人设）。
    module: str | None = Field(default=None, max_length=64)
    char_limit: int = Field(default=2000, ge=50, le=8000)


async def _owned_memory(memory: Any, memory_id: str, user_id: str) -> Any:
    """取记忆并校验属主；404 掩蔽（不存在与别人的记忆同响应）。"""
    existing = await memory.get_memory(memory_id)
    if existing is None or existing.user_id != user_id:
        raise HTTPException(status_code=404, detail=f"记忆不存在：{memory_id}")
    return existing


@router.post("")
async def create_memory(
    body: MemoryCreateRequest, request: Request, user_id: str = Depends(get_current_user)
) -> dict[str, Any]:
    """手工写入一条长期记忆（M6b 管理端点）。

    与管线写入（M6c）同一条路径：内容落库前尽力向量化，embedding
    不可用时存纯文本、检索自动走关键词路径。响应不含 embedding 字节。
    """
    memory = require_memory(get_runtime(request))
    if body.kind not in KNOWN_MEMORY_KINDS:
        raise HTTPException(
            status_code=400,
            detail=f"kind 非法：{body.kind!r}；允许 {list(KNOWN_MEMORY_KINDS)}",
        )
    try:
        record = await memory.add_memory(
            user_id=user_id,
            agent_id=body.agent_id or "*",
            content=body.content,
            kind=body.kind,
            tags=body.tags,
            salience=body.salience,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return record.meta()


@router.get("")
async def search_memory(
    request: Request,
    user_id: str = Depends(get_current_user),
    q: str | None = None,
    module: str | None = None,
    kind: str | None = None,
    status: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """检索/浏览长期记忆（M8 人工复核扩展）。

    ``q`` 存在 → 混合检索（向量 + BM25 + 时间衰减 + 显著度），按相关
    度降序（只搜启用中的记忆）；不存在 → 按更新时间浏览全部状态。
    ``module`` 限定 agent 作用域（该模块 + 全局共享；缺省查全部）。
    ``kind`` / ``status``（active/archived/superseded，缺省全部）在
    浏览模式下生效。每条记忆附带 ``expired`` 标志：episodic 记忆
    超过 TTL 即为已过期（召回时自动失效，是否保留由人工判断）。
    """
    rt = get_runtime(request)
    memory = require_memory(rt)
    if kind is not None and kind not in KNOWN_MEMORY_KINDS:
        raise HTTPException(
            status_code=400, detail=f"kind 非法：{kind!r}；允许 {list(KNOWN_MEMORY_KINDS)}"
        )
    status_filter = {name: (name,) for name in KNOWN_MEMORY_STATUSES if name != "expired"}.get(
        status or "", ()
    )
    if status is not None and not status_filter:
        raise HTTPException(
            status_code=400,
            detail=f"status 非法：{status!r}；允许 {list(KNOWN_MEMORY_STATUSES)}",
        )
    limit = max(1, min(limit, 100))
    ttl_days = rt.settings.memory.episodic_ttl_days

    def _expired(record: Any) -> bool:
        return (
            record.kind == "episodic"
            and ttl_days > 0
            and record.updated_at < time.time() - ttl_days * 86400
        )

    if q:
        scored = await memory.search(user_id=user_id, agent_id=module, query=q, top_k=limit)
        results = [
            {
                "score": round(item.score, 4),
                **item.record.meta(),
                "expired": _expired(item.record),
            }
            for item in scored
            if kind is None or item.record.kind == kind
        ]
        return {"query": q, "memories": results}
    records = await memory.list_memories(
        user_id,
        agent_id=module,
        kinds=[kind] if kind else None,
        statuses=status_filter,
        limit=limit,
    )
    return {"memories": [{**record.meta(), "expired": _expired(record)} for record in records]}


@router.patch("/{memory_id}")
async def patch_memory(
    memory_id: str,
    body: MemoryPatchRequest,
    request: Request,
    user_id: str = Depends(get_current_user),
) -> dict[str, Any]:
    """部分更新一条长期记忆（改内容会重新向量化）。"""
    memory = require_memory(get_runtime(request))
    if body.status is not None and body.status not in KNOWN_MEMORY_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=f"status 非法：{body.status!r}；允许 {list(KNOWN_MEMORY_STATUSES)}",
        )
    await _owned_memory(memory, memory_id, user_id)
    try:
        updated = await memory.update_memory(
            memory_id,
            content=body.content,
            tags=body.tags,
            salience=body.salience,
            status=body.status,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return updated.meta() if updated is not None else {}


@router.delete("/{memory_id}")
async def delete_memory(
    memory_id: str, request: Request, user_id: str = Depends(get_current_user)
) -> dict[str, Any]:
    """删除一条长期记忆（硬删除；版本墓碑与审计保留）。"""
    memory = require_memory(get_runtime(request))
    await _owned_memory(memory, memory_id, user_id)
    deleted = await memory.delete_memory(memory_id)
    return {"deleted": deleted}


@router.get("/{memory_id}/versions")
async def list_memory_versions(
    memory_id: str,
    request: Request,
    user_id: str = Depends(get_current_user),
    limit: int = 50,
) -> dict[str, Any]:
    """一条记忆的内容版本史（create/update/delete/restore 快照），最新在前。"""
    memory = require_memory(get_runtime(request))
    await _owned_memory(memory, memory_id, user_id)
    versions = await memory.list_versions(memory_id, limit=max(1, min(limit, 200)))
    return {
        "versions": [
            {
                "version_id": v.version_id,
                "op": v.op,
                "content": v.content,
                "previous_content": v.previous_content or None,
                "status": v.status,
                "created_at": v.created_at,
            }
            for v in versions
        ]
    }


@router.post("/{memory_id}/versions/{version_id}/restore")
async def restore_memory_version(
    memory_id: str,
    version_id: str,
    request: Request,
    user_id: str = Depends(get_current_user),
) -> dict[str, Any]:
    """把记忆内容恢复到指定版本（产生一条 op=restore 的新版本）。"""
    memory = require_memory(get_runtime(request))
    await _owned_memory(memory, memory_id, user_id)
    restored = await memory.restore_version(memory_id, version_id)
    if restored is None:
        raise HTTPException(status_code=400, detail="版本不可恢复：version_id 无效或对应删除墓碑")
    return restored.meta()


@router.get("/blocks")
async def list_memory_blocks(
    request: Request,
    user_id: str = Depends(get_current_user),
    module: str | None = None,
) -> dict[str, Any]:
    """列出当前用户的常驻记忆块（M6f；module 缺省查全局块）。"""
    memory = require_memory(get_runtime(request))
    blocks = await memory.store.list_blocks(user_id, module or "*")
    return {
        "blocks": [
            {
                "agent_id": block.agent_id,
                "label": block.label,
                "content": block.content,
                "char_limit": block.char_limit,
                "version": block.version,
                "updated_at": block.updated_at,
            }
            for block in blocks
        ]
    }


@router.put("/blocks/{label}")
async def put_memory_block(
    label: str,
    body: MemoryBlockPutRequest,
    request: Request,
    user_id: str = Depends(get_current_user),
) -> dict[str, Any]:
    """手工写一个常驻记忆块（覆盖式；与 agent 的 memory_update_block
    工具同表，版本号递增）。"""
    memory = require_memory(get_runtime(request))
    agent_id = body.module or "*"
    existing = await memory.store.get_block(user_id, agent_id, label)
    block = MemoryBlock(
        user_id=user_id,
        agent_id=agent_id,
        label=label,
        content=body.content,
        char_limit=body.char_limit,
        version=(existing.version if existing else 0) + 1,
    )
    await memory.store.upsert_block(block)
    return {
        "agent_id": block.agent_id,
        "label": block.label,
        "content": block.content,
        "char_limit": block.char_limit,
        "version": block.version,
        "updated_at": block.updated_at,
    }


@router.delete("/blocks/{label}")
async def delete_memory_block(
    label: str,
    request: Request,
    user_id: str = Depends(get_current_user),
    module: str | None = None,
) -> dict[str, Any]:
    """删除一个常驻记忆块。"""
    memory = require_memory(get_runtime(request))
    deleted = await memory.store.delete_block(user_id, module or "*", label)
    return {"deleted": deleted}


@router.get("/profile")
async def get_memory_profile(
    request: Request, user_id: str = Depends(get_current_user)
) -> dict[str, Any]:
    """读取当前用户的结构化画像（M6c 形成，M6f 暴露）。"""
    memory = require_memory(get_runtime(request))
    return {"profile": await memory.get_profile(user_id)}


@router.get("/audit")
async def list_memory_audit(
    request: Request,
    user_id: str = Depends(get_current_user),
    limit: int = 50,
) -> dict[str, Any]:
    """记忆系统的操作审计（M6）：抽取/整合/画像/摘要/摄取/手工增删。"""
    memory = require_memory(get_runtime(request))
    ops = await memory.store.list_ops(user_id, limit=max(1, min(limit, 200)))
    return {
        "ops": [
            {
                "op_id": op.op_id,
                "op": op.op,
                "user_id": op.user_id,
                "agent_id": op.agent_id,
                "thread_id": op.thread_id,
                "detail": op.detail,
                "status": op.status,
                "error_text": op.error_text or None,
                "duration_ms": op.duration_ms,
                "created_at": op.created_at,
            }
            for op in ops
        ]
    }
