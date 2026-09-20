"""附件与知识库文件端点（H1 拆分）：上传 / 撤销 / 预览 / 切片 / 原始字节。

属主模型（S1）：上传时记录 user_id；读取/撤销端点做属主比对（404
掩蔽）。``user_id == ""`` 视为无主遗留（属主校验引入前的旧行），允许
撤销——file_id 本身 48 位随机不可猜。
"""

from __future__ import annotations

import uuid
from typing import Any, cast
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from fastapi.responses import Response

from agent_base.entrypoints.server.background import spawn_background
from agent_base.entrypoints.server.deps import (
    get_current_user,
    get_runtime,
    known_module,
    memory_or_none,
)
from agent_base.entrypoints.server.serializers import (
    IMAGE_EXTENSIONS,
    IMAGE_STORED_FORMATS,
    _sniff_image_mime,
)
from agent_base.extensions.filestore import UploadedFileInfo
from agent_base.tools.parsing import DocumentParseError, parse_document

router = APIRouter(prefix="/v1", tags=["files"])

# 上传读取的分块大小：流式读取 + 边读边判大小（M5），声明的上限
# 真正挡住内存/磁盘压力，而不是"先全量读入再判"。
_UPLOAD_CHUNK_BYTES = 1 << 20


async def _owned_file(rt: Any, file_id: str, user_id: str) -> UploadedFileInfo:
    """取附件并校验属主；404 掩蔽（filestore 缺席由调用方先判）。"""
    file_store = rt.file_store
    if file_store is None:
        raise HTTPException(status_code=503, detail="附件存储未启用（存储后端不可用）")
    infos = await file_store.get_many([file_id])
    info = infos[0] if infos else None
    if info is None or info.user_id not in ("", user_id):
        # 404 而非 403：不向第三方泄露文件 id 的存在性。
        raise HTTPException(status_code=404, detail=f"文件不存在：{file_id}")
    return cast(UploadedFileInfo, info)


@router.post("/agents/{module}/files")
async def upload_file(
    module: str, request: Request, file: UploadFile, user_id: str = Depends(get_current_user)
) -> dict[str, Any]:
    """上传附件（M4a 解析层 + M4c 图片多模态的 HTTP 入口）。

    文档（pdf/docx/txt/md）走 ``tools/parsing`` 提取文本，注入对话
    上下文；图片（png/jpg/webp/gif）按 magic bytes 嗅探后整字节入库，
    invoke 时以多模态 content blocks 注入 vision 模型。两类都返回
    元信息（不含全文与字节）；解析/校验失败以 400 返回可读原因。
    """
    rt = get_runtime(request)
    if not known_module(rt, module):
        raise HTTPException(status_code=404, detail=f"unknown module {module!r}")
    file_store = rt.file_store
    if file_store is None:
        raise HTTPException(status_code=503, detail="附件存储未启用（存储后端不可用）")
    # 流式分块读取，边读边判大小（M5）：超限立即 413，不再把整个
    # 超大文件读进内存。
    max_bytes = rt.settings.doc_parse.max_input_bytes
    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(_UPLOAD_CHUNK_BYTES):
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"文件超出大小上限：> {max_bytes} 字节",
            )
        chunks.append(chunk)
    data = b"".join(chunks)
    if not data:
        raise HTTPException(status_code=400, detail="上传文件为空")
    suffix = (file.filename or "").rsplit(".", 1)[-1].lower()
    if suffix in IMAGE_EXTENSIONS:
        # 图片：不走文本解析，按 magic bytes 校验后整字节入库，
        # invoke 时以多模态 content blocks 注入 vision 模型。
        mime = _sniff_image_mime(data)
        if mime is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"文件内容不是有效的图片：扩展名 {suffix!r} 与实际内容不符（magic 校验失败）"
                ),
            )
        info = UploadedFileInfo(
            file_id=uuid.uuid4().hex[:12],
            filename=file.filename or "",
            format=mime.split("/", 1)[1],
            pages=None,
            paragraphs=None,
            truncated=False,
            text_len=0,
            extracted_text="",
            content=data,
            user_id=user_id,
        )
        await file_store.save(info)
        spawn_background(file_store.purge_orphans(), name="purge-orphans")
        return info.meta()
    try:
        doc = parse_document(data, filename=file.filename or "")
    except DocumentParseError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    # 空提取文本（扫描件/纯图片 PDF）不是错误，但要显式告知前端。
    warning = ""
    if not doc.text.strip():
        warning = "未能从文件中提取到文本：可能是扫描件或纯图片 PDF"
    info = UploadedFileInfo(
        file_id=uuid.uuid4().hex[:12],
        filename=file.filename or "",
        format=doc.format,
        pages=doc.pages,
        paragraphs=doc.paragraphs,
        truncated=doc.truncated,
        text_len=len(doc.text),
        extracted_text=doc.text,
        content=data,
        warning=warning,
        user_id=user_id,
    )
    await file_store.save(info)
    memory = memory_or_none(rt)
    # 知识库摄取（M6e）：文档文本切块 + 向量化入 doc_chunks（用户级
    # 知识资产，独立于线程存活）。fire-and-forget，失败不影响上传。
    if memory is not None and info.extracted_text.strip():
        spawn_background(
            memory.ingest_document(
                file_id=info.file_id,
                user_id=user_id,
                agent_id=module,
                text=info.extracted_text,
            ),
            name="memory-ingest",
        )
    # 机会式清理 24h 未绑定的孤儿附件（fire-and-forget）。
    spawn_background(file_store.purge_orphans(), name="purge-orphans")
    return info.meta()


@router.delete("/agents/{module}/files/{file_id}")
async def revoke_file(
    module: str,
    file_id: str,
    request: Request,
    user_id: str = Depends(get_current_user),
) -> dict[str, Any]:
    """撤销一份上传文件及其知识库分块（边界加固）。

    上传错误文件的显式收回路径：文件原始行 + 该文件的全部知识库
    分块一并删除。属主校验基于上传时记录的 user_id。
    """
    rt = get_runtime(request)
    if not known_module(rt, module):
        raise HTTPException(status_code=404, detail=f"unknown module {module!r}")
    # 属主校验（不存在/非属主即 404）；返回值本身在撤销路径不需要。
    await _owned_file(rt, file_id, user_id)
    assert rt.file_store is not None
    deleted = await rt.file_store.delete_file(file_id)
    memory = memory_or_none(rt)
    chunks_removed = 0
    if memory is not None:
        chunks_removed = await memory.revoke_document(file_id=file_id, user_id=user_id)
    spawn_background(rt.file_store.purge_orphans(), name="purge-orphans")
    return {"deleted": deleted, "chunks_removed": chunks_removed}


@router.get("/agents/{module}/files/{file_id}/preview")
async def preview_file(
    module: str,
    file_id: str,
    request: Request,
    user_id: str = Depends(get_current_user),
) -> dict[str, Any]:
    """文档预览数据：元信息 + 提取正文（M7 前端预览的数据源）。

    ``extracted_text`` 与附件注入给模型的内容同源（M4b），前端展示
    的即系统"记得"的内容。
    """
    rt = get_runtime(request)
    if not known_module(rt, module):
        raise HTTPException(status_code=404, detail=f"unknown module {module!r}")
    info = await _owned_file(rt, file_id, user_id)
    return {
        "file_id": info.file_id,
        "filename": info.filename,
        "format": info.format,
        "pages": info.pages,
        "text_len": info.text_len,
        "truncated": info.truncated,
        **({"warning": info.warning} if info.warning else {}),
        "extracted_text": info.extracted_text,
    }


@router.get("/agents/{module}/files/{file_id}/chunks")
async def list_file_chunks(
    module: str,
    file_id: str,
    request: Request,
    user_id: str = Depends(get_current_user),
    limit: int = 200,
) -> dict[str, Any]:
    """文档的切片存储视图：序号/原文区间/向量化状态（M7 可视化）。

    ``offsets`` 是各段在原文（extracted_text）中的字符区间列表——
    段落打包块是多区间，固定窗口切片是单区间；旧数据为 null，前端
    降级为纯卡片视图。按请求方 user_id 过滤，天然用户隔离。
    """
    rt = get_runtime(request)
    if not known_module(rt, module):
        raise HTTPException(status_code=404, detail=f"unknown module {module!r}")
    await _owned_file(rt, file_id, user_id)
    memory = memory_or_none(rt)
    chunks = []
    if memory is not None:
        chunks = await memory.store.list_chunks(
            user_id, file_id=file_id, limit=max(1, min(limit, 2000))
        )
    return {
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "ordinal": chunk.ordinal,
                "text": chunk.text,
                "offsets": [list(seg) for seg in chunk.offsets] if chunk.offsets else None,
                "char_len": len(chunk.text),
                "has_embedding": chunk.embedding is not None,
                "embedding_dim": chunk.embedding_dim,
            }
            for chunk in chunks
        ]
    }


@router.get("/agents/{module}/files/{file_id}/raw")
async def download_file_raw(
    module: str,
    file_id: str,
    request: Request,
    user_id: str = Depends(get_current_user),
) -> Response:
    """原始文件字节：图片按存储 mime 直出（前端 <img> 预览），
    文档以 attachment 下载（RFC 5987 编码中文文件名）。"""
    rt = get_runtime(request)
    if not known_module(rt, module):
        raise HTTPException(status_code=404, detail=f"unknown module {module!r}")
    info = await _owned_file(rt, file_id, user_id)
    if not info.content:
        raise HTTPException(status_code=404, detail="文件内容不可用")
    headers: dict[str, str] = {}
    if info.format in IMAGE_STORED_FORMATS:
        media_type = f"image/{info.format}"
    else:
        media_type = "application/octet-stream"
        headers["Content-Disposition"] = f"attachment; filename*=UTF-8''{quote(info.filename)}"
    return Response(content=info.content, media_type=media_type, headers=headers)


@router.get("/knowledge/search")
async def search_knowledge(
    request: Request,
    q: str,
    user_id: str = Depends(get_current_user),
    file_id: str | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    """知识库检索直出（成本治理 T1.1：Tier 0 零 LLM 分诊档）。

    查找类问题（"那份保单的等待期"）是检索不是推理——本端点把混合检索
    结果原样返回，全程不调用任何模型；只有需要综合/解释的查询才走
    invoke。字段命名与 M7 切片视图一致（``ordinal`` / ``offsets`` /
    ``has_embedding``），前端切片卡片可直接复用。空结果返回空列表而非
    404（没有命中是合法答案）。
    """
    rt = get_runtime(request)
    memory = memory_or_none(rt)
    if memory is None:
        raise HTTPException(status_code=503, detail="记忆系统未启用（MEMORY_ENABLED=false）")
    query = q.strip()
    if not query:
        raise HTTPException(status_code=422, detail="q 不能为空")
    scored = await memory.search_knowledge(
        user_id=user_id, agent_id=None, query=query, top_k=max(1, min(limit, 20))
    )
    if file_id is not None:
        scored = [item for item in scored if item.chunk.file_id == file_id]
    # 文件名补全（尽力而为）：分块已按 user_id 隔离，名字只做展示。
    names: dict[str, str] = {}
    if rt.file_store is not None and scored:
        infos = await rt.file_store.get_many(sorted({item.chunk.file_id for item in scored}))
        names = {info.file_id: info.filename for info in infos}
    return {
        "results": [
            {
                "score": round(item.score, 4),
                "file_id": item.chunk.file_id,
                "file_name": names.get(item.chunk.file_id, ""),
                "chunk_id": item.chunk.chunk_id,
                "ordinal": item.chunk.ordinal,
                "text": item.chunk.text,
                "offsets": (
                    [list(seg) for seg in item.chunk.offsets] if item.chunk.offsets else None
                ),
                "has_embedding": item.chunk.embedding is not None,
            }
            for item in scored
        ]
    }


@router.get("/knowledge/files")
async def list_knowledge_files(
    request: Request, user_id: str = Depends(get_current_user)
) -> dict[str, Any]:
    """知识库文件列表（M8：独立知识库页面的数据源）。

    用户级视图：该用户上传的全部文件 + 每个文件的切片数（切片
    归属由 user_id 天然隔离）。记忆系统未启用时切片数降级为 0
    （文件列表仍可用——文件是附件链路的基础设施）。
    """
    rt = get_runtime(request)
    file_store = rt.file_store
    if file_store is None:
        raise HTTPException(status_code=503, detail="附件存储未启用（存储后端不可用）")
    files = await file_store.list_for_user(user_id)
    counts: dict[str, int] = {}
    memory = memory_or_none(rt)
    if memory is not None:
        counts = await memory.store.chunk_counts_by_file(user_id)
    return {
        "files": [
            {
                **info.meta(),
                "chunks": counts.get(info.file_id, 0),
                "created_at": info.created_at,
            }
            for info in files
        ]
    }
