"""对话与会话线程端点（H1 拆分）：invoke（SSE）、线程列表/历史/删除、工具审计。

S1 线程作用域：每个线程在 ``thread_index`` 表里有属主记录（invoke 时
upsert）。列表端点只返回**当前用户**的线程；历史/删除/工具审计端点
校验属主（404 掩蔽，不泄露线程存在性）。索引存储缺席（后端不可用或
旧版本升级前）时降级为旧行为——全库扫描 + 无属主校验，仅限本地开发。
"""

from __future__ import annotations

import base64
import time
from dataclasses import replace
from datetime import datetime
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from agent_base.core.bootstrap import UnknownModuleError
from agent_base.entrypoints.server.deps import (
    get_current_user,
    get_runtime,
    known_module,
    memory_or_none,
)
from agent_base.entrypoints.server.serializers import IMAGE_STORED_FORMATS, _serialize_message
from agent_base.entrypoints.server.sse import _SSE_HEADERS, _build_capture_hook, _event_stream
from agent_base.extensions.filestore import UploadedFileInfo
from agent_base.extensions.observability import new_request_id
from agent_base.memory.context import ATTACHMENT_CONTEXT_PREFIX
from agent_base.memory.store import ThreadIndex

router = APIRouter(prefix="/v1/agents", tags=["agents"])

# 线程列表端点的扫描/数量上限（旧版回退路径仍用）：alist 按 checkpoint
# 粒度迭代（一个线程有多轮 checkpoint），无上限会在大库上退化成全表遍历。
_THREAD_LIST_SCAN_LIMIT = 2000
_THREAD_LIST_TOTAL_LIMIT = 20_000
_THREAD_LIST_LIMIT = 50


class InvokeRequest(BaseModel):
    """一次对话轮。"""

    message: str = Field(min_length=1, max_length=100_000)
    # thread_id 是 checkpointer 的一部分键：限定字符集与长度，避免任意
    # 字符串直接落库/进日志。
    thread_id: str | None = Field(
        default=None, max_length=128, pattern=r"^[\w.-]+$"
    )  # None -> 创建一个新 thread
    # 附件对话（M4b）：本条消息引用的上传文件 id（≤5 个），服务端把解析
    # 文本作为 SystemMessage 注入上下文——用户气泡保持干净。
    attachments: list[str] = Field(default_factory=list, max_length=5)
    # 难度分流（M9）：chat=普通对话；plan=强制走 planner；auto=启发式
    # （消息 >200 字或带附件 → plan，planner 缺席时静默降级 chat）。
    mode: Literal["chat", "plan", "auto"] = "chat"
    # 模型档位（成本治理 T1.2）：main=主力模型；fast=LLM_FAST_* 快档
    # （Tier 1 轻综合）。与 mode 正交——mode 决定图，profile 决定图的
    # 模型。快档未配置时显式 400，不静默降级。
    profile: Literal["main", "fast"] = "main"


def _graph_or_404(rt: Any, module: str) -> Any:
    """按模块取图；未知模块给干净的 404（UnknownModuleError 是 KeyError
    子类，str() 会带引号包装，取 args[0]）。"""
    try:
        return rt.graph(module)
    except UnknownModuleError as exc:
        detail = exc.args[0] if exc.args else "unknown module"
        raise HTTPException(status_code=404, detail=str(detail)) from exc


async def _thread_owner_or_404(rt: Any, module: str, thread_id: str, user_id: str) -> None:
    """线程属主校验（S1）：索引在位时，非属主一律 404 掩蔽。

    历史线程（索引行不存在——升级前的旧数据）同样 404，与审查报告
    S1 的迁移策略一致："旧线程不可访问"。
    """
    index = rt.thread_index
    if index is None:
        return
    entry = await index.get_thread_index(f"{module}:{thread_id}")
    if entry is None or entry.user_id != user_id:
        raise HTTPException(status_code=404, detail=f"线程不存在：{thread_id}")


@router.post("/{module}/invoke")
async def invoke(module: str, body: InvokeRequest, request: Request) -> StreamingResponse:
    """一次对话轮：SSE 流输出契约事件（ping/step/delta/sources/done/error）。"""
    rt = get_runtime(request)
    # 难度分流（M9）：mode 只决定调哪张图——thread 前缀、记忆 agent_id、
    # thread_index 属主、附件绑定、记忆形成钩子全部维持请求模块不变
    # （决策 5：图选择与线程命名空间解耦）。
    use_plan = body.mode == "plan" or (
        body.mode == "auto" and (len(body.message) > 200 or bool(body.attachments))
    )
    if use_plan and "planner" not in rt.modules:
        if body.mode == "plan":
            raise HTTPException(
                status_code=503,
                detail="planner 模块未启用（AGENT_MODULES 不含 planner）",
            )
        use_plan = False  # auto：静默降级
    # 模型档位（T1.2）：profile 决定图用哪个模型——快档视图共享
    # checkpointer/memory，线程历史跨档连续；未配置显式 400。
    view = rt.profile_view(body.profile)
    if view is None:
        raise HTTPException(
            status_code=400,
            detail='LLM_FAST_* 未配置：profile="fast" 不可用（配置快速档或使用默认档）',
        )
    graph = _graph_or_404(view, "planner" if use_plan else module)
    user_thread_id = body.thread_id or new_request_id()
    user_id = get_current_user(request)
    # 按模块划分的 thread id：图共用一个 checkpointer；未划分命名空间
    # 的 id 会把它们的状态混在一起。user_id（M6）进 configurable：
    # 记忆工具经 RunnableConfig 注入读取（M6e）。
    full_thread_id = f"{module}:{user_thread_id}"
    config: RunnableConfig = {
        "recursion_limit": rt.settings.agent_recursion_limit,
        "configurable": {"thread_id": full_thread_id, "user_id": user_id},
    }
    messages_input: list[Any] = []
    # 记忆上下文注入（M6d）：画像/记忆块/摘要/相关记忆打包成注入块，
    # 置于全部消息之前（attachments 注入块在其后）。失败静默降级。
    memory = memory_or_none(rt)
    if memory is not None:
        context_block = await memory.compose_context(
            user_id=user_id,
            agent_id=module,
            thread_id=full_thread_id,
            query=body.message,
        )
        if context_block:
            messages_input.append(SystemMessage(content=context_block))
    # 附件（M4b）：解析引用的 file_id → 取记录 → 校验属主（S1：可用的
    # file_id 不能是别人的——IDOR 组合链的入口）→ 绑定线程 → 注入。
    if body.attachments:
        file_store = rt.file_store
        if file_store is None:
            raise HTTPException(status_code=503, detail="附件存储未启用（存储后端不可用）")
        attachment_infos = await file_store.get_many(body.attachments)
        missing = set(body.attachments) - {info.file_id for info in attachment_infos}
        # 附件属主校验（S1）：他人的 file_id 不能注入自己的对话（IDOR
        # 组合链的入口）。缺失与越权同文案同状态码——不给第三方探测
        # file_id 存在性的通道。
        unauthorized = {
            info.file_id for info in attachment_infos if info.user_id not in ("", user_id)
        }
        if missing or unauthorized:
            detail = "，".join(sorted(missing | unauthorized))
            raise HTTPException(
                status_code=400,
                detail=f"附件不存在或无权访问：{detail}；请重新上传",
            )
        await file_store.bind_thread(body.attachments, full_thread_id)
        blocks = []
        image_parts: list[UploadedFileInfo] = []
        for info in attachment_infos:
            if info.format in IMAGE_STORED_FORMATS:
                image_parts.append(info)
                continue
            if info.extracted_text.strip():
                pages = f"{info.pages} 页，" if info.pages else ""
                blocks.append(
                    f"[附件文件：{info.filename}（{info.format}，{pages}"
                    f"{info.text_len} 字符）]\n{info.extracted_text}"
                )
            else:
                note = info.warning or "无法提取文本"
                blocks.append(f"[附件文件：{info.filename}]（{note}）")
        if blocks:
            messages_input.append(
                SystemMessage(
                    content=f"{ATTACHMENT_CONTEXT_PREFIX}，供回答时参考：\n\n" + "\n\n".join(blocks)
                )
            )
        # 图片：多模态 content blocks（vision 模型直接看图）。
        human_content: Any = body.message
        if image_parts:
            content: list[dict[str, Any]] = [{"type": "text", "text": body.message}]
            for img in image_parts:
                encoded = base64.b64encode(img.content).decode("ascii")
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{img.format};base64,{encoded}"},
                    }
                )
            human_content = content
        attachments_meta = [info.meta() for info in attachment_infos]
        messages_input.append(
            HumanMessage(
                content=human_content,
                additional_kwargs={"attachments": attachments_meta},
            )
        )
    else:
        messages_input.append(HumanMessage(content=body.message))
    # 会话索引 upsert（S1）：属主/模块/标题（首条用户消息前 60 字符）。
    if rt.thread_index is not None:
        now = time.time()
        existing = await rt.thread_index.get_thread_index(full_thread_id)
        if existing is None:
            entry = ThreadIndex(
                thread_id=full_thread_id,
                user_id=user_id,
                module=module,
                title=body.message[:60],
                created_at=now,
                updated_at=now,
            )
        else:
            entry = replace(existing, updated_at=now)
        await rt.thread_index.upsert_thread_index(entry)
    # 记忆形成（M6c）：轮次正常完成后在后台跑抽取/整合/画像/摘要。
    # 需要形成管线可用（llm 已装配）且记忆系统启用。
    on_complete = _build_capture_hook(memory, module, user_thread_id, graph, config)
    return StreamingResponse(
        _event_stream(graph, messages_input, config, user_thread_id, on_complete),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@router.get("/{module}/threads")
async def list_threads(
    module: str, request: Request, user_id: str = Depends(get_current_user)
) -> dict[str, Any]:
    """列出**当前用户**在某模块命名空间下的线程（S1 作用域修复）。

    索引在位时直接查 ``thread_index``（属主过滤，无全表扫描）；缺席时
    降级为旧版 checkpointer 扫描（仅本地开发语义）。
    """
    rt = get_runtime(request)
    if not known_module(rt, module):
        raise HTTPException(status_code=404, detail=f"unknown module {module!r}")
    index = rt.thread_index
    if index is not None:
        entries = await index.list_thread_index(user_id, module=module, limit=_THREAD_LIST_LIMIT)
        return {
            "threads": [
                {
                    "thread_id": entry.thread_id.split(":", 1)[1]
                    if ":" in entry.thread_id
                    else entry.thread_id,
                    "module": entry.module or module,
                    "title": entry.title or entry.thread_id,
                    "updated_at": int(entry.updated_at * 1000),
                }
                for entry in entries
            ]
        }
    if rt.checkpointer is None:
        return {"threads": []}
    prefix = f"{module}:"
    threads: dict[str, dict[str, Any]] = {}
    scanned = 0
    total = 0
    async for tp in rt.checkpointer.alist(None):
        # 扫描预算只计目标模块的 checkpoint：否则繁忙模块的行会耗尽
        # 预算，让目标模块在大库里返回空列表。总行数上限兜底，避免
        # 无界遍历。
        total += 1
        if total > _THREAD_LIST_TOTAL_LIMIT:
            break
        full_id = tp.config["configurable"].get("thread_id", "")
        if not full_id.startswith(prefix):
            continue
        scanned += 1
        if scanned > _THREAD_LIST_SCAN_LIMIT:
            break
        short_id = full_id[len(prefix) :]
        if short_id in threads or not short_id:
            continue
        messages = tp.checkpoint.get("channel_values", {}).get("messages", [])
        title = next(
            (str(m.content)[:60] for m in messages if getattr(m, "type", "") == "human"),
            short_id,
        )
        try:
            updated_at = int(
                datetime.fromisoformat(str(tp.checkpoint.get("ts", ""))).timestamp() * 1000
            )
        except ValueError:
            updated_at = 0
        threads[short_id] = {
            "thread_id": short_id,
            "module": module,
            "title": title,
            "updated_at": updated_at,
        }
        if len(threads) >= _THREAD_LIST_LIMIT:
            break
    ordered = sorted(threads.values(), key=lambda t: t["updated_at"], reverse=True)
    return {"threads": ordered}


@router.get("/{module}/threads/{thread_id}")
async def get_thread_history(
    module: str, thread_id: str, request: Request, user_id: str = Depends(get_current_user)
) -> dict[str, Any]:
    """返回某会话已持久化的消息历史（供前端恢复会话显示）。"""
    rt = get_runtime(request)
    graph = _graph_or_404(rt, module)
    await _thread_owner_or_404(rt, module, thread_id, user_id)
    # thread id 按模块划分命名空间，与 invoke 端点保持一致。
    config: RunnableConfig = {"configurable": {"thread_id": f"{module}:{thread_id}"}}
    snapshot = await graph.aget_state(config)
    raw_messages = (snapshot.values or {}).get("messages", []) if snapshot else []
    messages = [
        serialized
        for serialized in (_serialize_message(m) for m in raw_messages)
        if serialized is not None
    ]
    return {"thread_id": thread_id, "module": module, "messages": messages}


@router.get("/{module}/threads/{thread_id}/tool-calls")
async def list_tool_calls(
    module: str,
    thread_id: str,
    request: Request,
    user_id: str = Depends(get_current_user),
    limit: int = 100,
) -> dict[str, Any]:
    """返回某会话的工具调用审计记录（按时间升序，最新在末尾）。"""
    rt = get_runtime(request)
    if not known_module(rt, module):
        raise HTTPException(status_code=404, detail=f"unknown module {module!r}")
    await _thread_owner_or_404(rt, module, thread_id, user_id)
    recorder = rt.tool_recorder
    if recorder is None:
        return {"tool_calls": []}
    records = await recorder.list_for_thread(f"{module}:{thread_id}", limit=max(1, min(limit, 500)))
    return {"tool_calls": records}


@router.get("/{module}/threads/{thread_id}/plan")
async def get_thread_plan(
    module: str, thread_id: str, request: Request, user_id: str = Depends(get_current_user)
) -> dict[str, Any]:
    """只读计划状态（M9）。

    plan 状态存在 planner 图的通道里，但线程命名空间属于请求模块
    （决策 5：invoke 的 thread 永远是 ``{module}:{thread_id}``）——
    读取必须经 **planner 图** 做 ``aget_state``（chat 图的通道表里没有
    tasks，用它读会丢字段）。线程从未跑过 plan 时返回空默认值
    （决策 6 的读取侧体现）。
    """
    rt = get_runtime(request)
    if not known_module(rt, module):
        raise HTTPException(status_code=404, detail=f"unknown module {module!r}")
    if "planner" not in rt.modules:
        raise HTTPException(status_code=503, detail="planner 模块未启用，无计划状态可读")
    await _thread_owner_or_404(rt, module, thread_id, user_id)
    planner_graph = rt.graph("planner")
    config: RunnableConfig = {"configurable": {"thread_id": f"{module}:{thread_id}"}}
    snapshot = await planner_graph.aget_state(config)
    values = (snapshot.values or {}) if snapshot else {}
    return {
        "thread_id": thread_id,
        "module": module,
        "tasks": values.get("tasks") or [],
        "cursor": values.get("cursor") or 0,
        "replans": values.get("replans") or 0,
    }


@router.delete("/{module}/threads/{thread_id}")
async def delete_thread(
    module: str, thread_id: str, request: Request, user_id: str = Depends(get_current_user)
) -> dict[str, Any]:
    """删除某模块命名空间下的一个已持久化线程（仅属主）。"""
    rt = get_runtime(request)
    if not known_module(rt, module):
        raise HTTPException(status_code=404, detail=f"unknown module {module!r}")
    await _thread_owner_or_404(rt, module, thread_id, user_id)
    if rt.checkpointer is None:
        return {"deleted": False, "reason": "checkpointer unconfigured"}
    # adelete_thread 接收原始 thread_id 字符串；与 invoke/get 一致地
    # 使用 module:thread_id 命名空间。
    full_thread_id = f"{module}:{thread_id}"
    await rt.checkpointer.adelete_thread(full_thread_id)
    if rt.file_store is not None:
        await rt.file_store.delete_for_thread(full_thread_id)
    if rt.thread_index is not None:
        # 索引行随线程消亡（store.delete_for_thread 也会清，这里显式
        # 覆盖 memory 未启用、purge_thread 不可用的部署形态）。
        await rt.thread_index.delete_thread_index(full_thread_id)
    memory = memory_or_none(rt)
    if memory is not None:
        # 记忆级联（M6）：清该线程的滚动摘要与知识库分块；跨会话
        # 记忆刻意保留（出处仍在 source_thread_id）。
        await memory.purge_thread(full_thread_id)
    return {"deleted": True}
