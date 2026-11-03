"""对话与会话线程端点（H1 拆分）：invoke（SSE）、线程列表/历史/删除、工具审计。

S1 线程作用域：每个线程在 ``thread_index`` 表里有属主记录（invoke 时
upsert）。列表端点只返回**当前用户**的线程；历史/删除/工具审计端点
校验属主（404 掩蔽，不泄露线程存在性）。索引存储缺席（后端不可用或
旧版本升级前）时降级为旧行为——全库扫描 + 无属主校验，仅限本地开发。
"""

from __future__ import annotations

import asyncio
import base64
import time
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Path, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from pydantic import BaseModel, Field

from agent_base.core.bootstrap import UnknownModuleError
from agent_base.core.context import ATTACHMENT_CONTEXT_PREFIX
from agent_base.core.storage import ThreadIndex
from agent_base.core.storage import tree as tree_mod
from agent_base.core.storage.tree import TreeBusyError, tree_write_lock
from agent_base.core.threads import ThreadId
from agent_base.entrypoints.server.deps import (
    get_current_user,
    get_runtime,
    known_module,
    memory_or_none,
)
from agent_base.entrypoints.server.serializers import IMAGE_STORED_FORMATS, _serialize_message
from agent_base.entrypoints.server.sse import (
    _SSE_HEADERS,
    SSE_QUEUE_MAXSIZE,
    _build_capture_hook,
    _event_stream,
    _heartbeat,
    sse_channel,
)
from agent_base.extensions.costmeter import CostBudgetExceeded
from agent_base.extensions.events import AgentEvent, PingEvent, decode_event, encode_sse
from agent_base.extensions.filestore import UploadedFileInfo
from agent_base.extensions.observability import new_request_id

router = APIRouter(prefix="/v1/agents", tags=["agents"])

# 线程列表端点的扫描/数量上限（旧版回退路径仍用）：alist 按 checkpoint
# 粒度迭代（一个线程有多轮 checkpoint），无上限会在大库上退化成全表遍历。
_THREAD_LIST_SCAN_LIMIT = 2000
_THREAD_LIST_TOTAL_LIMIT = 20_000
_THREAD_LIST_LIMIT = 50

# 路径参数形态的 thread_id：与 InvokeRequest.thread_id 同一约束
# （ThreadId 构造会拒绝含分隔符的 raw——非法形态在这里以 422 先行拒绝，
# 而不是落到 500）。
_ThreadPathParam = Annotated[str, Path(max_length=128, pattern=r"^[\w.-]+$")]


class InvokeRequest(BaseModel):
    """一次对话轮。"""

    # regenerate 模式（ADR-013）忽略本字段（重发既有 human 节点的原文），
    # 因此放宽到空串、由 handler 按模式显式校验——pydantic 的 min_length
    # 无法按模式豁免（前端重新生成发空串，真实 E2E 踩过 422）。
    message: str = Field(default="", max_length=100_000)
    # thread_id 是 checkpointer 的一部分键：限定字符集与长度，避免任意
    # 字符串直接落库/进日志。
    thread_id: str | None = Field(
        default=None, max_length=128, pattern=r"^[\w.-]+$"
    )  # None -> 创建一个新 thread
    # 消息树分岔（ADR-013）：新消息挂在哪个节点之下。缺省挂在当前活跃
    # 叶子后（正常续聊）；指定某个历史节点即"从此分岔"。
    parent_node_id: str | None = Field(default=None, max_length=32)
    # 重新生成（ADR-013）：不追加新的 human 节点，在 parent_node_id（一
    # 条既有 human 消息）之下重新跑一轮，产生新的 AI 兄弟分支。
    regenerate: bool = False
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


class ApprovalDecision(BaseModel):
    """对单个工具调用的审批决策（ADR-005）。"""

    action: Literal["approve", "reject", "modify"]
    # 仅 modify 使用：替换后的完整参数（工具自身 schema 校验兜底）。
    args: dict[str, Any] | None = None


class ResumeRequest(BaseModel):
    """审批恢复：对 approval_request 事件里每个 tool_call_id 给出决策。"""

    decisions: dict[str, ApprovalDecision] = Field(min_length=1)


@router.post("/{module}/threads/{thread_id}/resume")
async def resume_thread(
    module: str,
    thread_id: _ThreadPathParam,
    body: ResumeRequest,
    request: Request,
    user_id: str = Depends(get_current_user),
) -> StreamingResponse:
    """审批门恢复（ADR-005）：携带决策续走暂停的对话流（SSE）。

    线程必须处于待审批暂停态（图快照存在未决 interrupt），否则 409——
    避免把 resume 当成普通输入重放。恢复流与 invoke 流同契约；记忆形成
    钩子同样在正常完成后触发。
    """
    rt = get_runtime(request)
    if not known_module(rt, module):
        raise HTTPException(status_code=404, detail=f"unknown module {module!r}")
    await _thread_owner_or_404(rt, module, thread_id, user_id)
    # 审批恢复必须命中原请求模块的图（thread 命名空间属于请求模块——
    # 与 invoke 的决策 5 一致：mode=plan 的轮也挂在请求模块名下）。
    graph = _graph_or_404(rt, module)
    full_thread_id = str(ThreadId(module=module, raw=thread_id))
    config: RunnableConfig = {
        "recursion_limit": rt.settings.agent_recursion_limit,
        "configurable": {"thread_id": full_thread_id, "user_id": user_id},
    }
    snapshot = await graph.aget_state(config)
    pending = any(
        getattr(task, "interrupts", None) for task in (snapshot.tasks if snapshot else [])
    )
    if not pending:
        raise HTTPException(status_code=409, detail="该线程没有待决策的审批请求")
    resume_value = {
        call_id: {"action": d.action, "args": d.args} for call_id, d in body.decisions.items()
    }
    memory = memory_or_none(rt)
    # 树回写（ADR-013）：恢复轮的尾部挂在中断轮的 human 节点之下——
    # 回写按消息 id 判重，中断时已落树的前缀自动跳过；锚点缺失（指针
    # 漂移/树被并发改动）时回写自弃，下轮物化自愈。
    turn_node_id = await _interrupt_turn_anchor(rt, full_thread_id)
    on_complete = _build_capture_hook(
        memory,
        module,
        thread_id,
        graph,
        config,
        tree_store=rt.thread_index,
        full_thread_id=full_thread_id,
        turn_node_id=turn_node_id,
        redis=rt.redis,
    )
    return StreamingResponse(
        _event_stream(
            graph, Command(resume=resume_value), config, thread_id, on_complete, bus=rt.redis
        ),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


async def _interrupt_turn_anchor(rt: Any, full_thread_id: str) -> str | None:
    """审批恢复轮的树回写锚点：指针节点向上最近的 human 节点（中断轮
    的用户消息）。树不在位或路径断裂时返回 None（回写自弃）。"""
    store = rt.thread_index
    if store is None:
        return None
    nodes = await store.list_nodes(full_thread_id)
    pointer = await store.get_current_node(full_thread_id)
    if not pointer:
        return nodes[-1].node_id if nodes and getattr(nodes[-1], "role", "") == "human" else None
    try:
        path = tree_mod.build_path(nodes, pointer)
    except tree_mod.TreeError:
        return None
    for node in reversed(path):
        if node.role == "human":
            return node.node_id
    return None


def _graph_or_404(rt: Any, module: str) -> Any:
    """按模块取图；未知模块给干净的 404（UnknownModuleError 是 KeyError
    子类，str() 会带引号包装，取 args[0]）。"""
    try:
        return rt.graph(module)
    except UnknownModuleError as exc:
        detail = exc.args[0] if exc.args else "unknown module"
        raise HTTPException(status_code=404, detail=str(detail)) from exc


async def _resolve_tree_parent(
    store: Any,
    full_thread_id: str,
    nodes: list[Any],
    parent_node_id: str | None,
    *,
    require_role: str | None = None,
) -> Any:
    """解析新消息的挂载父节点：显式指定即"从此分岔"（校验存在性与角
    色），缺省为当前活跃叶子（指针优先，退化为 seq 最大的存活节点——
    树按 seq 单调追加，最大 seq 即最新消息）。空树返回 None（新根）。"""
    if parent_node_id is None:
        pointer = await store.get_current_node(full_thread_id)
        if pointer:
            node = next((n for n in nodes if n.node_id == pointer and not n.deleted), None)
            if node is not None:
                return node
        return nodes[-1] if nodes else None
    node = next((n for n in nodes if n.node_id == parent_node_id), None)
    if node is None or node.deleted:
        raise HTTPException(status_code=404, detail=f"分岔节点不存在：{parent_node_id}")
    if require_role is not None and node.role != require_role:
        raise HTTPException(
            status_code=400,
            detail=f"重新生成的挂载点必须是用户消息（节点 {parent_node_id} 不是）",
        )
    return node


async def _thread_owner_or_404(rt: Any, module: str, thread_id: str, user_id: str) -> None:
    """线程属主校验（S1）：索引在位时，非属主一律 404 掩蔽。

    历史线程（索引行不存在——升级前的旧数据）同样 404，与审查报告
    S1 的迁移策略一致："旧线程不可访问"。
    """
    index = rt.thread_index
    if index is None:
        return
    entry = await index.get_thread_index(str(ThreadId(module=module, raw=thread_id)))
    if entry is None or entry.user_id != user_id:
        raise HTTPException(status_code=404, detail=f"线程不存在：{thread_id}")


@router.post("/{module}/invoke")
async def invoke(module: str, body: InvokeRequest, request: Request) -> StreamingResponse:
    """一次对话轮：SSE 流输出契约事件（ping/step/delta/sources/done/error）。"""
    rt = get_runtime(request)
    # 预算熔断（T4.2）：在建流之前检查——429 优先于半截 SSE。只挡新的
    # LLM 消耗；非 LLM 端点不经过这里，不受熔断影响。acheck（ADR-012）：
    # 共享计数器在场时读跨 worker 累计。
    governor = getattr(request.app.state, "cost_governor", None)
    if governor is not None:
        try:
            await governor.acheck()
        except CostBudgetExceeded as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
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
    if not body.message.strip() and not body.regenerate:
        raise HTTPException(status_code=400, detail="消息不能为空")
    user_thread_id = body.thread_id or new_request_id()
    user_id = await get_current_user(request)
    # 按模块划分的 thread id：图共用一个 checkpointer；未划分命名空间
    # 的 id 会把它们的状态混在一起。user_id（M6）进 configurable：
    # 记忆工具经 RunnableConfig 注入读取（M6e）。
    full_thread_id = str(ThreadId(module=module, raw=user_thread_id))
    config: RunnableConfig = {
        "recursion_limit": rt.settings.agent_recursion_limit,
        "configurable": {"thread_id": full_thread_id, "user_id": user_id},
    }
    messages_input: list[Any] = []
    # 记忆上下文注入（M6d + T2.1）：冻结段（画像/记忆块/摘要，线程内
    # 快照）在前，召回段（查询相关，逐轮变化）紧随其后、贴着附件与
    # 本轮 human——头部稳定保前缀缓存，尾部变化不伤历史缓存。
    # 任何一段失败都静默降级。
    memory = memory_or_none(rt)
    # ── 消息树（ADR-013）：树模式的分岔语义与执行物化 ──────────────────
    # 树在位（thread_index 即基座存储）时：新 human 消息先落树（客户端
    # 断开也不丢消息）、指针推进；物化把父链写进 checkpointer（切分支
    # 后的 state 重写）；turn 结束后的回写由捕获钩子完成。
    store = rt.thread_index
    turn_node_id: str | None = None
    summary_nodes: list[str] = []
    human_message: HumanMessage | None = None
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
        human_message = HumanMessage(
            content=human_content,
            additional_kwargs={"attachments": attachments_meta},
        )
    else:
        human_message = HumanMessage(content=body.message)
    # 消息 id 必须在落树前就位（ADR-013）：树的 msg_id 判重、物化重建、
    # checkpointer 消息身份全靠它——id 缺席时 langgraph 入 state 才另发
    # 新 id，树与 state 就对不上了（human 会被重复落树）。
    if not human_message.id:
        human_message.id = tree_mod.new_node_id()
    if store is not None:
        # 越权防护（S1 对齐）：invoke 是唯一带写语义的会话端点——不校验
        # 属主的话，知道他人 thread_id 即可向其消息树落节点、推进活跃
        # 指针、物化覆盖执行缓存。策略是"存在即校验"：索引无行 = 新线程
        # （放行，由下方 upsert 建立属主记录），有行则属主必须匹配（404
        # 掩蔽，与其余线程端点同一语义）。
        existing_entry = await store.get_thread_index(full_thread_id)
        if existing_entry is not None and existing_entry.user_id != user_id:
            raise HTTPException(status_code=404, detail=f"线程不存在：{user_thread_id}")
        # 树写临界区（ADR-013）：落 human 节点 + 推指针。Redis 在场用
        # 分布式锁，缺席退化进程内锁（本地开发 / 测试语义）。
        try:
            async with tree_write_lock(rt.redis, full_thread_id):
                nodes = await tree_mod.ensure_thread_tree(store, graph, config, full_thread_id)
                if body.regenerate:
                    # 重新生成：不追加 human，在既有 human 节点下重跑——
                    # 输入消息从节点重建（id 与节点 msg_id 一致，checkpointer
                    # 里的消息身份稳定）。
                    parent_node = await _resolve_tree_parent(
                        store, full_thread_id, nodes, body.parent_node_id, require_role="human"
                    )
                    if parent_node is None:
                        raise HTTPException(
                            status_code=400, detail="会话还没有消息，无法重新生成"
                        )
                    human_message = tree_mod.node_to_message(parent_node)
                else:
                    parent_node = await _resolve_tree_parent(
                        store, full_thread_id, nodes, body.parent_node_id
                    )
                    seq = await store.next_node_seq(full_thread_id)
                    human_node = tree_mod.message_to_node(
                        human_message, thread_id=full_thread_id, parent_id=parent_node.node_id
                        if parent_node is not None else None,
                        seq=seq,
                    )
                    assert human_node is not None  # human 消息恒可入树
                    if human_message.id:
                        human_node = replace(human_node, msg_id=str(human_message.id))
                    await store.append_node(human_node)
                    nodes.append(human_node)  # 锁内的路径构建需要它（避免重读）
                    turn_node_id = human_node.node_id
                    parent_node = human_node
                await store.set_current_node(
                    full_thread_id,
                    turn_node_id
                    or (parent_node.node_id if parent_node is not None else None),
                )
        except TreeBusyError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        # 摘要候选（先近后远）：冻结段的会话摘要按"当前分支最近的祖先
        # 摘要"取——分支各自的摘要独立演进。
        if parent_node is not None:
            summary_nodes = list(
                reversed(
                    [
                        node.node_id
                        for node in tree_mod.build_path(nodes, parent_node.node_id)[:-1]
                    ]
                )
            )
            # 物化父链（root→parent）：state 与目标分支不一致时全量替换。
            await tree_mod.materialize_branch(
                graph, config, tree_mod.build_path(nodes, parent_node.node_id)[:-1]
            )
    messages_input.append(human_message)
    if memory is not None:
        context_block = await memory.compose_context(
            user_id=user_id,
            agent_id=module,
            thread_id=full_thread_id,
            query=body.message,
            summary_nodes=summary_nodes,
        )
        recall_block = await memory.recall_context(user_id=user_id, query=body.message)
        for block in (context_block, recall_block):
            if block:
                messages_input.insert(-1, SystemMessage(content=block))
    # 会话索引 upsert（S1）：属主/模块/标题（首条用户消息前 60 字符）。
    # existing 已在树块的属主校验处查过（两者之间没有其他写入方），复用。
    if rt.thread_index is not None:
        now = time.time()
        if store is not None:
            existing = existing_entry
        else:
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
    # 记忆形成（M6c + ADR-013）：轮次正常完成后在后台跑树回写与
    # 抽取/整合/画像/摘要。需要形成管线可用（llm 已装配）或树在位。
    on_complete = _build_capture_hook(
        memory,
        module,
        user_thread_id,
        graph,
        config,
        tree_store=rt.thread_index,
        full_thread_id=full_thread_id,
        turn_node_id=turn_node_id,
        redis=rt.redis,
    )
    return StreamingResponse(
        _event_stream(
            graph,
            {"messages": messages_input},
            config,
            user_thread_id,
            on_complete,
            bus=rt.redis,
        ),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@router.get("/{module}/threads/{thread_id}/events")
async def observe_thread_events(
    module: str,
    thread_id: _ThreadPathParam,
    request: Request,
    user_id: str = Depends(get_current_user),
) -> StreamingResponse:
    """线程事件的只读 SSE 镜像（ADR-012；REDIS_ENABLED=true 时可用）。

    订阅 Redis 频道 ``sse:{thread_id}`` 并转发镜像事件：多实例部署下可
    从任一实例观察正在发生（或后续发生）的对话轮——前端在连接漂移后
    重挂、运维旁路观察都走这里。Pub/Sub 是 at-most-once：订阅建立前与
    Redis 抖动期间的事件不补发；主对话流仍是 invoke/resume 请求内闭环，
    镜像缺席不影响正确性。未启用 Redis 时 503（能力未开启，不是故障）。
    """
    rt = get_runtime(request)
    if not known_module(rt, module):
        raise HTTPException(status_code=404, detail=f"unknown module {module!r}")
    await _thread_owner_or_404(rt, module, thread_id, user_id)
    redis = rt.redis
    if redis is None:
        raise HTTPException(status_code=503, detail="SSE 事件镜像未启用（REDIS_ENABLED=false）")

    async def _mirror() -> AsyncIterator[str]:
        queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue(maxsize=SSE_QUEUE_MAXSIZE)

        async def _relay() -> None:
            # 订阅→解码→入队；连接断开（Redis 抖动/关停）时送哨兵结束流
            # ——客户端按 SSE 惯例重连，期间事件不补发（at-most-once）。
            try:
                async for payload in redis.pubsub_messages(sse_channel(thread_id)):
                    event = decode_event(payload)
                    if event is not None:
                        await queue.put(event)
            finally:
                while True:
                    try:
                        queue.put_nowait(None)
                        break
                    except asyncio.QueueFull:
                        await asyncio.sleep(0.05)

        relay = asyncio.create_task(_relay())
        heartbeat = asyncio.create_task(_heartbeat(queue))
        try:
            # 立即发出存活信号：订阅在 Redis 侧生效前客户端就知道流已打开。
            yield encode_sse(PingEvent())
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield encode_sse(event)
        finally:
            for task in (relay, heartbeat):
                task.cancel()
            await asyncio.gather(relay, heartbeat, return_exceptions=True)

    return StreamingResponse(_mirror(), media_type="text/event-stream", headers=_SSE_HEADERS)


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

        def _user_part(thread_id: str) -> str:
            # 索引行总是命名空间形态；异常行（历史数据）整串回退。
            parsed = ThreadId.try_parse(thread_id)
            return parsed.raw if parsed is not None else thread_id

        return {
            "threads": [
                {
                    "thread_id": _user_part(entry.thread_id),
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
    module: str,
    thread_id: _ThreadPathParam,
    request: Request,
    user_id: str = Depends(get_current_user),
) -> dict[str, Any]:
    """返回某会话沿当前活跃分支的消息历史（供前端恢复会话显示）。

    树在位（ADR-013）时从 ``message_nodes`` 沿活跃指针读——树是会话的
    唯一事实源，每条消息带 ``node_id``（前端树交互的锚点）；遗留线程
    首次访问时从 checkpointer 幂等迁移。树缺席（存储后端不可用）时
    降级为旧版 checkpointer 快照读取（本地开发语义）。
    """
    rt = get_runtime(request)
    graph = _graph_or_404(rt, module)
    await _thread_owner_or_404(rt, module, thread_id, user_id)
    # thread id 按模块划分命名空间，与 invoke 端点保持一致。
    full_thread_id = str(ThreadId(module=module, raw=thread_id))
    config: RunnableConfig = {"configurable": {"thread_id": full_thread_id}}
    store = rt.thread_index
    if store is not None:
        nodes = await tree_mod.ensure_thread_tree(store, graph, config, full_thread_id)
        pointer = await store.get_current_node(full_thread_id)
        path: list[Any] = []
        if pointer:
            try:
                path = tree_mod.build_path(nodes, pointer)
            except tree_mod.TreeError:
                path = []
        messages = [
            serialized
            for serialized in (tree_mod.serialize_history_node(node) for node in path)
            if serialized is not None
        ]
        return {"thread_id": thread_id, "module": module, "messages": messages}
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
    thread_id: _ThreadPathParam,
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
    records = await recorder.list_for_thread(
        str(ThreadId(module=module, raw=thread_id)), limit=max(1, min(limit, 500))
    )
    return {"tool_calls": records}


@router.get("/{module}/threads/{thread_id}/plan")
async def get_thread_plan(
    module: str,
    thread_id: _ThreadPathParam,
    request: Request,
    user_id: str = Depends(get_current_user),
) -> dict[str, Any]:
    """只读计划状态（M9；M10 起读快照表）。

    plan 快照存在 ``plan_snapshots`` 表（planner 图每个状态变更点
    upsert）——**chat 轮穿插不再重置计划**（chat 图的 checkpoint 只含
    messages 通道，但表独立于图状态存活）。线程命名空间属于请求模块
    （决策 5：invoke 的 thread 永远是 ``{module}:{thread_id}``）。表缺席
    （存储后端不可用）时回退读 planner 图状态；线程从未跑过 plan 时
    返回空默认值。
    """
    rt = get_runtime(request)
    if not known_module(rt, module):
        raise HTTPException(status_code=404, detail=f"unknown module {module!r}")
    if "planner" not in rt.modules:
        raise HTTPException(status_code=503, detail="planner 模块未启用，无计划状态可读")
    await _thread_owner_or_404(rt, module, thread_id, user_id)
    full_thread_id = str(ThreadId(module=module, raw=thread_id))
    if rt.plan_store is not None:
        stored = await rt.plan_store.get_plan(full_thread_id)
        if stored is not None:
            return {
                "thread_id": thread_id,
                "module": stored.get("module") or module,
                "tasks": stored.get("tasks") or [],
                "cursor": int(stored.get("cursor") or 0),
                "replans": int(stored.get("replans") or 0),
            }
    planner_graph = rt.graph("planner")
    config: RunnableConfig = {"configurable": {"thread_id": full_thread_id}}
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
    module: str,
    thread_id: _ThreadPathParam,
    request: Request,
    user_id: str = Depends(get_current_user),
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
    full_thread_id = str(ThreadId(module=module, raw=thread_id))
    await rt.checkpointer.adelete_thread(full_thread_id)
    if rt.file_store is not None:
        await rt.file_store.delete_for_thread(full_thread_id)
    # 消息树的节点 id 先取后删：记忆侧的节点摘要级联需要它（ADR-013）。
    tree_node_ids: list[str] = []
    if rt.thread_index is not None:
        tree_node_ids = [
            node.node_id for node in await rt.thread_index.list_nodes(full_thread_id)
        ]
        await rt.thread_index.delete_nodes_for_thread(full_thread_id)
        # 索引行随线程消亡（store.delete_for_thread 也会清，这里显式
        # 覆盖 memory 未启用、purge_thread 不可用的部署形态）；计划
        # 快照同属线程作用域数据面（M10）。
        await rt.thread_index.delete_thread_index(full_thread_id)
        await rt.thread_index.delete_plan(full_thread_id)
    memory = memory_or_none(rt)
    if memory is not None:
        # 记忆级联（M6）：清该线程的滚动摘要与知识库分块；跨会话
        # 记忆刻意保留（出处仍在 source_thread_id）。树模式下摘要
        # 键控到节点，按节点 id 级联（ADR-013）。
        await memory.purge_thread(full_thread_id)
        purge_nodes = getattr(memory, "purge_nodes", None)
        if tree_node_ids and callable(purge_nodes):
            await purge_nodes(tree_node_ids)
    return {"deleted": True}
