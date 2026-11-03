"""会话消息树：领域逻辑与执行编排（ADR-013）。

树（``message_nodes``）是会话的**唯一事实源**：任意节点可分岔（从此
分岔 / 编辑重发 / 重新生成都是"在指定节点下追加子节点"的特例），
LangGraph checkpointer 退化为"活跃路径的执行缓存"——每次 invoke 前把
目标路径物化进 state（``REMOVE_ALL_MESSAGES`` 全量替换；历史快照仍
不可变，符合"checkpointer 历史永不改写"哲学），turn 结束后按消息
``id`` 把新增消息（AI / tool）追加回树。

三层职责：

- **纯函数**（可单测）：路径/子树/孩子索引计算、消息 ↔ 节点互转、
  序列化、内容预览；
- **物化**（``materialize_branch``）：树路径 → checkpointer state，
  按 ``msg_id`` 逐条比对，已一致就跳过写（同分支续聊的快路径）；
- **回写**（``append_turn_tail``）：state 新增消息 → 树，按 ``msg_id``
  幂等（中断→审批恢复的重叠消息自动跳过），失败自愈——下轮物化会
  把 state 重写回树的事实。

历史遗留线程的迁移（``ensure_thread_tree``）：checkpointer 里的线性
历史一次性生成根链节点，幂等。
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent_base.core.context import content_text, is_injected_system
from agent_base.core.storage.models import MessageNode

logger = logging.getLogger(__name__)

# 树面板的节点内容预览长度（前端请求 with_content=False 时的摘要截断）。
PREVIEW_LIMIT = 80

# 树写锁（lock:tree:{thread_id}）：TTL 兜底持有者崩溃；等待超时给干净
# 的 503（"会话正忙"）而不是无限排队。
_TREE_LOCK_TTL_SECONDS = 60
_TREE_LOCK_TIMEOUT_SECONDS = 10

_ROLE_TO_FRONT = {"human": "human", "ai": "assistant", "tool": "tool"}


class TreeError(ValueError):
    """树操作的结构性错误（节点不存在 / 路径断裂 / 环）。"""


class TreeBusyError(RuntimeError):
    """会话树写锁等待超时（另一写方持锁）；调用方映射为 503。"""


_process_locks: dict[tuple[int, str], asyncio.Lock] = {}


def _process_lock_for(full_thread_id: str) -> asyncio.Lock:
    """进程内锁按事件循环隔离：asyncio.Lock 绑定创建它的循环，跨循环
    复用会抛 "bound to a different event loop"（同进程多循环的测试/
    嵌入场景会踩到）。生产 uvicorn 是单循环——键里带上循环 id 后行为
    不变；残留的旧循环锁对象随字典驻留（与原实现同量级，可接受）。"""
    loop_key = id(asyncio.get_running_loop())
    return _process_locks.setdefault((loop_key, full_thread_id), asyncio.Lock())


@asynccontextmanager
async def tree_write_lock(redis: Any, full_thread_id: str) -> AsyncIterator[None]:
    """会话树写临界区：Redis 在场用分布式锁（多 worker 串行），缺席退化
    进程内锁（本地开发 / 测试语义）。"""
    if redis is None:
        # 进程内兜底：每个 thread 一把锁，随进程生命周期驻留（Lock 对象
        # 极小，且仅在 REDIS_ENABLED=false 的本地开发/测试路径启用）——
        # 不能事后从 dict 摘除：并发等待者可能仍持有该对象引用。
        lock = _process_lock_for(full_thread_id)
        async with lock:
            yield
        return
    key = f"lock:tree:{full_thread_id}"
    token = await redis.acquire_lock(
        key, ttl_seconds=_TREE_LOCK_TTL_SECONDS, acquire_timeout_seconds=_TREE_LOCK_TIMEOUT_SECONDS
    )
    if token is None:
        raise TreeBusyError("会话正忙（另一写方持锁），请稍后再试")
    try:
        yield
    finally:
        try:
            await redis.release_lock(key, token)
        except Exception:
            logger.warning("tree: 会话锁释放失败（TTL 兜底过期）", exc_info=True)


def new_node_id() -> str:
    """节点 id（uuid4 hex32，与 memory_id 等同风格）。"""
    return uuid.uuid4().hex


# ─────────────────────────── 纯函数：树结构 ────────────────────────────


def children_index(nodes: list[MessageNode]) -> dict[str | None, list[MessageNode]]:
    """parent_id → 孩子列表（按 seq 升序）。"""
    index: dict[str | None, list[MessageNode]] = {}
    for node in nodes:
        index.setdefault(node.parent_id, []).append(node)
    for siblings in index.values():
        siblings.sort(key=lambda node: node.seq)
    return index


def build_path(nodes: list[MessageNode], leaf_id: str) -> list[MessageNode]:
    """leaf → root 上溯再反转成 root → leaf 的路径；断链/成环抛 TreeError。"""
    by_id = {node.node_id: node for node in nodes}
    if leaf_id not in by_id:
        raise TreeError(f"节点不存在：{leaf_id}")
    chain: list[MessageNode] = []
    seen: set[str] = set()
    cursor: str | None = leaf_id
    while cursor is not None:
        if cursor in seen:
            raise TreeError(f"检测到环：{cursor}")
        seen.add(cursor)
        node = by_id[cursor]
        chain.append(node)
        cursor = node.parent_id
    chain.reverse()
    return chain


def subtree_ids(
    nodes: list[MessageNode], root_id: str, *, include_deleted: bool = False
) -> set[str]:
    """以 root_id 为根的子树节点 id 集合（含根；默认只算存活节点）。"""
    children = children_index([node for node in nodes if include_deleted or not node.deleted])
    if root_id not in {node.node_id for node in nodes}:
        raise TreeError(f"节点不存在：{root_id}")
    ids: set[str] = set()
    stack = [root_id]
    while stack:
        current = stack.pop()
        ids.add(current)
        stack.extend(child.node_id for child in children.get(current, []))
    return ids


# ─────────────────────── 纯函数：消息 ↔ 节点互转 ───────────────────────


def message_to_node(
    message: Any, *, thread_id: str, parent_id: str | None, seq: int = 0
) -> MessageNode | None:
    """LangChain 消息 → 树节点；不入树的消息返回 None（system/空 AI）。

    注入型系统消息（记忆上下文/附件全文）在调用方过滤（``plain_view``）；
    其余 system 消息也不入树——树只承载真实对话轮。
    """
    mtype = getattr(message, "type", "")
    content = getattr(message, "content", "")
    if mtype == "system":
        return None
    if isinstance(content, str):
        stored, is_json = content, 0
    else:
        stored = json.dumps(content, ensure_ascii=False)
        is_json = 1
    msg_id = str(getattr(message, "id", "") or "")
    # 兜底：无 id 的消息（理论上 langgraph 入 state 时都会分配；仅迁移
    # 极旧数据时可能出现）生成稳定 id——下次物化会把该 id 写回 state，
    # 树与 checkpointer 的消息身份重新对齐（自愈）。
    if not msg_id:
        msg_id = new_node_id()
    if mtype == "human":
        attachments = (getattr(message, "additional_kwargs", {}) or {}).get("attachments")
        return MessageNode(
            node_id=new_node_id(),
            thread_id=thread_id,
            parent_id=parent_id,
            role="human",
            content=stored,
            content_json=is_json,
            attachments_json=(
                json.dumps(attachments, ensure_ascii=False)
                if isinstance(attachments, list) and attachments
                else "[]"
            ),
            msg_id=msg_id,
            seq=seq,
        )
    if mtype == "ai":
        tool_calls: list[dict[str, Any]] = [
            {
                "id": str(tc.get("id") or ""),
                "name": str(tc.get("name") or ""),
                "args": tc.get("args") if isinstance(tc.get("args"), dict) else {},
            }
            for tc in (getattr(message, "tool_calls", None) or [])
            if isinstance(tc, dict)
        ]
        if not stored and not tool_calls:
            return None  # 既无文本也无调用的空 AI 消息不落树
        return MessageNode(
            node_id=new_node_id(),
            thread_id=thread_id,
            parent_id=parent_id,
            role="ai",
            content=stored,
            content_json=is_json,
            tool_calls_json=json.dumps(tool_calls, ensure_ascii=False),
            msg_id=msg_id,
            seq=seq,
        )
    if mtype == "tool":
        return MessageNode(
            node_id=new_node_id(),
            thread_id=thread_id,
            parent_id=parent_id,
            role="tool",
            content=stored,
            content_json=is_json,
            tool_call_id=str(getattr(message, "tool_call_id", "") or ""),
            name=str(getattr(message, "name", "") or ""),
            status=str(getattr(message, "status", "") or "success"),
            msg_id=msg_id,
            seq=seq,
        )
    return None


def node_to_message(node: MessageNode) -> Any:
    """树节点 → LangChain 消息（物化用）。多段 content 与 tool_calls
    完整重建；``id`` 带回节点存的 ``msg_id``——checkpointer 里的消息身份
    跨物化稳定，状态比对与追加全部按 id 精确匹配。"""
    content = json.loads(node.content) if node.content_json else node.content
    if node.role == "human":
        message = HumanMessage(content=content, id=node.msg_id or None)
        try:
            attachments = json.loads(node.attachments_json or "[]")
        except ValueError:
            attachments = []
        if attachments:
            message.additional_kwargs["attachments"] = attachments
        return message
    if node.role == "ai":
        try:
            tool_calls = json.loads(node.tool_calls_json or "[]")
        except ValueError:
            tool_calls = []
        rebuilt = [
            {
                "name": tc.get("name", ""),
                "args": tc.get("args", {}),
                "id": tc.get("id", ""),
                "type": "tool_call",
            }
            for tc in tool_calls
            if isinstance(tc, dict) and tc.get("id")
        ]
        return AIMessage(content=content, tool_calls=rebuilt, id=node.msg_id or None)
    if node.role == "tool":
        return ToolMessage(
            content=content,
            tool_call_id=node.tool_call_id,
            name=node.name or None,
            status=node.status if node.status in ("success", "error") else "success",
            id=node.msg_id or None,
        )
    return None


def plain_view(messages: list[Any]) -> list[Any]:
    """过滤注入型系统消息（记忆上下文/附件全文）后的"真实对话"视图。"""
    return [message for message in messages if not is_injected_system(message)]


def node_text(node: MessageNode) -> str:
    """节点的展示文本（多段 content 提取文本段）。"""
    content = json.loads(node.content) if node.content_json else node.content
    return content if isinstance(content, str) else content_text(content)


def preview(text: str, limit: int = PREVIEW_LIMIT) -> str:
    """树面板的节点内容预览（单行截断）。"""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def serialize_node(node: MessageNode, *, with_content: bool = False) -> dict[str, Any]:
    """树 API 的节点序列化（预览 or 全文）。role 用前端命名
    （human/assistant/tool），与历史接口一致。"""
    out: dict[str, Any] = {
        "node_id": node.node_id,
        "parent_id": node.parent_id,
        "role": _ROLE_TO_FRONT.get(node.role, node.role),
        "seq": node.seq,
        "label": node.label,
        "starred": node.starred,
        "archived": node.archived,
        "created_at": int(node.created_at * 1000),
        "deleted": node.deleted,
    }
    if node.role == "tool":
        out["tool_call_id"] = node.tool_call_id
        out["status"] = "error" if node.status == "error" else "ok"
    if with_content:
        out["content"] = node_text(node)
        if node.role == "human":
            try:
                attachments = json.loads(node.attachments_json or "[]")
            except ValueError:
                attachments = []
            if attachments:
                out["attachments"] = attachments
        if node.role == "ai":
            try:
                tool_calls = json.loads(node.tool_calls_json or "[]")
            except ValueError:
                tool_calls = []
            if tool_calls:
                out["tool_calls"] = tool_calls
    else:
        out["preview"] = preview(node_text(node))
    return out


def serialize_history_node(node: MessageNode) -> dict[str, Any] | None:
    """历史接口的节点序列化：与 ``_serialize_message`` 的形状逐字段
    对齐（前端 replayToUiMessages 无感），外加 ``node_id`` 供树交互。"""
    text = node_text(node)
    out: dict[str, Any] = {"node_id": node.node_id}
    if node.label:
        out["label"] = node.label
    if node.starred:
        out["starred"] = True
    if node.role == "human":
        out["role"] = "human"
        out["content"] = text
        try:
            attachments = json.loads(node.attachments_json or "[]")
        except ValueError:
            attachments = []
        if attachments:
            out["attachments"] = attachments
        return out
    if node.role == "ai":
        out["role"] = "assistant"
        out["content"] = text
        try:
            tool_calls = json.loads(node.tool_calls_json or "[]")
        except ValueError:
            tool_calls = []
        clean = [
            {"id": tc.get("id", ""), "name": tc.get("name", ""), "args": tc.get("args", {})}
            for tc in tool_calls
            if isinstance(tc, dict) and tc.get("id")
        ]
        if clean:
            out["tool_calls"] = clean
        return out
    if node.role == "tool":
        out["role"] = "tool"
        out["name"] = node.name
        out["content"] = text
        out["tool_call_id"] = node.tool_call_id
        out["status"] = "error" if node.status == "error" else "ok"
        return out
    return None


# ─────────────────────────── 编排：迁移 / 物化 / 回写 ───────────────────────


async def ensure_thread_tree(
    store: Any, graph: Any, config: Any, full_thread_id: str
) -> list[MessageNode]:
    """线程的存活节点集（调用前保证树存在）：无节点的遗留线程从
    checkpointer 快照一次性迁移生成根链（幂等——有节点即直接返回）。

    迁移只做一次线性化：注入消息过滤、消息 id 原样保留，物化路径与
    历史行为完全一致。树不可用（store 缺席）时返回空列表，调用方走
    旧线性语义。
    """
    if store is None:
        return []
    nodes = await store.list_nodes(full_thread_id)
    if nodes:
        return list(nodes)
    snapshot = await graph.aget_state(config)
    raw = (snapshot.values or {}).get("messages", []) if snapshot else []
    plain = plain_view(list(raw))
    parent_id: str | None = None
    seq = await store.next_node_seq(full_thread_id)
    for message in plain:
        node = message_to_node(message, thread_id=full_thread_id, parent_id=parent_id, seq=seq)
        if node is None:
            continue
        await store.append_node(node)
        parent_id = node.node_id
        seq += 1
    if parent_id is not None:
        await store.set_current_node(full_thread_id, parent_id)
    return list(await store.list_nodes(full_thread_id))


async def materialize_branch(graph: Any, config: Any, path_nodes: list[MessageNode]) -> None:
    """把树路径物化进 checkpointer state（目标分支的执行缓存）。

    按 ``msg_id`` 逐条比对（过滤注入消息后的视图）；已一致就跳过——
    同分支续聊零额外写。不一致（切分支 / 首次物化）时用
    ``REMOVE_ALL_MESSAGES`` 全量替换：只产生新 checkpoint，历史快照
    仍然不可变。
    """
    snapshot = await graph.aget_state(config)
    raw = (snapshot.values or {}).get("messages", []) if snapshot else []
    state_ids = [str(getattr(m, "id", "") or "") for m in plain_view(list(raw))]
    path_ids = [node.msg_id for node in path_nodes]
    if state_ids == path_ids:
        return
    from langchain_core.messages import RemoveMessage
    from langgraph.graph import START
    from langgraph.graph.message import REMOVE_ALL_MESSAGES

    if not path_nodes:
        # 目标路径为空（挂载点是根 human：path[:-1] == []）——只需要清空
        # 上一分支的残留。清空写入**必须归因到 START**：默认归因会落到
        # 带条件边的节点上，LangGraph 在写入后立刻做条件路由，
        # tools_condition 对空 messages 直接抛
        # "No messages found in input state to tool_edge"（真实 500，E2E
        # 抓到）。START 无条件边，路由安全；空 state 由紧随的 astream
        # 输入（注入 + 重建的 human）补齐。
        await graph.aupdate_state(
            config, {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}, as_node=START
        )
        return
    rebuilt = [node_to_message(node) for node in path_nodes]
    rebuilt = [message for message in rebuilt if message is not None]
    await graph.aupdate_state(
        config, {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *rebuilt]}
    )


async def append_turn_tail(
    store: Any, full_thread_id: str, messages: list[Any], *, after_node_id: str | None
) -> str | None:
    """turn 结束后把 state 的新增消息回写进树（幂等；返回新叶子 id）。

    ``after_node_id`` 是本轮起点节点（invoke 的 human 节点）——回写只
    挂在它的子链上。按 ``msg_id`` 判重：中断时已落树的前缀（审批恢复
    轮）自动跳过；``after_node`` 缺席（树被并发删空）时放弃并交由
    下轮物化自愈。
    """
    plain = plain_view(list(messages))
    if store is None:
        return None
    # 会话已删除（索引行消亡）则不再回写：删除与会话树写锁之间没有
    # 全局串行——不设这道闸，钩子会在删除完成后把节点追加回已销毁的
    # 会话，留下孤儿行。
    index_entry = await store.get_thread_index(full_thread_id)
    if index_entry is None:
        logger.info("tree: 会话 %s 已删除，跳过尾部回写", full_thread_id)
        return None
    nodes = await store.list_nodes(full_thread_id)
    by_msg_id = {node.msg_id: node for node in nodes if node.msg_id}
    anchor = None
    if after_node_id is not None:
        anchor = next((node for node in nodes if node.node_id == after_node_id), None)
        if anchor is None:
            logger.warning("tree: 回写锚点丢失（after=%s），跳过尾部追加", after_node_id)
            return None
    # 找到第一条未入库消息：它的前驱节点就是挂载点。
    parent_id: str | None = anchor.node_id if anchor is not None else None
    pending: list[Any] = []
    matched_any = False
    for message in plain:
        msg_id = str(getattr(message, "id", "") or "")
        if msg_id and msg_id in by_msg_id:
            parent_id = by_msg_id[msg_id].node_id
            pending = []
            matched_any = True
            continue
        pending.append(message)
    if not pending:
        # 本轮没有新消息（重复回调 / 审批恢复的重叠已覆盖）。
        return parent_id
    if anchor is None and not matched_any and nodes:
        # 无锚点且快照与树完全对不上（树被并发改动/恢复后状态错位）：
        # 无法定位挂载点——全量追加会造出第二棵根链污染树。放弃本次回写，
        # 下轮物化把 state 重写回树的事实，自愈。
        logger.warning("tree: 回写无锚点且快照与树不匹配，跳过尾部追加（下轮物化自愈）")
        pointer: str | None = await store.get_current_node(full_thread_id)
        return pointer
    seq = await store.next_node_seq(full_thread_id)
    leaf: str | None = parent_id
    for message in pending:
        node = message_to_node(message, thread_id=full_thread_id, parent_id=parent_id, seq=seq)
        if node is None:
            continue
        await store.append_node(node)
        parent_id = node.node_id
        seq += 1
        leaf = node.node_id
    if leaf is not None:
        await store.set_current_node(full_thread_id, leaf)
    return leaf


async def tree_turn_context(store: Any, full_thread_id: str, leaf_node_id: str) -> dict[str, Any]:
    """捕获钩子的树上下文：叶子 + 祖先链（摘要先近后远的查找序）。"""
    nodes = await store.list_nodes(full_thread_id)
    try:
        path = build_path(nodes, leaf_node_id)
    except TreeError:
        return {"leaf": leaf_node_id, "ancestors": []}
    return {
        "leaf": leaf_node_id,
        "ancestors": [node.node_id for node in reversed(path[:-1])],
    }
