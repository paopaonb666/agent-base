"""会话消息树的维护端点（ADR-013）：树查询 / 节点维护 / 子树删除与恢复 /
分支跳转 / 导出。

树是会话的**唯一事实源**（``core/storage/tree.py``）；本路由只做参数
校验、属主校验（404 掩蔽，复用 agents 路由的语义）与树操作。写操作
全部在会话树写锁内执行——与 invoke 的落节点共用同一把锁。
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request
from pydantic import BaseModel, Field

from agent_base.core.storage import tree as tree_mod
from agent_base.core.storage.tree import TreeBusyError, TreeError, tree_write_lock
from agent_base.core.threads import ThreadId
from agent_base.entrypoints.server.deps import get_current_user, get_runtime, known_module
from agent_base.entrypoints.server.routes.agents import _graph_or_404, _thread_owner_or_404

router = APIRouter(prefix="/v1/agents", tags=["tree"])

_ThreadPathParam = Annotated[str, Path(max_length=128, pattern=r"^[\w.-]+$")]
_NodePathParam = Annotated[str, Path(max_length=32, pattern=r"^[0-9a-f]{32}$")]


class NodePatchRequest(BaseModel):
    """节点维护元数据的按需更新；None 的字段不动。"""

    label: str | None = Field(default=None, max_length=120)
    starred: bool | None = None
    archived: bool | None = None


class JumpRequest(BaseModel):
    """切换活跃分支：目标节点成为新的活跃叶子（懒物化——下次 invoke
    前才把该路径写进 checkpointer）。"""

    node_id: str = Field(max_length=32)


def _store_or_503(rt: Any) -> Any:
    """树存储缺席（存储后端不可用）时 503——能力未启用，不是故障。"""
    store = rt.thread_index
    if store is None:
        raise HTTPException(status_code=503, detail="消息树未启用（存储后端不可用）")
    return store


def _check_module(rt: Any, module: str) -> None:
    if not known_module(rt, module):
        raise HTTPException(status_code=404, detail=f"unknown module {module!r}")


async def _load_tree(store: Any, graph: Any, full_thread_id: str, config: dict[str, Any]) -> None:
    """首次访问的遗留线程迁移（幂等；无节点时从 checkpointer 生成根链）。

    迁移会写节点表——调用方必须已持有会话树写锁，否则两个并发首访会
    各迁移一条根链（树出现双根）。读端点（tree/export）同样要走锁。
    """
    await tree_mod.ensure_thread_tree(store, graph, config, full_thread_id)


@router.get("/{module}/threads/{thread_id}/tree")
async def get_thread_tree(
    module: str,
    thread_id: _ThreadPathParam,
    request: Request,
    with_content: bool = False,
    user_id: str = Depends(get_current_user),
) -> dict[str, Any]:
    """整棵消息树（前端侧边树面板的数据源）。

    默认只带 80 字内容预览（长会话的树也保持轻量）；``with_content=true``
    携带全文与附件/工具调用明细。软删节点随树返回（前端的"已删除"
    分组与恢复入口从这里渲染）；兄弟分支与父子结构由前端本地计算。
    """
    rt = get_runtime(request)
    _check_module(rt, module)
    store = _store_or_503(rt)
    graph = _graph_or_404(rt, module)
    await _thread_owner_or_404(rt, module, thread_id, user_id)
    full_thread_id = str(ThreadId(module=module, raw=thread_id))
    config: dict[str, Any] = {"configurable": {"thread_id": full_thread_id}}
    try:
        # 迁移是写操作：与 invoke/维护端点共用同一把树锁，防止并发首访
        # 双重迁移（锁内无节点写入时开销只有一次锁获取）。
        async with tree_write_lock(rt.redis, full_thread_id):
            await _load_tree(store, graph, full_thread_id, config)
    except TreeBusyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    nodes = await store.list_nodes(full_thread_id, include_deleted=True)
    pointer = await store.get_current_node(full_thread_id)
    return {
        "thread_id": thread_id,
        "module": module,
        "current_node_id": pointer,
        "nodes": [tree_mod.serialize_node(node, with_content=with_content) for node in nodes],
    }


@router.patch("/{module}/threads/{thread_id}/nodes/{node_id}")
async def patch_node(
    module: str,
    thread_id: _ThreadPathParam,
    node_id: _NodePathParam,
    body: NodePatchRequest,
    request: Request,
    user_id: str = Depends(get_current_user),
) -> dict[str, Any]:
    """维护节点元数据：重命名标签 / 星标 / 归档（不影响消息内容与树形）。"""
    rt = get_runtime(request)
    _check_module(rt, module)
    store = _store_or_503(rt)
    await _thread_owner_or_404(rt, module, thread_id, user_id)
    full_thread_id = str(ThreadId(module=module, raw=thread_id))
    if body.label == "":
        raise HTTPException(status_code=400, detail="标签不能为空串（不修改请省略该字段）")
    try:
        async with tree_write_lock(rt.redis, full_thread_id):
            updated = await store.update_node_meta(
                full_thread_id,
                node_id,
                label=body.label,
                starred=body.starred,
                archived=body.archived,
            )
    except TreeBusyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if not updated:
        raise HTTPException(status_code=404, detail=f"节点不存在：{node_id}")
    node = await store.get_node(full_thread_id, node_id)
    if node is None:  # pragma: no cover - update 与 get 同库，防御式
        raise HTTPException(status_code=404, detail=f"节点不存在：{node_id}")
    return {"node": tree_mod.serialize_node(node, with_content=True)}


@router.delete("/{module}/threads/{thread_id}/nodes/{node_id}")
async def delete_subtree(
    module: str,
    thread_id: _ThreadPathParam,
    node_id: _NodePathParam,
    request: Request,
    cascade: bool = Query(default=False, description="有子节点时强制整棵子树软删"),
    user_id: str = Depends(get_current_user),
) -> dict[str, Any]:
    """软删一棵子树（可恢复）。

    默认仅允许删除叶子（无存活后代）；``cascade=true`` 连同整棵子树打
    删除标记。活跃路径被删时指针回退到最近的存活祖先；根节点不可删
    （删根等价于删会话——走 DELETE /threads/{thread_id}）。
    """
    rt = get_runtime(request)
    _check_module(rt, module)
    store = _store_or_503(rt)
    graph = _graph_or_404(rt, module)
    await _thread_owner_or_404(rt, module, thread_id, user_id)
    full_thread_id = str(ThreadId(module=module, raw=thread_id))
    config: dict[str, Any] = {"configurable": {"thread_id": full_thread_id}}
    try:
        async with tree_write_lock(rt.redis, full_thread_id):
            await _load_tree(store, graph, full_thread_id, config)
            nodes = await store.list_nodes(full_thread_id, include_deleted=False)
            target = next((n for n in nodes if n.node_id == node_id), None)
            if target is None:
                raise HTTPException(status_code=404, detail=f"节点不存在：{node_id}")
            if target.parent_id is None:
                raise HTTPException(
                    status_code=400,
                    detail="根节点不可删除（删除整个会话请用 DELETE /threads/{thread_id}）",
                )
            subtree = tree_mod.subtree_ids(nodes, node_id)
            if len(subtree) > 1 and not cascade:
                raise HTTPException(
                    status_code=409,
                    detail=f"该节点下有 {len(subtree) - 1} 个后代；确认删除请带 ?cascade=true",
                )
            changed = await store.set_nodes_deleted(full_thread_id, sorted(subtree), deleted=True)
            # 活跃路径被删 → 指针回退到最近的存活祖先。
            pointer = await store.get_current_node(full_thread_id)
            rolled_back = False
            if pointer is not None and pointer in subtree:
                try:
                    path = tree_mod.build_path(nodes, pointer)
                except TreeError:
                    path = []
                fallback = next(
                    (n.node_id for n in reversed(path[:-1]) if n.node_id not in subtree),
                    None,
                )
                await store.set_current_node(full_thread_id, fallback)
                rolled_back = True
    except TreeBusyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"deleted": changed, "pointer_rolled_back": rolled_back}


@router.post("/{module}/threads/{thread_id}/nodes/{node_id}/restore")
async def restore_subtree(
    module: str,
    thread_id: _ThreadPathParam,
    node_id: _NodePathParam,
    request: Request,
    user_id: str = Depends(get_current_user),
) -> dict[str, Any]:
    """恢复一棵软删的子树（连同其全部软删后代清掉删除标记）。"""
    rt = get_runtime(request)
    _check_module(rt, module)
    store = _store_or_503(rt)
    await _thread_owner_or_404(rt, module, thread_id, user_id)
    full_thread_id = str(ThreadId(module=module, raw=thread_id))
    try:
        async with tree_write_lock(rt.redis, full_thread_id):
            every = await store.list_nodes(full_thread_id, include_deleted=True)
            target = next((n for n in every if n.node_id == node_id), None)
            if target is None:
                raise HTTPException(status_code=404, detail=f"节点不存在：{node_id}")
            subtree = tree_mod.subtree_ids(every, node_id, include_deleted=True)
            changed = await store.set_nodes_deleted(full_thread_id, sorted(subtree), deleted=False)
    except TreeBusyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"restored": changed}


@router.post("/{module}/threads/{thread_id}/jump")
async def jump_branch(
    module: str,
    thread_id: _ThreadPathParam,
    body: JumpRequest,
    request: Request,
    user_id: str = Depends(get_current_user),
) -> dict[str, Any]:
    """切换活跃分支（懒物化）：只更新指针，checkpointer 的 state 在下次
    invoke 前由 ``materialize_branch`` 重写到目标路径。"""
    rt = get_runtime(request)
    _check_module(rt, module)
    store = _store_or_503(rt)
    graph = _graph_or_404(rt, module)
    await _thread_owner_or_404(rt, module, thread_id, user_id)
    full_thread_id = str(ThreadId(module=module, raw=thread_id))
    config: dict[str, Any] = {"configurable": {"thread_id": full_thread_id}}
    try:
        async with tree_write_lock(rt.redis, full_thread_id):
            await _load_tree(store, graph, full_thread_id, config)
            nodes = await store.list_nodes(full_thread_id, include_deleted=False)
            try:
                tree_mod.build_path(nodes, body.node_id)
            except TreeError as exc:
                raise HTTPException(
                    status_code=404, detail=f"跳转节点不存在：{body.node_id}"
                ) from exc
            await store.set_current_node(full_thread_id, body.node_id)
    except TreeBusyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"thread_id": thread_id, "current_node_id": body.node_id}


@router.get("/{module}/threads/{thread_id}/export")
async def export_tree(
    module: str,
    thread_id: _ThreadPathParam,
    request: Request,
    format: str = Query(default="md", pattern="^(md|json)$"),
    user_id: str = Depends(get_current_user),
) -> dict[str, Any]:
    """导出整棵消息树：Markdown（主线 + 各分支章节）或 JSON（完整结构）。"""
    rt = get_runtime(request)
    _check_module(rt, module)
    store = _store_or_503(rt)
    graph = _graph_or_404(rt, module)
    await _thread_owner_or_404(rt, module, thread_id, user_id)
    full_thread_id = str(ThreadId(module=module, raw=thread_id))
    config: dict[str, Any] = {"configurable": {"thread_id": full_thread_id}}
    try:
        async with tree_write_lock(rt.redis, full_thread_id):
            await _load_tree(store, graph, full_thread_id, config)
    except TreeBusyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    nodes = await store.list_nodes(full_thread_id, include_deleted=True)
    pointer = await store.get_current_node(full_thread_id)
    index = rt.thread_index
    title = ""
    if index is not None:
        entry = await index.get_thread_index(full_thread_id)
        title = entry.title if entry is not None else ""
    if format == "json":
        return {
            "thread_id": thread_id,
            "module": module,
            "title": title,
            "current_node_id": pointer,
            "nodes": [tree_mod.serialize_node(node, with_content=True) for node in nodes],
        }
    return {
        "thread_id": thread_id,
        "module": module,
        "format": "md",
        "content": render_tree_markdown(title or thread_id, nodes, pointer),
    }


# ───────────────────────────── Markdown 渲染 ─────────────────────────────


def _md_line(node: Any) -> str:
    """一个节点渲染为一行：角色 + 星标 + 标签 + 截断的内容。"""
    role = tree_mod._ROLE_TO_FRONT.get(node.role, node.role)
    star = "★ " if node.starred else ""
    label = f"「{node.label}」" if node.label else ""
    text = " ".join(tree_mod.node_text(node).split())
    if len(text) > 160:
        text = text[:159] + "…"
    return f"- `{role}` {star}{label}{text}"


def render_tree_markdown(title: str, nodes: list[Any], pointer: str | None) -> str:
    """整棵树渲染为 Markdown：主线（当前路径）+ 每条旁支一个章节。

    旁支章节在"分岔点在主线 / 已渲染旁支上"处切入，子树深度优先展开；
    软删节点不进导出（恢复入口在树面板）。
    """
    children = tree_mod.children_index(nodes)
    by_id = {node.node_id: node for node in nodes}
    main_path: list[Any] = []
    if pointer and pointer in by_id:
        try:
            main_path = tree_mod.build_path(nodes, pointer)
        except TreeError:
            main_path = []
    on_main = {node.node_id for node in main_path}
    rendered: set[str] = set(on_main)

    lines: list[str] = [f"# {title}", "", "## 主线", ""]
    for node in main_path:
        lines.append(_md_line(node))

    def _render_subtree(node_id: str, lines: list[str]) -> None:
        for child in children.get(node_id, []):
            if child.deleted or child.node_id in rendered:
                continue
            rendered.add(child.node_id)
            lines.append(_md_line(child))
            _render_subtree(child.node_id, lines)

    # 旁支 = 存活节点里不在主线上的，按父节点（主线/已渲染旁支）分组
    # 切章节；根级旁支（理论不存在，防御）也覆盖。
    parents: list[str | None] = [node.node_id for node in main_path] + [None]
    for parent_id in parents:
        branches = [
            child
            for child in children.get(parent_id, [])
            if not child.deleted and child.node_id not in rendered
        ]
        if not branches:
            continue
        if parent_id is None:
            parent_label = "会话开头"
        elif parent_id in on_main:
            parent_label = "主线"
        else:
            parent_label = "上游分支"
        for head in branches:
            rendered.add(head.node_id)
            lines += [
                "",
                f"## 分支「{head.label or head.node_id[:8]}」"
                f"{' ★' if head.starred else ''}（分叉自{parent_label}）",
                "",
            ]
            lines.append(_md_line(head))
            _render_subtree(head.node_id, lines)
    return "\n".join(lines)


__all__ = ["render_tree_markdown", "router"]
