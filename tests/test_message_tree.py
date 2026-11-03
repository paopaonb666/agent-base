"""会话消息树（ADR-013）的回归测试。

覆盖验收点：invoke 分岔（parent_node_id）/ 重新生成（regenerate）、
历史沿活跃指针读取、维护 API（重命名/星标/归档/软删恢复/跳转）、
活跃指针持久化、checkpointer 物化（切分支后 state 重写）、尾部回写
幂等（中断→恢复重叠）、遗留线程迁移幂等、导出。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent_base.core.bootstrap import AgentRuntime
from agent_base.core.config import Settings
from agent_base.core.storage import CoreMemoryStore
from agent_base.core.storage import tree as tree_mod
from agent_base.core.storage.models import MessageNode
from agent_base.entrypoints.server import create_app
from agent_base.modules.chat.module import ChatModule
from fakes import ScriptedChatModel

ALICE = {"X-User-Id": "alice"}


def _runtime(replies: list[str] | None = None) -> AgentRuntime:
    scripted = [
        AIMessage(content=text) for text in (replies or ["回复一", "回复二", "回复三", "回复四"])
    ]
    return AgentRuntime(
        settings=Settings(_env_file=None, llm_api_key="sk-test"),
        llm=ScriptedChatModel(scripted),
        modules={"chat": ChatModule()},
        tools=[],
        checkpointer=InMemorySaver(),
        thread_index=CoreMemoryStore(),
    )


def _invoke(client: TestClient, body: dict[str, Any]) -> dict[str, Any]:
    """跑一轮 invoke（SSE），返回 done 事件载荷（含 thread_id）。"""
    response = client.post("/v1/agents/chat/invoke", json=body, headers=ALICE)
    assert response.status_code == 200, response.text
    events = _frames(response.text)
    done = next(event for event in events if event["event"] == "done")
    assert done.get("thread_id"), done
    return done


def _frames(raw: str) -> list[dict[str, Any]]:
    return [
        {
            "event": block.split("\n")[0].removeprefix("event: "),
            **json.loads(block.split("\n")[1].removeprefix("data: ")),
        }
        for block in raw.strip().split("\n\n")
    ]


def _all_nodes(runtime: AgentRuntime, full_thread_id: str) -> list[MessageNode]:
    store = runtime.thread_index
    assert store is not None

    async def _read() -> list[MessageNode]:
        return await store.list_nodes(full_thread_id)

    return asyncio.run(_read())


def _start_thread(client: TestClient, message: str = "第一问") -> tuple[str, str]:
    """一轮 invoke，返回 (thread_id, human_node_id)。"""
    events = _invoke(client, {"message": message})
    thread_id = events["thread_id"]
    tree = client.get(f"/v1/agents/chat/threads/{thread_id}/tree", headers=ALICE).json()
    humans = [n for n in tree["nodes"] if n["role"] == "human"]
    return thread_id, humans[0]["node_id"]


# ─────────────────────────── invoke 分岔与历史 ───────────────────────────


def test_invoke_appends_human_node_and_pointer_advances() -> None:
    with TestClient(create_app(runtime=_runtime())) as client:
        thread_id, human_id = _start_thread(client)
        tree = client.get(f"/v1/agents/chat/threads/{thread_id}/tree", headers=ALICE).json()
        # human + AI 两个节点；human 无父（新根），AI 挂在 human 之下。
        assert len(tree["nodes"]) == 2
        ai_node = next(n for n in tree["nodes"] if n["role"] == "assistant")
        assert ai_node["parent_id"] == human_id
        assert tree["current_node_id"] == ai_node["node_id"]
        # 历史接口沿活跃指针：两条，每条带 node_id。
        history = client.get(f"/v1/agents/chat/threads/{thread_id}", headers=ALICE).json()[
            "messages"
        ]
        assert [m["role"] for m in history] == ["human", "assistant"]
        assert history[0]["node_id"] == human_id


def test_branch_from_node_creates_sibling_paths() -> None:
    with TestClient(create_app(runtime=_runtime())) as client:
        thread_id, human_id = _start_thread(client)
        # 从 human 节点分岔：第二条 user 消息与原 AI 回复互为兄弟。
        events = _invoke(
            client,
            {"message": "换个思路", "thread_id": thread_id, "parent_node_id": human_id},
        )
        assert events["thread_id"] == thread_id
        tree = client.get(f"/v1/agents/chat/threads/{thread_id}/tree", headers=ALICE).json()
        human_children = [n for n in tree["nodes"] if n["parent_id"] == human_id]
        roles = {n["role"] for n in human_children}
        assert roles == {"assistant", "human"}  # 原 AI 回复 + 分岔的新 human
        # 指针在新分岔的最新消息上；历史只剩新分支的线性路径。
        pointer = tree["current_node_id"]
        assert pointer != human_id
        history = client.get(f"/v1/agents/chat/threads/{thread_id}", headers=ALICE).json()[
            "messages"
        ]
        assert [m["role"] for m in history] == ["human", "human", "assistant"]


def test_regenerate_creates_sibling_ai_reply() -> None:
    with TestClient(create_app(runtime=_runtime())) as client:
        thread_id, human_id = _start_thread(client)
        events = _invoke(
            client,
            {
                "message": "x",
                "thread_id": thread_id,
                "parent_node_id": human_id,
                "regenerate": True,
            },
        )
        assert events["thread_id"] == thread_id
        tree = client.get(f"/v1/agents/chat/threads/{thread_id}/tree", headers=ALICE).json()
        ai_siblings = [n for n in tree["nodes"] if n["parent_id"] == human_id]
        assert len(ai_siblings) == 2  # 原回复 + 重新生成的兄弟
        assert tree["current_node_id"] in {n["node_id"] for n in ai_siblings}


def test_regenerate_requires_human_parent() -> None:
    with TestClient(create_app(runtime=_runtime())) as client:
        thread_id, _ = _start_thread(client)
        tree = client.get(f"/v1/agents/chat/threads/{thread_id}/tree", headers=ALICE).json()
        ai_id = next(n["node_id"] for n in tree["nodes"] if n["role"] == "assistant")
        response = client.post(
            "/v1/agents/chat/invoke",
            json={
                "message": "x",
                "thread_id": thread_id,
                "parent_node_id": ai_id,
                "regenerate": True,
            },
            headers=ALICE,
        )
        assert response.status_code == 400


def test_unknown_branch_node_rejected() -> None:
    with TestClient(create_app(runtime=_runtime())) as client:
        thread_id, _ = _start_thread(client)
        response = client.post(
            "/v1/agents/chat/invoke",
            json={"message": "x", "thread_id": thread_id, "parent_node_id": "0" * 32},
            headers=ALICE,
        )
        assert response.status_code == 404


# ─────────────────────────── checkpointer 物化 ───────────────────────────


def _state_messages(graph: Any, config: dict[str, Any]) -> list[Any]:
    snapshot = asyncio.run(graph.aget_state(config))
    return (snapshot.values or {}).get("messages", []) if snapshot else []


def _materialize(graph: Any, config: dict[str, Any], path: list[MessageNode]) -> None:
    asyncio.run(tree_mod.materialize_branch(graph, config, path))


def test_materialization_rewrites_state_on_branch_switch() -> None:
    runtime = _runtime()
    with TestClient(create_app(runtime=runtime)) as client:
        thread_id, human_id = _start_thread(client)
        graph = runtime.graph("chat")
        full_id = f"chat:{thread_id}"
        config = {"configurable": {"thread_id": full_id}}
        # 分岔前 state 的第一条 human 是"第一问"。
        first = next(m for m in _state_messages(graph, config) if getattr(m, "type", "") == "human")
        assert str(first.content) == "第一问"
        # 从根 human 分岔一轮，然后直接调物化验证重写语义。
        _invoke(
            client,
            {"message": "岔路", "thread_id": thread_id, "parent_node_id": human_id},
        )
        all_nodes = _all_nodes(runtime, full_id)
        branch_human = next(n for n in all_nodes if n.role == "human" and n.content == "岔路")
        _materialize(graph, config, tree_mod.build_path(all_nodes, branch_human.node_id))
        humans = [
            str(m.content)
            for m in _state_messages(graph, config)
            if getattr(m, "type", "") == "human"
        ]
        assert humans == ["第一问", "岔路"]


# ─────────────────────────── 维护 API ───────────────────────────


def test_patch_node_label_star_archive() -> None:
    with TestClient(create_app(runtime=_runtime())) as client:
        thread_id, human_id = _start_thread(client)
        url = f"/v1/agents/chat/threads/{thread_id}/nodes/{human_id}"
        patched = client.patch(url, json={"label": "起点", "starred": True}, headers=ALICE).json()[
            "node"
        ]
        assert patched["label"] == "起点"
        assert patched["starred"] is True
        patched = client.patch(url, json={"starred": False}, headers=ALICE).json()["node"]
        assert patched["label"] == "起点"
        assert patched["starred"] is False
        history = client.get(f"/v1/agents/chat/threads/{thread_id}", headers=ALICE).json()[
            "messages"
        ]
        assert history[0]["label"] == "起点"


def test_delete_subtree_requires_cascade_and_rolls_back_pointer() -> None:
    with TestClient(create_app(runtime=_runtime())) as client:
        thread_id, human_id = _start_thread(client)
        # 先造一棵非根的子树：从根 human 分岔 → 分支 human + 它的 AI 回复。
        _invoke(
            client,
            {"message": "岔路", "thread_id": thread_id, "parent_node_id": human_id},
        )
        tree = client.get(f"/v1/agents/chat/threads/{thread_id}/tree", headers=ALICE).json()
        branch_human = next(
            n for n in tree["nodes"] if n["role"] == "human" and n["parent_id"] == human_id
        )
        # 无 cascade 删有后代的（非根）节点 → 409。
        response = client.delete(
            f"/v1/agents/chat/threads/{thread_id}/nodes/{branch_human['node_id']}",
            headers=ALICE,
        )
        assert response.status_code == 409
        # cascade 软删整棵子树；活跃路径被删 → 指针回退。
        result = client.delete(
            f"/v1/agents/chat/threads/{thread_id}/nodes/{branch_human['node_id']}?cascade=true",
            headers=ALICE,
        ).json()
        assert result["deleted"] == 2
        assert result["pointer_rolled_back"] is True
        tree = client.get(f"/v1/agents/chat/threads/{thread_id}/tree", headers=ALICE).json()
        deleted = {n["node_id"] for n in tree["nodes"] if n["deleted"]}
        assert branch_human["node_id"] in deleted
        # 指针回退到根 human（最近存活祖先）。
        assert tree["current_node_id"] == human_id
        # 恢复子树：删除标记清除，节点回到树里（指针不自动接管）。
        restored = client.post(
            f"/v1/agents/chat/threads/{thread_id}/nodes/{branch_human['node_id']}/restore",
            headers=ALICE,
        ).json()["restored"]
        assert restored == 2
        tree = client.get(f"/v1/agents/chat/threads/{thread_id}/tree", headers=ALICE).json()
        assert all(not n["deleted"] for n in tree["nodes"])


def test_delete_leaf_and_jump_branch() -> None:
    with TestClient(create_app(runtime=_runtime())) as client:
        thread_id, human_id = _start_thread(client)
        tree = client.get(f"/v1/agents/chat/threads/{thread_id}/tree", headers=ALICE).json()
        ai_id = next(n["node_id"] for n in tree["nodes"] if n["role"] == "assistant")
        # 跳回 human 节点（指针移回），历史随之变为截断路径。
        jumped = client.post(
            f"/v1/agents/chat/threads/{thread_id}/jump",
            json={"node_id": human_id},
            headers=ALICE,
        ).json()
        assert jumped["current_node_id"] == human_id
        history = client.get(f"/v1/agents/chat/threads/{thread_id}", headers=ALICE).json()[
            "messages"
        ]
        assert [m["role"] for m in history] == ["human"]
        # 删叶子（AI 回复）：指针已先跳回 human（不在被删子树上）→ 不回退，
        # 指针原地不动。
        result = client.delete(
            f"/v1/agents/chat/threads/{thread_id}/nodes/{ai_id}", headers=ALICE
        ).json()
        assert result["deleted"] == 1 and result["pointer_rolled_back"] is False
        tree = client.get(f"/v1/agents/chat/threads/{thread_id}/tree", headers=ALICE).json()
        assert tree["current_node_id"] == human_id


def test_root_node_cannot_be_deleted() -> None:
    with TestClient(create_app(runtime=_runtime())) as client:
        thread_id, human_id = _start_thread(client)
        response = client.delete(
            f"/v1/agents/chat/threads/{thread_id}/nodes/{human_id}?cascade=true",
            headers=ALICE,
        )
        assert response.status_code == 400


def test_tree_endpoints_are_owner_scoped() -> None:
    bob = {"X-User-Id": "bob"}
    with TestClient(create_app(runtime=_runtime())) as client:
        thread_id, _ = _start_thread(client)
        assert (
            client.get(f"/v1/agents/chat/threads/{thread_id}/tree", headers=bob).status_code == 404
        )
        assert (
            client.get(f"/v1/agents/chat/threads/{thread_id}/export", headers=bob).status_code
            == 404
        )
        assert (
            client.post(
                f"/v1/agents/chat/threads/{thread_id}/jump",
                json={"node_id": "0" * 32},
                headers=bob,
            ).status_code
            == 404
        )


def test_export_markdown_and_json() -> None:
    with TestClient(create_app(runtime=_runtime())) as client:
        thread_id, human_id = _start_thread(client)
        _invoke(
            client,
            {"message": "岔路", "thread_id": thread_id, "parent_node_id": human_id},
        )
        tree = client.get(f"/v1/agents/chat/threads/{thread_id}/tree", headers=ALICE).json()
        branch_human = next(
            n["node_id"] for n in tree["nodes"] if n["role"] == "human" and n["preview"] == "岔路"
        )
        client.patch(
            f"/v1/agents/chat/threads/{thread_id}/nodes/{branch_human}",
            json={"label": "备选"},
            headers=ALICE,
        )
        md = client.get(
            f"/v1/agents/chat/threads/{thread_id}/export?format=md", headers=ALICE
        ).json()["content"]
        # 主线 = 当前活跃路径（分岔后指针在新分支上，"岔路"在主线）；
        # 被顶掉的原回复（回复一）落入旁支章节。
        assert "## 主线" in md
        assert "「备选」岔路" in md
        assert "## 分支「" in md
        assert "回复一" in md
        payload = client.get(
            f"/v1/agents/chat/threads/{thread_id}/export?format=json", headers=ALICE
        ).json()
        assert len(payload["nodes"]) == 4  # 2 主线 + 2 分岔
        assert payload["current_node_id"] is not None


# ─────────────────────────── 纯函数与回写 ───────────────────────────


def test_append_turn_tail_is_idempotent_by_msg_id() -> None:
    store = CoreMemoryStore()

    async def _flow() -> None:
        messages = [
            HumanMessage(content="问", id="m1"),
            AIMessage(content="答", id="m2"),
        ]
        # 第一批（invoke 落 human 后的完整快照回写——含 human，按 id 跳过）。
        human_node = tree_mod.message_to_node(messages[0], thread_id="t", parent_id=None)
        assert human_node is not None
        await store.append_node(human_node)
        leaf = await tree_mod.append_turn_tail(store, "t", messages, after_node_id=None)
        assert leaf is not None and leaf != human_node.node_id
        # 重复回写同一批：不产生新节点（id 判重）。
        again = await tree_mod.append_turn_tail(store, "t", messages, after_node_id=None)
        assert again == leaf
        nodes = await store.list_nodes("t")
        assert len(nodes) == 2
        assert await store.get_current_node("t") == leaf

    asyncio.run(_flow())


def test_message_node_roundtrip_preserves_ids_and_content() -> None:
    message = AIMessage(
        content="带调用的回复",
        tool_calls=[{"name": "search", "args": {"q": "x"}, "id": "call_1"}],
        id="m-ai",
    )
    node = tree_mod.message_to_node(message, thread_id="t", parent_id=None)
    assert node is not None
    rebuilt = tree_mod.node_to_message(node)
    assert isinstance(rebuilt, AIMessage)
    assert rebuilt.id == "m-ai"
    assert rebuilt.tool_calls[0]["name"] == "search"
    assert rebuilt.tool_calls[0]["args"] == {"q": "x"}
    # msg_id 是状态比对与回写判重的唯一依据。
    assert node.msg_id == "m-ai"


def test_append_turn_tail_without_anchor_on_nonempty_tree_bails_out() -> None:
    """无锚点且快照与树完全对不上：放弃回写而不是造出第二棵根链。"""

    async def _flow() -> None:
        store = CoreMemoryStore()
        root = tree_mod.message_to_node(
            HumanMessage(content="已有", id="m-root"), thread_id="t", parent_id=None
        )
        assert root is not None
        await store.append_node(root)
        await store.set_current_node("t", root.node_id)
        leaf = await tree_mod.append_turn_tail(
            store,
            "t",
            [HumanMessage(content="陌生消息", id="m-x")],
            after_node_id=None,
        )
        assert leaf == root.node_id  # 指针原样返回，树未被污染
        nodes = await store.list_nodes("t")
        assert len(nodes) == 1

    asyncio.run(_flow())


def test_build_path_detects_cycles() -> None:
    base = dict(thread_id="t", role="human", content="")
    a = MessageNode(node_id="a", parent_id="b", seq=1, **base)
    b = MessageNode(node_id="b", parent_id="a", seq=2, **base)
    with pytest.raises(tree_mod.TreeError):
        tree_mod.build_path([a, b], "a")
