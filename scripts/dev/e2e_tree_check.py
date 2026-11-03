"""消息树端到端验收脚本（ADR-013）：对**真实运行的服务器**跑完整流程。

用法（配合 scripts/dev/mock_llm.py）：
    python scripts/dev/e2e_tree_check.py --base-url http://127.0.0.1:8210
    # 服务器重启后验证持久化：
    python scripts/dev/e2e_tree_check.py --base-url http://127.0.0.1:8210 \\
        --phase restart --thread-id <第一步输出的 id>

覆盖：首轮对话 → 从此分岔 → 重新生成 → 分支切换 → 重命名/星标 →
软删子树与恢复 → 导出 MD/JSON → （restart 阶段）树/指针跨重启存活与
活跃分支续聊。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

import httpx

USER = {"X-User-Id": "alice"}


def _sse_done(text: str) -> dict[str, Any]:
    for block in text.strip().split("\n\n"):
        if block.startswith("event: done"):
            payload = block.split("\n", 1)[1].removeprefix("data: ")
            return dict(json.loads(payload))
    raise AssertionError(f"SSE 流里没有 done 事件：{text[-400:]}")


class Checker:
    def __init__(self, base_url: str) -> None:
        self.client = httpx.Client(base_url=base_url, timeout=60.0)
        self.failed: list[str] = []

    def check(self, name: str, condition: bool, detail: str = "") -> None:
        mark = "PASS" if condition else "FAIL"
        print(f"[{mark}] {name}" + (f" — {detail}" if detail and not condition else ""))
        if not condition:
            self.failed.append(name)

    def invoke(self, body: dict[str, Any]) -> str:
        response = self.client.post("/v1/agents/chat/invoke", json=body, headers=USER)
        response.raise_for_status()
        return str(_sse_done(response.text)["thread_id"])

    def tree(self, thread_id: str) -> dict[str, Any]:
        response = self.client.get(f"/v1/agents/chat/threads/{thread_id}/tree", headers=USER)
        response.raise_for_status()
        return dict(response.json())

    def history(self, thread_id: str) -> list[dict[str, Any]]:
        response = self.client.get(f"/v1/agents/chat/threads/{thread_id}", headers=USER)
        response.raise_for_status()
        return list(response.json()["messages"])

    def wait_nodes(self, thread_id: str, count: int, *, timeout: float = 3.0) -> dict[str, Any]:
        """轮询等后台回写落树（done 事件先于回写返回）。"""
        deadline = time.monotonic() + timeout
        tree = self.tree(thread_id)
        while len(tree["nodes"]) < count and time.monotonic() < deadline:
            time.sleep(0.1)
            tree = self.tree(thread_id)
        return tree

    def run_main(self) -> str | None:
        health = self.client.get("/health")
        self.check("健康检查", health.status_code == 200 and health.json()["status"] == "ok")

        thread_id = self.invoke({"message": "第一问"})
        tree = self.wait_nodes(thread_id, 2)
        nodes = tree["nodes"]
        self.check("首轮落树（human+AI 两节点）", len(nodes) == 2, f"实际 {len(nodes)}")
        root = next(n for n in nodes if n["role"] == "human")
        pointer = tree["current_node_id"]
        self.check("活跃指针推进到叶子", pointer == nodes[-1]["node_id"])

        # 从此分岔
        self.invoke(
            {"message": "换个思路试试", "thread_id": thread_id, "parent_node_id": root["node_id"]}
        )
        tree = self.wait_nodes(thread_id, 4)
        human_children = [n for n in tree["nodes"] if n["parent_id"] == root["node_id"]]
        roles = sorted(n["role"] for n in human_children)
        self.check("从此分岔产生兄弟分支", roles == ["assistant", "human"], f"实际 {roles}")
        history = self.history(thread_id)
        self.check(
            "历史沿新分支线性", [m["role"] for m in history] == ["human", "human", "assistant"]
        )

        # 重新生成（挂在 root human 下，产生第二个 AI 兄弟）
        self.invoke(
            {
                "message": "x",
                "thread_id": thread_id,
                "parent_node_id": root["node_id"],
                "regenerate": True,
            }
        )
        tree = self.wait_nodes(thread_id, 5)
        ai_siblings = [
            n
            for n in tree["nodes"]
            if n["parent_id"] == root["node_id"] and n["role"] == "assistant"
        ]
        self.check("重新生成产生 AI 兄弟", len(ai_siblings) == 2, f"实际 {len(ai_siblings)}")
        siblings_current = tree["current_node_id"] in {n["node_id"] for n in ai_siblings}
        self.check("指针在新生成的回复上", siblings_current)

        # 分支切换：跳回 root human，历史截断
        jumped = self.client.post(
            f"/v1/agents/chat/threads/{thread_id}/jump",
            json={"node_id": root["node_id"]},
            headers=USER,
        )
        jumped.raise_for_status()
        self.check("jump 返回新指针", jumped.json()["current_node_id"] == root["node_id"])
        self.check("切换后历史截断", [m["role"] for m in self.history(thread_id)] == ["human"])

        # 维护：重命名 + 星标
        patched = self.client.patch(
            f"/v1/agents/chat/threads/{thread_id}/nodes/{root['node_id']}",
            json={"label": "起点", "starred": True},
            headers=USER,
        )
        patched.raise_for_status()
        node = patched.json()["node"]
        self.check("重命名+星标生效", node["label"] == "起点" and node["starred"] is True)

        # 软删子树（root 不可删；删分支 human 需 cascade）+ 恢复
        branch_human = next(
            n for n in tree["nodes"] if n["role"] == "human" and n["parent_id"] == root["node_id"]
        )
        denied = self.client.delete(
            f"/v1/agents/chat/threads/{thread_id}/nodes/{branch_human['node_id']}",
            headers=USER,
        )
        self.check("有后代删除默认 409", denied.status_code == 409, f"实际 {denied.status_code}")
        # 指针先跳到被删子树内（验证回退路径；指针不在子树上时不会回退）。
        jumped_back = self.client.post(
            f"/v1/agents/chat/threads/{thread_id}/jump",
            json={"node_id": branch_human["node_id"]},
            headers=USER,
        )
        jumped_back.raise_for_status()
        deleted = self.client.delete(
            f"/v1/agents/chat/threads/{thread_id}/nodes/{branch_human['node_id']}?cascade=true",
            headers=USER,
        )
        deleted.raise_for_status()
        body = deleted.json()
        self.check(
            "cascade 软删子树", body["deleted"] == 2 and body["pointer_rolled_back"], f"实际 {body}"
        )
        tree = self.tree(thread_id)
        self.check("指针回退到根", tree["current_node_id"] == root["node_id"])
        restored = self.client.post(
            f"/v1/agents/chat/threads/{thread_id}/nodes/{branch_human['node_id']}/restore",
            headers=USER,
        )
        restored.raise_for_status()
        tree = self.tree(thread_id)
        alive = [n for n in tree["nodes"] if not n["deleted"]]
        self.check("恢复子树", len(alive) == 5)

        # 导出
        md = self.client.get(f"/v1/agents/chat/threads/{thread_id}/export?format=md", headers=USER)
        md.raise_for_status()
        content = md.json()["content"]
        self.check(
            "导出 Markdown（主线+分支章节）", "## 主线" in content and "## 分支「" in content
        )
        payload = self.client.get(
            f"/v1/agents/chat/threads/{thread_id}/export?format=json", headers=USER
        )
        payload.raise_for_status()
        self.check("导出 JSON 全节点", len(payload.json()["nodes"]) == 5)
        return thread_id

    def run_restart(self, thread_id: str) -> None:
        tree = self.tree(thread_id)
        nodes = tree["nodes"]
        pointer = tree["current_node_id"]
        self.check("重启后树完整", len(nodes) == 5, f"实际 {len(nodes)}")
        self.check("重启后指针存活", pointer is not None)
        root = next(n for n in nodes if n["parent_id"] is None)
        self.check("重启后节点元数据存活", root["label"] == "起点" and root["starred"] is True)

        def path_roles(t: dict[str, Any]) -> list[str]:
            by_id = {n["node_id"]: n for n in t["nodes"]}
            roles: list[str] = []
            seen: set[str] = set()
            cursor = t["current_node_id"]
            while cursor and cursor in by_id and cursor not in seen:
                seen.add(cursor)
                roles.append(by_id[cursor]["role"])
                cursor = by_id[cursor]["parent_id"]
            return list(reversed(roles))

        # 历史沿指针续读：预期从树+指针自洽推导（指针在主流程末尾被删除
        # 回退留在根上，历史就是根 human 单节点——这本身就是正确行为）。
        expected = path_roles(tree)
        history = self.history(thread_id)
        self.check(
            "重启后历史沿指针续读",
            [m["role"] for m in history] == expected,
            f"实际 {[m['role'] for m in history]}，预期 {expected}",
        )
        # 活跃分支续聊（默认挂载 = 指针叶子），历史随之线性延伸两条。
        self.invoke({"message": "重启后的追问", "thread_id": thread_id})
        tree = self.wait_nodes(thread_id, len(nodes) + 2)
        self.check("重启后新轮落在活跃分支", len(tree["nodes"]) == len(nodes) + 2)
        expected = path_roles(tree)
        history = self.history(thread_id)
        self.check(
            "续聊历史线性延伸",
            [m["role"] for m in history] == expected,
            f"实际 {[m['role'] for m in history]}，预期 {expected}",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8210")
    parser.add_argument("--phase", choices=["main", "restart"], default="main")
    parser.add_argument("--thread-id", default=None, help="restart 阶段的会话 id")
    args = parser.parse_args()
    checker = Checker(args.base_url)
    try:
        if args.phase == "main":
            thread_id = checker.run_main()
            print(f"\nTHREAD_ID={thread_id}")
        else:
            if not args.thread_id:
                parser.error("restart 阶段需要 --thread-id")
            checker.run_restart(args.thread_id)
    finally:
        checker.client.close()
    if checker.failed:
        print(f"\n共 {len(checker.failed)} 项失败：{checker.failed}")
        return 1
    print("\n全部通过 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
