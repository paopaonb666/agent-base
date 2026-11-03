"""消息树持久化验收（ADR-013）：树与活跃指针跨进程重启存活。

生产同构装配（``create_app()`` 全量组合：真实 sqlite checkpointer +
sqlite 基座表 + mock LLM），两个 TestClient 生命周期 = 两次"进程重启"：
第一进程建会话→分岔→维护；第二进程在同一库上重启——树、节点元数据、
活跃指针必须原样存活，历史沿指针续读且可继续对话。
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient

from agent_base.entrypoints.server import create_app

# ── 进程内 mock LLM（与 scripts/dev/mock_llm.py 同契约，端口随机） ────────

_REPLIES = (
    "回复一：mock 固定回答。",
    "回复二：mock 另一条回答。",
    "回复三：mock 轮换文案。",
    "回复四：mock 继续轮换。",
)
_seq = {"n": 0}


def _mock_llm_app() -> FastAPI:
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> Any:
        body = await request.json()
        _seq["n"] += 1
        cid = f"cmpl-{_seq['n']}"
        reply = _REPLIES[_seq["n"] % len(_REPLIES)]
        if body.get("stream"):

            async def _stream() -> Iterator[str]:
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "id": cid,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": "mock",
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"role": "assistant", "content": reply},
                                    "finish_reason": None,
                                },
                            ],
                        }
                    )
                    + "\n\n"
                )
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "id": cid,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": "mock",
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        }
                    )
                    + "\n\n"
                )
                yield "data: [DONE]\n\n"

            return StreamingResponse(_stream(), media_type="text/event-stream")
        return JSONResponse(
            {
                "id": cid,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "mock",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": reply},
                        "finish_reason": "stop",
                    }
                ],
            }
        )

    return app


@pytest.fixture()
def mock_llm_base_url() -> Iterator[str]:
    config = uvicorn.Config(_mock_llm_app(), host="127.0.0.1", port=0, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    assert server.started, "mock LLM 未能在 10s 内启动"
    sock = server.servers[0].sockets[0]
    host, port = sock.getsockname()[:2]
    yield f"http://{host}:{port}/v1"
    server.should_exit = True
    thread.join(timeout=5)


def _env(monkeypatch: pytest.MonkeyPatch, db_path: Path, base_url: str) -> None:
    overrides = {
        "AGENT_MODULES": "chat",
        "CHECKPOINTER_BACKEND": "sqlite",
        "CHECKPOINTER_SQLITE_PATH": str(db_path),
        "REDIS_ENABLED": "false",
        "MEMORY_ENABLED": "false",
        "LLM_PROVIDER": "openai-compatible",
        "LLM_API_KEY": "test",
        "LLM_BASE_URL": base_url,
        "LLM_MODEL": "mock",
        "LLM_FAST_BASE_URL": "",
        "LLM_FAST_MODEL": "",
    }
    for key, value in overrides.items():
        monkeypatch.setenv(key, value)


def _invoke(client: TestClient, body: dict[str, object]) -> str:
    response = client.post("/v1/agents/chat/invoke", json=body, headers={"X-User-Id": "alice"})
    assert response.status_code == 200, response.text
    frames = [
        block for block in response.text.strip().split("\n\n") if block.startswith("event: done")
    ]
    assert frames, response.text
    payload = frames[0].split("\n", 1)[1].removeprefix("data: ")
    return str(json.loads(payload)["thread_id"])


def _tree(client: TestClient, thread_id: str) -> dict[str, object]:
    response = client.get(
        f"/v1/agents/chat/threads/{thread_id}/tree", headers={"X-User-Id": "alice"}
    )
    assert response.status_code == 200
    return dict(response.json())


def _wait_nodes(client: TestClient, thread_id: str, count: int) -> dict[str, object]:
    """轮询等树回写：invoke 的 done 事件先于后台回写返回（spawn_background），
    立刻读树会抢跑——等到节点数到位（2s 兜底）再继续断言。"""
    deadline = time.monotonic() + 2.0
    tree = _tree(client, thread_id)
    while len(tree["nodes"]) < count and time.monotonic() < deadline:  # type: ignore[arg-type]
        time.sleep(0.05)
        tree = _tree(client, thread_id)
    return tree


def test_tree_and_pointer_survive_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mock_llm_base_url: str
) -> None:
    db_path = tmp_path / "agent.db"
    _env(monkeypatch, db_path, mock_llm_base_url)

    # ── 第一"进程"：两轮对话 + 从首条 human 分岔 + 重命名/星标 ──────────
    with TestClient(create_app()) as client:
        health = client.get("/health")
        assert health.status_code == 200, health.text
        thread_id = _invoke(client, {"message": "第一问"})
        tree = _wait_nodes(client, thread_id, 2)
        root_human = next(n for n in tree["nodes"] if n["role"] == "human")
        _invoke(
            client,
            {"message": "岔路", "thread_id": thread_id, "parent_node_id": root_human["node_id"]},
        )
        client.patch(
            f"/v1/agents/chat/threads/{thread_id}/nodes/{root_human['node_id']}",
            json={"label": "起点", "starred": True},
            headers={"X-User-Id": "alice"},
        )
        before = _wait_nodes(client, thread_id, 4)
        assert before["current_node_id"] is not None
        assert len(before["nodes"]) == 4

    # ── 第二"进程"：同一数据库重新装配，验收持久化 ──────────────────────
    with TestClient(create_app()) as client:
        # 指针跨重启存活：历史沿重启前的活跃分支续读。
        history = client.get(
            f"/v1/agents/chat/threads/{thread_id}", headers={"X-User-Id": "alice"}
        ).json()["messages"]
        assert [m["role"] for m in history] == ["human", "human", "assistant"]
        tree = client.get(
            f"/v1/agents/chat/threads/{thread_id}/tree", headers={"X-User-Id": "alice"}
        ).json()
        assert tree["current_node_id"] == before["current_node_id"]
        assert len(tree["nodes"]) == 4
        root_human = next(n for n in tree["nodes"] if n["parent_id"] is None)
        assert root_human["label"] == "起点"
        assert root_human["starred"] is True
        # 重启后继续对话：默认挂载点 = 指针叶子（活跃分支延续，不回主线）。
        _invoke(client, {"message": "重启后的追问", "thread_id": thread_id})
        tree = _wait_nodes(client, thread_id, 6)
        assert len(tree["nodes"]) == 6
        history = client.get(
            f"/v1/agents/chat/threads/{thread_id}", headers={"X-User-Id": "alice"}
        ).json()["messages"]
        assert [m["role"] for m in history] == [
            "human",
            "human",
            "assistant",
            "human",
            "assistant",
        ]
