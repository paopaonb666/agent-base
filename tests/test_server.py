"""SSE 服务入口的测试（阶段 4）。

覆盖验收点：SSE 事件序列、健康降级、路由模板指标 label、以及取消传播
（客户端断开必须取消正在进行的 LLM 调用）。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent_base.core.bootstrap import AgentRuntime
from agent_base.core.config import Settings
from agent_base.entrypoints import server as server_module
from agent_base.entrypoints.server import _event_stream, create_app
from agent_base.modules.chat.module import ChatModule
from agent_base.modules.writer.module import WriterModule
from fakes import CancellableChatModel, ScriptedChatModel


def _runtime(model: Any = None) -> AgentRuntime:
    return AgentRuntime(
        settings=Settings(_env_file=None, llm_api_key="sk-test"),
        llm=model if model is not None else ScriptedChatModel([]),
        modules={"chat": ChatModule(), "writer": WriterModule()},
        tools=[],
        checkpointer=InMemorySaver(),
    )


def _client(model: Any = None) -> TestClient:
    app = create_app(runtime=_runtime(model))
    return TestClient(app)


def _frames(raw: str) -> list[dict[str, Any]]:
    """把一个 SSE 响应体解析为 (事件类型, 载荷) 字典列表。"""
    events = []
    for block in raw.strip().split("\n\n"):
        lines = block.split("\n")
        etype = lines[0].removeprefix("event: ")
        payload = json.loads(lines[1].removeprefix("data: "))
        events.append({"event": etype, **payload})
    return events


def test_health_reports_components() -> None:
    with _client() as client:
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["components"]["checkpointer"] == "ok"
    assert body["components"]["model"] == "ok"


def test_health_degrades_without_model_key() -> None:
    runtime = _runtime()
    runtime.settings = Settings(_env_file=None, llm_api_key="")
    with TestClient(create_app(runtime=runtime)) as client:
        body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["components"]["model"] == "unconfigured"


def test_lifespan_wires_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    """LOG_JSON / request_id 日志契约必须在 server 路径同样生效（A1）。"""
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(server_module, "setup_logging", lambda **kw: calls.append(kw))
    with TestClient(create_app(runtime=_runtime())):
        pass
    assert calls == [{"json_lines": False}]


def test_invoke_unknown_module_404() -> None:
    with _client() as client:
        response = client.post("/v1/agents/nobody/invoke", json={"message": "hi"})
    assert response.status_code == 404


def test_cors_preflight_allowed() -> None:
    """浏览器预检（OPTIONS）必须通过 CORS，而不是以 405 失败。"""
    with _client() as client:
        response = client.options(
            "/v1/agents/chat/invoke",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert "POST" in response.headers["access-control-allow-methods"]


def test_cors_headers_on_invoke() -> None:
    model = ScriptedChatModel([AIMessage(content="hi")])
    with (
        _client(model) as client,
        client.stream(
            "POST",
            "/v1/agents/chat/invoke",
            json={"message": "hi"},
            headers={"Origin": "http://localhost:3000"},
        ) as response,
    ):
        assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
        "".join(response.iter_text())


def test_cors_rejects_unknown_origin() -> None:
    with _client() as client:
        response = client.options(
            "/v1/agents/chat/invoke",
            headers={
                "Origin": "http://evil.example.com",
                "Access-Control-Request-Method": "POST",
            },
        )
    assert "access-control-allow-origin" not in response.headers


def test_invoke_sse_sequence() -> None:
    model = ScriptedChatModel([AIMessage(content="hello there")])
    with (
        _client(model) as client,
        client.stream("POST", "/v1/agents/chat/invoke", json={"message": "hi"}) as response,
    ):
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["X-Request-ID"]
        raw = "".join(response.iter_text())
    events = _frames(raw)
    types = [e["event"] for e in events]
    # ping ... step(agent 运行) ... delta(s) ... step(agent 完成) ... done
    assert types[0] == "ping"
    assert "delta" in types
    assert "done" in types
    assert types[-1] == "done"
    deltas = "".join(e["content"] for e in events if e["event"] == "delta")
    assert deltas == "hello there"
    done = events[-1]
    assert done["thread_id"]


def test_invoke_error_becomes_error_event() -> None:
    class ExplodingModel(ScriptedChatModel):
        async def _astream(self, *args: Any, **kwargs: Any) -> Any:
            # 一个在产出前就失败的生成器（astream 期望的形状）。
            raise RuntimeError("llm exploded")
            yield  # pragma: no cover - 不可达；用于把本函数标记为生成器

    with (
        _client(ExplodingModel([])) as client,
        client.stream("POST", "/v1/agents/chat/invoke", json={"message": "hi"}) as response,
    ):
        raw = "".join(response.iter_text())
    events = _frames(raw)
    assert events[-1]["event"] == "error"
    assert "llm exploded" in events[-1]["message"]


def test_thread_id_resumes_conversation() -> None:
    model = ScriptedChatModel([AIMessage(content="first"), AIMessage(content="second")])
    with _client(model) as client:
        with client.stream("POST", "/v1/agents/chat/invoke", json={"message": "one"}) as response:
            done = _frames("".join(response.iter_text()))[-1]
        thread_id = done["thread_id"]
        with client.stream(
            "POST",
            "/v1/agents/chat/invoke",
            json={"message": "two", "thread_id": thread_id},
        ) as response:
            raw = "".join(response.iter_text())
    # 第 2 轮必须在其 step 载荷中包含恢复出的第 1 轮历史；运行时状态通过
    # delta 只包含新回复、且第二次调用在同一个 thread 上成功来证明这一点。
    deltas = "".join(e["content"] for e in _frames(raw) if e["event"] == "delta")
    assert deltas == "second"


def test_get_thread_history_returns_messages() -> None:
    """只读端点把 checkpointer 里的历史序列化为 human/assistant 消息。"""
    model = ScriptedChatModel([AIMessage(content="first"), AIMessage(content="second")])
    with _client(model) as client:
        with client.stream("POST", "/v1/agents/chat/invoke", json={"message": "one"}) as response:
            done = _frames("".join(response.iter_text()))[-1]
        thread_id = done["thread_id"]
        with client.stream(
            "POST",
            "/v1/agents/chat/invoke",
            json={"message": "two", "thread_id": thread_id},
        ) as response:
            "".join(response.iter_text())
        history = client.get(f"/v1/agents/chat/threads/{thread_id}")

    assert history.status_code == 200
    body = history.json()
    assert body["thread_id"] == thread_id
    assert body["module"] == "chat"
    assert body["messages"] == [
        {"role": "human", "content": "one"},
        {"role": "assistant", "content": "first"},
        {"role": "human", "content": "two"},
        {"role": "assistant", "content": "second"},
    ]


def test_get_thread_history_unknown_module_404() -> None:
    with _client() as client:
        response = client.get("/v1/agents/nobody/threads/abc")
    assert response.status_code == 404


def test_metrics_uses_route_template_labels() -> None:
    model = ScriptedChatModel([AIMessage(content="ok")])
    with _client(model) as client:
        client.get("/health")
        with client.stream("POST", "/v1/agents/chat/invoke", json={"message": "hi"}):
            pass
        text = client.get("/metrics").text
    assert 'route="/health"' in text
    assert 'route="/v1/agents/{module}/invoke"' in text  # 模板，而非原始路径
    assert "/v1/agents/chat/invoke" not in text.replace("/v1/agents/{module}/invoke", "")
    assert 'status="200"' in text


def test_metrics_unmatched_route_uses_constant_label() -> None:
    """404 请求没有路由模板：必须用常量兜底，否则每个攻击路径都会
    永久新增一条指标时间序列（基数爆炸）。"""
    with _client() as client:
        client.get("/no-such-path-1")
        client.get("/no-such-path-2")
        text = client.get("/metrics").text
    assert 'route="unmatched"' in text
    assert "/no-such-path" not in text


def test_untrusted_request_id_is_replaced() -> None:
    """客户端提供的 X-Request-ID 过白名单才回显：否则换行符可以
    伪造日志行、非法字符会炸掉响应头。"""
    with _client() as client:
        ok = client.get("/health", headers={"X-Request-ID": "abc-123_456.xyz"})
        forged = client.get("/health", headers={"X-Request-ID": "bad id\ninjected: yes"})
    assert ok.headers["X-Request-ID"] == "abc-123_456.xyz"
    forged_id = forged.headers["X-Request-ID"]
    assert "\n" not in forged_id and forged_id != "bad id\ninjected: yes"


def test_invoke_rejects_empty_message_and_bad_thread_id() -> None:
    with _client() as client:
        assert client.post("/v1/agents/chat/invoke", json={"message": ""}).status_code == 422
        assert (
            client.post(
                "/v1/agents/chat/invoke", json={"message": "hi", "thread_id": "../escape"}
            ).status_code
            == 422
        )
        assert (
            client.post("/v1/agents/chat/invoke", json={"message": "x" * 100_001}).status_code
            == 422
        )


async def test_client_disconnect_cancels_llm_call() -> None:
    """阶段 4 验收：断开连接会停止正在进行的 LLM 请求。"""
    model = CancellableChatModel()
    runtime = _runtime(model)
    graph = runtime.graph("chat")
    config: dict[str, Any] = {"configurable": {"thread_id": "chat:t-cancel"}}
    stream = _event_stream(graph, "hi", config, "t-cancel")  # type: ignore[arg-type]

    # 像 SSE 客户端一样消费；模型会挂起，因此在最初的 ping 之后，
    # 流会像一次真实的慢速 LLM 调用一样处于空闲状态。
    task = asyncio.ensure_future(_consume(stream))
    await asyncio.sleep(0.2)  # 生产者已到达挂起的模型调用
    task.cancel()  # 客户端断开时 starlette 所做的动作
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert model.cancelled is True


async def _consume(stream: Any) -> None:
    async for _ in stream:
        pass


def test_list_modules_endpoint() -> None:
    """模块列表端点：配置面板的合法模块名来源（免试错）。"""
    with _client() as client:
        response = client.get("/v1/modules")
    assert response.status_code == 200
    names = [m["name"] for m in response.json()["modules"]]
    assert names == ["chat", "writer", "supervisor"]
    assert all(m["description"] for m in response.json()["modules"])


def test_cors_exposes_request_id_header() -> None:
    """X-Request-ID 必须对浏览器 JS 可读（expose_headers）。"""
    with _client() as client:
        response = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert response.headers.get("access-control-expose-headers") == "X-Request-ID"


def test_health_degrades_when_checkpointer_fails() -> None:
    """checkpointer 探针失败 -> degraded（核心降级路径）。"""

    class _BrokenSaver:
        async def aget_tuple(self, config: dict[str, Any]) -> None:
            raise RuntimeError("db down")

    runtime = _runtime()
    runtime.checkpointer = _BrokenSaver()  # type: ignore[assignment]
    with TestClient(create_app(runtime=runtime)) as client:
        body = client.get("/health").json()
    assert body["components"]["checkpointer"] == "error"
    assert body["status"] == "degraded"


def test_health_flags_misconfigured_base_url() -> None:
    runtime = _runtime()
    runtime.settings = Settings(_env_file=None, llm_api_key="sk-test", llm_base_url="not-a-url")
    with TestClient(create_app(runtime=runtime)) as client:
        body = client.get("/health").json()
    assert body["components"]["model"] == "misconfigured"
    assert body["status"] == "degraded"


def test_health_model_probe_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """探活开关默认关闭：不应该打真实网络。"""

    def _no_probe(settings: Any) -> str:
        raise AssertionError("probe must not run when health_probe_model is False")

    monkeypatch.setattr(server_module, "_probe_model", _no_probe)
    with _client() as client:
        body = client.get("/health").json()
    assert body["components"]["model"] == "ok"


def test_health_model_probe_error_degrades(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _runtime()
    runtime.settings = Settings(_env_file=None, llm_api_key="sk-test", health_probe_model=True)

    async def _failing_probe(settings: Any) -> str:
        return "error"

    monkeypatch.setattr(server_module, "_probe_model", _failing_probe)
    with TestClient(create_app(runtime=runtime)) as client:
        body = client.get("/health").json()
    assert body["components"]["model"] == "error"
    assert body["status"] == "degraded"


async def test_list_threads_endpoint() -> None:
    """线程列表端点：按模块命名空间过滤、每线程取最新、标题为首条用户消息。"""
    runtime = _runtime(ScriptedChatModel([AIMessage("回复A"), AIMessage("回复B")]))
    graph = runtime.graph("chat")
    await graph.ainvoke(
        {"messages": [HumanMessage(content="第一条线程的问题")]},
        {"configurable": {"thread_id": "chat:thread-a"}},
    )
    await graph.ainvoke(
        {"messages": [HumanMessage(content="第二条线程的问题")]},
        {"configurable": {"thread_id": "chat:thread-b"}},
    )
    with TestClient(create_app(runtime=runtime)) as client:
        body = client.get("/v1/agents/chat/threads").json()
    ids = {t["thread_id"] for t in body["threads"]}
    assert {"thread-a", "thread-b"} <= ids
    titles = {t["thread_id"]: t["title"] for t in body["threads"]}
    assert titles["thread-a"] == "第一条线程的问题"
    assert all(t["module"] == "chat" for t in body["threads"])
    assert all(t["updated_at"] > 0 for t in body["threads"])


def test_list_threads_unknown_module_404() -> None:
    with _client() as client:
        response = client.get("/v1/agents/nonexist/threads")
    assert response.status_code == 404


async def test_delete_thread_endpoint() -> None:
    """删除端点按 module:thread_id 命名空间删掉整个线程，且不影响其他线程。"""
    runtime = _runtime(ScriptedChatModel([AIMessage("回复A"), AIMessage("回复B")]))
    graph = runtime.graph("chat")
    config_a = {"configurable": {"thread_id": "chat:thread-a"}}
    config_b = {"configurable": {"thread_id": "chat:thread-b"}}
    await graph.ainvoke({"messages": [HumanMessage(content="待删除")]}, config_a)
    await graph.ainvoke({"messages": [HumanMessage(content="保留")]}, config_b)
    with TestClient(create_app(runtime=runtime)) as client:
        assert client.delete("/v1/agents/chat/threads/thread-a").json() == {"deleted": True}
        remaining = client.get("/v1/agents/chat/threads").json()
    assert [t["thread_id"] for t in remaining["threads"]] == ["thread-b"]


def test_delete_thread_unknown_module_404() -> None:
    with _client() as client:
        response = client.delete("/v1/agents/nonexist/threads/t1")
    assert response.status_code == 404


# ─────────────────── Tier 0：零 LLM 知识库检索（T1.1） ───────────────────


def _memory_runtime() -> AgentRuntime:
    """带记忆服务的运行时：HashEmbedding 确定性向量，ScriptedChatModel 空
    队列——检索端点若碰了 LLM 会以 500 暴露，通过即证明零 LLM。"""
    from agent_base.memory.service import MemoryService
    from agent_base.memory.store import MemoryMemoryStore
    from fakes import HashEmbedding

    rt = _runtime()
    settings = Settings(_env_file=None, llm_api_key="sk-test")
    service = MemoryService(
        store=MemoryMemoryStore(),
        embedder=HashEmbedding(),
        settings=settings,
        llm=rt.llm,
    )
    rt.memory = service
    rt.thread_index = service.store
    return rt


def _seed_knowledge(rt: AgentRuntime) -> None:
    assert rt.memory is not None
    asyncio.run(
        rt.memory.ingest_document(
            file_id="f1",
            user_id="alice",
            agent_id="chat",
            text="本保险的等待期为 90 天，等待期内出险不予赔付。\n\n第二条讲续保规则与宽限期。",
        )
    )


def test_knowledge_search_returns_chunks_without_llm() -> None:
    runtime = _memory_runtime()
    _seed_knowledge(runtime)
    with TestClient(create_app(runtime=runtime)) as client:
        response = client.get(
            "/v1/knowledge/search", params={"q": "等待期"}, headers={"X-User-Id": "alice"}
        )
    assert response.status_code == 200
    body = response.json()
    assert body["results"], "关键词命中的切片必须出现在结果里"
    top = body["results"][0]
    assert top["file_id"] == "f1"
    assert "等待期" in top["text"]
    assert top["ordinal"] == 0
    assert isinstance(top["offsets"], list)
    assert top["has_embedding"] is True
    assert top["score"] > 0


def test_knowledge_search_scopes_by_user() -> None:
    """用户作用域：bob 检索不到 alice 的知识分块（空结果而非 404）。"""
    runtime = _memory_runtime()
    _seed_knowledge(runtime)
    with TestClient(create_app(runtime=runtime)) as client:
        response = client.get(
            "/v1/knowledge/search", params={"q": "等待期"}, headers={"X-User-Id": "bob"}
        )
    assert response.status_code == 200
    assert response.json() == {"results": []}


def test_knowledge_search_file_filter_and_limit() -> None:
    runtime = _memory_runtime()
    _seed_knowledge(runtime)
    assert runtime.memory is not None
    asyncio.run(
        runtime.memory.ingest_document(
            file_id="f2",
            user_id="alice",
            agent_id="chat",
            text="等待期条款的补充说明：意外医疗无等待期。",
        )
    )
    with TestClient(create_app(runtime=runtime)) as client:
        single = client.get(
            "/v1/knowledge/search",
            params={"q": "等待期", "file_id": "f2"},
            headers={"X-User-Id": "alice"},
        )
        capped = client.get(
            "/v1/knowledge/search",
            params={"q": "等待期", "limit": 1},
            headers={"X-User-Id": "alice"},
        )
    assert single.status_code == 200
    assert {r["file_id"] for r in single.json()["results"]} == {"f2"}
    assert len(capped.json()["results"]) == 1


def test_knowledge_search_requires_query_and_memory() -> None:
    runtime = _memory_runtime()
    with TestClient(create_app(runtime=runtime)) as client:
        missing_q = client.get("/v1/knowledge/search", headers={"X-User-Id": "alice"})
    assert missing_q.status_code == 422

    bare = _runtime()  # 无记忆服务
    with TestClient(create_app(runtime=bare)) as client:
        no_memory = client.get(
            "/v1/knowledge/search", params={"q": "x"}, headers={"X-User-Id": "alice"}
        )
    assert no_memory.status_code == 503
