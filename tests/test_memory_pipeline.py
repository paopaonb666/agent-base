"""记忆形成管线（M6c）的测试：抽取/整合/画像/摘要 + 失败安全 + 端到端钩子。"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent_base.core.bootstrap import AgentRuntime
from agent_base.core.config import Settings
from agent_base.entrypoints.server import create_app
from agent_base.memory.pipeline import (
    MemoryPipeline,
    _parse_json_block,
    render_transcript,
)
from agent_base.memory.service import MemoryService
from agent_base.memory.store import MemoryMemoryStore, SessionSummary
from agent_base.modules.chat.module import ChatModule
from fakes import HashEmbedding, ScriptedChatModel


def _settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {"llm_api_key": "sk-test"}
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


def _service(responses: list[AIMessage], **overrides: object) -> MemoryService:
    settings = _settings(**overrides)
    return MemoryService(
        store=MemoryMemoryStore(),
        embedder=HashEmbedding(),
        settings=settings,
        llm=ScriptedChatModel(responses),
    )


def _json_msg(payload: Any) -> AIMessage:
    return AIMessage(content=json.dumps(payload, ensure_ascii=False))


def _rearm(service: MemoryService, responses: list[AIMessage]) -> None:
    """替换脚本模型并同步重建管线（pipeline 持有 llm 引用）。"""
    service._llm = ScriptedChatModel(responses)  # type: ignore[attr-defined]
    service.pipeline = MemoryPipeline(service._llm, service._settings, service)


def _split_pipeline(
    main_responses: list[AIMessage],
    fast_responses: list[AIMessage],
    **overrides: object,
) -> tuple[MemoryPipeline, ScriptedChatModel, ScriptedChatModel, MemoryService]:
    """构建双档管线：主力与快档各持一个可录制的脚本模型。"""
    settings = _settings(**overrides)
    main = ScriptedChatModel(main_responses)
    fast = ScriptedChatModel(fast_responses)
    service = MemoryService(
        store=MemoryMemoryStore(),
        embedder=HashEmbedding(),
        settings=settings,
        llm=main,
        fast_llm=fast,
    )
    pipeline = MemoryPipeline(main, settings, service, fast_llm=fast)
    return pipeline, main, fast, service


# ─────────────────────────── JSON 解析 ───────────────────────────


def test_parse_json_block_variants() -> None:
    assert _parse_json_block('[{"content": "a"}]') == [{"content": "a"}]
    assert _parse_json_block('```json\n{"op": "ADD"}\n```') == {"op": "ADD"}
    assert _parse_json_block('结果如下：\n{"op": "NONE"}\n以上。') == {"op": "NONE"}
    with pytest.raises(ValueError):
        _parse_json_block("完全不是 JSON 的回答")


class _FakeMsg:
    def __init__(self, mtype: str, content: str) -> None:
        self.type = mtype
        self.content = content


def test_render_transcript_and_truncation() -> None:
    messages: list[Any] = [
        _FakeMsg("human", "你好"),
        _FakeMsg("ai", "你好！"),
        _FakeMsg("tool", "搜索结果……"),
        _FakeMsg("human", "帮我记住：用户在开发记忆系统"),
        _FakeMsg("ai", "好的"),
    ]
    transcript = render_transcript(messages, max_chars=8000)
    assert "用户：你好" in transcript
    assert "助手：你好！" in transcript
    assert "工具[" in transcript
    assert "记忆系统" in transcript
    # 超限保尾部。
    short = render_transcript(messages, max_chars=20)
    assert short.startswith("……（前文截断）")
    assert "好的" in short  # 尾部保留


# ─────────────────────────── 抽取 ───────────────────────────


async def test_extract_candidates_parses_llm_output() -> None:
    service = _service(
        [
            AIMessage(
                content='```json\n[{"content": "用户在开发记忆系统", "kind": "semantic",'
                ' "salience": 0.9, "tags": ["项目"]},'
                ' {"content": "", "kind": "semantic"},'
                ' {"content": "非法 kind 的条目", "kind": "gossip"}]\n```'
            )
        ]
    )
    assert service.pipeline is not None
    candidates = await service.pipeline.extract_candidates("用户：我在开发记忆系统")
    # 空内容剔除、非法 kind 归一为 semantic。
    assert [c.content for c in candidates] == ["用户在开发记忆系统", "非法 kind 的条目"]
    assert candidates[0].salience == pytest.approx(0.9)
    assert candidates[1].kind == "semantic"


async def test_extract_failure_records_error_op() -> None:
    service = _service([])  # 队列为空 → 模型调用抛 RuntimeError
    assert service.pipeline is not None
    detail = await service.capture_turn(
        user_id="u", agent_id="chat", thread_id="chat:t", messages=[HumanMessage(content="hi")]
    )
    assert detail is not None and detail["candidates"] == 0
    ops = await service.store.list_ops()
    assert any(op.op == "extract" and op.status == "error" for op in ops)


# ─────────────────────────── 双档分派（T0.2） ───────────────────────────


def _extraction_json() -> AIMessage:
    return _json_msg([{"content": "用户在写小说", "kind": "semantic", "salience": 0.8}])


async def test_pipeline_routes_bulk_to_fast_and_decision_to_main() -> None:
    """split 档：抽取/画像/摘要打快档，整合裁决打主力——快档不被裁决消费。"""
    pipeline, main, fast, service = _split_pipeline(
        # main：仅整合裁决应答
        [_json_msg({"op": "ADD", "content": "用户在写小说"})],
        # fast：抽取 → 画像 → 摘要
        [
            _extraction_json(),
            _json_msg({"基本信息": [], "偏好": ["写小说"]}),
            AIMessage(content="本会话聊了写作。"),
        ],
    )
    candidates = await pipeline.extract_candidates("用户：我在写小说")
    assert [c.content for c in candidates] == ["用户在写小说"]
    assert main.received == []  # 抽取没碰主力
    assert len(fast.received) == 1

    op = await pipeline.consolidate(candidates[0], user_id="u", agent_id="chat", thread_id="chat:t")
    assert op == "ADD"
    assert len(main.received) == 1  # 裁决在主力
    assert len(fast.received) == 1  # 快档未被裁决消费

    assert await pipeline.merge_profile("u", ["用户在写小说"]) is True
    assert len(main.received) == 1
    assert len(fast.received) == 2

    assert await pipeline.update_summary("u", "chat", "chat:t", "用户：写小说") is True
    assert len(main.received) == 1
    assert len(fast.received) == 3
    _ = service  # service 参与检索装配，显式消费避免未用告警


async def test_pipeline_all_main_ignores_fast_side() -> None:
    """all_main 档：全部走主力（历史行为），快档一次都不被调用。"""
    pipeline, main, fast, _service = _split_pipeline(
        [_extraction_json()],
        [_json_msg([{"content": "不应被消费"}])],
        memory_pipeline_profile="all_main",
    )
    candidates = await pipeline.extract_candidates("用户：我在写小说")
    assert candidates
    assert len(main.received) == 1
    assert fast.received == []


async def test_pipeline_all_fast_sends_decision_to_fast() -> None:
    """all_fast 档（显式覆盖）：裁决也走快档。"""
    pipeline, main, fast, _service = _split_pipeline(
        [],  # main：一次都不应被调用
        [
            _extraction_json(),
            _json_msg({"op": "ADD", "content": "用户在写小说"}),
        ],
        memory_pipeline_profile="all_fast",
    )
    candidates = await pipeline.extract_candidates("用户：我在写小说")
    op = await pipeline.consolidate(candidates[0], user_id="u", agent_id="chat", thread_id="chat:t")
    assert op == "ADD"
    assert main.received == []
    assert len(fast.received) == 2


async def test_pipeline_fast_unconfigured_degrades_to_main() -> None:
    """fast_llm 未装配（LLM_FAST_* 未配置）时 split 行为等同 all_main。"""
    settings = _settings()
    main = ScriptedChatModel([_extraction_json()])
    service = MemoryService(
        store=MemoryMemoryStore(),
        embedder=HashEmbedding(),
        settings=settings,
        llm=main,
    )
    pipeline = MemoryPipeline(main, settings, service)  # fast_llm 缺省 None
    candidates = await pipeline.extract_candidates("用户：我在写小说")
    assert candidates
    assert len(main.received) == 1


async def test_build_memory_service_passes_fast_llm_through(tmp_path: Any) -> None:
    """组合根传入的 fast_llm 必须原样到达管线。"""
    from agent_base.memory.service import build_memory_service

    settings = _settings(
        checkpointer_backend="memory",
        memory_mysql_pool_size=1,
    )
    svc = await build_memory_service(
        settings, ScriptedChatModel([]), fast_llm=object(), store=MemoryMemoryStore()
    )
    assert svc is not None and svc.pipeline is not None
    assert svc.pipeline._fast_llm is not None
    await svc.aclose()


# ─────────────────────────── 整合 ───────────────────────────


async def test_consolidate_all_ops() -> None:
    service = _service([])
    # 先放一条既有记忆供 UPDATE/DELETE 引用。
    existing = await service.add_memory(
        user_id="u", agent_id="chat", content="用户住在北京", source_thread_id="chat:t"
    )
    cases: list[tuple[dict[str, Any], str, str | None]] = [
        ({"op": "ADD", "content": "用户在开发记忆系统"}, "ADD", None),
        (
            {"op": "UPDATE", "memory_id": existing.memory_id, "content": "用户住在上海"},
            "UPDATE",
            None,
        ),
        ({"op": "DELETE", "memory_id": existing.memory_id}, "DELETE", None),
        ({"op": "NONE"}, "NONE", None),
        ({"op": "UPDATE", "memory_id": "ghost-id", "content": "x"}, "NONE", None),
        ({"op": "DELETE", "memory_id": "ghost-id"}, "NONE", None),
        ({"op": "BOGUS"}, "NONE", None),
    ]
    for decision, expected, _ in cases:
        # 每次整合前把决策放进模型队列；既有记忆让 UPDATE/DELETE 命中。
        _rearm(service, [_json_msg(decision)])
        assert service.pipeline is not None
        from agent_base.memory.pipeline import Candidate

        op = await service.pipeline.consolidate(
            Candidate(content=decision.get("content", "用户在开发记忆系统"), kind="semantic"),
            user_id="u",
            agent_id="chat",
            thread_id="chat:t",
        )
        assert op == expected, f"{decision} -> {op}（期望 {expected}）"


async def test_consolidate_add_creates_memory_with_provenance() -> None:
    service = _service([_json_msg({"op": "ADD", "content": "用户在开发记忆系统"})])
    assert service.pipeline is not None
    from agent_base.memory.pipeline import Candidate

    await service.pipeline.consolidate(
        Candidate(content="用户在开发记忆系统", kind="semantic", salience=0.8, tags=["项目"]),
        user_id="u",
        agent_id="chat",
        thread_id="chat:t1",
    )
    records = await service.store.list_memories("u", agent_id="chat")
    assert len(records) == 1
    assert records[0].source_thread_id == "chat:t1"
    assert records[0].salience == pytest.approx(0.8)


# ─────────────────────── 画像与摘要 ───────────────────────


async def test_profile_merge_creates_and_updates() -> None:
    service = _service(
        [
            _json_msg({"偏好": ["深色主题"], "基本信息": ["用户是工程师"]}),
            _json_msg({"偏好": ["浅色主题"], "基本信息": ["用户是工程师", "住在杭州"]}),
        ]
    )
    assert service.pipeline is not None
    assert await service.pipeline.merge_profile("u", ["用户喜欢深色主题"]) is True
    profile = await service.get_profile("u")
    assert profile is not None and "深色主题" in json.dumps(profile, ensure_ascii=False)
    # 空事实不调用模型。
    assert await service.pipeline.merge_profile("u", []) is False
    await service.pipeline.merge_profile("u", ["用户搬到杭州了，现在喜欢浅色主题"])
    updated = await service.get_profile("u")
    assert updated is not None and "浅色主题" in json.dumps(updated, ensure_ascii=False)
    # 画像记录按确定性 id 存放，且不会被检索召回。
    from agent_base.memory.store import profile_memory_id

    assert await service.store.get_memory(profile_memory_id("u")) is not None
    hits = await service.search(user_id="u", agent_id="chat", query="深色主题")
    assert all(not item.record.memory_id.startswith("profile:") for item in hits)


async def test_update_summary_creates_and_merges() -> None:
    service = _service([AIMessage(content="用户在调试记忆系统，讨论了存储层设计。")])
    assert service.pipeline is not None
    assert (
        await service.pipeline.update_summary(
            "u", "chat", "chat:t", "用户：记忆系统怎么做？\n助手：先建存储层。"
        )
        is True
    )
    summary = await service.store.get_summary("u", "chat:t")
    assert summary is not None and "存储层" in summary.summary
    # 第二次更新（触发摘要阈值后的滚动合并）。
    _rearm(service, [AIMessage(content="更新后的摘要。")])
    assert service.pipeline is not None
    await service.pipeline.update_summary("u", "chat", "chat:t", "用户：继续摘要话题。")
    merged = await service.store.get_summary("u", "chat:t")
    assert merged is not None and merged.summary == "更新后的摘要。"
    assert isinstance(merged, SessionSummary)


# ─────────────────────── 编排与端到端钩子 ───────────────────────


async def test_capture_turn_full_flow() -> None:
    service = _service(
        [
            _json_msg(
                [
                    {"content": "用户在开发企业级记忆系统", "kind": "semantic", "salience": 0.8},
                ]
            ),
            _json_msg({"op": "ADD", "content": "用户在开发企业级记忆系统"}),
            _json_msg({"偏好": ["企业级工程实践"]}),
        ]
    )
    detail = await service.capture_turn(
        user_id="u",
        agent_id="chat",
        thread_id="chat:t",
        messages=[HumanMessage(content="我在开发企业级记忆系统"), AIMessage(content="好的")],
    )
    assert detail is not None
    assert detail["candidates"] == 1
    assert detail["profile_updated"] is True
    records = await service.store.list_memories("u", agent_id="chat")
    assert any(r.content == "用户在开发企业级记忆系统" for r in records)
    ops = {op.op for op in await service.store.list_ops()}
    assert {"extract", "consolidate", "profile"} <= ops


async def test_capture_turn_respects_switches() -> None:
    service = _service([], memory_capture_enabled=False, memory_summary_enabled=False)
    detail = await service.capture_turn(
        user_id="u", agent_id="chat", thread_id="chat:t", messages=[HumanMessage(content="hi")]
    )
    assert detail is not None and detail["candidates"] == 0
    assert await service.store.list_ops() == []


def _runtime_with_memory(responses: list[AIMessage], **overrides: object) -> AgentRuntime:
    rt = AgentRuntime(
        settings=_settings(**overrides),
        llm=ScriptedChatModel(responses),
        modules={"chat": ChatModule()},
        tools=[],
        checkpointer=InMemorySaver(),
    )
    rt.memory = MemoryService(
        store=MemoryMemoryStore(),
        embedder=HashEmbedding(),
        settings=rt.settings,
        llm=rt.llm,  # 与 chat 共用同一个脚本模型：第 1 条给对话，其后给管线
    )
    return rt


def test_invoke_triggers_background_capture() -> None:
    """SSE 对话轮正常完成后，后台形成管线应把事实写进记忆库。"""
    extract = json.dumps(
        [{"content": "用户正在测试记忆钩子", "kind": "semantic", "salience": 0.7}],
        ensure_ascii=False,
    )
    consolidate = json.dumps({"op": "ADD", "content": "用户正在测试记忆钩子"}, ensure_ascii=False)
    profile = json.dumps({"偏好": ["记忆系统测试协作"]}, ensure_ascii=False)
    responses = [
        AIMessage(content="好的，我了解了。"),  # chat 对话轮
        AIMessage(content=extract),  # 管线：抽取
        AIMessage(content=consolidate),  # 管线：整合
        AIMessage(content=profile),  # 管线：画像合并
    ]
    with TestClient(create_app(runtime=_runtime_with_memory(responses))) as client:
        frames = client.post(
            "/v1/agents/chat/invoke",
            json={"message": "请记住：我正在测试记忆钩子"},
            headers={"X-User-Id": "hook-user"},
        )
        assert frames.status_code == 200
        assert "done" in frames.text
        # 后台任务轮询等待（最多 ~5s）：记忆出现即说明钩子生效。
        service = client.app.state.runtime.memory
        deadline = time.time() + 5.0
        found = False
        while time.time() < deadline and not found:
            records = asyncio.run(service.store.list_memories("hook-user", agent_id="chat"))
            found = any(r.content == "用户正在测试记忆钩子" for r in records)
            if not found:
                time.sleep(0.05)
        assert found, "后台记忆形成未生效"
        assert asyncio.run(service.get_profile("hook-user")) is not None


def test_invoke_without_pipeline_skips_capture() -> None:
    """管线缺席（无 LLM）时钩子返回 None，对话照常完成。"""
    rt = AgentRuntime(
        settings=_settings(),
        llm=ScriptedChatModel([AIMessage(content="ok")]),
        modules={"chat": ChatModule()},
        tools=[],
        checkpointer=InMemorySaver(),
    )
    from agent_base.memory.store import MemoryMemoryStore

    rt.memory = MemoryService(
        store=MemoryMemoryStore(), embedder=HashEmbedding(), settings=rt.settings
    )
    with TestClient(create_app(runtime=rt)) as client:
        response = client.post("/v1/agents/chat/invoke", json={"message": "hi"})
    assert response.status_code == 200
    assert asyncio.run(rt.memory.store.list_ops()) == []


def test_tool_message_rendering_in_transcript() -> None:
    tool_msg = ToolMessage(content="工具结果", tool_call_id="x", name="web_search")
    transcript = render_transcript([tool_msg], max_chars=8000)
    assert "web_search" in transcript and "工具结果" in transcript


async def test_capture_transcript_window_configurable() -> None:
    """抽取窗口走配置：窗口外的更早消息不进转写（中段事实靠摘要承载）。"""
    from langchain_core.messages import HumanMessage

    service = _service([], memory_extraction_max_messages=2)
    seen: dict[str, str] = {}

    class RecordingPipeline:
        async def capture_turn(self, **kwargs: object) -> dict[str, object]:
            seen["transcript"] = kwargs.get("transcript", "")
            return {}

    service.pipeline = RecordingPipeline()  # type: ignore[assignment]
    messages = [
        HumanMessage(content="最早的问题"),
        HumanMessage(content="中间的问题"),
        HumanMessage(content="最新的问题"),
    ]
    await service.capture_turn(user_id="u", agent_id="chat", thread_id="chat:t", messages=messages)
    transcript = seen["transcript"]
    assert "最新的问题" in transcript
    assert "中间的问题" in transcript
    assert "最早的问题" not in transcript  # 窗口外被裁掉
