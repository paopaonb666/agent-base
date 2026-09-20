"""上下文组装与短期压缩（M6d）的测试：注入块组装 + 模型输入修剪。"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent_base.core.context import MEMORY_CONTEXT_FROZEN_PREFIX
from agent_base.memory.context import (
    ATTACHMENT_CONTEXT_PREFIX,
    MEMORY_CONTEXT_PREFIX,
    build_model_input,
    compose_context,
    compose_frozen_context,
    compose_recall_context,
    estimate_tokens,
    is_injected_system,
    render_profile,
)
from agent_base.memory.retrieval import ScoredMemory
from agent_base.memory.service import MemoryService
from agent_base.memory.store import MemoryBlock, MemoryMemoryStore, MemoryRecord
from fakes import HashEmbedding, ScriptedChatModel

# ─────────────────────────── 估算与判定 ───────────────────────────


def test_estimate_tokens_cjk_aware() -> None:
    # 中文按 0.6 token/字。
    assert estimate_tokens("四个汉字！") == int(4 * 0.6 + 1 * 0.25)
    # ASCII 约 4 字符/token。
    assert estimate_tokens("abcdefgh") == int(8 * 0.25)
    assert estimate_tokens("") == 0


def test_is_injected_system_recognition() -> None:
    assert is_injected_system(SystemMessage(content=MEMORY_CONTEXT_PREFIX + "……"))
    assert is_injected_system(SystemMessage(content=ATTACHMENT_CONTEXT_PREFIX + "，供参考：\n\nx"))
    assert not is_injected_system(SystemMessage(content="普通系统提示"))
    assert not is_injected_system(HumanMessage(content=MEMORY_CONTEXT_PREFIX))
    # 列表型 content 取 text 段。
    assert is_injected_system(
        SystemMessage(content=[{"type": "text", "text": MEMORY_CONTEXT_PREFIX + "x"}])
    )


# ─────────────────────────── 注入块组装 ───────────────────────────


def _scored(content: str, score: float) -> ScoredMemory:
    return ScoredMemory(
        record=MemoryRecord(
            memory_id=f"m-{content[:6]}",
            user_id="u",
            agent_id="chat",
            kind="semantic",
            content=content,
        ),
        score=score,
    )


def test_compose_context_sections_and_budget() -> None:
    blocks = [MemoryBlock(user_id="u", agent_id="chat", label="persona", content="务实的工程师")]
    recalled = [_scored("用户喜欢深色主题", 0.9), _scored("用户用 Python", 0.5)]
    text = compose_context(
        profile={"偏好": ["简洁回答"]},
        blocks=blocks,
        summary="此前讨论了存储层设计",
        recalled=recalled,
        max_chars=8000,
    )
    assert text is not None
    assert text.startswith(MEMORY_CONTEXT_PREFIX)
    assert "## 用户画像" in text and "简洁回答" in text
    assert "[persona] 务实的工程师" in text
    assert "## 会话摘要" in text and "存储层" in text
    assert "## 相关记忆" in text and "深色主题" in text

    # 预算超限：得分低的记忆先被丢弃，高分与画像保留。
    many = recalled + [_scored(f"填充记忆{i}" * 5, 0.4 - i * 0.05) for i in range(8)]
    tight = compose_context(
        profile={"偏好": ["简洁回答"]},
        blocks=blocks,
        summary=None,
        recalled=many,
        max_chars=350,
    )
    assert tight is not None
    assert "深色主题" in tight  # 最高分保留
    assert "填充记忆7" not in tight  # 最低分被丢弃
    assert "简洁回答" in tight  # 画像最后动

    # 全空 → None。
    empty = compose_context(profile=None, blocks=[], summary=None, recalled=[], max_chars=100)
    assert empty is None


def test_render_profile_shapes() -> None:
    text = render_profile(
        {"偏好": ["深色主题", "简洁"], "工作背景": {"公司": "Acme"}, "备注": "单行"}
    )
    assert "- 偏好：深色主题；简洁" in text
    assert "- 工作背景：公司：Acme" in text
    assert "- 备注：单行" in text


# ─────────────────────────── 模型输入修剪 ───────────────────────────


def _inj(text: str = "旧注入") -> SystemMessage:
    return SystemMessage(content=MEMORY_CONTEXT_PREFIX + text)


def test_build_model_input_dedupes_injected_systems() -> None:
    messages = [
        _inj("第一轮注入"),
        HumanMessage(content="问题一"),
        AIMessage(content="回答一"),
        SystemMessage(content=ATTACHMENT_CONTEXT_PREFIX + "，供参考：\n\n附件全文……"),
        HumanMessage(content="问题二"),
        AIMessage(content="回答二"),
        _inj("本轮注入"),  # 注入块在最前（server insert(0) 的时序效果）
        HumanMessage(content="问题三"),
    ]
    model_input = build_model_input(messages, max_tokens=100_000)
    injected = [m for m in model_input if is_injected_system(m)]
    assert len(injected) == 1
    assert injected[0].content.endswith("本轮注入")
    # 非注入消息一个不少。
    assert len(model_input) == len(messages) - 2
    # 预算充足时保持原顺序：唯一的注入块留在它原来的位置（回答二之后）。
    assert model_input.index(injected[0]) > model_input.index(
        next(m for m in model_input if getattr(m, "content", "") == "回答二")
    )


def test_build_model_input_trims_tail_with_budget() -> None:
    messages: list[Any] = [HumanMessage(content="旧" * 200), AIMessage(content="答" * 200)]
    for i in range(5):
        messages.extend([HumanMessage(content=f"近{i}" * 50), AIMessage(content=f"回{i}" * 50)])
    trimmed = build_model_input(messages, max_tokens=600)
    assert len(trimmed) < len(messages)
    assert trimmed[-1] == messages[-1]  # 尾部保留
    # 被剪掉的是最早的消息。
    assert messages[0] not in trimmed
    # 全量预算下原样返回。
    assert build_model_input(messages, max_tokens=10**9) == messages


def test_build_model_input_keeps_both_injections_on_attachment_turn() -> None:
    """回归（UI 实测缺陷 F1）：附件轮同时注入记忆块与附件全文，两者都必须保留。"""
    messages = [
        _inj("第一轮的旧注入"),  # 历史里的旧注入 → 应被剔除
        HumanMessage(content="旧问题"),
        AIMessage(content="旧回答"),
        SystemMessage(content=MEMORY_CONTEXT_PREFIX + "本轮记忆块"),
        SystemMessage(content=ATTACHMENT_CONTEXT_PREFIX + "，供参考：\n\n附件全文"),
        HumanMessage(content="附件说了什么？"),
    ]
    model_input = build_model_input(messages, max_tokens=100_000)
    injected = [m for m in model_input if is_injected_system(m)]
    assert len(injected) == 2
    kinds = {str(m.content)[:12] for m in injected}
    assert any(k.startswith(MEMORY_CONTEXT_PREFIX) for k in kinds), "记忆块被丢弃"
    assert any(k.startswith(ATTACHMENT_CONTEXT_PREFIX) for k in kinds), "附件块被丢弃"


def test_build_model_input_never_splits_tool_pairs() -> None:
    ai_with_tools = AIMessage(
        content="", tool_calls=[{"name": "web_search", "args": {"q": "x"}, "id": "c1"}]
    )
    tool_msg = ToolMessage(content="搜索结果", tool_call_id="c1", name="web_search")
    messages: list[Any] = [HumanMessage(content="早" * 300), AIMessage(content="答" * 300)]
    messages.extend(
        [HumanMessage(content="查一下"), ai_with_tools, tool_msg, AIMessage(content="结论")]
    )
    trimmed = build_model_input(messages, max_tokens=200)
    # 修剪后第一条不能是 ToolMessage（否则与它的 AIMessage 拆散）。
    assert getattr(trimmed[0], "type", "") != "tool"
    if ai_with_tools in trimmed:
        assert tool_msg in trimmed


def test_build_model_input_single_oversized_message_kept() -> None:
    big = HumanMessage(content="巨" * 50_000)
    assert build_model_input([big], max_tokens=10) == [big]


# ─────────────────────── service 组装 + 图集成 ───────────────────────


async def test_service_compose_context_end_to_end() -> None:
    """冻结段（画像/记忆块/摘要）与召回段分离：各自只含自己的内容。"""
    service = MemoryService(
        store=MemoryMemoryStore(), embedder=HashEmbedding(), settings=_settings()
    )
    await service.add_memory(user_id="alice", agent_id="chat", content="用户喜欢深色主题")
    await service.save_profile("alice", {"偏好": ["简洁回答"]})
    await service.store.upsert_block(
        MemoryBlock(user_id="alice", agent_id="chat", label="human", content="在开发记忆系统")
    )
    frozen = await service.compose_context(
        user_id="alice",
        agent_id="chat",
        thread_id="chat:t1",
        query="我该用什么主题？",
    )
    recall = await service.recall_context(user_id="alice", query="我该用什么主题？")
    assert frozen is not None and recall is not None
    # 冻结段：画像 + 记忆块；不含逐轮召回。
    assert "简洁回答" in frozen and "记忆系统" in frozen
    assert frozen.startswith(MEMORY_CONTEXT_FROZEN_PREFIX)
    assert "深色主题" not in frozen
    # 召回段：查询相关记忆，不含画像。
    assert "深色主题" in recall and "简洁回答" not in recall
    await service.aclose()


async def test_service_compose_context_recall_is_user_level() -> None:
    """注入召回为用户级：chat 写入的记忆在 supervisor 线程的注入块可见。"""
    service = MemoryService(
        store=MemoryMemoryStore(), embedder=HashEmbedding(), settings=_settings()
    )
    await service.add_memory(user_id="alice", agent_id="chat", content="用户的项目代号是雨燕")
    block = await service.recall_context(user_id="alice", query="项目代号")
    assert block is not None and "雨燕" in block
    await service.aclose()


def _settings(**overrides: object):
    from agent_base.core.config import Settings

    defaults: dict[str, object] = {"llm_api_key": "sk-test"}
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


async def test_chat_graph_trims_model_input() -> None:
    """chat 图的 call_model 把超预算的旧历史剪出模型输入。"""
    from langgraph.checkpoint.memory import InMemorySaver

    from agent_base.core.contracts import ModuleContext
    from agent_base.modules.chat.module import ChatModule

    llm = ScriptedChatModel([AIMessage(content="ok")])
    ctx = ModuleContext(
        settings=_settings(memory_context_max_tokens=300),
        llm=llm,
        checkpointer=InMemorySaver(),
        tools=[],
    )
    graph = ChatModule().build_graph(ctx)
    messages: list[Any] = []
    for i in range(12):
        messages.extend(
            [HumanMessage(content=f"历史问题{i}" + "很长" * 60), AIMessage(content="历史回答" * 60)]
        )
    await graph.ainvoke({"messages": messages}, {"configurable": {"thread_id": "chat:trim-test"}})
    assert llm.received, "模型未被调用"
    seen = llm.received[0]
    assert len(seen) < len(messages)
    # 最新一轮（问题 11 与它的回答）保留在尾部。
    assert seen[-1].content == "历史回答" * 60
    assert seen[-2].content.startswith("历史问题11")


# ─────────────────── 冻结/召回分离与快照（T2.1 成本治理） ───────────────────


def test_compose_frozen_and_recall_split() -> None:
    """冻结段=画像+记忆块+摘要；召回段=相关记忆——互不混入。"""
    frozen = compose_frozen_context(
        profile={"偏好": ["简洁回答"]},
        blocks=[MemoryBlock(user_id="u", agent_id="chat", label="human", content="在写小说")],
        summary="聊过写作进度",
        max_chars=10_000,
    )
    assert frozen is not None
    assert frozen.startswith(MEMORY_CONTEXT_FROZEN_PREFIX)
    assert "简洁回答" in frozen and "写小说" in frozen and "写作进度" in frozen
    assert "混不进来" not in frozen

    recall = compose_recall_context(recalled=[_scored("查询相关的记忆", 0.9)], max_chars=10_000)
    assert recall is not None
    assert recall.startswith(MEMORY_CONTEXT_PREFIX)
    assert "查询相关" in recall
    assert "简洁回答" not in recall

    # 空内容语义与 compose_context 一致：None = 本轮不注入。
    assert compose_frozen_context(profile=None, blocks=[], summary=None, max_chars=100) is None
    assert compose_recall_context(recalled=[], max_chars=100) is None


async def test_service_snapshot_frozen_within_thread() -> None:
    """同线程第二次 compose 逐字节相同——即使画像/记忆块/摘要中途已变。"""
    service = MemoryService(
        store=MemoryMemoryStore(), embedder=HashEmbedding(), settings=_settings()
    )
    await service.save_profile("alice", {"偏好": ["旧偏好"]})
    first = await service.compose_context(
        user_id="alice", agent_id="chat", thread_id="chat:t1", query="q"
    )
    assert first is not None
    # 中途变更：数据层照常更新。
    await service.save_profile("alice", {"偏好": ["新偏好"]})
    await service.store.upsert_block(
        MemoryBlock(user_id="alice", agent_id="chat", label="human", content="中途改的块")
    )
    second = await service.compose_context(
        user_id="alice", agent_id="chat", thread_id="chat:t1", query="q"
    )
    assert second == first
    assert "旧偏好" in second and "新偏好" not in second and "中途改的块" not in second
    # 新线程不共享快照：反映最新数据。
    fresh = await service.compose_context(
        user_id="alice", agent_id="chat", thread_id="chat:t2", query="q"
    )
    assert fresh is not None and "新偏好" in fresh and "中途改的块" in fresh
    await service.aclose()


async def test_service_snapshot_lru_eviction() -> None:
    """快照缓存按 LRU 上限逐出：被逐出的线程重新定型。"""
    service = MemoryService(
        store=MemoryMemoryStore(),
        embedder=HashEmbedding(),
        settings=_settings(memory_context_snapshot_max=1),
    )
    await service.save_profile("alice", {"偏好": ["v1"]})
    first = await service.compose_context(
        user_id="alice", agent_id="chat", thread_id="chat:t1", query="q"
    )
    # t2 进缓存 → t1 被逐出。
    await service.compose_context(user_id="alice", agent_id="chat", thread_id="chat:t2", query="q")
    await service.save_profile("alice", {"偏好": ["v2"]})
    recomputed = await service.compose_context(
        user_id="alice", agent_id="chat", thread_id="chat:t1", query="q"
    )
    assert first is not None and recomputed is not None
    assert recomputed != first
    assert "v2" in recomputed
    await service.aclose()


async def test_service_purge_thread_drops_snapshot() -> None:
    """线程删除的级联清理包括注入快照：删除后重建反映空数据。"""
    service = MemoryService(
        store=MemoryMemoryStore(), embedder=HashEmbedding(), settings=_settings()
    )
    await service.save_profile("alice", {"偏好": ["x"]})
    before = await service.compose_context(
        user_id="alice", agent_id="chat", thread_id="chat:t1", query="q"
    )
    assert before is not None and "x" in before
    # 数据层改为 y 后 purge：快照被级联清理 → 重新组装反映 y（而非缓存的 x）。
    await service.save_profile("alice", {"偏好": ["y"]})
    await service.purge_thread("chat:t1")
    after = await service.compose_context(
        user_id="alice", agent_id="chat", thread_id="chat:t1", query="q"
    )
    assert after is not None and "y" in after and "x" not in after
    await service.aclose()


def test_build_model_input_hoists_frozen_head() -> None:
    """冻结段提升到模型输入头部且只出现一次；本轮召回贴尾保留。"""
    messages = [
        SystemMessage(content=MEMORY_CONTEXT_FROZEN_PREFIX + "会话固定段"),
        SystemMessage(content=MEMORY_CONTEXT_PREFIX + "第一轮召回"),
        HumanMessage(content="第一问"),
        AIMessage(content="第一答"),
        SystemMessage(content=MEMORY_CONTEXT_FROZEN_PREFIX + "会话固定段"),
        SystemMessage(content=MEMORY_CONTEXT_PREFIX + "本轮召回"),
        HumanMessage(content="第二问"),
    ]
    model_input = build_model_input(messages, max_tokens=10**9)
    # 冻结段在头部且只出现一次（历史副本剔除）。
    assert str(model_input[0].content).startswith(MEMORY_CONTEXT_FROZEN_PREFIX)
    assert (
        sum(1 for m in model_input if str(m.content).startswith(MEMORY_CONTEXT_FROZEN_PREFIX)) == 1
    )
    # 第一轮召回（历史注入）剔除；本轮召回保留在 human 之前。
    assert not any("第一轮召回" in str(m.content) for m in model_input)
    assert any("本轮召回" in str(m.content) for m in model_input)
    assert model_input.index(
        next(m for m in model_input if "本轮召回" in str(m.content))
    ) < model_input.index(messages[-1])
    # 非注入消息一个不少、顺序不乱。
    assert [m for m in model_input if not is_injected_system(m)] == [
        HumanMessage(content="第一问"),
        AIMessage(content="第一答"),
        HumanMessage(content="第二问"),
    ]


def test_build_model_input_frozen_head_survives_budget() -> None:
    """预算修剪只作用于历史体；冻结头部始终保留。"""
    messages = [
        SystemMessage(content=MEMORY_CONTEXT_FROZEN_PREFIX + "会话固定段"),
        HumanMessage(content="旧" * 500),
        AIMessage(content="答" * 500),
        HumanMessage(content="新问题"),
    ]
    model_input = build_model_input(messages, max_tokens=100)
    assert str(model_input[0].content).startswith(MEMORY_CONTEXT_FROZEN_PREFIX)
    assert model_input[-1] == messages[-1]
    assert len(model_input) < len(messages)  # 历史体确实被修剪
