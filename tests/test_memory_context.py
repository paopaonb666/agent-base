"""上下文组装与短期压缩（M6d）的测试：注入块组装 + 模型输入修剪。"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent_base.memory.context import (
    ATTACHMENT_CONTEXT_PREFIX,
    MEMORY_CONTEXT_PREFIX,
    build_model_input,
    compose_context,
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
    service = MemoryService(
        store=MemoryMemoryStore(), embedder=HashEmbedding(), settings=_settings()
    )
    await service.add_memory(user_id="alice", agent_id="chat", content="用户喜欢深色主题")
    await service.save_profile("alice", {"偏好": ["简洁回答"]})
    await service.store.upsert_block(
        MemoryBlock(user_id="alice", agent_id="chat", label="human", content="在开发记忆系统")
    )
    block = await service.compose_context(
        user_id="alice",
        agent_id="chat",
        thread_id="chat:t1",
        query="我该用什么主题？",
    )
    assert block is not None
    assert "深色主题" in block and "简洁回答" in block and "记忆系统" in block
    await service.aclose()


async def test_service_compose_context_recall_is_user_level() -> None:
    """注入召回为用户级：chat 写入的记忆在 supervisor 线程的注入块可见。"""
    service = MemoryService(
        store=MemoryMemoryStore(), embedder=HashEmbedding(), settings=_settings()
    )
    await service.add_memory(user_id="alice", agent_id="chat", content="用户的项目代号是雨燕")
    block = await service.compose_context(
        user_id="alice",
        agent_id="supervisor",
        thread_id="supervisor:t1",
        query="项目代号",
    )
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
