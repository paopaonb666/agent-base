"""验证循环（ADR-006，Loop 2）的测试：聚合语义、重试与最佳回退、LLM 判官解析。"""

from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from agent_base.core.verification import (
    Grade,
    GradingResult,
    LLMJudgeGrader,
    VerificationLoop,
)


class StaticGrader:
    """按脚本序列返回结果的确定性评分器替身。"""

    def __init__(self, results: list[GradingResult]) -> None:
        self.results = results
        self.calls: list[str] = []

    async def check(self, output: str) -> GradingResult:
        self.calls.append(output)
        return self.results.pop(0)


def _result(grade: Grade, score: float, feedback: str = "") -> GradingResult:
    return GradingResult(grade=grade, score=score, feedback=feedback)


# ─────────────────────────── 构造校验 ───────────────────────────


def test_loop_rejects_bad_config() -> None:
    grader = StaticGrader([])
    with pytest.raises(ValueError):
        VerificationLoop([])
    with pytest.raises(ValueError):
        VerificationLoop([grader], max_retries=0)
    with pytest.raises(ValueError):
        VerificationLoop([grader], pass_threshold=0)
    with pytest.raises(ValueError):
        VerificationLoop([grader], pass_threshold=1.5)
    with pytest.raises(ValueError):
        LLMJudgeGrader(chat=object(), rubric="  ")


# ─────────────────────────── run 语义 ───────────────────────────


async def test_first_pass_returns_immediately() -> None:
    grader = StaticGrader([_result(Grade.PASS, 0.95, "不错")])
    loop = VerificationLoop([grader], max_retries=3)

    async def agent(task: str, feedback: str | None) -> str:
        return "产出"

    outcome = await loop.run("任务", agent)
    assert outcome.grade == Grade.PASS
    assert outcome.attempts == 1
    assert outcome.score == pytest.approx(0.95)
    assert len(grader.calls) == 1


async def test_retry_feedback_passed_to_agent_then_pass() -> None:
    grader = StaticGrader(
        [
            _result(Grade.FAIL, 0.2, "缺少要点"),
            _result(Grade.PASS, 0.9, "补齐了"),
        ]
    )
    loop = VerificationLoop([grader], max_retries=3)
    received: list[tuple[str, str | None]] = []

    async def agent(task: str, feedback: str | None) -> str:
        received.append((task, feedback))
        return f"产出{len(received)}"

    outcome = await loop.run("任务", agent)
    assert outcome.grade == Grade.PASS
    assert outcome.attempts == 2
    assert received[0][1] is None
    assert "缺少要点" in (received[1][1] or "")


async def test_exhausted_returns_best_attempt() -> None:
    """重试耗尽：返回历史最佳产出（不是最后一轮），评级如实 FAIL。"""
    grader = StaticGrader(
        [
            _result(Grade.FAIL, 0.6, "第一轮"),
            _result(Grade.FAIL, 0.1, "更差了"),
            _result(Grade.FAIL, 0.3, "还是不行"),
        ]
    )
    loop = VerificationLoop([grader], max_retries=3)

    outputs = ["产出1", "产出2", "产出3"]

    async def agent(task: str, feedback: str | None) -> str:
        return outputs.pop(0)

    outcome = await loop.run("任务", agent)
    assert outcome.grade == Grade.FAIL
    assert outcome.output == "产出1"  # 0.6 是历史最佳
    assert outcome.score == pytest.approx(0.6)
    assert outcome.attempts == 3
    assert [h["attempt"] for h in outcome.history] == [1, 2, 3]


async def test_grader_exception_counts_as_fail() -> None:
    class ExplodingGrader:
        async def check(self, output: str) -> GradingResult:
            raise RuntimeError("boom")

    loop = VerificationLoop([ExplodingGrader()], max_retries=1)

    async def agent(task: str, feedback: str | None) -> str:
        return "产出"

    outcome = await loop.run("任务", agent)
    assert outcome.grade == Grade.FAIL
    assert outcome.score == pytest.approx(0.0)
    assert "boom" in outcome.history[0]["feedback"]


# ─────────────────────────── 聚合语义 ───────────────────────────


async def test_aggregate_average_and_threshold() -> None:
    """多评分器算术平均：0.9+0.7=0.8 ≥ 阈值 0.8 → PASS。"""
    graders = [
        StaticGrader([_result(Grade.NEEDS_REVISION, 0.9, "a")]),
        StaticGrader([_result(Grade.NEEDS_REVISION, 0.7, "b")]),
    ]
    loop = VerificationLoop(graders, pass_threshold=0.8)

    async def agent(task: str, feedback: str | None) -> str:
        return "x"

    outcome = await loop.run("t", agent)
    assert outcome.grade == Grade.PASS
    assert outcome.score == pytest.approx(0.8)


async def test_aggregate_require_all_pass() -> None:
    """require_all_pass：任一非 PASS 即 FAIL——即使平均分过线。"""
    graders = [
        StaticGrader([_result(Grade.PASS, 1.0)]),
        StaticGrader([_result(Grade.NEEDS_REVISION, 0.7)]),
    ]
    loop = VerificationLoop(graders, pass_threshold=0.5, require_all_pass=True)

    async def agent(task: str, feedback: str | None) -> str:
        return "x"

    outcome = await loop.run("t", agent)
    assert outcome.grade == Grade.FAIL


async def test_aggregate_needs_revision_band() -> None:
    """平均分落在 [0.5, 阈值) → NEEDS_REVISION（不算通过，触发重试）。"""
    grader = StaticGrader([_result(Grade.NEEDS_REVISION, 0.6)] * 3)
    loop = VerificationLoop([grader], max_retries=2, pass_threshold=0.8)

    async def agent(task: str, feedback: str | None) -> str:
        return "x"

    outcome = await loop.run("t", agent)
    assert outcome.grade == Grade.FAIL  # 重试耗尽后标 FAIL
    assert outcome.attempts == 2


# ─────────────────────────── LLM 判官 ───────────────────────────


class ScriptedChat:
    """带 ainvoke 的最小判官模型替身。"""

    def __init__(self, reply: Any) -> None:
        self.reply = reply
        self.received: list[Any] = []

    async def ainvoke(self, messages: Any) -> Any:
        self.received = list(messages)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


async def test_llm_judge_parses_verdict() -> None:
    chat = ScriptedChat(
        AIMessage(
            content=json.dumps(
                {"overall_score": 0.85, "verdict": "PASS", "feedback": "结构完整"},
                ensure_ascii=False,
            )
        )
    )
    grader = LLMJudgeGrader(chat, rubric="1. 结构完整\n2. 事实准确")
    result = await grader.check("输出")
    assert result.grade == Grade.PASS
    assert result.score == pytest.approx(0.85)
    assert result.components["llm_judge"] == pytest.approx(0.85)
    # 判官收到 rubric 系统提示 + 待评估输出。
    assert "事实准确" in str(chat.received[0].content)
    assert "输出" in str(chat.received[1].content)


async def test_llm_judge_fences_and_prose() -> None:
    fenced = (
        "评审如下：\n```json\n"
        '{"overall_score": 0.3, "verdict": "FAIL", "feedback": "缺引用"}\n```\n以上。'
    )
    chat = ScriptedChat(AIMessage(content=fenced))
    grader = LLMJudgeGrader(chat, rubric="有引用")
    result = await grader.check("输出")
    assert result.grade == Grade.FAIL
    assert result.score == pytest.approx(0.3)


async def test_llm_judge_malformed_fails_closed() -> None:
    for bad in ("完全不是 JSON", '{"overall_score": 2.0, "verdict": "PASS"}', '{"score": 1}'):
        chat = ScriptedChat(AIMessage(content=bad))
        grader = LLMJudgeGrader(chat, rubric="r")
        result = await grader.check("输出")
        assert result.grade == Grade.FAIL, bad
        assert result.score == 0.0


async def test_llm_judge_chat_exception_fails_closed() -> None:
    chat = ScriptedChat(RuntimeError("network down"))
    grader = LLMJudgeGrader(chat, rubric="r")
    result = await grader.check("输出")
    assert result.grade == Grade.FAIL


async def test_llm_judge_in_loop_end_to_end() -> None:
    """判官与循环的组合冒烟：首轮 FAIL → 反馈 → 次轮 PASS。"""
    verdicts = [
        AIMessage(
            content=json.dumps({"overall_score": 0.2, "verdict": "FAIL", "feedback": "太短"})
        ),
        AIMessage(content=json.dumps({"overall_score": 0.9, "verdict": "PASS", "feedback": "好"})),
    ]

    class QueueChat:
        async def ainvoke(self, messages: Any) -> Any:
            return verdicts.pop(0)

    loop = VerificationLoop([LLMJudgeGrader(QueueChat(), rubric="长度 ≥ 100 字")], max_retries=2)

    async def agent(task: str, feedback: str | None) -> str:
        return "足够长的产出" if feedback else "短"

    outcome = await loop.run("写总结", agent)
    assert outcome.grade == Grade.PASS
    assert outcome.attempts == 2
