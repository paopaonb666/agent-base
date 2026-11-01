"""Agent 输出验证循环（ADR-006，循环工程 Loop 2）：评分契约与循环引擎。

成本治理（T4）回答"花多少钱"，本模块回答"值不值"：给 Agent 输出加
评分器（``Grader``），不合格则带反馈重试，重试耗尽返回**历史最佳**。
四个原语全部基座级、零业务绑定——模块按需取用，不强制接线（planner
已有自己的 check/replan；记忆形成管线有自己的封闭决策集）。

评分器两类：

- **确定性评分器**（正则/语法/结构检查等纯代码）直接实现 ``Grader``；
- **代理式评分器**用 ``LLMJudgeGrader``（LLM as a judge）：prompt 强制
  JSON verdict，解析失败一律 FAIL——宁可错杀不放水。
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class Grade(str, Enum):
    """三态评级（与外部方法论的 PASS/FAIL/NEEDS_REVISION 对齐）。"""

    PASS = "PASS"
    FAIL = "FAIL"
    NEEDS_REVISION = "NEEDS_REVISION"


@dataclass(frozen=True)
class GradingResult:
    """一次评分的结果：评级 + 分数 + 反馈 + 分量明细。"""

    grade: Grade
    score: float  # 0.0 - 1.0
    feedback: str = ""
    components: dict[str, float] = field(default_factory=dict)


class Grader(Protocol):
    """评分器的最小接口（确定性评分器直接实现）。"""

    async def check(self, output: str) -> GradingResult: ...


# agent_fn 契约：task 与可选的上轮反馈 → Agent 产出文本。
AgentFn = Callable[[str, str | None], Awaitable[str]]


@dataclass(frozen=True)
class VerificationOutcome:
    """验证循环的整体结果（含完整尝试历史）。"""

    output: str
    grade: Grade
    score: float
    attempts: int
    history: list[dict[str, Any]] = field(default_factory=list)


def _aggregate(
    results: Sequence[GradingResult], *, pass_threshold: float, require_all_pass: bool
) -> GradingResult:
    """聚合多个评分器的结果（算术平均分 + 全量反馈拼接）。

    语义（照外部方法论）：``require_all_pass=True`` 时全部 PASS 才 PASS；
    否则平均分 ≥ 阈值判 PASS、≥ 0.5 判 NEEDS_REVISION、余者 FAIL。
    """
    if not results:
        return GradingResult(grade=Grade.PASS, score=1.0, feedback="无评分器")
    score = sum(r.score for r in results) / len(results)
    if require_all_pass:
        grade = Grade.PASS if all(r.grade == Grade.PASS for r in results) else Grade.FAIL
    elif score >= pass_threshold:
        grade = Grade.PASS
    elif score >= 0.5:
        grade = Grade.NEEDS_REVISION
    else:
        grade = Grade.FAIL
    feedback = "\n".join(f"[{r.grade.value}] {r.feedback}" for r in results if r.feedback)
    return GradingResult(grade=grade, score=score, feedback=feedback)


class VerificationLoop:
    """验证循环引擎：执行 → 全评分器打分 → 聚合 → 反馈重试 / 最佳回退。

    用法（模块侧自行包装 agent_fn，见图工厂外的任何调用形态）::

        loop = VerificationLoop([my_grader], max_retries=3, pass_threshold=0.8)
        outcome = await loop.run("写一份总结", my_agent_fn)
    """

    def __init__(
        self,
        graders: Sequence[Grader],
        *,
        max_retries: int = 3,
        pass_threshold: float = 0.8,
        require_all_pass: bool = False,
    ) -> None:
        if not graders:
            raise ValueError("VerificationLoop 需要至少一个评分器")
        if max_retries < 1:
            raise ValueError(f"max_retries must be >= 1, got {max_retries}")
        if not (0.0 < pass_threshold <= 1.0):
            raise ValueError(f"pass_threshold must be in (0, 1], got {pass_threshold}")
        self.graders = list(graders)
        self.max_retries = max_retries
        self.pass_threshold = pass_threshold
        self.require_all_pass = require_all_pass

    async def run(self, task: str, agent_fn: AgentFn) -> VerificationOutcome:
        """跑一轮验证循环；失败轮的反馈拼进下一轮输入。"""
        history: list[dict[str, Any]] = []
        best_output: str = ""
        best_score = -1.0
        feedback: str | None = None
        for attempt in range(1, self.max_retries + 1):
            output = await agent_fn(task, feedback)
            graded: list[GradingResult] = []
            for grader in self.graders:
                try:
                    graded.append(await grader.check(output))
                except Exception as exc:  # 评分器异常按 FAIL 计，不中断其他评分器。
                    logger.warning("verification: 评分器 %s 异常：%s", type(grader).__name__, exc)
                    graded.append(
                        GradingResult(grade=Grade.FAIL, score=0.0, feedback=str(exc)[:200])
                    )
            aggregated = _aggregate(
                graded, pass_threshold=self.pass_threshold, require_all_pass=self.require_all_pass
            )
            history.append(
                {
                    "attempt": attempt,
                    "grade": aggregated.grade.value,
                    "score": aggregated.score,
                    "feedback": aggregated.feedback,
                }
            )
            if aggregated.score > best_score:
                best_score = aggregated.score
                best_output = output
            if aggregated.grade == Grade.PASS:
                return VerificationOutcome(
                    output=output,
                    grade=aggregated.grade,
                    score=aggregated.score,
                    attempts=attempt,
                    history=history,
                )
            feedback = f"上一轮输出未通过质量检查：\n{aggregated.feedback}\n请改进后重新完成。"
        # 重试耗尽：返回历史最佳（best_score 跟踪），评级如实标 FAIL。
        return VerificationOutcome(
            output=best_output,
            grade=Grade.FAIL,
            score=best_score,
            attempts=self.max_retries,
            history=history,
        )


_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")

_JUDGE_PROMPT = """你是一个输出质量评审专家。请依据下列标准评估给定输出的质量：

{rubric}

请输出 JSON（不要输出其他内容）：
{{
    "overall_score": 0.0,
    "verdict": "PASS|FAIL|NEEDS_REVISION",
    "feedback": "总体评价与改进建议"
}}

overall_score 取 0.0-1.0；verdict 只能取三个枚举值之一。
"""


class LLMJudgeGrader:
    """代理式评分器（LLM as a judge）：rubric 打分 + JSON verdict。

    ``chat`` 是任何带 ``ainvoke(messages)`` 的对象（与 ``ChatInvokerPort``
    同形——经构造注入，不绑具体模型类）。响应解析失败（非 JSON / 缺字段 /
    非法枚举）一律 FAIL——宁可错杀不放水（照外部方法论契约）。
    """

    def __init__(self, chat: Any, *, rubric: str, name: str = "llm_judge") -> None:
        if not rubric.strip():
            raise ValueError("LLMJudgeGrader 需要 rubric（评分标准）")
        self._chat = chat
        self._rubric = rubric
        self.name = name

    async def check(self, output: str) -> GradingResult:
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=_JUDGE_PROMPT.format(rubric=self._rubric)),
            HumanMessage(content=f"待评估输出：\n{output}"),
        ]
        try:
            response = await self._chat.ainvoke(messages)
            content = str(getattr(response, "content", response))
            match = _JSON_BLOCK_RE.search(content)
            if match is None:
                raise ValueError("响应中没有 JSON 对象")
            data = json.loads(match.group())
            verdict = Grade(str(data["verdict"]).upper())
            score = float(data["overall_score"])
            if not (0.0 <= score <= 1.0):
                raise ValueError(f"overall_score 超界：{score}")
            return GradingResult(
                grade=verdict,
                score=score,
                feedback=str(data.get("feedback", ""))[:500],
                components={"llm_judge": score},
            )
        except Exception as exc:
            logger.warning("verification: LLM 判官解析失败（按 FAIL 计）：%s", exc)
            return GradingResult(
                grade=Grade.FAIL,
                score=0.0,
                feedback=f"判官输出解析失败：{exc}"[:200],
            )


__all__ = [
    "AgentFn",
    "Grade",
    "Grader",
    "GradingResult",
    "LLMJudgeGrader",
    "VerificationLoop",
    "VerificationOutcome",
]
