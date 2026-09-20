"""记忆形成管线（M6c）：抽取 → 整合 → 画像合并 → 滚动摘要。

四件事都由 LLM 驱动（复用运行时的对话模型），全部在对话响应完成后的
后台任务里执行（fire-and-forget），每一步落 ``memory_ops`` 审计：

- **抽取**（mem0 式）：从最近对话转写中提取候选事实（JSON 列表），
  候选只收"值得跨会话记住"的信息——身份/偏好/约束/事实/经验教训；
- **整合**：对每条候选检索相似既有记忆，让 LLM 决策
  ``ADD / UPDATE / DELETE / NONE``，保持记忆库的一致性（去重、纠错、
  失效剔除）而不是无限堆砌；
- **画像合并**（Memobase 式）：把新事实合并进结构化用户画像 JSON
  （memories 表里的确定性记录，``profile:<user_id>``）；
- **滚动摘要**：把旧摘要与新对话合并为更新的会话摘要，供 M6d 的
  短期压缩把超预算的旧历史替换掉。

鲁棒性约定：LLM 输出的 JSON 解析失败、决策引用不存在的 memory_id、
以及任何异常，都只记审计与日志，绝不向上抛——记忆形成是锦上添花，
主对话流永远优先。
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage

from agent_base.extensions.metrics import MEMORY_METRICS
from agent_base.memory.store import KNOWN_MEMORY_KINDS

if TYPE_CHECKING:  # pragma: no cover - import avoided at runtime
    from agent_base.core.config import Settings
    from agent_base.memory.service import MemoryService

logger = logging.getLogger(__name__)

# 整合决策的封闭集合（mem0 语义）。
DECISION_OPS: tuple[str, ...] = ("ADD", "UPDATE", "DELETE", "NONE")

# 画像的默认分节（Memobase 式结构化画像）。LLM 可以增删分节，这只是
# 首次建画像时的骨架。
PROFILE_SECTIONS: tuple[str, ...] = ("基本信息", "偏好", "工作背景", "沟通风格", "长期目标")


@dataclass(frozen=True)
class Candidate:
    """抽取管线产出的一条候选记忆。"""

    content: str
    kind: str = "semantic"
    salience: float = 0.5
    tags: list[str] = field(default_factory=list)


# ───────────────────────────── Prompt 常量 ─────────────────────────────

EXTRACTION_PROMPT = """你是记忆抽取器。从下面的对话转写中提取"值得跨会话记住"的信息。

只提取这些类别：
- 用户的身份、背景、长期目标（semantic）
- 用户的偏好、习惯、约束（semantic）
- 对话中发生的重要事件与结论（episodic）
- 用户明确要求记住的规则或工作方式（procedural）

不要提取：闲聊、寒暄、一次性的临时请求、对话本身的字面内容。
没有值得记的就返回空数组。宁缺毋滥，最多 5 条。

输出纯 JSON 数组（不要 markdown 围栏、不要解释），每条：
{{"content": "第三人称、自包含的一句话",
  "kind": "semantic|episodic|procedural", "salience": 0.0到1.0, "tags": ["标签"]}}

对话转写：
{transcript}"""

CONSOLIDATION_PROMPT = """你是记忆整合器。新候选记忆要与既有记忆保持一致：去重、纠错、剔除失效。

新候选：
{candidate}

既有相似记忆（id: 内容）：
{existing}

决策规则：
- 既有记忆已覆盖候选信息 → NONE（或内容更完整时 UPDATE 该条）
- 候选是既有记忆的更新/修正 → UPDATE（给出修正后的完整 content）
- 候选与某条既有记忆直接矛盾（信息已失效，如"已换工作"）→ DELETE 该条；
  只 DELETE 还是"DELETE+ADD"按信息价值判断
- 全新的信息 → ADD

输出纯 JSON（不要 markdown 围栏、不要解释）：
{{"op": "ADD|UPDATE|DELETE|NONE",
  "memory_id": "仅 UPDATE/DELETE 需要", "content": "仅 UPDATE/ADD 的最终内容"}}"""

PROFILE_MERGE_PROMPT = """你是用户画像维护器。把新事实合并进既有画像 JSON。

规则：
- 保留仍然有效的既有内容；与新事实矛盾的以新事实为准
- 画像条目要简短、第三人称、可执行
- 新事实与画像无关时保持画像不变（只格式化）
- 总长度控制在 600 字以内

既有画像 JSON：
{profile}

本轮新事实（逐条）：
{facts}

输出合并后的完整画像 JSON（不要 markdown 围栏、不要解释）。"""

SUMMARY_PROMPT = """你是会话摘要器。把既有摘要与新对话合并为一份更新的滚动摘要。

规则：
- 摘要覆盖"到目前为止这个会话聊过什么"：主题、结论、未决事项、
  用户表达的关键诉求
- 既有摘要中仍然成立的内容保留；被新对话推翻的以新对话为准
- 第三人称、简洁，不超过 300 字

既有摘要（可能为空）：
{prior}

最近对话转写：
{transcript}

输出新摘要的纯文本（不要 JSON、不要标题、不要解释）。"""


def _parse_json_block(text: str) -> Any:
    """从容解析 LLM 输出里的 JSON：剥 markdown 围栏、取首个平衡块。"""
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        return json.loads(cleaned)
    except ValueError:
        pass
    # 兜底：截取第一个 [ 或 { 到与之匹配的最后一个 ] 或 }。
    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        start = cleaned.find(open_ch)
        end = cleaned.rfind(close_ch)
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except ValueError:
                continue
    raise ValueError(f"无法从 LLM 输出解析 JSON：{cleaned[:200]}")


def _content_text(response: Any) -> str:
    """从模型响应提取纯文本（列表型 content 拼接 text 段）。"""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        return "".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content
        )
    return str(content)


def render_transcript(messages: list[Any], max_chars: int) -> str:
    """把 LangChain 消息渲染成抽取/摘要用的对话转写（保尾部，超限截断）。"""
    lines: list[str] = []
    for message in messages:
        mtype = getattr(message, "type", "")
        text = _content_text(getattr(message, "content", "")).strip()
        if not text:
            continue
        if mtype == "human":
            lines.append(f"用户：{text}")
        elif mtype == "ai":
            # 纯 tool_calls 的 AI 消息（无文本）没有转写价值。
            lines.append(f"助手：{text}")
        elif mtype == "tool":
            lines.append(f"工具[{getattr(message, 'name', '')}]：{text[:300]}")
    transcript = "\n".join(lines)
    if len(transcript) <= max_chars:
        return transcript
    # 保尾部：最近的内容对记忆形成更重要。
    return "……（前文截断）\n" + transcript[-max_chars:]


class MemoryPipeline:
    """LLM 驱动的记忆形成管线；所有步骤失败安全（绝不打断主对话）。

    模型分派（成本治理）：抽取/画像合并/摘要属于"简单、大量、重复"的
    批量工作，走快档（``fast_llm``——``LLM_FAST_*`` 装配的限流回退包装）；
    整合裁决是唯一会不可逆改动记忆库的破坏性决策步，默认永远走主力
    模型（``pipeline_profile="split"``）。``fast_llm`` 未装配时一切回落
    主力模型——快档是优化不是依赖。
    """

    def __init__(
        self, llm: Any, settings: Settings, service: MemoryService, fast_llm: Any | None = None
    ) -> None:
        self._llm = llm
        self._fast_llm = fast_llm
        self._settings = settings
        self._service = service

    def _llm_for(self, profile: str) -> Any:
        """按档位取模型；快档缺席时回落主力（不报错）。"""
        if profile == "fast":
            return self._fast_llm if self._fast_llm is not None else self._llm
        return self._llm

    @property
    def _bulk_profile(self) -> str:
        """抽取/画像/摘要的档位：split 与 all_fast 都交给快档侧。"""
        return "main" if self._settings.memory.pipeline_profile == "all_main" else "fast"

    @property
    def _decision_profile(self) -> str:
        """整合裁决的档位：只有显式 all_fast 才交给快档侧。"""
        return "fast" if self._settings.memory.pipeline_profile == "all_fast" else "main"

    async def _complete(self, prompt: str, *, profile: str = "main") -> str:
        response = await self._llm_for(profile).ainvoke([HumanMessage(content=prompt)])
        return _content_text(response)

    # -- 抽取 -----------------------------------------------------------------
    async def extract_candidates(self, transcript: str) -> list[Candidate]:
        raw = await self._complete(
            EXTRACTION_PROMPT.format(transcript=transcript), profile=self._bulk_profile
        )
        parsed = _parse_json_block(raw)
        if not isinstance(parsed, list):
            raise ValueError("抽取输出不是 JSON 数组")
        candidates: list[Candidate] = []
        for item in parsed[:5]:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            kind = str(item.get("kind", "semantic"))
            if kind not in KNOWN_MEMORY_KINDS:
                kind = "semantic"
            try:
                salience = min(max(float(item.get("salience", 0.5)), 0.0), 1.0)
            except (TypeError, ValueError):
                salience = 0.5
            tags = item.get("tags")
            candidates.append(
                Candidate(
                    content=content,
                    kind=kind,
                    salience=salience,
                    tags=[str(t) for t in tags][:10] if isinstance(tags, list) else [],
                )
            )
        return candidates

    # -- 整合 -----------------------------------------------------------------
    async def consolidate(
        self, candidate: Candidate, *, user_id: str, agent_id: str, thread_id: str
    ) -> str:
        """对一条候选拿到整合决策并落库；返回实际执行的 op。"""
        similar = await self._service.search(
            user_id=user_id, agent_id=agent_id, query=candidate.content, top_k=3
        )
        existing_block = (
            "\n".join(f"{item.record.memory_id}: {item.record.content}" for item in similar)
            or "（无）"
        )
        raw = await self._complete(
            CONSOLIDATION_PROMPT.format(
                candidate=json.dumps(
                    {"content": candidate.content, "kind": candidate.kind}, ensure_ascii=False
                ),
                existing=existing_block,
            ),
            profile=self._decision_profile,
        )
        parsed = _parse_json_block(raw)
        if not isinstance(parsed, dict):
            raise ValueError("整合输出不是 JSON 对象")
        op = str(parsed.get("op", "NONE")).upper()
        if op not in DECISION_OPS:
            op = "NONE"
        memory_id = parsed.get("memory_id")
        content = parsed.get("content")
        if op == "ADD":
            await self._service.add_memory(
                user_id=user_id,
                agent_id=agent_id,
                content=str(content or candidate.content),
                kind=candidate.kind,
                tags=candidate.tags,
                salience=candidate.salience,
                source_thread_id=thread_id,
            )
        elif op == "UPDATE":
            if not memory_id:
                op = "NONE"
            else:
                updated = await self._service.update_memory(
                    str(memory_id), content=str(content or candidate.content)
                )
                if updated is None:
                    op = "NONE"
        elif op == "DELETE":
            if not memory_id:
                op = "NONE"
            else:
                deleted = await self._service.delete_memory(str(memory_id))
                if not deleted:
                    op = "NONE"
        return op

    # -- 画像 -----------------------------------------------------------------
    async def merge_profile(self, user_id: str, facts: list[str]) -> bool:
        """把新事实合并进用户画像；画像不存在时先建骨架。返回是否更新。"""
        if not facts:
            return False
        profile = await self._service.get_profile(user_id) or {
            section: [] for section in PROFILE_SECTIONS
        }
        raw = await self._complete(
            PROFILE_MERGE_PROMPT.format(
                profile=json.dumps(profile, ensure_ascii=False, indent=1),
                facts="\n".join(f"- {fact}" for fact in facts),
            ),
            profile=self._bulk_profile,
        )
        merged = _parse_json_block(raw)
        if not isinstance(merged, dict) or not merged:
            raise ValueError("画像合并输出不是 JSON 对象")
        await self._service.save_profile(user_id, merged)
        return True

    # -- 滚动摘要 ----------------------------------------------------------------
    async def update_summary(
        self, user_id: str, agent_id: str, thread_id: str, transcript: str
    ) -> bool:
        """把既有摘要与最新对话合并为新的滚动摘要。"""
        prior_summary = await self._service.store.get_summary(user_id, thread_id)
        raw = await self._complete(
            SUMMARY_PROMPT.format(
                prior=prior_summary.summary if prior_summary else "（空）",
                transcript=transcript,
            ),
            profile=self._bulk_profile,
        )
        text = raw.strip()
        if not text:
            return False
        from agent_base.memory.store import SessionSummary

        await self._service.store.upsert_summary(
            SessionSummary(
                user_id=user_id,
                thread_id=thread_id,
                agent_id=agent_id,
                summary=text[:1000],
                covered_message_count=prior_summary.covered_message_count if prior_summary else 0,
            )
        )
        return True

    # -- 编排 ---------------------------------------------------------------------
    async def capture_turn(
        self,
        *,
        user_id: str,
        agent_id: str,
        thread_id: str,
        transcript: str,
        human_count: int,
        force: bool = False,
    ) -> dict[str, Any]:
        """一轮对话结束后的完整形成流程；每步独立审计、失败安全。

        ``force=True`` 旁路 ``MEMORY_CAPTURE_ENABLED`` 门控（夜间批脚本
        T3.2 的替代形成路径：每轮管线关闭时，一天一次批量抽取+整合）；
        服务端每轮路径永远用默认 False。
        """
        detail: dict[str, Any] = {
            "candidates": 0,
            "ops": {},
            "profile_updated": False,
            "summary_updated": False,
            "forced": force,
        }

        async def _run(op: str, coro: Any) -> Any:
            started = time.perf_counter()
            try:
                result = await coro
            except Exception as exc:
                duration_ms = int((time.perf_counter() - started) * 1000)
                logger.warning("memory: %s 失败（不影响对话）：%s", op, exc)
                await self._service.record_op(
                    op=op,
                    user_id=user_id,
                    agent_id=agent_id,
                    thread_id=thread_id,
                    status="error",
                    error_text=f"{type(exc).__name__}: {exc}"[:300],
                    duration_ms=duration_ms,
                    detail={"human_count": human_count},
                )
                MEMORY_METRICS.observe(op, "error", time.perf_counter() - started)
                return None
            duration_ms = int((time.perf_counter() - started) * 1000)
            if isinstance(result, dict):
                detail = result
            elif isinstance(result, list):
                # 候选记忆等数据类列表：取可读摘要（Candidate.content 等）。
                detail = {"result": [getattr(item, "content", str(item))[:120] for item in result]}
            else:
                detail = {"result": result}
            await self._service.record_op(
                op=op,
                user_id=user_id,
                agent_id=agent_id,
                thread_id=thread_id,
                duration_ms=duration_ms,
                detail=detail,
            )
            MEMORY_METRICS.observe(op, "ok", time.perf_counter() - started)
            return result

        # 1) 抽取 + 逐条整合（合并计为一次 extract 审计 + 各 consolidate）。
        if force or self._settings.memory.capture_enabled:
            extracted = await _run("extract", self.extract_candidates(transcript))
            if extracted:
                detail["candidates"] = len(extracted)
                accepted: list[str] = []
                for candidate in extracted:
                    op = await _run(
                        "consolidate",
                        self.consolidate(
                            candidate, user_id=user_id, agent_id=agent_id, thread_id=thread_id
                        ),
                    )
                    if op is not None:
                        detail["ops"][candidate.content[:50]] = op
                        if op in ("ADD", "UPDATE"):
                            accepted.append(candidate.content)
                # 2) 画像合并（用被接受的记忆作为事实源）。
                if self._settings.memory.profile_enabled and accepted:
                    updated = await _run("profile", self.merge_profile(user_id, accepted))
                    detail["profile_updated"] = bool(updated)

        # 3) 滚动摘要（独立于抽取：即使本轮不抽取，长会话也需要摘要）。
        if (
            self._settings.memory.summary_enabled
            and human_count >= self._settings.memory.summary_trigger_messages
        ):
            summary_updated = await _run(
                "summary", self.update_summary(user_id, agent_id, thread_id, transcript)
            )
            detail["summary_updated"] = bool(summary_updated)
        return detail


__all__ = [
    "DECISION_OPS",
    "EXTRACTION_PROMPT",
    "PROFILE_SECTIONS",
    "Candidate",
    "MemoryPipeline",
    "render_transcript",
]
