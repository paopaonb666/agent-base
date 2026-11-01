"""共享图工厂（P1-4 / H3）：单 LLM 节点 + 可选工具循环的公共装配。

chat / writer / hello 这类"单 LLM 节点（可选工具循环）"模块过去各自
复制同一套 ``StateGraph → ToolNode → compile`` 样板，且上下文工程
（注入去重 + token 预算修剪）只在 chat 里生效——writer/hello 的历史里
注入块逐轮累积，模型输入持续膨胀。工厂把样板与上下文工程一并收敛：

- 模块只声明 ``name`` / ``system_prompt`` / ``tools``，几行即可接入；
- 发给模型的输入统一经过 ``core.context.build_model_input``（注入去重
  + 预算修剪，工具配对不拆散）——**对所有模块生效**；
- 模块级 ``system_prompt`` 是临时拼装（不落库、不参与修剪去重）；
- 流式累积保持不变：SSE 层看到 token 级 delta，取消传播到 LLM 请求；
- 审批门（ADR-005，HITL）：``TOOL_APPROVAL_ENABLED`` 且名单非空时，
  工具节点换成带 interrupt 的审批执行器；未启用时保持 ToolNode 原状，
  行为零变化。

需要自定义拓扑的模块保留 escape hatch：直接使用 ``core.context`` 与
LangGraph 原生 API 自行组装。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from typing import Any, cast

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt

from agent_base.core.context import build_model_input
from agent_base.core.contracts import Graph, ModuleContext
from agent_base.core.tools import handle_tool_error
from agent_base.core.tool_routing import recent_tool_names

logger = logging.getLogger(__name__)

# 错误归一的统一签名：(异常, 工具名?) → ToolMessage 文本。第二参带默认：
# ToolNode 回调只传异常，审批门/planner 收口知道工具名、按位置传入。
# 有状态实现（ToolErrorReporter，ADR-009）需要工具名执行修复预算。
ToolErrorFunc = Callable[..., str]


def _stateless_tool_error(exc: Exception, tool: str = "") -> str:
    """未装配韧性层的兜底（语义与阶段 3 的 handle_tool_error 一致）。"""
    return handle_tool_error(exc)


def _last_user_text(messages: Sequence[Any]) -> str:
    """最近一条用户消息文本——路由与 few-shot 的语义查询源。

    多模态 content（list）直接 str 化：嵌入端有自己的截断策略；
    没有用户消息（理论上只在异常重放里）返回空串，调用方各自降级。
    """
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


async def _stream_and_accumulate(model: Any, model_input: list[Any]) -> AIMessage | None:
    """流式调用模型并累积分块；显式归一为 AIMessage。

    AIMessageChunk 的 tool_calls 由 tool_call_chunks 惰性聚合，分块
    不一致时 ToolNode 解析失败、循环提前终止（症状：工具只调一次）。
    累积完成后显式构造 AIMessage，让 tool_calls 完成聚合校验，图状态
    （checkpointer 历史 / 序列化 / ToolNode）只见到完整消息。
    """
    final: AIMessageChunk | None = None
    async for chunk in model.astream(model_input):
        # astream 声明的产出类型是消息联合体；运行时的分块是
        # AIMessageChunk（仅对话模型）。
        piece = cast(AIMessageChunk, chunk)
        final = piece if final is None else final + piece
    if final is None:
        return None
    return AIMessage(
        content=final.content,
        tool_calls=final.tool_calls,
        id=final.id,
        response_metadata=final.response_metadata,
        usage_metadata=final.usage_metadata,
    )


def make_approval_tool_executor(
    pool: Sequence[BaseTool],
    approval_names: frozenset[str],
    error_handler: ToolErrorFunc | None = None,
) -> Any:
    """带审批门的工具执行节点（ADR-005）：命中名单的调用先 interrupt。

    执行语义与 ToolNode 对齐（并发执行 + 异常归一成 ToolMessage 反馈），
    差别仅在执行前对命中名单的 tool_call 做一次 interrupt——暂停点由
    checkpointer 持久化，``/resume`` 端点经 ``Command(resume=...)`` 恢复。

    恢复值契约：``{call_id: {"action": "approve"|"reject"|"modify",
    "args": {...}}}``——approve 放行原参数、reject 不执行并注入拒绝
    ToolMessage、modify 以新参数执行。同轮多个待审批调用只 interrupt
    一次（载荷携带全部待审批项），一次 resume 携带全部决策。

    ``error_handler``（ADR-009）是错误归一收口，签名 (异常, 工具名)；
    缺省退回无状态 ``handle_tool_error``（阶段 3 语义）。

    刻意不改写 AIMessage 的 tool_calls（消息序列保持 [ai, tool_msgs…]
    的合法形态）：拒绝的调用同样产出 ToolMessage，模型看到拒绝原因后
    自行决定下一步。
    """

    handler = error_handler or _stateless_tool_error

    tool_map = {tool.name: tool for tool in pool}

    async def execute_with_approval(state: MessagesState, config: RunnableConfig) -> dict[str, Any]:
        last = state["messages"][-1]
        calls: list[dict[str, Any]] = list(getattr(last, "tool_calls", None) or [])
        pending = [tc for tc in calls if str(tc.get("name")) in approval_names]
        decisions: dict[str, dict[str, Any]] = {}
        if pending:
            decision_value: Any = interrupt(
                {
                    "pending": [
                        {
                            "tool_call_id": str(tc.get("id")),
                            "tool": str(tc.get("name")),
                            "args": dict(tc.get("args") or {}),
                        }
                        for tc in pending
                    ]
                }
            )
            if isinstance(decision_value, dict):
                decisions = {
                    str(call_id): dict(decision)
                    for call_id, decision in decision_value.items()
                    if isinstance(decision, dict)
                }
            logger.info(
                "approval: 工具审批决策（thread=%s）：%s",
                (config.get("configurable") or {}).get("thread_id", ""),
                {cid: d.get("action") for cid, d in decisions.items()},
            )

        async def _run_one(tc: dict[str, Any]) -> ToolMessage:
            call_id = str(tc.get("id"))
            tool_name = str(tc.get("name"))
            decision = decisions.get(call_id)
            if decision is not None and decision.get("action") == "reject":
                return ToolMessage(
                    content=f"[审批] 人工拒绝执行工具 {tool_name}：请改用其他方式满足请求。",
                    tool_call_id=call_id,
                    name=tool_name,
                )
            args = dict(tc.get("args") or {})
            if (
                decision is not None
                and decision.get("action") == "modify"
                and isinstance(decision.get("args"), dict)
            ):
                args = dict(decision["args"])
            tool = tool_map.get(tool_name)
            if tool is None:
                return ToolMessage(
                    content=f"tool execution failed: unknown tool {tool_name!r}",
                    tool_call_id=call_id,
                    name=tool_name,
                )
            try:
                result = await tool.ainvoke(
                    {"name": tool_name, "args": args, "id": call_id, "type": "tool_call"},
                    config,
                )
                if isinstance(result, ToolMessage):
                    return result
                return ToolMessage(content=str(result), tool_call_id=call_id, name=tool_name)
            except Exception as exc:
                return ToolMessage(
                    content=handler(exc, tool_name), tool_call_id=call_id, name=tool_name
                )

        results = await asyncio.gather(*(_run_one(tc) for tc in calls))
        return {"messages": list(results)}

    return execute_with_approval


def build_single_agent_graph(
    ctx: ModuleContext,
    *,
    name: str,
    system_prompt: str | None = None,
    tools: Sequence[BaseTool] | None = None,
) -> Graph:
    """构建并编译"单 LLM 节点 + 可选工具循环"的图。

    ``tools`` 缺省用共享工具池（``ctx.tools``）；传空列表可显式禁用
    工具循环。图以 ``name=<模块名>`` 编译——supervisor 要求每个
    sub-agent 图携带其模块名。

    三个可选增强（各自 None 时行为回到阶段 3，逐字节一致）：

    - ``ctx.tool_resilience``（ADR-009）——工具节点的错误归一换成有
      状态 reporter（结构化参数反馈 + 修复预算）；
    - ``ctx.tool_router``（ADR-010）——active 时**每轮**按查询语义选择
      绑定子集（basic 全集 ∪ top-k advanced ∪ 近轮已用），inactive 时
      保持建图时的全量绑定快路径；
    - ``ctx.tool_fewshot``（ADR-011）——按查询相似度动态选样，作为
      追加的临时 SystemMessage 注入（不落库、不参与修剪去重）。
    """
    pool = list(ctx.tools) if tools is None else list(tools)
    model = ctx.llm.bind_tools(pool) if pool else ctx.llm
    max_tokens = ctx.settings.memory.context_max_tokens
    approval_names = _resolve_approval(ctx)
    reporter = ctx.tool_resilience.reporter if ctx.tool_resilience is not None else None
    error_handler: ToolErrorFunc = reporter.handle if reporter is not None else _stateless_tool_error
    router = ctx.tool_router
    fewshot = ctx.tool_fewshot
    recent_rounds = ctx.settings.toolkit.routing_recent_rounds

    async def call_model(state: MessagesState, config: RunnableConfig) -> dict[str, Any]:
        # 上下文工程（H3）：注入去重 + token 预算修剪，对所有模块生效。
        model_input = build_model_input(state["messages"], max_tokens=max_tokens)
        if system_prompt is not None:
            # 模块提示词临时拼装在最前：不落库，也不属于"注入型系统消息"。
            model_input = [SystemMessage(content=system_prompt), *model_input]
        query = _last_user_text(state["messages"])
        # 向量路由（ADR-010）：active 时每轮动态绑定；降级路径在
        # router.select 内部收口（任何失败回全量池）。
        bound = model
        bound_names = [t.name for t in pool]
        if router is not None and router.active and pool:
            recent = recent_tool_names(state["messages"], rounds=recent_rounds)
            selected = await router.select(query, recent_tools=recent)
            bound = ctx.llm.bind_tools(selected) if selected else ctx.llm
            bound_names = [t.name for t in (selected or [])]
        if fewshot is not None and bound_names:
            # few-shot（ADR-011）在路由之后：只注入本轮实际绑定了的工具
            # 的示例——教模型调不在绑定集里的工具等于教它幻觉。
            segment = await fewshot.render_segment(query, available=bound_names)
            if segment:
                # 与模块提示词同规则：临时拼装、不落库、不参与去重。
                model_input = [SystemMessage(content=segment), *model_input]
        final = await _stream_and_accumulate(bound, model_input)
        if final is None:
            return {"messages": []}
        return {"messages": [final]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    if pool:
        if approval_names:
            graph.add_node(
                "tools", make_approval_tool_executor(pool, approval_names, error_handler)
            )
        else:
            # 错误归一收口把工具失败转成 ToolMessage 反馈，坏掉的工具
            # 绝不会打断对话（阶段 3 验收；ADR-009 起优先走有状态版）。
            graph.add_node("tools", ToolNode(pool, handle_tool_errors=error_handler))
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", tools_condition)
        graph.add_edge("tools", "agent")
    else:
        graph.add_edge(START, "agent")
        graph.add_edge("agent", END)
    return graph.compile(checkpointer=ctx.checkpointer, name=name)


def _resolve_approval(ctx: ModuleContext) -> frozenset[str]:
    """解析审批名单（延迟导入：core 不在模块顶层依赖 tools 包）。"""
    from agent_base.tools.registry import resolve_approval_names

    return resolve_approval_names(ctx.settings)
