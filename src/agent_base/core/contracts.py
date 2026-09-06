"""``AgentModule`` 契约 + ``ModuleContext``。

这是每个 agent 模块都要实现的唯一扩展点。基座从不导入业务逻辑；
模块实现这个协议，并由 registry（``core/registry.py``）把它们接入
运行时。

契约刻意保持最小化，并按阶段增长：
- 阶段 1：``name`` / ``description`` / ``build_graph`` / ``get_tools``
- 阶段 3：工具池和 checkpointer 被接入 ``ModuleContext``；模块绑定
  ``ctx.tools`` 并用 ``ctx.checkpointer`` 编译
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, runtime_checkable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

if TYPE_CHECKING:  # pragma: no cover - import avoided at runtime
    from agent_base.core.config import Settings

# CompiledStateGraph 对 (StateT, ContextT, InputT, OutputT) 是泛型的。
# 基座把编译后的图当作不透明对象处理——它只会把图回传给入口——
# 因此四个参数都绑定为 Any。
Graph: TypeAlias = CompiledStateGraph[Any, Any, Any, Any]


@dataclass
class ModuleContext:
    """基座交给每个模块的运行时服务。

    ``settings``     —— 已校验的配置
    ``llm``          —— 装配好的对话模型（openai 兼容）
    ``checkpointer`` —— 对话状态存储器（阶段 3）；仅当调用方手动构造
                        context（测试）时才是 ``None``
    ``tools``        —— 共享工具池（阶段 3）：每个模块 ``get_tools()``
                        的贡献，加上超时包装；绑定到 LLM 并通过
                        ``ToolNode`` 执行
    """

    settings: Settings
    llm: BaseChatModel
    checkpointer: BaseCheckpointSaver[Any] | None = None
    tools: list[BaseTool] = field(default_factory=list)


@runtime_checkable
class AgentModule(Protocol):
    """一个可运行的 agent 模块。

    ``name``        —— 稳定标识符；必须与 AGENT_MODULES 条目及模块的
                      目录名一致（registry 强制执行）
    ``description`` —— 用于发现 / supervisor 的人类可读摘要
    ``build_graph`` —— 构建该模块编译后的 LangGraph；以
                      ``name=<模块名>`` 编译，这样 supervisor 就能把图
                      作为 sub-agent 来编排
    ``get_tools``   —— 该模块贡献给共享池的工具（阶段 3：收集进
                      ``ModuleContext.tools``）
    """

    name: str
    description: str

    def build_graph(self, ctx: ModuleContext) -> Graph: ...

    def get_tools(self) -> list[BaseTool]: ...
