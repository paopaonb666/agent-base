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


@runtime_checkable
class MemoryPort(Protocol):
    """记忆子系统暴露给模块的最小面（依赖倒置，P0-4）。

    core 契约只认这个端口，不 import ``memory.service`` 的具体类——
    替换记忆实现（mem0/Letta/自研）时，模块与基座契约都不用动。
    实现方（MemoryService）是这个协议的超集。
    """

    async def search(
        self, *, user_id: str, agent_id: str | None, query: str, top_k: int | None = None
    ) -> Any: ...

    async def compose_context(
        self, *, user_id: str, agent_id: str, thread_id: str, query: str
    ) -> str | None: ...

    async def capture_turn(
        self, *, user_id: str, agent_id: str, thread_id: str, messages: list[Any]
    ) -> Any: ...

    async def aclose(self) -> None: ...


@runtime_checkable
class MemoryServicePort(MemoryPort, Protocol):
    """入口层需要的记忆服务完整面（0.3.0 拆分后由记忆包实现）。

    ``MemoryPort`` 是业务模块看到的最小面；HTTP 层还需要知识库入库/
    检索、线程级联清理与健康探针。这些方法同样不 import 记忆包的数据
    类——跨边界的载荷一律是 ``Any``/``dict``（依赖倒置：基座源码中
    不出现 ``agent_base_memory``）。
    """

    def capture_ready(self) -> bool:
        """记忆形成管线是否可用（未配置 LLM 时不可用）。"""
        ...

    async def recall_context(self, *, user_id: str, query: str) -> str | None: ...

    async def purge_thread(self, thread_id: str) -> None: ...

    async def ingest_document(
        self,
        *,
        file_id: str,
        user_id: str,
        agent_id: str,
        text: str,
        thread_id: str = "",
    ) -> int: ...

    async def revoke_document(self, *, file_id: str, user_id: str) -> int: ...

    async def search_knowledge(
        self, *, user_id: str, agent_id: str | None, query: str, top_k: int | None = None
    ) -> list[Any]: ...

    async def list_file_chunks(
        self, *, user_id: str, file_id: str, limit: int = 200
    ) -> list[dict[str, Any]]: ...

    async def chunk_counts_by_file(self, user_id: str) -> dict[str, int]: ...

    async def health_probe(self) -> str: ...


@runtime_checkable
class ChatInvokerPort(Protocol):
    """后台管线用到的快档模型最小面（0.3.0 拆分引入）。

    成本治理的快档是 ResilientLLM——它刻意不继承 BaseChatModel
    （只在后台管线做单次 ainvoke，不进图执行路径）。记忆包的装配
    点只需要这个能力，因此契约按端口声明，双方都不绑定具体实现类。
    """

    async def ainvoke(self, messages: Any) -> Any: ...


@runtime_checkable
class PlanStorePort(Protocol):
    """计划快照存储的最小端口（M10 跨轮延续，依赖倒置同 MemoryPort）。

    planner 图在每个状态变更点 upsert 快照；plan 检查端点读表而非图
    状态——chat 轮穿插（chat 图 checkpoint 只含 messages 通道）不再
    重置计划。载荷刻意用 dict 进出：core 契约不 import 存储层的数据
    类，实现在组合根替换时契约不动。
    """

    async def save_plan(
        self,
        *,
        thread_id: str,
        module: str,
        tasks: list[dict[str, Any]],
        cursor: int,
        replans: int,
    ) -> None: ...

    async def get_plan(self, thread_id: str) -> dict[str, Any] | None: ...

    async def delete_plan(self, thread_id: str) -> None: ...


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
    # 记忆服务端口（M6）：None = 未启用（MEMORY_ENABLED=false 或后端
    # 不可用）。模块按需取用；不关心记忆的模块可以完全忽略它。
    memory: MemoryPort | None = None
    # 计划快照端口（M10）：None = 存储不可用（planner 退化为只写图
    # 状态，跨轮延续不可用）。planner 之外的模块可以完全忽略它。
    plan_store: PlanStorePort | None = None


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
