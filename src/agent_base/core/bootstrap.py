"""装配运行时：加载模块、构建 LLM、接线状态与工具。

这是 config / registry / llm / memory / tools 与各入口之间的薄“接线”层。
它不包含业务逻辑——只负责组合基座所拥有的各个部件，并把一个就绪的
运行时交给 CLI / 服务器。

阶段 3 新增：checkpointer（``extensions/memory``）与共享工具池
（``core/tools``）在此装配，并流入每个模块的 ``ModuleContext``。
``create_runtime`` 之所以是异步的，是因为 sqlite checkpointer 会绑定到
调用它的事件循环。

M1（工具库）新增：``TOOLKIT_ENABLED`` 选中的基座内置工具在装配进池，
与模块工具共用重名检查与超时包装。

阶段 4 新增：保留的模块名 ``supervisor`` 会构建一个 supervisor 图
（``extensions/collab``），把每个已注册的模块编排为 sub-agent。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver

from agent_base.core.config import Settings
from agent_base.core.contracts import AgentModule, Graph, ModuleContext
from agent_base.core.llm import build_llm
from agent_base.core.registry import RegistryError, load_modules
from agent_base.core.tools import build_tool_pool
from agent_base.extensions.memory import build_checkpointer, close_checkpointer
from agent_base.extensions.toollog import ToolCallRecorder, build_tool_call_recorder
from agent_base.tools.registry import build_toolkit_tools, toolkit_timeouts

# 保留的模块名：不构建单个模块的图，而是在所有已加载模块之上构建
# supervisor 图（阶段 4）。
SUPERVISOR_MODULE = "supervisor"


class UnknownModuleError(KeyError):
    """请求的模块不在 AGENT_MODULES 清单中（入口层据此给出友好报错）。"""


def validate_module_names(modules: dict[str, AgentModule]) -> None:
    """启动时的清单完整性检查：supervisor 是保留名，不能被业务模块占用。"""
    if SUPERVISOR_MODULE in modules:
        raise RegistryError(
            f"module name {SUPERVISOR_MODULE!r} is reserved for the multi-agent "
            "supervisor; rename the module"
        )


@dataclass
class AgentRuntime:
    """一个装配完毕的运行时：settings + llm + modules + state + tools。"""

    settings: Settings
    llm: BaseChatModel
    modules: dict[str, AgentModule]
    tools: list[BaseTool] = field(default_factory=list)
    checkpointer: BaseCheckpointSaver[Any] | None = None
    tool_recorder: ToolCallRecorder | None = None
    _graphs: dict[str, Graph] = field(default_factory=dict, repr=False)
    _supervisor: Graph | None = field(default=None, repr=False)

    def context(self) -> ModuleContext:
        """交给每个模块的 ``build_graph`` 的 ``ModuleContext``。"""
        return ModuleContext(
            settings=self.settings,
            llm=self.llm,
            checkpointer=self.checkpointer,
            tools=self.tools,
        )

    def graph(self, module_name: str) -> Graph:
        """构建（并编译）指定模块的图；编译结果按模块缓存。

        server 的每个请求都会调用这里——不缓存的话每次请求都要重新
        bind_tools + compile。``supervisor`` 是保留名：它返回编排所有
        已注册模块的多 Agent supervisor 图（同样缓存，阶段 4）。
        """
        if module_name == SUPERVISOR_MODULE:
            return self.supervisor_graph()
        cached = self._graphs.get(module_name)
        if cached is not None:
            return cached
        try:
            module = self.modules[module_name]
        except KeyError:
            available = ", ".join([*sorted(self.modules), SUPERVISOR_MODULE]) or "(none)"
            raise UnknownModuleError(
                f"module {module_name!r} is not enabled; "
                f"available: {available}. Add it to AGENT_MODULES."
            ) from None
        graph = module.build_graph(self.context())
        self._graphs[module_name] = graph
        return graph

    def supervisor_graph(self) -> Graph:
        """惰性地在所有已注册模块之上构建 supervisor 图。"""
        if self._supervisor is None:
            # 延迟导入：扩展可能会增多；core 不应在模块加载时依赖
            # 每个扩展的导入图。
            from agent_base.extensions.collab import build_supervisor_graph

            self._supervisor = build_supervisor_graph(self.context(), self.modules)
        return self._supervisor

    async def close(self) -> None:
        """释放运行时持有的资源（checkpointer 连接、审计写线程等）。"""
        if self.checkpointer is not None:
            await close_checkpointer(self.checkpointer)
        if self.tool_recorder is not None and hasattr(self.tool_recorder, "aclose"):
            await self.tool_recorder.aclose()


async def create_runtime(settings: Settings | None = None) -> AgentRuntime:
    """根据 settings（或默认值）创建运行时；配置错误时快速失败。

    必须在将要执行这些图的事件循环中运行（sqlite checkpointer 会把
    aiosqlite 绑定到调用它的循环）。
    """
    resolved = settings if settings is not None else Settings()
    resolved.ensure_production_ready()
    llm = build_llm(resolved)
    modules = load_modules(resolved.agent_modules)
    validate_module_names(modules)
    # 工具库的内置工具按 TOOLKIT_ENABLED 装配后并入共享池（模块工具在
    # 前，工具库在后）；重名在任何一侧发生都会快速失败。注册表声明的
    # per-tool 超时在这里下发给池；审计记录器挂上收口，全量落库。
    recorder = build_tool_call_recorder(resolved)
    toolkit = build_toolkit_tools(resolved)
    tools = build_tool_pool(
        modules,
        timeout=resolved.tool_timeout_seconds,
        extra_tools=toolkit,
        timeouts=toolkit_timeouts(resolved),
        recorder=recorder,
    )
    checkpointer = await build_checkpointer(resolved)
    return AgentRuntime(
        settings=resolved,
        llm=llm,
        modules=modules,
        tools=tools,
        checkpointer=checkpointer,
        tool_recorder=recorder,
    )
