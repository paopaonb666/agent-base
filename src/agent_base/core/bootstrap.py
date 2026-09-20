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

import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver

from agent_base.core.config import Settings
from agent_base.core.contracts import AgentModule, Graph, MemoryPort, ModuleContext
from agent_base.core.llm import ResilientLLM, build_llm
from agent_base.core.registry import RegistryError, load_modules
from agent_base.core.tools import build_tool_pool
from agent_base.extensions.filestore import UploadedFileStore, build_uploaded_file_store
from agent_base.extensions.memory import build_checkpointer, close_checkpointer
from agent_base.extensions.toollog import ToolCallRecorder, build_tool_call_recorder
from agent_base.memory.store import MemoryStore
from agent_base.tools.registry import build_toolkit_tools, toolkit_timeouts

# 保留的模块名：不构建单个模块的图，而是在所有已加载模块之上构建
# supervisor 图（阶段 4）。
SUPERVISOR_MODULE = "supervisor"

logger = logging.getLogger(__name__)


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
    file_store: UploadedFileStore | None = None
    # 记忆服务端口（M6）：None = 未启用；模块通过 ModuleContext.memory 取用。
    memory: MemoryPort | None = None
    # 会话索引存储（S1）：thread_index 表的后端（跟随 checkpointer 后端）。
    # 与 memory 服务独立——MEMORY_ENABLED=false 时仍提供线程属主数据面；
    # memory 启用时它就是 memory.store 同一实例（随 memory.aclose 关闭）。
    thread_index: MemoryStore | None = None
    _graphs: dict[str, Graph] = field(default_factory=dict, repr=False)
    _supervisor: Graph | None = field(default=None, repr=False)
    # 快档视图缓存（成本治理 T1.2）：profile -> 共享状态的运行时副本。
    _profile_views: dict[str, AgentRuntime] = field(default_factory=dict, repr=False)

    def profile_view(self, profile: str) -> AgentRuntime | None:
        """同状态、不同模型的运行时视图（成本治理 T1.2：Tier 1/2 分流）。

        ``profile="main"`` 返回自身；``"fast"`` 在 ``LLM_FAST_*`` 已配置时
        返回共享 checkpointer/memory/tools 的轻量副本（仅 llm 与图缓存
        独立——线程历史跨档连续，同一 checkpointer 键空间）；未配置返回
        None（调用方显式 400，不静默降级——对齐 planner 503 的哲学）。
        """
        if profile == "main":
            return self
        if profile != "fast":
            raise ValueError(f"unknown llm profile {profile!r}; expected 'main' or 'fast'")
        cached = self._profile_views.get(profile)
        if cached is not None:
            return cached
        if not self.settings.llm_fast.is_configured:
            return None
        view = AgentRuntime(
            settings=self.settings,
            llm=build_llm(self.settings, profile="fast"),
            modules=self.modules,
            tools=self.tools,
            checkpointer=self.checkpointer,
            tool_recorder=self.tool_recorder,
            file_store=self.file_store,
            memory=self.memory,
            thread_index=self.thread_index,
        )
        self._profile_views[profile] = view
        return view

    def context(self) -> ModuleContext:
        """交给每个模块的 ``build_graph`` 的 ``ModuleContext``。"""
        return ModuleContext(
            settings=self.settings,
            llm=self.llm,
            checkpointer=self.checkpointer,
            tools=self.tools,
            memory=self.memory,
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
        if self.file_store is not None and hasattr(self.file_store, "aclose"):
            await self.file_store.aclose()
        if self.memory is not None:
            await self.memory.aclose()
        elif self.thread_index is not None:
            closer = getattr(self.thread_index, "aclose", None)
            if closer is not None:
                await closer()


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
    # 记忆服务（M6）先于工具池装配：记忆工具（M6e）作为 extra_tools 进
    # 池，与模块工具/工具库工具同样受超时包装与审计收口。延迟导入：
    # core 装配层不在模块加载时依赖 memory 子系统的导入图（依赖倒置——
    # 契约只认 MemoryPort）。
    checkpointer = await build_checkpointer(resolved)
    file_store = await build_uploaded_file_store(resolved)
    from agent_base.memory.service import build_memory_service
    from agent_base.memory.store import build_memory_store

    # 存储后端常备（S1）：thread_index 的线程属主数据面不随
    # MEMORY_ENABLED 关闭；记忆服务启用时复用同一 store 实例。
    try:
        memory_store = await build_memory_store(resolved)
    except Exception:
        if resolved.memory_enabled:
            raise
        logger.warning("memory: 存储后端装配失败（记忆已禁用，仅线程索引降级）", exc_info=True)
        memory_store = None
    memory_service = None
    if memory_store is not None:
        # 快档侧（成本治理）：LLM_FAST_* 已配置且非 all_main 时，为形成管线
        # 装配限流回退包装（抽取/画像/摘要走它，整合裁决按 profile 决定）。
        # 组合根接线而非 memory 自建——memory 不在运行时依赖 core.llm。
        fast_llm = None
        if (
            llm is not None
            and resolved.memory.pipeline_profile != "all_main"
            and resolved.llm_fast.is_configured
        ):
            fast_llm = ResilientLLM(
                primary=build_llm(resolved, profile="fast"),
                fallback=llm,
                max_concurrency=resolved.llm_fast.concurrency,
            )
        memory_service = await build_memory_service(
            resolved, llm, fast_llm=fast_llm, store=memory_store
        )
    # 工具库的内置工具按 TOOLKIT_ENABLED 装配后并入共享池（模块工具在
    # 前，工具库在后）；重名在任何一侧发生都会快速失败。注册表声明的
    # per-tool 超时在这里下发给池；审计记录器挂上收口，全量落库。
    recorder = build_tool_call_recorder(resolved)
    toolkit = build_toolkit_tools(resolved)
    extra_tools: list[BaseTool] = list(toolkit)
    if memory_service is not None:
        from agent_base.memory.tools import build_memory_tools

        extra_tools.extend(build_memory_tools(memory_service))
    tools = build_tool_pool(
        modules,
        timeout=resolved.tool_timeout_seconds,
        extra_tools=extra_tools,
        timeouts=toolkit_timeouts(resolved),
        recorder=recorder,
    )
    return AgentRuntime(
        settings=resolved,
        llm=llm,
        modules=modules,
        tools=tools,
        checkpointer=checkpointer,
        tool_recorder=recorder,
        file_store=file_store,
        memory=memory_service,
        thread_index=memory_store,
    )
