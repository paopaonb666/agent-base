package io.github.paopaonb666.agentbase.core;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.core.graph.Checkpointer;
import io.github.paopaonb666.agentbase.core.graph.CompiledGraph;
import io.github.paopaonb666.agentbase.core.llm.LlmFactory;
import io.github.paopaonb666.agentbase.core.llm.ModelRuntime;
import io.github.paopaonb666.agentbase.core.llm.ResilientLlm;
import io.github.paopaonb666.agentbase.core.storage.CoreStore;
import io.github.paopaonb666.agentbase.core.storage.CoreStores;
import io.github.paopaonb666.agentbase.core.tools.AgentTool;
import io.github.paopaonb666.agentbase.core.tools.ToolPool;
import io.github.paopaonb666.agentbase.extensions.collab.SupervisorGraphs;
import io.github.paopaonb666.agentbase.extensions.filestore.UploadedFileStore;
import io.github.paopaonb666.agentbase.extensions.filestore.UploadedFileStores;
import io.github.paopaonb666.agentbase.extensions.toollog.ToolCallLoggers;
import io.github.paopaonb666.agentbase.extensions.toollog.ToolCallRecorder;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * 装配运行时：加载模块、构建 LLM、接线状态与工具。
 *
 * <p>这是 config / registry / llm / memory / tools 与各入口之间的薄
 * "接线"层。它不包含业务逻辑——只负责组合基座所拥有的各个部件，并把
 * 一个就绪的运行时交给 CLI / 服务器。
 *
 * <p>保留的模块名 {@code supervisor} 会构建一个 supervisor 图
 * （extensions/collab），把每个已注册的模块编排为 sub-agent。
 *
 * <p>对应 Python 侧 {@code core/bootstrap.py}。MCP 工具来源按第一期
 * 裁剪决策：{@code MCP_ENABLED=true} 时启动即报错（启用即承诺的契约
 * 语义保留——承诺的工具来源无法兑现时拒绝静默启动）。
 */
public final class AgentRuntime {

    private static final Logger LOG = LoggerFactory.getLogger(AgentRuntime.class);

    /** 保留的模块名：在所有已加载模块之上构建 supervisor 图。 */
    public static final String SUPERVISOR_MODULE = "supervisor";

    private final Settings settings;
    private final ModelRuntime llm;
    private final Map<String, AgentModule> modules;
    private final ToolPool tools;
    private final Checkpointer checkpointer;
    private final ToolCallRecorder toolRecorder;
    private final UploadedFileStore fileStore;
    private final AgentModule.MemoryServicePort memory;
    private final CoreStore threadIndex;
    private final AgentModule.PlanStorePort planStore;
    // 惰性缓存被并发 HTTP 请求线程读写（Python dict 有事件循环串行化 +
    // GIL 兜底；Java 侧必须自己保证并发安全——LinkedHashMap 并发 put 可
    // 能丢条目甚至损坏结构）。
    private final Map<String, CompiledGraph> graphs = new java.util.concurrent.ConcurrentHashMap<>();
    private volatile CompiledGraph supervisor;
    private final Map<String, AgentRuntime> profileViews = new java.util.concurrent.ConcurrentHashMap<>();

    AgentRuntime(Settings settings, ModelRuntime llm, Map<String, AgentModule> modules,
                 ToolPool tools, Checkpointer checkpointer, ToolCallRecorder toolRecorder,
                 UploadedFileStore fileStore, AgentModule.MemoryServicePort memory,
                 CoreStore threadIndex, AgentModule.PlanStorePort planStore) {
        this.settings = settings;
        this.llm = llm;
        this.modules = modules;
        this.tools = tools;
        this.checkpointer = checkpointer;
        this.toolRecorder = toolRecorder;
        this.fileStore = fileStore;
        this.memory = memory;
        this.threadIndex = threadIndex;
        this.planStore = planStore;
    }

    /** 请求的模块不在 AGENT_MODULES 清单中（入口层据此给出友好报错）。 */
    public static final class UnknownModuleError extends IllegalArgumentException {
        public UnknownModuleError(String message) { super(message); }
    }

    public Settings settings() { return settings; }

    public ModelRuntime llm() { return llm; }

    public Map<String, AgentModule> modules() { return modules; }

    public ToolPool tools() { return tools; }

    public Checkpointer checkpointer() { return checkpointer; }

    public ToolCallRecorder toolRecorder() { return toolRecorder; }

    public UploadedFileStore fileStore() { return fileStore; }

    public AgentModule.MemoryServicePort memory() { return memory; }

    public CoreStore threadIndex() { return threadIndex; }

    public AgentModule.PlanStorePort planStore() { return planStore; }

    /** 启动时的清单完整性检查：supervisor 是保留名，不能被业务模块占用。 */
    public static void validateModuleNames(Map<String, AgentModule> modules) {
        if (modules.containsKey(SUPERVISOR_MODULE)) {
            throw new ModuleRegistry.RegistryError(
                    "module name '%s' is reserved for the multi-agent "
                            .formatted(SUPERVISOR_MODULE)
                            + "supervisor; rename the module");
        }
    }

    /**
     * 同状态、不同模型的运行时视图（成本治理 T1.2：Tier 1/2 分流）。
     * {@code profile="main"} 返回自身；{@code "fast"} 在 LLM_FAST_* 已配置
     * 时返回共享 checkpointer/memory/tools 的轻量副本（仅 llm 与图缓存
     * 独立——线程历史跨档连续，同一 checkpointer 键空间）；未配置返回
     * null（调用方显式 400，不静默降级——对齐 planner 503 的哲学）。
     */
    public AgentRuntime profileView(String profile) {
        if ("main".equals(profile)) {
            return this;
        }
        if (!"fast".equals(profile)) {
            throw new IllegalArgumentException("unknown llm profile '%s'; expected 'main' or 'fast'".formatted(profile));
        }
        AgentRuntime cached = profileViews.get(profile);
        if (cached != null) {
            return cached;
        }
        if (!settings.llmFast.isConfigured()) {
            return null;
        }
        // 快档视图的计量独立挂（对应 Python build_llm(profile="fast") 在
        // cost.enabled 时同样挂 CostMeterHandler(profile="fast")）——图执行
        // 走视图自己的 runtime，不会经过主 sink。
        ModelRuntime fastLlm = LlmFactory.buildLlm(settings, "fast",
                settings.cost.enabled
                        ? io.github.paopaonb666.agentbase.core.llm.UsageSinks.costMeterSink("fast")
                        : ModelRuntime.noopUsageSink());
        AgentRuntime view = new AgentRuntime(settings, fastLlm, modules, tools, checkpointer,
                toolRecorder, fileStore, memory, threadIndex, planStore);
        profileViews.put(profile, view);
        return view;
    }

    /** 交给每个模块的 {@code buildGraph} 的 {@code ModuleContext}。 */
    public AgentModule.ModuleContext context() {
        return new AgentModule.ModuleContext(settings, llm, checkpointer,
                tools == null ? java.util.List.of() : moduleToolViews(), memory, planStore);
    }

    /**
     * 交给 ModuleContext 的工具视图：共享池里的工具（已含超时包装与审计
     * 收口）。池的包装在 Java 侧是池内部实现，这里交回池登记的原始工具
     * 面供模块 bind（ToolSpecification 从池取，执行走池收口）。
     */
    private java.util.List<AgentTool> moduleToolViews() {
        return tools == null ? java.util.List.of() : tools.registeredTools();
    }

    /**
     * 构建（并编译）指定模块的图；编译结果按模块缓存。server 的每个请求
     * 都会调用这里——不缓存的话每次请求都要重新 bind_tools + compile。
     * {@code supervisor} 是保留名：它返回编排所有已注册模块的 supervisor
     * 图（同样缓存）。
     */
    public CompiledGraph graph(String moduleName) {
        if (SUPERVISOR_MODULE.equals(moduleName)) {
            return supervisorGraph();
        }
        CompiledGraph cached = graphs.get(moduleName);
        if (cached != null) {
            return cached;
        }
        AgentModule module = modules.get(moduleName);
        if (module == null) {
            // 对应 Python bootstrap.graph：available 按 sorted 排序、空清单
            // 回退 "(none)"。
            java.util.List<String> available = new java.util.ArrayList<>(modules.keySet());
            java.util.Collections.sort(available);
            available.add(SUPERVISOR_MODULE);
            String availableText = String.join(", ", available);
            if (modules.isEmpty()) {
                availableText = "(none)";
            }
            throw new UnknownModuleError(
                    "module '%s' is not enabled; available: %s. Add it to AGENT_MODULES."
                            .formatted(moduleName, availableText));
        }
        CompiledGraph graph = module.buildGraph(context());
        // 共享池覆写：图的工具执行统一走共享池（超时 + 审计 + 指标收口）。
        graph.setToolPool(tools);
        graphs.put(moduleName, graph);
        return graph;
    }

    /** 惰性地在所有已注册模块之上构建 supervisor 图。 */
    public CompiledGraph supervisorGraph() {
        if (supervisor == null) {
            supervisor = SupervisorGraphs.buildSupervisorGraph(context(), modules);
            supervisor.setToolPool(tools);
        }
        return supervisor;
    }

    /** 释放运行时持有的资源（checkpointer 连接、审计写线程等）。 */
    public void close() {
        if (tools != null) {
            tools.close();
        }
        if (checkpointer != null) {
            checkpointer.close();
        }
        if (toolRecorder != null) {
            toolRecorder.close();
        }
        if (fileStore != null) {
            fileStore.close();
        }
        if (memory != null) {
            memory.close();
        }
        // 基座存储与记忆存储相互独立（0.3.0）：两个都要关，不能 elif。
        if (threadIndex != null) {
            threadIndex.close();
        }
    }

    /** 根据 settings 创建运行时；配置错误时快速失败。 */
    public static AgentRuntime create(Settings resolved) {
        resolved.ensureProductionReady();
        // 无鉴权部署横幅（#196）：secret 空时 HMAC 整层跳过、全部请求落
        // default 用户——development 默认配置就是如此，必须醒目提示而不是
        // 让暴露面静默存在（production 已被 ensureProductionReady 拦下）。
        if (io.github.paopaonb666.agentbase.entrypoints.server.AuthBackends
                .unauthenticatedDeployment(resolved)) {
            LOG.warn("\n{}", io.github.paopaonb666.agentbase.entrypoints.server.AuthBackends
                    .unauthenticatedBanner());
        }
        // MCP（第一期未迁移）：启用即承诺——配置打开而能力未迁移时拒绝启动。
        if (resolved.mcp.enabled) {
            throw new IllegalStateException(
                    "MCP_ENABLED=true 但 MCP 工具源第一期未迁移（Java 版）：关闭 MCP_ENABLED 或等待后续版本。"
                            + "启用即承诺：拒绝静默忽略该配置。");
        }
        // usage 采集（T4.1）：cost.enabled 时挂成本计量 sink，未启用时完全不
        // 挂（对应 Python build_llm 的条件挂载——否则 pending 队列没有
        // flusher 消费、metrics 也会泄漏 Python 不会输出的指标）。
        ModelRuntime llm = LlmFactory.buildLlm(resolved, "main",
                resolved.cost.enabled
                        ? io.github.paopaonb666.agentbase.core.llm.UsageSinks.costMeterSink("main")
                        : ModelRuntime.noopUsageSink());
        // 主路径弹性包（#1/#10/#73/#75）：并发信号量 + 指数退避重试 +
        // 滑窗重试预算，图执行节点经 ModuleContext.llm() 透明获得。usage
        // 仍由内层 runtime 的 sink 按每次真实调用记账。
        llm = new io.github.paopaonb666.agentbase.core.llm.ResilientModelRuntime(
                llm, resolved.llm.mainRetries, resolved.llm.mainConcurrency);
        Map<String, AgentModule> modules = ModuleRegistry.loadModules(
                resolved.agentModules, null,
                resolved.agentModulePackages.stream().map(ModuleRegistry::normalizePackage).toList());
        validateModuleNames(modules);
        Checkpointer checkpointer = io.github.paopaonb666.agentbase.core.graph.CheckpointerFactory.build(resolved);
        UploadedFileStore fileStore = UploadedFileStores.build(resolved);
        // 基座自有存储（S1/M10）：线程索引与计划快照的数据面由基座自己拥有，
        // 不随 MEMORY_ENABLED 关闭，也不依赖记忆包是否安装。装配失败只降级。
        CoreStore coreStore;
        try {
            coreStore = CoreStores.build(resolved);
        } catch (Exception e) {
            LOG.warn("core store: 装配失败（线程索引/计划快照降级为不可用）", e);
            coreStore = null;
        }
        // 快档侧（成本治理）：LLM_FAST_* 已配置且非 all_main 时，为形成管线
        // 装配限流回退包装（抽取/画像/摘要走它，整合裁决按 profile 决定）。
        ResilientLlm fastLlm = null;
        if (resolved.memory.enabled
                && !"all_main".equals(resolved.memory.pipelineProfile)
                && resolved.llmFast.isConfigured()) {
            fastLlm = new ResilientLlm(
                    LlmFactory.buildLlm(resolved, "fast",
                            resolved.cost.enabled
                                    ? io.github.paopaonb666.agentbase.core.llm.UsageSinks.costMeterSink("fast")
                                    : ModelRuntime.noopUsageSink()),
                    llm,
                    resolved.llmFast.concurrency);
        }
        // 记忆服务经 ServiceLoader SPI 发现（对应 Python entry point 0.3.0
        // 拆分）：基座源码不 import 记忆包；MEMORY_ENABLED=true 而发行包
        // 缺失时 fail-fast 提示安装。
        io.github.paopaonb666.agentbase.core.memory.MemoryPlugins.MemoryPlugin plugin =
                io.github.paopaonb666.agentbase.core.memory.MemoryPlugins.build(resolved, llm, fastLlm);
        AgentModule.MemoryServicePort memoryService = plugin == null ? null : plugin.service();
        // 工具库的内置工具按 TOOLKIT_ENABLED 装配后并入共享池（模块工具在
        // 前，工具库在后）；重名在任何一侧发生都会快速失败。
        ToolCallRecorder recorder = ToolCallLoggers.build(resolved);
        java.util.List<AgentTool> extraTools = new java.util.ArrayList<>(
                io.github.paopaonb666.agentbase.toolkit.Toolkits.buildToolkitTools(resolved));
        if (plugin != null) {
            // 记忆工具（M6e）由插件随服务一起返回；与模块/工具库工具同样受
            // 超时包装与审计收口。
            extraTools.addAll(plugin.tools());
        }
        ToolPool pool = ToolPool.build(moduleTools(modules), extraTools,
                resolved.toolkit.timeoutSeconds,
                io.github.paopaonb666.agentbase.toolkit.Toolkits.toolkitTimeouts(resolved),
                recorder);
        return new AgentRuntime(resolved, llm, modules, pool, checkpointer, recorder,
                fileStore, memoryService, coreStore,
                coreStore == null ? null : new io.github.paopaonb666.agentbase.core.storage.PlanStoreAdapter(coreStore));
    }

    private static java.util.List<AgentTool> moduleTools(Map<String, AgentModule> modules) {
        java.util.List<AgentTool> list = new java.util.ArrayList<>();
        for (AgentModule module : modules.values()) {
            list.addAll(module.getTools());
        }
        return list;
    }
}
