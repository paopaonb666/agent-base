package io.github.paopaonb666.agentbase.core;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.core.graph.Checkpointer;
import io.github.paopaonb666.agentbase.core.graph.CompactionService;
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
 * <p>对应 Python 侧 {@code core/bootstrap.py}。MCP 工具来源（#331/#333）
 * 在 bootstrap 阶段装配：{@code MCP_ENABLED=true} 时 stdio server 的工具
 * 经 {@code McpTool} 适配并入共享池（启用即承诺——配置非法或连不上启动
 * 即失败，绝不静默缺席）。
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
    /**
     * 会话历史压缩任务（#328）：null = 未装配（开关关/记忆端口缺席，全链路
     * 零行为变化）。钩子经 ModuleContext 进图编译，写回走
     * {@link CompiledGraph#compactThread}（#330 方案 B）。
     */
    private final CompactionService compaction;
    /** MCP 装配句柄（#29 健康可见）：null 或空 servers = MCP 未启用。 */
    private final io.github.paopaonb666.agentbase.toolkit.mcp.McpSources.McpAssembly mcpAssembly;
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
        this(settings, llm, modules, tools, checkpointer, toolRecorder, fileStore,
                memory, threadIndex, planStore, null);
    }

    AgentRuntime(Settings settings, ModelRuntime llm, Map<String, AgentModule> modules,
                 ToolPool tools, Checkpointer checkpointer, ToolCallRecorder toolRecorder,
                 UploadedFileStore fileStore, AgentModule.MemoryServicePort memory,
                 CoreStore threadIndex, AgentModule.PlanStorePort planStore,
                 io.github.paopaonb666.agentbase.toolkit.mcp.McpSources.McpAssembly mcpAssembly) {
        this(settings, llm, modules, tools, checkpointer, toolRecorder, fileStore,
                memory, threadIndex, planStore, mcpAssembly, null);
    }

    AgentRuntime(Settings settings, ModelRuntime llm, Map<String, AgentModule> modules,
                 ToolPool tools, Checkpointer checkpointer, ToolCallRecorder toolRecorder,
                 UploadedFileStore fileStore, AgentModule.MemoryServicePort memory,
                 CoreStore threadIndex, AgentModule.PlanStorePort planStore,
                 io.github.paopaonb666.agentbase.toolkit.mcp.McpSources.McpAssembly mcpAssembly,
                 CompactionService compaction) {
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
        this.mcpAssembly = mcpAssembly;
        this.compaction = compaction;
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

    /**
     * MCP server 连接健康态（#29）：server 名 → connected()。MCP 未启用
     * 返回 null（/health 据此省略 mcp 组件，与 memory 可选先例同构）。
     */
    public java.util.Map<String, Boolean> mcpServerHealth() {
        return mcpAssembly == null || mcpAssembly.servers().isEmpty()
                ? null : mcpAssembly.health();
    }

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
        if (!LlmFactory.PROFILES.contains(profile)) {
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
                toolRecorder, fileStore, memory, threadIndex, planStore, null, compaction);
        profileViews.put(profile, view);
        return view;
    }

    /** 交给每个模块的 {@code buildGraph} 的 {@code ModuleContext}。 */
    public AgentModule.ModuleContext context() {
        return new AgentModule.ModuleContext(settings, llm, checkpointer,
                tools == null ? java.util.List.of() : moduleToolViews(), memory, planStore, tools,
                // 压缩终态钩子（#328）：未装配为 null（图执行零开销）。
                compaction == null ? null : compaction::onRunTerminated,
                // 工具结果外置通道（条目 6）：store 未装配（未知 checkpointer
                // 后端）为 null——外置停用，行为退回原样。
                fileStore);
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
        // 共享池经 ModuleContext 在 compile 期绑定（#238）：图的工具执行
        // 统一走共享池（超时 + 审计 + 指标收口），不再编译后覆写。
        // A-6（审查 P2-6）：构建收进 computeIfAbsent——并发冷启动下同一
        // 模块的 buildGraph 恰好一次（此前的 get→build→put 非原子，竞态
        // 会构建两个等价图、丢一个引用；良性但浪费且短暂违反「编译结果按
        // 模块缓存」的单实例预期）。
        return graphs.computeIfAbsent(moduleName, k -> module.buildGraph(context()));
    }

    /**
     * 惰性地在所有已注册模块之上构建 supervisor 图。A-6（审查 P2-6）：
     * volatile 双检 + synchronized——并发冷启动下构建恰好一次（与模块图
     * 的 computeIfAbsent 同口径）。
     */
    public CompiledGraph supervisorGraph() {
        CompiledGraph cached = supervisor;
        if (cached == null) {
            synchronized (this) {
                cached = supervisor;
                if (cached == null) {
                    cached = SupervisorGraphs.buildSupervisorGraph(context(), modules);
                    supervisor = cached;
                }
            }
        }
        return cached;
    }

    /** 释放运行时持有的资源（checkpointer 连接、审计写线程、MCP 子进程等）。 */
    public void close() {
        if (tools != null) {
            // MCP server 子进程随池的工具收敛关闭（McpTool.close 幂等）。
            for (AgentTool tool : tools.registeredTools()) {
                if (tool instanceof io.github.paopaonb666.agentbase.toolkit.mcp.McpTool mcpTool) {
                    mcpTool.close();
                }
            }
            tools.close();
        }
        if (compaction != null) {
            // 压缩任务执行器（#328）：在途任务中断、队列丢弃（守护线程，
            // 不阻塞进程退出；close 前到达的终态不再评估）。
            // F4（R4-a）：先停压缩、后关存储——顺序安排<b>降低</b>撞窗概率
            // （写回发生在存储仍打开的可能性更大），但不提供「在途压缩写回
            // 必然先于存储关闭完成」的强保证：shutdownNow 不 join（守护线程
            // 快速退出取舍），数秒级 LLM 摘要期间的写回仍可能与存储关闭竞速
            // ——写回失败被 persistSnapshot 吞成欠账计数，/health 欠账计数
            // 的关停瞬态污染仍在（A-2/审查 P2-2 注释与实现强度对齐）。
            compaction.close();
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
    /** #72：按配置给链元素包熔断器（阈值 0 = 禁用直通）。 */
    private static ModelRuntime breakered(ModelRuntime inner, Settings settings) {
        if (settings.llm.circuitFailureThreshold < 1) {
            return inner;
        }
        return new io.github.paopaonb666.agentbase.core.llm.CircuitBreakedModelRuntime(
                inner, settings.llm.circuitFailureThreshold, settings.llm.circuitCooldownSeconds);
    }

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
        // 重放防护弱模式横幅（#112）：配了 secret 但窗口为 0——防重放退化
        // 为仅签 user_id。合法配置不拒启，但必须醒目提示而不是静默弱化。
        if (io.github.paopaonb666.agentbase.entrypoints.server.AuthBackends
                .weakReplayMode(resolved)) {
            LOG.warn("\n{}", io.github.paopaonb666.agentbase.entrypoints.server.AuthBackends
                    .weakReplayBanner());
        }
        // MCP 工具源（#331/#333）：启用即承诺——配置非法或 server 连不上
        // 启动即失败；未启用零解析、零连接。工具与 per-server 超时并入
        // 共享池，与模块/工具库/记忆工具同规。
        io.github.paopaonb666.agentbase.toolkit.mcp.McpSources.McpAssembly mcp =
                io.github.paopaonb666.agentbase.toolkit.mcp.McpSources.load(resolved);
        // usage 采集（T4.1）：cost.enabled 时挂成本计量 sink，未启用时完全不
        // 挂（对应 Python build_llm 的条件挂载——否则 pending 队列没有
        // flusher 消费、metrics 也会泄漏 Python 不会输出的指标）。
        ModelRuntime llm = LlmFactory.buildLlm(resolved, "main",
                resolved.cost.enabled
                        ? io.github.paopaonb666.agentbase.core.llm.UsageSinks.costMeterSink("main")
                        : ModelRuntime.noopUsageSink());
        // 主路径弹性包（#1/#10/#73/#75/#71/#72）：并发信号量 + 指数退避重试 +
        // 滑窗重试预算 + LLM_FALLBACKS 备用模型链 + 逐链元素熔断器（主力
        // open 时链自动跳备用模型），图执行节点经 ModuleContext.llm() 透明
        // 获得。usage 仍由内层 runtime 的 sink 按每次真实调用记账（fallback
        // 各自带 modelName，按实际生效模型入账）。
        java.util.List<ModelRuntime> chain = new java.util.ArrayList<>();
        chain.add(breakered(llm, resolved));
        for (ModelRuntime fallback : LlmFactory.buildFallbacks(resolved,
                resolved.cost.enabled
                        ? io.github.paopaonb666.agentbase.core.llm.UsageSinks.costMeterSink("main")
                        : ModelRuntime.noopUsageSink())) {
            chain.add(breakered(fallback, resolved));
        }
        llm = new io.github.paopaonb666.agentbase.core.llm.ResilientModelRuntime(
                chain.get(0), chain.subList(1, chain.size()),
                resolved.llm.mainRetries, resolved.llm.mainConcurrency);
        Map<String, AgentModule> modules = ModuleRegistry.loadModules(
                resolved.agentModules, null,
                resolved.agentModulePackages.stream().map(ModuleRegistry::normalizePackage).toList());
        validateModuleNames(modules);
        // planner 递归预算推导（#229）：最坏步数（H5 公式）超过全局上限时
        // 启动即提示——触顶从「首跑才炸、报错让用户翻文档」变成「启动横幅
        // 给建议值」。全局上限仍只做兜底（H5 裁决），不自动放宽。
        if (modules.containsKey("planner")) {
            int worst = io.github.paopaonb666.agentbase.core.graphs.Graphs.plannerWorstSteps(resolved);
            if (resolved.agentRecursionLimit < worst) {
                LOG.warn("planner 递归预算不足（#229）：最坏步数 {} > AGENT_RECURSION_LIMIT={} "
                                + "——极端输入下会先触递归上限而非优雅返回；"
                                + "建议 AGENT_RECURSION_LIMIT >= {}，或调低 PLANNER_MAX_SUBTASKS/"
                                + "PLANNER_SUBTASK_TOOL_ROUNDS/PLANNER_MAX_REPLANS",
                        worst, resolved.agentRecursionLimit, worst);
            }
        }
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
        // 快档侧（成本治理）：非 all_main 时为形成管线装配限流回退包装
        // （抽取/画像/摘要走它，整合裁决按 profile 决定）。LLM_FAST_* 未
        // 配置时 buildLlm("fast") 回落主力配置的工具安全形态（#128，对齐
        // Python build_llm 恒返回客户端的语义）——分流能力不缺位。
        ResilientLlm fastLlm = null;
        if (resolved.memory.enabled
                && !"all_main".equals(resolved.memory.pipelineProfile)) {
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
        // 会话历史压缩（#328/#330）：开关关/记忆端口缺席 = 不装配、零行为
        // 变化。budget 按主力 runtime 窗口经 resolveInputBudget 同口径推导，
        // 触发估算乘主力 tokenFactor（R4-a F5 对齐 #340 口径；system 扣减与
        // fast profile 共用主力预算两处偏差在 CompactionService 构造 javadoc
        // 登记）；摘要走记忆端口的 compaction 通道（快档，未配 LLM 时端口
        // 返回 null → 放弃本轮）；写回经 CompiledGraph.compactThread 持锁互斥。
        CompactionService compaction = null;
        if (resolved.memory.compactionEnabled && memoryService != null) {
            compaction = new CompactionService(checkpointer, resolved.memory,
                    io.github.paopaonb666.agentbase.core.context.ContextEngine
                            .resolveInputBudget(resolved, llm),
                    llm.tokenFactor(),
                    memoryService::compactTranscript);
        }
        // 工具库的内置工具按 TOOLKIT_ENABLED 装配后并入共享池（模块工具在
        // 前，工具库在后）；重名在任何一侧发生都会快速失败。
        ToolCallRecorder recorder = ToolCallLoggers.build(resolved);
        // A-31：记忆插件的批量嵌入函数注入 web_search（语义 rerank 用）——
        // 端口 default null（未配 memory/embedding 时 web_search rerank
        // 恒降级原序，零成本）。
        java.util.List<AgentTool> extraTools = new java.util.ArrayList<>(
                io.github.paopaonb666.agentbase.toolkit.Toolkits.buildToolkitTools(resolved,
                        memoryService == null ? null : memoryService.embedBatch()));
        if (plugin != null) {
            // 记忆工具（M6e）由插件随服务一起返回；与模块/工具库工具同样受
            // 超时包装与审计收口。
            extraTools.addAll(plugin.tools());
        }
        extraTools.addAll(mcp.tools());
        // 条目 6：read_tool_result 直装（不进 TOOLKIT_ENABLED 注册表——它随
        // 外置机制存在，无独立启用语义；KNOWN_TOOLKIT_NAMES 不动）。store
        // 缺席或阈值 0（外置停用）时不装：工具不存在，行为退回原样。
        if (fileStore != null && resolved.toolkit.toolResultSpillThreshold > 0) {
            extraTools.add(new io.github.paopaonb666.agentbase.toolkit.ReadToolResultTool(fileStore));
        }
        // MCP per-server 超时并入池的超时表（键 = 前缀化工具名）。
        java.util.Map<String, Double> timeouts =
                new java.util.LinkedHashMap<>(
                        io.github.paopaonb666.agentbase.toolkit.Toolkits.toolkitTimeouts(resolved));
        timeouts.putAll(mcp.timeouts());
        ToolPool pool = ToolPool.build(moduleTools(modules), extraTools,
                resolved.toolkit.timeoutSeconds, timeouts, recorder);
        return new AgentRuntime(resolved, llm, modules, pool, checkpointer, recorder,
                fileStore, memoryService, coreStore,
                coreStore == null ? null : new io.github.paopaonb666.agentbase.core.storage.PlanStoreAdapter(coreStore),
                mcp, compaction);
    }

    private static java.util.List<AgentTool> moduleTools(Map<String, AgentModule> modules) {
        java.util.List<AgentTool> list = new java.util.ArrayList<>();
        for (AgentModule module : modules.values()) {
            list.addAll(module.getTools());
        }
        return list;
    }
}
