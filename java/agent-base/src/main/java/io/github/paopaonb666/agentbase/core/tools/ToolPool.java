package io.github.paopaonb666.agentbase.core.tools;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import dev.langchain4j.agent.tool.ToolSpecification;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.UUID;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

import io.github.paopaonb666.agentbase.core.ThreadId;
import io.github.paopaonb666.agentbase.core.graph.NodeContext;
import io.github.paopaonb666.agentbase.core.graph.RunConfig;
import io.github.paopaonb666.agentbase.extensions.Metrics;
import io.github.paopaonb666.agentbase.extensions.Observability;
import io.github.paopaonb666.agentbase.extensions.events.Events;
import io.github.paopaonb666.agentbase.extensions.toollog.ToolCallRecord;
import io.github.paopaonb666.agentbase.extensions.toollog.ToolCallRecorder;

/**
 * 共享工具池（阶段 3）。
 *
 * <p>每个模块通过 {@code AgentModule.getTools()} 贡献工具；每个运行时
 * 装配一次池，再通过 {@code ModuleContext.tools} 交还给模块。工具失败
 * 会被归一成模型可读的文本反馈（模型看到错误并能恢复——工具异常绝不
 * 会让图崩溃）。
 *
 * <p>每个工具都用挂钟超时（TOOL_TIMEOUT_SECONDS）包装：截止时间在调用
 * 方一侧得到遵守，超时后调用方立即收到超时文本；失控工具留在后台线程
 * 跑完直至自然结束（有界泄漏，已记录在案的权衡）——超时即
 * {@code cancel(true)}，被取消任务的工具体结束时计入
 * {@link #leakedTaskCount()}（累计口径，不回零）。执行线程池有界
 * （#25，此前是全仓唯一的 {@code newCachedThreadPool} 无界池）：
 * 常驻 8 / 峰值 64 / 队列 200 / AbortPolicy，容量远超现实负载
 * （AGENT_MAIN_CONCURRENCY 默认 8），只防 OOM 级失控；池满被拒的提交
 * 归一成模型可读文本（"tool execution failed: pool saturated"，
 * LOG.warn，metrics 走 outcome=error 既有通道），池水位见
 * {@link #activeThreadCount()} / {@link #queuedCount()}（暂不接
 * /metrics，由指标收口任务统一接线）。包装层同时是
 * <b>工具调用可观测的唯一收口</b>（M5）：执行前后发 {@code tool_call}
 * 流事件，完结时把记录交给 recorder 落库——成功、超时、异常、参数校验
 * 失败（outcome=invalid，#334/#335：不执行、回喂契约文本）四种结局
 * 都会被记录。
 *
     * <p>对应 Python 侧 {@code core/tools.py}。
     *
     * <p>#16 工具表稳定排序：{@link #specifications()} /
     * {@link #registeredTools()} 按工具名字典序返回——装配序会随模块清单
     * 漂移，排序保证 prompt 工具表前缀逐字节稳定（KV-cache 友好）。Python
     * 侧 {@code build_tool_pool} 仍按装配序返回（不排序），Java 侧排序属
     * 单边增强（差异表 #315 后续补行）。
     */
public final class ToolPool {

    private static final Logger LOG = LoggerFactory.getLogger(ToolPool.class);

    /** 默认挂钟预算（对应 DEFAULT_TOOL_TIMEOUT_SECONDS）。 */
    public static final double DEFAULT_TOOL_TIMEOUT_SECONDS = 30.0;

    /** 无法装配工具池（工具名重复）时抛出。 */
    public static final class ToolPoolError extends IllegalArgumentException {
        public ToolPoolError(String message) { super(message); }
    }

    /** 某个工具超出了其挂钟预算而被截断。 */
    public static final class ToolTimeoutError extends RuntimeException {
        public ToolTimeoutError(String message) { super(message); }
    }

    // 工具错误文本会进入 ToolMessage（模型 + 会话历史回放都会看到）：
    // URL 脱敏、长度封顶，避免把内部端点/巨型负载泄给前端。
    private static final java.util.regex.Pattern TOOL_ERROR_URL_RE =
            java.util.regex.Pattern.compile("[a-zA-Z][a-zA-Z0-9+.\\-]*://\\S+");
    private static final int TOOL_ERROR_MAX_CHARS = 300;

    private static final ObjectMapper MAPPER = new ObjectMapper();

    // #25 有界池参数：容量远超现实负载（AGENT_MAIN_CONCURRENCY 默认 8，
    // callAll 并发槽复用同一池），只防 OOM 级失控——队列满且线程到峰值
    // 后才拒绝（AbortPolicy），拒绝由 call/callAll 捕获并归一。
    private static final int POOL_CORE_SIZE = 8;
    private static final int POOL_MAX_SIZE = 64;
    private static final long POOL_KEEP_ALIVE_SECONDS = 60L;
    private static final int POOL_QUEUE_CAPACITY = 200;

    /** 工具执行用的有界守护线程池（对应 Python 的 daemon 线程语义；#25 前为 newCachedThreadPool）。 */
    private final ThreadPoolExecutor executor = new ThreadPoolExecutor(
            POOL_CORE_SIZE, POOL_MAX_SIZE, POOL_KEEP_ALIVE_SECONDS, TimeUnit.SECONDS,
            new LinkedBlockingQueue<>(POOL_QUEUE_CAPACITY),
            r -> {
                Thread t = new Thread(r, "agent-tool-worker");
                t.setDaemon(true);
                return t;
            },
            new ThreadPoolExecutor.AbortPolicy());

    /** 超时被 cancel(true) 的任务在其工具体结束时的累计计数（#25 泄漏口径，不回零）。 */
    private final AtomicLong leakedTasks = new AtomicLong();

    private final Map<String, Entry> tools;
    private final ToolCallRecorder recorder;

    private record Entry(AgentTool inner, ToolSpecification spec, double timeout) {}

    private ToolPool(Map<String, Entry> tools, ToolCallRecorder recorder) {
        this.tools = tools;
        this.recorder = recorder;
    }

    /**
     * 把模块工具与工具库工具收集进一个带超时包装、无冲突的池。
     * 重复的工具名会中止启动——两个同名工具会让工具调用路由变得模糊，
     * 这是一个快速失败的配置错误；该检查覆盖模块工具与工具库工具的整体。
     */
    public static ToolPool build(List<AgentTool> moduleTools, List<AgentTool> extraTools,
                                 double defaultTimeout, Map<String, Double> timeouts,
                                 ToolCallRecorder recorder) {
        Map<String, Entry> pool = new LinkedHashMap<>();
        for (AgentTool tool : moduleTools) {
            put(pool, tool, "module", defaultTimeout, timeouts);
        }
        for (AgentTool tool : extraTools) {
            put(pool, tool, "toolkit", defaultTimeout, timeouts);
        }
        // #33：装配期把工具名闭集显式交给指标层（基数防漂移；未注册名照常计数并 WARN 一次）。
        Metrics.tool().registerNames(pool.keySet());
        return new ToolPool(Map.copyOf(pool), recorder);
    }

    private static void put(Map<String, Entry> pool, AgentTool tool, String sourceKind,
                            double defaultTimeout, Map<String, Double> timeouts) {
        if (pool.containsKey(tool.name())) {
            throw new ToolPoolError(
                    "duplicate tool name '%s' contributed by %s; tool names must be unique "
                            .formatted(tool.name(), sourceKind)
                            + "across modules and toolkit");
        }
        pool.put(tool.name(), new Entry(tool, tool.specification(),
                timeouts.getOrDefault(tool.name(), defaultTimeout)));
    }

    /**
     * 池内全部工具的 ToolSpecification（按名字稳定排序，#16），供绑定模型。
     *
     * <p>#16：装配序会随模块清单漂移（且 Map.copyOf 本身不保序），模型看到
     * 的工具表顺序漂移会破坏 prompt 前缀稳定性（KV-cache 命中率）。排序在
     * 返回时按工具名字典序做（内部存储不动），保证工具表前缀逐字节稳定。
     */
    public List<ToolSpecification> specifications() {
        return sortedEntries().stream().map(entry -> entry.getValue().spec()).toList();
    }

    /** 池内登记的原始工具视图（按名字稳定排序，同 {@link #specifications()}；ModuleContext 交给模块用）。 */
    public List<AgentTool> registeredTools() {
        return sortedEntries().stream().map(entry -> entry.getValue().inner()).toList();
    }

    /** #16：池内条目按工具名字典序（键即工具名）；排序在返回时做，内部存储保持装配态。 */
    private List<Map.Entry<String, Entry>> sortedEntries() {
        return tools.entrySet().stream()
                .sorted(Map.Entry.comparingByKey())
                .toList();
    }

    /** 池是否为空。 */
    public boolean isEmpty() {
        return tools.isEmpty();
    }

    /**
     * 执行工具并返回结果文本（对应 ToolNode + _TimeoutTool._arun）。
     * 任何失败都被归一成模型可读文本，绝不上抛。
     */
    public String call(String name, String argsJson, RunConfig config, NodeContext ctx) {
        Entry entry = tools.get(name);
        if (entry == null) {
            return handleToolError(new IllegalArgumentException("unknown tool '%s'".formatted(name)));
        }
        String callId = UUID.randomUUID().toString().replace("-", "").substring(0, 12);
        Map<String, Object> safeArgs = jsonSafe(argsJson);
        emitToolCall(ctx, callId, name, "start", safeArgs, null, null, null, null);
        long started = System.nanoTime();
        String outcome = "ok";
        String resultText = "";
        String errorText = "";
        try {
            // #334/#335：执行前对声明的参数面做轻校验，失败短路（不进工具
            // 线程），回喂结构化错误让模型一轮自愈。契约见
            // ToolArgsValidator.contractText，由 ToolPoolValidationTest 锁定。
            java.util.List<String> validationErrors = validateArgs(entry.spec(), argsJson);
            if (!validationErrors.isEmpty()) {
                outcome = "invalid";
                errorText = io.github.paopaonb666.agentbase.toolkit.ToolArgsValidator
                        .contractText(validationErrors);
                return errorText;
            }
            // #162：MDC（request id + MemoryScope 的 user/thread）随任务
            // 进工具线程——工具体内的日志与作用域读取和调用方一致（此前
            // 工具线程 MDC 全空，记忆工具经池调用会落到 default 用户作用域）。
            // #25 泄漏计数：超时路径先置 per-call 标志再 cancel(true)；
            // 被取消任务的工具体结束时（finally）发现标志 → 计一次泄漏。
            AtomicBoolean cancelledByTimeout = new AtomicBoolean(false);
            Future<String> future;
            try {
                future = executor.submit(Observability.wrapCallable(() -> {
                    try {
                        return entry.inner().call(argsJson, new AgentTool.ToolContext(config, ctx));
                    } finally {
                        if (cancelledByTimeout.get()) {
                            leakedTasks.incrementAndGet();
                        }
                    }
                }));
            } catch (RejectedExecutionException e) {
                // #25：有界池饱和。照 unknown tool/timeout 的归一风格回喂
                // 模型可读文本；metrics 经 finally track() 走 outcome=error
                // 既有通道。绝不抛错打断对话。
                LOG.warn("tool {}: 线程池饱和，任务提交被拒（有界池队列 {}）",
                        name, POOL_QUEUE_CAPACITY, e);
                outcome = "error";
                errorText = handleToolError(new RejectedExecutionException("pool saturated"));
                return errorText;
            }
            try {
                resultText = future.get((long) (entry.timeout() * 1000), TimeUnit.MILLISECONDS);
            } catch (TimeoutException e) {
                cancelledByTimeout.set(true); // 先登记泄漏标志，再取消
                future.cancel(true);
                outcome = "timeout";
                errorText = "tool '%s' exceeded %ss timeout".formatted(name, entry.timeout());
                // 与 Python 侧一致：超时异常经 ToolNode 的 handle_tool_error
                // 归一后作为 ToolMessage 反馈给模型（带统一前缀）。
                throw new ToolTimeoutError(errorText);
            } catch (java.util.concurrent.ExecutionException e) {
                Throwable cause = e.getCause() == null ? e : e.getCause();
                outcome = "error";
                errorText = handleToolError(asException(cause));
                return errorText; // 归一文本直接作为工具结果反馈给模型
            }
            return resultText;
        } catch (ToolTimeoutError e) {
            return handleToolError(e); // 超时异常同样经 ToolNode 归一语义反馈给模型
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            outcome = "error";
            errorText = "tool execution failed: interrupted";
            return errorText;
        } finally {
            long durationMs = (System.nanoTime() - started) / 1_000_000;
            track(ctx, callId, name, safeArgs, started, outcome, resultText, errorText, durationMs);
        }
    }

    /** 执行前校验：解析失败按一条结构化错误处理（空参视作 {}）。 */
    private static java.util.List<String> validateArgs(ToolSpecification spec, String argsJson) {
        com.fasterxml.jackson.databind.JsonNode parsed;
        try {
            parsed = MAPPER.readTree(argsJson == null || argsJson.strip().isEmpty()
                    ? "{}" : argsJson);
        } catch (Exception e) {
            return java.util.List.of("参数必须是合法 JSON 对象（解析失败：" + e.getMessage() + "）");
        }
        return io.github.paopaonb666.agentbase.toolkit.ToolArgsValidator
                .validate(spec.parameters(), parsed);
    }

    /**
     * 批量执行一轮 tool_calls（#337）：池内只读工具（{@link AgentTool#readOnly()}）
     * 并发跑（tool_call 事件按 call_id 自配对，乱序完成安全），有副作用的
     * 串行；结果严格按请求顺序返回。
     *
     * <p>{@code runner} 是单请求执行通道：普通图节点传 {@code (req) ->
     * this.call(req.name(), req.arguments(), config, ctx)}；supervisor 的
     * handoff 传 {@code runHandoff}（handoff 工具不在池内 → 按非只读串行，
     * 语义与原实现一致）。串行槽的异常原样传播（handoff 依赖这一语义）；
     * 并发槽由池通道保证不抛（pool.call 全归一）。
     */
    public List<String> callAll(List<dev.langchain4j.agent.tool.ToolExecutionRequest> requests,
                                RunConfig config, NodeContext ctx,
                                java.util.function.Function<dev.langchain4j.agent.tool.ToolExecutionRequest, String> runner) {
        int size = requests.size();
        if (size <= 1) {
            return requests.stream().map(runner).toList();
        }
        // 槽位：String = 已完成的串行结果（异常在提交时原样抛出）；
        // CompletableFuture<String> = 并发中的只读任务。
        List<Object> slots = new java.util.ArrayList<>(size);
        for (dev.langchain4j.agent.tool.ToolExecutionRequest request : requests) {
            Entry entry = tools.get(request.name() == null ? "" : request.name());
            if (entry != null && entry.inner().readOnly()) {
                // #162：并发槽同样转发 MDC（wrapSupplier 在调用方线程捕获快照）。
                try {
                    slots.add(java.util.concurrent.CompletableFuture.supplyAsync(
                            Observability.wrapSupplier(() -> runner.apply(request)), executor));
                } catch (RejectedExecutionException e) {
                    // #25：并发槽提交被拒——归一文本占位该槽结果（并发槽
                    // 契约：由池通道保证不抛；pool.call 侧同款归一）。
                    LOG.warn("tool {}: 线程池饱和，并发槽提交被拒", request.name(), e);
                    slots.add(handleToolError(new RejectedExecutionException("pool saturated")));
                }
            } else {
                slots.add(runner.apply(request)); // 异常原样传播（语义同旧 for 循环）
            }
        }
        List<String> out = new java.util.ArrayList<>(size);
        for (Object slot : slots) {
            out.add(slot instanceof String text
                    ? text : ((java.util.concurrent.CompletableFuture<String>) slot).join());
        }
        return out;
    }

    private static Exception asException(Throwable t) {
        return t instanceof Exception e ? e : new IllegalStateException(t.getMessage(), t);
    }

    /** 超时被 cancel(true) 的任务在工具体结束时的累计泄漏计数（#25；不回零）。 */
    public long leakedTaskCount() {
        return leakedTasks.get();
    }

    /** 池水位：正在执行工具的线程数（#25；暂不接 /metrics，由指标收口任务统一接线）。 */
    public int activeThreadCount() {
        return executor.getActiveCount();
    }

    /** 池水位：排队等待执行的任务数（#25；暂不接 /metrics）。 */
    public int queuedCount() {
        return executor.getQueue().size();
    }

    /** 完结一支调用：发流事件 + 交审计记录（两者都绝不抛错打断对话）。 */
    private void track(NodeContext ctx, String callId, String name, Map<String, Object> safeArgs,
                       long startedNanos, String outcome, String resultText, String errorText,
                       long durationMs) {
        try {
            emitToolCall(ctx, callId, name, "end", safeArgs, resultText.isEmpty() ? null : resultText,
                    outcome, (int) durationMs, errorText.isEmpty() ? null : errorText);
        } catch (Exception e) {
            LOG.debug("tool {}: emit tool_call end failed", name, e);
        }
        if (recorder != null) {
            try {
                ThreadId parsed = ThreadId.tryParse(ctx != null ? ctx.config().threadId() : "");
                recorder.record(new ToolCallRecord(
                        callId,
                        name,
                        outcome,
                        trunc(jsonDump(safeArgs), ToolCallRecord.ARGS_MAX_CHARS),
                        trunc(resultText, ToolCallRecord.RESULT_MAX_CHARS),
                        trunc(errorText, ToolCallRecord.ERROR_MAX_CHARS),
                        durationMs,
                        parsed == null ? "" : ctx.config().threadId(),
                        parsed == null ? "" : parsed.module(),
                        Observability.requestId()));
            } catch (Exception e) {
                LOG.error("tool {}: recorder.record failed", name, e);
            }
        }
        Metrics.tool().observeTool(name, outcome, durationMs / 1000.0);
    }

    private static void emitToolCall(NodeContext ctx, String callId, String name, String phase,
                                     Map<String, Object> args, String result, String status,
                                     Integer durationMs, String error) {
        if (ctx != null) {
            ctx.emit(new Events.ToolCallEvent(callId, name, phase, args, result, status, durationMs, error));
        }
    }

    /**
     * 把任何工具失败归一成模型可读的反馈：URL 脱敏 + 长度封顶。基座的
     * 契约是"坏掉的工具绝不能打断对话"——每个异常都被转换为模型能够
     * 回应的文本。对应 Python {@code _normalize_tool_error}：str(exc)
     * 对无消息异常是空串（不是 "null"），且 300 字符上限包含前缀整体。
     */
    public static String handleToolError(Exception exc) {
        String raw = exc.getMessage() == null ? "" : exc.getMessage();
        String text = TOOL_ERROR_URL_RE.matcher(raw).replaceAll("<redacted-url>");
        String out = "tool execution failed: " + text;
        return out.length() > TOOL_ERROR_MAX_CHARS ? out.substring(0, TOOL_ERROR_MAX_CHARS) : out;
    }

    /** 把工具参数转成可 JSON 序列化的形式；解析失败降级为 {"_raw": ...}。 */
    private static Map<String, Object> jsonSafe(String argsJson) {
        try {
            Map<String, Object> loaded = MAPPER.readValue(argsJson == null ? "{}" : argsJson,
                    new TypeReference<Map<String, Object>>() {});
            return new TreeMap<>(loaded); // 排序稳定，事件渲染可复现
        } catch (Exception e) {
            Map<String, Object> raw = new LinkedHashMap<>();
            raw.put("_raw", argsJson == null ? "" : argsJson);
            return raw;
        }
    }

    private static String jsonDump(Map<String, Object> value) {
        try {
            return MAPPER.writeValueAsString(value);
        } catch (JsonProcessingException e) {
            return "{\"_raw\": \"unserializable\"}";
        }
    }

    private static String trunc(String text, int max) {
        if (text == null) {
            return "";
        }
        return text.length() <= max ? text : text.substring(0, max);
    }

    /** 关闭执行线程池（运行时 close 时调用）。 */
    public void close() {
        executor.shutdownNow();
    }
}
