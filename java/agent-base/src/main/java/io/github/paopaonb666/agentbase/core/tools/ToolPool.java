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
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

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
 * 跑完直至自然结束（有界泄漏，已记录在案的权衡）。包装层同时是
 * <b>工具调用可观测的唯一收口</b>（M5）：执行前后发 {@code tool_call}
 * 流事件，完结时把记录交给 recorder 落库——成功、超时、异常、参数校验
 * 失败（outcome=invalid，#334/#335：不执行、回喂契约文本）四种结局
 * 都会被记录。
 *
 * <p>对应 Python 侧 {@code core/tools.py}。
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

    /** 工具执行用的守护线程池（对应 Python 的 daemon 线程语义）。 */
    private final ExecutorService executor = Executors.newCachedThreadPool(r -> {
        Thread t = new Thread(r, "agent-tool-worker");
        t.setDaemon(true);
        return t;
    });

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

    /** 池内全部工具的 ToolSpecification（按装配顺序），供绑定模型。 */
    public List<ToolSpecification> specifications() {
        return tools.values().stream().map(Entry::spec).toList();
    }

    /** 池内登记的原始工具视图（按装配顺序；ModuleContext 交给模块用）。 */
    public List<AgentTool> registeredTools() {
        return tools.values().stream().map(Entry::inner).toList();
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
            Future<String> future = executor.submit(
                    (java.util.concurrent.Callable<String>) () ->
                            entry.inner().call(argsJson, new AgentTool.ToolContext(config, ctx)));
            try {
                resultText = future.get((long) (entry.timeout() * 1000), TimeUnit.MILLISECONDS);
            } catch (TimeoutException e) {
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

    private static Exception asException(Throwable t) {
        return t instanceof Exception e ? e : new IllegalStateException(t.getMessage(), t);
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
