package io.github.paopaonb666.agentbase.extensions.events;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyOrder;
import com.fasterxml.jackson.annotation.JsonValue;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * SSE 事件契约。agent 的进展是显式的——step 转换、token 增量、来源、
 * 终止——以一组封闭的事件类型呈现，因此线上格式是经过校验且可版本化
 * 的，而不是临时的字符串。
 *
 * <p>线上格式：Server-Sent Events，每个事件一帧：
 *
 * <pre>
 * event: delta
 * data: {"type":"delta","content":"he"}
 * </pre>
 *
 * <p>字段名/顺序/null 序列化与 Python 侧 {@code extensions/events.py} 的
 * Pydantic {@code model_dump_json()} 逐字节对齐（null 字段保留为
 * {@code null}，顺序按声明序）。{@code SourcesEvent} 刻意作为契约的
 * 一部分，尽管基座从不发出它：未来模块（如 RAG）可以通过它发布引用，
 * 而不必改动编码器。
 */
public final class Events {

    private Events() {}

    private static final ObjectMapper MAPPER = new ObjectMapper()
            .setSerializationInclusion(JsonInclude.Include.ALWAYS);

    /** agent 运行中的一个具名阶段（节点 running / completed / error）。 */
    @JsonPropertyOrder({"type", "name", "status", "detail"})
    public record StepEvent(String name, String status, String detail) {
        @JsonProperty("type")
        public String type() { return "step"; }
        public static StepEvent running(String name) { return new StepEvent(name, "running", null); }
        public static StepEvent completed(String name) { return new StepEvent(name, "completed", null); }
    }

    /** assistant 回复中 token 级别的一小段（流式）。 */
    @JsonPropertyOrder({"type", "content"})
    public record DeltaEvent(String content) {
        @JsonProperty("type")
        public String type() { return "delta"; }
    }

    /** 一条被引用的来源（未来的 RAG 模块会发布这些）。 */
    @JsonPropertyOrder({"title", "url"})
    public record Source(String title, String url) {}

    /** 附着在回复上的引用（基座从不发出；契约预留）。 */
    @JsonPropertyOrder({"type", "sources"})
    public record SourcesEvent(List<Source> sources) {
        @JsonProperty("type")
        public String type() { return "sources"; }
    }

    /**
     * 图运行终止点的统一封闭枚举（#9）：done 事件 termination 字段的取值，
     * 前端据此识别结束原因，不再从消息形状/error 文本反推。终止判定的权威
     * 在 {@code CompiledGraph}（运行结束处产出）；SSE 层只负责渲染。Java
     * 单边扩展（Python 侧 events.py 的 DoneEvent 无此字段，#307 登记）。
     *
     * <ul>
     *   <li>DONE —— 图正常走到 END（含 put 失败内存续跑后的正常收尾）；</li>
     *   <li>FAILED / PARTIAL —— 预留：supervisor/planner 判分（done/failed/
     *       partial 三态）落到 run 级的细粒度标注，当前无产生点；</li>
     *   <li>CANCELLED —— 消费者取消（客户端断连：图层 break 或 sink 取消传播）；</li>
     *   <li>RECURSION_LIMIT —— 步数触顶（GraphRecursionError）；</li>
     *   <li>THREAD_BUSY —— 同 thread 并发 run 快速失败（ThreadBusyError）；</li>
     *   <li>INTERNAL_ERROR —— 其余异常（节点执行失败/未知节点/出口层失败）。</li>
     * </ul>
     */
    public enum RunTermination {
        DONE, FAILED, PARTIAL, CANCELLED, RECURSION_LIMIT, THREAD_BUSY, INTERNAL_ERROR;

        /** wire 值：枚举名小写（RECURSION_LIMIT → "recursion_limit"）。 */
        @JsonValue
        public String wire() {
            return name().toLowerCase(Locale.ROOT);
        }
    }

    /**
     * 终止性的成功标记；携带 thread id 以便恢复会话。termination（#9）为
     * 统一终止标注；null 时不序列化——未标注的 done 与旧 wire 逐字节一致
     * （前端向后兼容）。
     */
    @JsonPropertyOrder({"type", "thread_id", "termination"})
    public record DoneEvent(String threadId, RunTermination termination) {
        /** 既有单参形态：不标注终止原因（wire 上不出现 termination 字段）。 */
        public DoneEvent(String threadId) {
            this(threadId, null);
        }

        @JsonProperty("type")
        public String type() { return "done"; }
        @JsonProperty("thread_id")
        public String threadId() { return threadId; }
        @JsonInclude(JsonInclude.Include.NON_NULL)
        @JsonProperty("termination")
        public RunTermination termination() { return termination; }
    }

    /** 终止性的失败标记，带有人类可读的消息。 */
    @JsonPropertyOrder({"type", "message"})
    public record ErrorEvent(String message) {
        @JsonProperty("type")
        public String type() { return "error"; }
    }

    /** 心跳，防止中间环节关闭空闲的流。 */
    @JsonPropertyOrder({"type"})
    public record PingEvent() {
        @JsonProperty("type")
        public String type() { return "ping"; }
    }

    /**
     * 一次工具调用的开始/结束记录。由池的超时包装在执行前后经 custom
     * 通道发出：start 携带模型给定的参数，end 携带结果/状态/耗时——
     * 成功、超时、异常三种结局都会发。前端按 call_id 把两半配对成完整
     * 的调用面板。
     */
    @JsonPropertyOrder({"type", "call_id", "name", "phase", "args", "result", "status", "duration_ms", "error"})
    public record ToolCallEvent(
            @JsonProperty("call_id") String callId,
            String name,
            String phase,
            Map<String, Object> args,
            String result,
            String status,
            @JsonProperty("duration_ms") Integer durationMs,
            String error) {
        @JsonProperty("type")
        public String type() { return "tool_call"; }
    }

    /** 计划清单里的一条子任务（前端渲染任务列表）。 */
    @JsonPropertyOrder({"id", "goal", "status"})
    public record PlanTask(int id, String goal, String status) {}

    /**
     * planner 的计划级进展（M9）：任务清单 + 阶段状态。created（拆解
     * 完成）/ progress（子任务状态变化）/ replanned（重规划）/ done
     * （执行结束）。每次携带全量清单，前端整表替换即可。
     */
    @JsonPropertyOrder({"type", "status", "plan", "detail"})
    public record PlanEvent(String status, List<PlanTask> plan, String detail) {
        @JsonProperty("type")
        public String type() { return "plan"; }
    }

    /** 把一个契约事件渲染为 Server-Sent Events 帧（对应 Python encode_sse）。 */
    public static String encodeSse(Object event) {
        return "event: " + typeOf(event) + "\ndata: " + dataJson(event) + "\n\n";
    }

    /** 事件 JSON（data 行内容），供 SseEmitter 的 data 部分复用。 */
    public static String dataJson(Object event) {
        try {
            return MAPPER.writeValueAsString(event);
        } catch (JsonProcessingException e) {
            throw new IllegalStateException("SSE 事件序列化失败", e);
        }
    }

    /** 事件的 type 字段（SSE event 行）。 */
    public static String typeOf(Object event) {
        if (event instanceof PingEvent) return "ping";
        if (event instanceof StepEvent) return "step";
        if (event instanceof DeltaEvent) return "delta";
        if (event instanceof SourcesEvent) return "sources";
        if (event instanceof DoneEvent) return "done";
        if (event instanceof ErrorEvent) return "error";
        if (event instanceof ToolCallEvent) return "tool_call";
        if (event instanceof PlanEvent) return "plan";
        throw new IllegalArgumentException("未知 SSE 事件类型: " + event.getClass().getName());
    }
}
