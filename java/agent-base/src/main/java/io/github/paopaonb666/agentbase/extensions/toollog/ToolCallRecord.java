package io.github.paopaonb666.agentbase.extensions.toollog;

import java.util.List;
import java.util.Map;

/**
 * 工具调用审计记录（M5 可观测）：成功、超时、异常全量落库。
 *
 * <p>设计要点（对应 Python 侧 {@code extensions/toollog.py}）：
 * <ul>
 *   <li><b>唯一事实源在池的收口</b>：ToolPool 在完结时组装
 *       {@link ToolCallRecord} 交给 recorder；无论成功、超时还是异常——
 *       审计不漏一条。</li>
 *   <li><b>写入绝不拖垮对话</b>：{@code record()} 同步、非阻塞，内部各自
 *       消化并发；写失败只记日志。</li>
 *   <li><b>存储后端跟随 checkpointer</b>：memory → 进程内环形缓冲（重启
 *       即丢）；sqlite → 同一文件里的 {@code tool_call_records} 表；
 *       mysql → JDBC 写表（建表 DDL 与 alembic 0002 一致，未跑迁移的库
 *       也能自举）。</li>
 *   <li><b>截断防膨胀</b>：result 4k / args 2k / error 500 字符封顶。</li>
 * </ul>
 */
public final class ToolCallRecord {

    /** 截断上限（字符）：审计是旁路数据，超长内容进日志/全量输出即可。 */
    public static final int RESULT_MAX_CHARS = 4000;
    public static final int ARGS_MAX_CHARS = 2000;
    public static final int ERROR_MAX_CHARS = 500;
    /** memory 后端的进程内环形缓冲上限。 */
    public static final int MEMORY_MAX_RECORDS = 1000;

    private final String callId;
    private final String tool;
    private final String status; // ok | timeout | error
    private final String argsJson;
    private final String resultText;
    private final String errorText;
    private final long durationMs;
    private final String threadId;
    private final String module;
    private final String requestId;

    public ToolCallRecord(String callId, String tool, String status, String argsJson,
                          String resultText, String errorText, long durationMs,
                          String threadId, String module, String requestId) {
        this.callId = callId == null ? "" : callId;
        this.tool = tool == null ? "" : tool;
        this.status = status == null ? "" : status;
        this.argsJson = argsJson == null ? "" : argsJson;
        this.resultText = resultText == null ? "" : resultText;
        this.errorText = errorText == null ? "" : errorText;
        this.durationMs = durationMs;
        this.threadId = threadId == null ? "" : threadId;
        this.module = module == null ? "" : module;
        this.requestId = requestId == null ? "" : requestId;
    }

    public String callId() { return callId; }

    public String tool() { return tool; }

    public String status() { return status; }

    public String argsJson() { return argsJson; }

    public String resultText() { return resultText; }

    public String errorText() { return errorText; }

    public long durationMs() { return durationMs; }

    public String threadId() { return threadId; }

    public String module() { return module; }

    public String requestId() { return requestId; }

    /** HTTP 端点的序列化视图（args 解析为对象；error 空串转 null）。 */
    public Map<String, Object> asDict() {
        return new java.util.LinkedHashMap<>(java.util.Map.ofEntries(
                java.util.Map.entry("call_id", callId),
                java.util.Map.entry("tool", tool),
                java.util.Map.entry("status", status),
                java.util.Map.entry("args", loadsOrRaw(argsJson)),
                java.util.Map.entry("result", resultText),
                java.util.Map.entry("error", errorText.isEmpty() ? null : errorText),
                java.util.Map.entry("duration_ms", durationMs),
                java.util.Map.entry("thread_id", threadId),
                java.util.Map.entry("module", module),
                java.util.Map.entry("request_id", requestId)));
    }

    /** args JSON → 对象（解析失败回退原文；空串 → {}）。 */
    public static Object loadsOrRaw(String text) {
        if (text == null || text.isEmpty()) {
            return new java.util.LinkedHashMap<String, Object>();
        }
        try {
            return new com.fasterxml.jackson.databind.ObjectMapper().readValue(text, Object.class);
        } catch (Exception e) {
            return text;
        }
    }

    /** 行查询结果 → asDict 同构视图（sqlite/mysql 后端共用）。 */
    public static Map<String, Object> rowToDict(Object[] row) {
        return new java.util.LinkedHashMap<>(java.util.Map.ofEntries(
                java.util.Map.entry("call_id", row[0]),
                java.util.Map.entry("thread_id", row[1]),
                java.util.Map.entry("module", row[2]),
                java.util.Map.entry("request_id", row[3]),
                java.util.Map.entry("tool", row[4]),
                java.util.Map.entry("args", loadsOrRaw((String) row[5])),
                java.util.Map.entry("result", row[6]),
                java.util.Map.entry("status", row[7]),
                java.util.Map.entry("error", row[8] == null || String.valueOf(row[8]).isEmpty() ? null : row[8]),
                java.util.Map.entry("duration_ms", row[9])));
    }

    /** INSERT 列清单（sqlite 与 MySQL 的占位符由各实现决定）。 */
    public static final List<String> INSERT_COLUMNS = List.of(
            "call_id", "thread_id", "module", "request_id", "tool",
            "args_json", "result_text", "status", "error_text", "duration_ms");
}
