package io.github.paopaonb666.agentbase.entrypoints.server;

import com.fasterxml.jackson.annotation.JsonValue;

/**
 * 结构化错误码（#91）：HTTP 错误体与 SSE error 事件统一携带的封闭枚举
 * ——前端只认 {@code code} 不认文案（文案可能随部署语言漂移）。wire 值
 * 小写下划线（与 #9 RunTermination 同风格）。message 保持人类可读；
 * 双端契约：Python 侧无对应物（差异登记簿承载的单边增强）。
 *
 * <p>产生点映射：
 * <ul>
 *   <li>rate_limited —— per-user 令牌桶 429（#78）</li>
 *   <li>budget_blocked —— 成本预算熔断 429</li>
 *   <li>run_gate_saturated —— 全局 run 并发闸门 429（#79）</li>
 *   <li>thread_busy —— 同 thread 并发 409（#197）</li>
 *   <li>unauthorized —— 鉴权 401；not_found —— 资源缺席/属主掩蔽 404</li>
 *   <li>validation —— 请求校验 422；bad_request —— 语义拒绝 400/413</li>
 *   <li>module_unavailable —— 模块/规划器缺席 503；storage_unavailable ——
 *       线程索引/附件/文件存储降级 503（fail-safe，#82）</li>
 *   <li>tool_timeout / context_overflow —— 预留位（当前无 HTTP 产生点：
 *       工具超时走 ToolMessage 文本自愈、context overflow 是静默预算截断）</li>
 *   <li>llm_unavailable —— SSE 失败分支：熔断 open / 流断（StreamStalled）</li>
 * </ul>
 */
public enum ErrorCode {

    RATE_LIMITED("rate_limited"),
    BUDGET_BLOCKED("budget_blocked"),
    RUN_GATE_SATURATED("run_gate_saturated"),
    THREAD_BUSY("thread_busy"),
    UNAUTHORIZED("unauthorized"),
    NOT_FOUND("not_found"),
    VALIDATION("validation"),
    BAD_REQUEST("bad_request"),
    MODULE_UNAVAILABLE("module_unavailable"),
    STORAGE_UNAVAILABLE("storage_unavailable"),
    /** 预留位：工具超时当前走 ToolMessage 文本自愈，无 HTTP/SSE 产生点。 */
    TOOL_TIMEOUT("tool_timeout"),
    /** 预留位：上下文超限当前是静默预算截断，无终止/HTTP 产生点。 */
    CONTEXT_OVERFLOW("context_overflow"),
    /** LLM 不可用：SSE 失败分支的熔断 open / 流断分类。 */
    LLM_UNAVAILABLE("llm_unavailable"),
    /** 递归触顶：SSE 失败分支专用（与 done.termination=recursion_limit 同源；HTTP 面无产生点）。 */
    RECURSION_LIMIT("recursion_limit"),
    INTERNAL_ERROR("internal_error");

    private final String wire;

    ErrorCode(String wire) {
        this.wire = wire;
    }

    @JsonValue
    public String wire() {
        return wire;
    }
}
