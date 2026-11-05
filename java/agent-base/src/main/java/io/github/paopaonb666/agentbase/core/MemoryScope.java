package io.github.paopaonb666.agentbase.core;

import org.slf4j.MDC;

/**
 * 记忆作用域（M6e）：图节点与记忆工具据此拿到当前请求的 user/thread。
 *
 * <p>实现为 MDC 键（对应 Python 侧 ContextVar）：server 在把图执行提交
 * 到工作线程前写入；工具池转发 MDC 到工具线程（Observability.wrap），
 * 记忆工具读取。Python 侧曾有基座与记忆包两个同名 ContextVar 的历史
 * 包袱——Java 侧统一为单一请求作用域（重构计划决策）。
 */
public final class MemoryScope {

    private MemoryScope() {}

    private static final String USER_KEY = "memory_user_id";
    private static final String THREAD_KEY = "memory_thread_id";

    /** 写入本轮请求的记忆作用域（SSE 生产者每轮调用）。 */
    public static void set(String userId, String threadId) {
        MDC.put(USER_KEY, userId == null || userId.isEmpty() ? "default" : userId);
        MDC.put(THREAD_KEY, threadId == null ? "" : threadId);
    }

    /** 当前用户 id（缺省 "default"）。 */
    public static String currentUserId() {
        String value = MDC.get(USER_KEY);
        return value == null || value.isEmpty() ? "default" : value;
    }

    /** 当前完整线程 id（可为空 = 不在图上下文）。 */
    public static String currentThreadId() {
        String value = MDC.get(THREAD_KEY);
        return value == null ? "" : value;
    }

    /** 清理（请求收尾）。 */
    public static void clear() {
        MDC.remove(USER_KEY);
        MDC.remove(THREAD_KEY);
    }
}
