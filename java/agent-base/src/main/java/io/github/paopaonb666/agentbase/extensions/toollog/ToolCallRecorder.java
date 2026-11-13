package io.github.paopaonb666.agentbase.extensions.toollog;

import java.util.List;
import java.util.Map;

/**
 * 审计记录器的最小接口；实现必须各自保证线程安全。
 *
 * <p>{@code record()} 同步、非阻塞（内部消化并发）；{@code listForThread()}
 * 返回该线程的记录（时间升序、封顶 {@code limit} 条），视图与
 * {@link ToolCallRecord#asDict()} 同构。
 */
public interface ToolCallRecorder {

    /** 记录一条已完结的工具调用（绝不抛错打断对话）。 */
    void record(ToolCallRecord record);

    /** 某线程的审计记录（时间升序，最多 limit 条）。 */
    List<Map<String, Object>> listForThread(String threadId, int limit);

    /**
     * A-#217 分页形态：{@code offset} 从<b>最新一端</b>跳过（与 #204 线程
     * 列表游标同语义——翻页向更旧的方向走）。default 委托 2 参旧形态
     * （忽略 offset——兼容垫片）；SQL 实现覆写为 LIMIT/OFFSET 下推。
     */
    default List<Map<String, Object>> listForThread(String threadId, int limit, int offset) {
        return listForThread(threadId, limit);
    }

    /** 释放后台资源（写线程/连接）；默认无操作。 */
    default void close() {}
}
