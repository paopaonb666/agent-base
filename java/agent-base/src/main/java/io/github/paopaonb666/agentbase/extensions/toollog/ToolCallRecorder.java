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

    /** 释放后台资源（写线程/连接）；默认无操作。 */
    default void close() {}
}
