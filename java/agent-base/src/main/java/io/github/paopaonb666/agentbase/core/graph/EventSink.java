package io.github.paopaonb666.agentbase.core.graph;

/**
 * 图执行的事件出口（对应 Python astream 三通道里 sink 到 SSE 生产者的一侧）。
 *
 * <p>运行时按序推送：{@code StepEvent(running)} → 节点执行期间的自定义
 * 事件（tool_call / plan / sources）与 token 增量（DeltaEvent）→
 * {@code StepEvent(completed)}。{@code DoneEvent}/{@code ErrorEvent}
 * 由服务入口层在终态追加。
 */
@FunctionalInterface
public interface EventSink {

    /** 接受一个流事件（StepEvent / DeltaEvent / 自定义契约事件）。 */
    void accept(Object event);

    /** sink 是否已被取消（客户端断开）；节点循环在步间检查后停止。 */
    default boolean cancelled() {
        return false;
    }
}
