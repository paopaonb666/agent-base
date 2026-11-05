package io.github.paopaonb666.agentbase.core.graph;

import io.github.paopaonb666.agentbase.core.tools.ToolPool;

/**
 * 节点执行期上下文（对应 Python 的 RunnableConfig + get_stream_writer +
 * 工具执行通道）。节点经它：
 * <ul>
 *   <li>读取本次执行的 {@link RunConfig}（thread_id / user_id）；</li>
 *   <li>向流推送自定义契约事件（tool_call / plan / sources，对应
 *       LangGraph 的 custom stream / get_stream_writer）；</li>
 *   <li>调用共享工具池里的工具（对应 ToolNode / tool runner）；</li>
 *   <li>查询取消状态（客户端断开 → 尽快让出）。</li>
 * </ul>
 */
public interface NodeContext {

    /** 本次执行的运行配置。 */
    RunConfig config();

    /**
     * 向流推送自定义事件（尽力而为：队列满/取消时静默丢弃，不中断
     * 节点执行——对应 Python get_stream_writer 的容错语义）。
     */
    void emit(Object event);

    /** 执行共享池里的工具；返回结果文本（异常已归一为模型可读文本）。 */
    String callTool(String name, String argsJson);

    /** 工具池（供 planner 自建 tool runner 复用）。 */
    ToolPool tools();

    /** 流是否已被取消（客户端断开）；长循环节点应在轮间检查。 */
    boolean cancelled();

    /** 流式调用模型时的增量回调通道（null = 非流式调用）。 */
    default void emitDelta(String content) {
        if (content != null && !content.isEmpty()) {
            emit(new io.github.paopaonb666.agentbase.extensions.events.Events.DeltaEvent(content));
        }
    }
}
