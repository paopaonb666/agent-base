package io.github.paopaonb666.agentbase.modules.writer.graph;

import io.github.paopaonb666.agentbase.core.AgentModule;
import io.github.paopaonb666.agentbase.core.graph.CompiledGraph;
import io.github.paopaonb666.agentbase.core.graphs.Graphs;

/**
 * writer 模块的图构建。
 *
 * <p>一个带写作取向系统提示词的 LLM 节点，经基座共享图工厂装配——结构
 * 与 chat 完全一致，唯一的区别就是提示词，而这正是要点所在：模块的
 * 区别在于行为，而不在于接线。工厂同时带来基座级的上下文保护（注入
 * 去重 + token 预算修剪），注入块不再随轮次无界累积（H3）。
 *
 * <p>对应 Python 侧 {@code modules/writer/graph.py}。
 */
public final class WriterGraphs {

    private WriterGraphs() {}

    public static final String MODULE_NAME = "writer";

    public static final String WRITER_PROMPT =
            "You are a professional writing agent. Draft, rewrite, summarize, or "
                    + "polish the requested text with clear structure and a consistent voice. "
                    + "Ask for nothing except the writing task itself.";

    /** 构建并编译 writer 图。 */
    public static CompiledGraph buildWriterGraph(AgentModule.ModuleContext ctx) {
        return Graphs.buildSingleAgentGraph(ctx, MODULE_NAME, WRITER_PROMPT, null);
    }
}
