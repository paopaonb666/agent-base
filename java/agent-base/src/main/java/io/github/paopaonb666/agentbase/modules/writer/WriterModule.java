package io.github.paopaonb666.agentbase.modules.writer;

import io.github.paopaonb666.agentbase.core.AgentModule;
import io.github.paopaonb666.agentbase.core.graph.CompiledGraph;
import io.github.paopaonb666.agentbase.core.tools.AgentTool;
import io.github.paopaonb666.agentbase.modules.writer.graph.WriterGraphs;

import java.util.List;

/**
 * {@code WriterModule} 的 AgentModule 实现——writer 样板模块（第二个
 * 内置 agent，阶段 4）。
 *
 * <p>它有两个用途：1) 作为第二个 supervisor sub-agent，让多 Agent 模板
 * 可以演示性地在专家之间路由（chat vs writer）；2) 证明新增模块遵循
 * 五步 module guide 且零基座改动。对应 Python 侧 {@code modules/writer}。
 */
public final class WriterModule implements AgentModule {

    @Override
    public String name() {
        return "writer";
    }

    @Override
    public String description() {
        return "Writing specialist: drafts, rewrites, summarizes, and polishes "
                + "text. Route any request that is about producing or improving a "
                + "piece of writing to this agent.";
    }

    @Override
    public String capabilities() {
        return "Writing deliverables: articles, emails, reports, summaries, "
                + "rewrites and polish passes. Not for factual Q&A that needs no "
                + "document artifact.";
    }

    @Override
    public CompiledGraph buildGraph(AgentModule.ModuleContext ctx) {
        return WriterGraphs.buildWriterGraph(ctx);
    }

    @Override
    public List<AgentTool> getTools() {
        return List.of();
    }
}
