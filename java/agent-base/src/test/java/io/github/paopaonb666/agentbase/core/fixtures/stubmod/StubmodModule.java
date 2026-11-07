package io.github.paopaonb666.agentbase.core.fixtures.stubmod;

import io.github.paopaonb666.agentbase.core.AgentModule;
import io.github.paopaonb666.agentbase.core.graph.CompiledGraph;
import io.github.paopaonb666.agentbase.core.graph.GraphBuilder;

import java.util.List;
import java.util.Map;

/** 契约合格的 fixture 模块（registry 加载路径测试用）。 */
public class StubmodModule implements AgentModule {

    @Override
    public String name() {
        return "stubmod";
    }

    @Override
    public String description() {
        return "fixture module for registry tests";
    }

    @Override
    public CompiledGraph buildGraph(ModuleContext ctx) {
        return GraphBuilder.singleNode("stubmod", (state, nodeCtx) -> Map.of(), ctx.checkpointer());
    }

    @Override
    public List<io.github.paopaonb666.agentbase.core.tools.AgentTool> getTools() {
        return List.of();
    }
}
