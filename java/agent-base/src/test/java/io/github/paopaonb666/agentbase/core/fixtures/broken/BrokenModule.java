package io.github.paopaonb666.agentbase.core.fixtures.broken;

import io.github.paopaonb666.agentbase.core.AgentModule;
import io.github.paopaonb666.agentbase.core.graph.CompiledGraph;

import java.util.List;

/** name() 与加载名不一致的 fixture（契约校验测试用）。 */
public class BrokenModule implements AgentModule {

    @Override
    public String name() {
        return "other";
    }

    @Override
    public String description() {
        return "declares a mismatched name";
    }

    @Override
    public CompiledGraph buildGraph(ModuleContext ctx) {
        throw new UnsupportedOperationException();
    }

    @Override
    public List<io.github.paopaonb666.agentbase.core.tools.AgentTool> getTools() {
        return List.of();
    }
}
