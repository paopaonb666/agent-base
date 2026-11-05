package io.github.paopaonb666.agentbase.modules.hello;

import io.github.paopaonb666.agentbase.core.AgentModule;
import io.github.paopaonb666.agentbase.core.graph.CompiledGraph;
import io.github.paopaonb666.agentbase.core.tools.AgentTool;
import io.github.paopaonb666.agentbase.modules.hello.graph.HelloGraphs;

import java.util.List;

/**
 * {@code HelloModule} 的 AgentModule 实现——hello 模块（module-guide
 * 第 5 步：通过 AGENT_MODULES 启用；默认未启用——AGENT_MODULES 是唯一
 * 的开关，ADR-002）。最小的可行 AgentModule：图构建委托给
 * {@link HelloGraphs}，不贡献任何工具。
 *
 * <p>对应 Python 侧 {@code modules/hello}。
 */
public final class HelloModule implements AgentModule {

    @Override
    public String name() {
        return "hello";
    }

    @Override
    public String description() {
        return "Hello-world sample module from the module-guide five-step recipe.";
    }

    @Override
    public CompiledGraph buildGraph(AgentModule.ModuleContext ctx) {
        return HelloGraphs.buildHelloGraph(ctx);
    }

    @Override
    public List<AgentTool> getTools() {
        return List.of();
    }
}
