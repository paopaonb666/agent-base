package io.github.paopaonb666.agentbase.modules.hello.graph;

import io.github.paopaonb666.agentbase.core.AgentModule;
import io.github.paopaonb666.agentbase.core.graph.CompiledGraph;
import io.github.paopaonb666.agentbase.core.graphs.Graphs;

/**
 * hello 模块的图构建——module-guide 五步配方在真实代码中的落地：最小
 * 的可行模块（一个 LLM 节点，无工具），经基座共享图工厂一行装配。它的
 * 存在是为了证明手册的承诺——新模块接入时无需触碰基座代码。
 *
 * <p>该图用运行时的 checkpointer 编译（按 thread 持久化对话状态），并以
 * {@code name="hello"} 编译，这样 supervisor 就能像编排其他已注册模块
 * 一样把它作为 sub-agent 编排。对应 Python 侧 {@code modules/hello/graph.py}。
 */
public final class HelloGraphs {

    private HelloGraphs() {}

    public static final String MODULE_NAME = "hello";

    /** 构建并编译 hello 图：单个 LLM 节点。 */
    public static CompiledGraph buildHelloGraph(AgentModule.ModuleContext ctx) {
        return Graphs.buildSingleAgentGraph(ctx, MODULE_NAME, null, null);
    }
}
