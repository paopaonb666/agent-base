package io.github.paopaonb666.agentbase.core.graphs;

import io.github.paopaonb666.agentbase.core.AgentModule;
import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.core.graph.CompiledGraph;
import io.github.paopaonb666.agentbase.core.graph.GraphBuilder;
import io.github.paopaonb666.agentbase.core.graph.InMemoryCheckpointer;
import io.github.paopaonb666.agentbase.core.tools.ToolPool;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;

/**
 * 工具池绑定收敛（roadmap #238 + #131）：池在 compile 期定死——经
 * ModuleContext 从装配方传入共享池，不再有编译后 volatile 覆写（竞态面
 * 与装配顺序陷阱）；缺省兜底池保留默认超时（#131 的"保留超时"分支）。
 */
class ToolPoolBindingTest {

    @Test
    void toolPoolIsBoundAtCompileAndImmutableAfterwards() {
        // 缺省编译：无池（NodeContext.callTool 走"工具不可用"文案）。
        CompiledGraph bare = GraphBuilder.singleNode("n", (state, ctx) -> Map.of(),
                new InMemoryCheckpointer());
        assertNull(bare.toolPool());

        // 显式池：编译期绑定，getter 返回同一实例。
        ToolPool pool = ToolPool.build(List.of(), List.of(),
                ToolPool.DEFAULT_TOOL_TIMEOUT_SECONDS, Map.of(), null);
        CompiledGraph bound = GraphBuilder.begin()
                .addNode("n", (state, ctx) -> Map.of())
                .entry("n")
                .addEdge("n", GraphBuilder.END)
                .compile(new InMemoryCheckpointer(), "bound", pool);
        assertSame(pool, bound.toolPool());
    }

    @Test
    void moduleContextCarriesSharedPoolIntoBuiltGraph() {
        ToolPool shared = ToolPool.build(List.of(), List.of(),
                ToolPool.DEFAULT_TOOL_TIMEOUT_SECONDS, Map.of(), null);
        AgentModule.ModuleContext ctx = new AgentModule.ModuleContext(
                Settings.load(Map.of("ENV", "development")), null, new InMemoryCheckpointer(),
                List.of(), null, null, shared);
        // 共享池经 ctx 到达 buildSingleAgentGraph → compile 参数（同一实例）。
        CompiledGraph graph = Graphs.buildSingleAgentGraph(ctx, "chat", "系统提示", null);
        assertSame(shared, graph.toolPool());

        // 六参便利构造 = 无池（测试直构形态不受影响）。
        AgentModule.ModuleContext legacy = new AgentModule.ModuleContext(
                Settings.load(Map.of("ENV", "development")), null, new InMemoryCheckpointer(),
                List.of(), null, null);
        assertNull(legacy.toolPool());
        CompiledGraph fallbackBuilt = Graphs.buildSingleAgentGraph(legacy, "chat", "系统提示", null);
        // 兜底池：非 null（#131：保留默认超时，不做无超时执行），但不是共享池。
        assertNull(legacy.toolPool());
        org.junit.jupiter.api.Assertions.assertNotSame(shared, fallbackBuilt.toolPool());
        org.junit.jupiter.api.Assertions.assertNotNull(fallbackBuilt.toolPool());
    }
}
