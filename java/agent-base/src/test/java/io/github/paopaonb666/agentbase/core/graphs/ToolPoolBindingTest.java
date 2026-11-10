package io.github.paopaonb666.agentbase.core.graphs;

import io.github.paopaonb666.agentbase.core.AgentModule;
import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.core.graph.CompiledGraph;
import io.github.paopaonb666.agentbase.core.graph.GraphBuilder;
import io.github.paopaonb666.agentbase.core.graph.GraphState;
import io.github.paopaonb666.agentbase.core.graph.InMemoryCheckpointer;
import io.github.paopaonb666.agentbase.core.tools.AgentTool;
import io.github.paopaonb666.agentbase.core.tools.ToolPool;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

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

    @Test
    void untrustedMarkerWrapsExternalToolResultsInContextOnly() {
        // #84①：开关开 → 外部数据工具（web_search）的 ToolMessage 带标记，
        // 豁免工具不带；开关关 → 逐字节旧行为。事件面（ToolCallEvent）在
        // 池内用原文——本图无 sink 侧断言，分离性由 ToolPool 侧测试覆盖
        // （事件发射先于 Graphs 组装，结构性保证）。
        AgentTool webSearch = new AgentTool() {
            @Override public String name() { return "web_search"; }
            @Override public String description() { return "搜索（测试桩）"; }
            @Override public ToolSpecification specification() {
                return ToolSpecification.builder().name(name()).description(description())
                        .parameters(JsonObjectSchema.builder().build()).build();
            }
            @Override public String call(String argsJson, ToolContext ctx) {
                return "网页结果正文";
            }
        };
        // 两段式 stub LLM：第一次回 tool_calls（触发 tools 节点）、第二次收尾
        // ——工厂形态，两次图运行各用独立实例（计数器互不串扰）。
        java.util.function.Supplier<io.github.paopaonb666.agentbase.core.llm.ModelRuntime>
                scriptedLlmFactory = () -> {
            java.util.concurrent.atomic.AtomicInteger rounds = new java.util.concurrent.atomic.AtomicInteger();
            return new io.github.paopaonb666.agentbase.core.llm.ModelRuntime() {
                    @Override
                    public dev.langchain4j.model.chat.ChatModel chat() {
                        throw new UnsupportedOperationException("测试桩不走向线");
                    }

                    @Override
                    public dev.langchain4j.model.chat.StreamingChatModel streaming() {
                        throw new UnsupportedOperationException("测试桩不走向线");
                    }

                    @Override
                    public dev.langchain4j.data.message.AiMessage stream(
                            List<dev.langchain4j.data.message.ChatMessage> messages,
                            List<ToolSpecification> tools,
                            java.util.function.Consumer<String> onDelta) {
                        if (rounds.incrementAndGet() == 1) {
                            return dev.langchain4j.data.message.AiMessage.from("",
                                    List.of(dev.langchain4j.agent.tool.ToolExecutionRequest.builder()
                                            .id("c1").name("web_search").arguments("{}").build()));
                        }
                        return dev.langchain4j.data.message.AiMessage.from("done");
                    }

                    @Override
                    public dev.langchain4j.data.message.AiMessage invoke(
                            List<dev.langchain4j.data.message.ChatMessage> messages) {
                        return dev.langchain4j.data.message.AiMessage.from("done");
                    }
            };
        };

        // 开关开：ToolMessage 头部带标记。
        Settings on = Settings.load(java.util.Map.of(
                "ENV", "development", "TOOL_RESULT_UNTRUSTED_MARKER", "true"), null);
        AgentModule.ModuleContext ctxOn = new AgentModule.ModuleContext(
                on, scriptedLlmFactory.get(), new InMemoryCheckpointer(), List.of(), null, null, null);
        CompiledGraph graphOn = Graphs.buildSingleAgentGraph(ctxOn, "chat", null, List.of(webSearch));
        GraphState withCalls = new GraphState(List.of(
                dev.langchain4j.data.message.AiMessage.from(
                    "", List.of(dev.langchain4j.agent.tool.ToolExecutionRequest.builder()
                        .id("c1").name("web_search").arguments("{}").build()))),
                Map.of());
        GraphState outOn = graphOn.invoke(withCalls, io.github.paopaonb666.agentbase.core.graph.RunConfig.of("chat:t84", "u1"));
        String toolText = outOn.messages().stream()
                .filter(m -> m instanceof dev.langchain4j.data.message.ToolExecutionResultMessage)
                .map(m -> ((dev.langchain4j.data.message.ToolExecutionResultMessage) m).text())
                .findFirst().orElseThrow();
        assertTrue(toolText.startsWith(
                io.github.paopaonb666.agentbase.core.context.ContextEngine.UNTRUSTED_TOOL_RESULT_PREFIX),
                "开关开时外部工具结果应带标记：" + toolText);

        // 开关关（默认）：逐字节旧行为。
        Settings off = Settings.load(Map.of("ENV", "development"), null);
        AgentModule.ModuleContext ctxOff = new AgentModule.ModuleContext(
                off, scriptedLlmFactory.get(), new InMemoryCheckpointer(), List.of(), null, null, null);
        CompiledGraph graphOff = Graphs.buildSingleAgentGraph(ctxOff, "chat", null, List.of(webSearch));
        GraphState outOff = graphOff.invoke(withCalls, io.github.paopaonb666.agentbase.core.graph.RunConfig.of("chat:t85", "u1"));
        String plainText = outOff.messages().stream()
                .filter(m -> m instanceof dev.langchain4j.data.message.ToolExecutionResultMessage)
                .map(m -> ((dev.langchain4j.data.message.ToolExecutionResultMessage) m).text())
                .findFirst().orElseThrow();
        assertEquals("网页结果正文", plainText);
    }
}
