package io.github.paopaonb666.agentbase.core.graphs;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import io.github.paopaonb666.agentbase.core.AgentModule;
import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.core.graph.CompiledGraph;
import io.github.paopaonb666.agentbase.core.graph.GraphState;
import io.github.paopaonb666.agentbase.core.graph.InMemoryCheckpointer;
import io.github.paopaonb666.agentbase.core.graph.RunConfig;
import io.github.paopaonb666.agentbase.core.llm.ModelRuntime;
import io.github.paopaonb666.agentbase.core.text.EstimatingTokenCounter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * SYSTEM 域预算扣减的图级集成用例（条目 15 / R2-F1 收口）：窗口生效时
 * systemPrompt 计入输入预算（Graphs.buildSingleAgentGraph 构建期扣减），
 * 挤占历史体 → 修剪更早；window=0 与旧口径逐字节一致。样板
 * ToolPoolBindingTest（ModuleContext 直构 + 桩运行时捕获模型输入）。
 */
class SystemPromptBudgetTest {

    @AfterEach
    void restoreDefaultCounter() {
        io.github.paopaonb666.agentbase.core.context.ContextEngine.setTokenCounter(null);
    }

    /** 捕获模型输入的单轮桩运行时（窗口可配；stream 记录入参后收尾）。 */
    private static ModelRuntime capturingRuntime(int window,
                                                 AtomicReference<List<ChatMessage>> received) {
        return new ModelRuntime() {
            @Override public dev.langchain4j.model.chat.ChatModel chat() {
                throw new UnsupportedOperationException("测试桩不走向线");
            }

            @Override public dev.langchain4j.model.chat.StreamingChatModel streaming() {
                return null;
            }

            @Override public AiMessage stream(List<ChatMessage> messages,
                    List<dev.langchain4j.agent.tool.ToolSpecification> tools,
                    Consumer<String> onDelta) {
                received.set(List.copyOf(messages));
                return AiMessage.from("done");
            }

            @Override public AiMessage invoke(List<ChatMessage> messages) {
                throw new UnsupportedOperationException("测试桩不走向线");
            }

            @Override public int contextWindowTokens() {
                return window;
            }
        };
    }

    @Test
    void systemPromptIsDeductedFromBudgetWhenWindowConfigured() {
        // 确定性计量：60 条历史 × (25 token 估算 + 4 固定开销) = 1740；
        // systemPrompt 4000 token；窗口 8192 → 预算 min(12000, 8192-4096)=4096。
        io.github.paopaonb666.agentbase.core.context.ContextEngine.setTokenCounter(
                EstimatingTokenCounter.INSTANCE);
        List<ChatMessage> history = new ArrayList<>();
        for (int i = 0; i < 60; i++) {
            history.add(UserMessage.from("x".repeat(100))); // ASCII 0.25/字 → 25 token
        }
        Settings settings = Settings.load(Map.of("ENV", "development"), null);

        // window=0：现状逐字节——全部历史保留（60）+ system 头部 = 61 条。
        AtomicReference<List<ChatMessage>> noWindow = new AtomicReference<>();
        AgentModule.ModuleContext ctxNoWindow = new AgentModule.ModuleContext(
                settings, capturingRuntime(0, noWindow), new InMemoryCheckpointer(),
                List.of(), null, null);
        CompiledGraph graphNoWindow = Graphs.buildSingleAgentGraph(
                ctxNoWindow, "chat", "系统提示", List.of());
        graphNoWindow.invoke(new GraphState(List.copyOf(history), Map.of()),
                RunConfig.of("chat:budget0", "u1"));
        assertEquals(61, noWindow.get().size());
        assertEquals(SystemMessage.from("系统提示"), noWindow.get().get(0));

        // window>0、无 systemPrompt：预算 4096 装下全部 1740 → 60 条原样。
        AtomicReference<List<ChatMessage>> windowOnly = new AtomicReference<>();
        AgentModule.ModuleContext ctxWindowOnly = new AgentModule.ModuleContext(
                settings, capturingRuntime(8192, windowOnly), new InMemoryCheckpointer(),
                List.of(), null, null);
        CompiledGraph graphWindowOnly = Graphs.buildSingleAgentGraph(
                ctxWindowOnly, "chat", null, List.of());
        graphWindowOnly.invoke(new GraphState(List.copyOf(history), Map.of()),
                RunConfig.of("chat:budget1", "u1"));
        assertEquals(60, windowOnly.get().size());

        // window>0 + systemPrompt（4000 token）：扣减诚实生效 → 预算
        // 4096-4000=96 → 保尾部 3 条（29/条：3×29=87 ≤ 96 < 4×29）+ system
        // 头部原样 = 捕获输入 4 条（不截断 prompt，挤占的是历史体；floor 仅在
        // raw < 1 时兜底——本例 raw=96 ≥ 1）。
        AtomicReference<List<ChatMessage>> deducted = new AtomicReference<>();
        AgentModule.ModuleContext ctxDeducted = new AgentModule.ModuleContext(
                settings, capturingRuntime(8192, deducted), new InMemoryCheckpointer(),
                List.of(), null, null);
        String bigPrompt = "y".repeat(16000); // 4000 token
        CompiledGraph graphDeducted = Graphs.buildSingleAgentGraph(
                ctxDeducted, "chat", bigPrompt, List.of());
        graphDeducted.invoke(new GraphState(List.copyOf(history), Map.of()),
                RunConfig.of("chat:budget2", "u1"));
        assertEquals(4, deducted.get().size());
        assertEquals(SystemMessage.from(bigPrompt), deducted.get().get(0));
    }
}
