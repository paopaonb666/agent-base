package io.github.paopaonb666.agentbase.extensions.collab;

import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.agent.tool.ToolSpecification;
import io.github.paopaonb666.agentbase.core.AgentModule;
import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.core.graph.CompiledGraph;
import io.github.paopaonb666.agentbase.core.graph.GraphState;
import io.github.paopaonb666.agentbase.core.graph.InMemoryCheckpointer;
import io.github.paopaonb666.agentbase.core.graph.RunConfig;
import io.github.paopaonb666.agentbase.core.llm.ModelRuntime;
import io.github.paopaonb666.agentbase.modules.chat.ChatModule;
import org.junit.jupiter.api.Test;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * handoff 预算（#68）：run 级计数器超限后不再执行子图，supervisor 收到
 * 结构化收敛提示；计数器每轮（init 节点）清零。
 */
class SupervisorHandoffBudgetTest {

    /** 脚本化替身（与 SupervisorGraphTest 同构，局部自包含）。 */
    private static final class ScriptedModel implements ModelRuntime {
        private final Deque<AiMessage> script = new ArrayDeque<>();

        ScriptedModel(AiMessage... responses) {
            for (AiMessage response : responses) {
                script.add(response);
            }
        }

        private AiMessage next() {
            AiMessage next = script.poll();
            if (next == null) {
                throw new IllegalStateException("scripted responses exhausted");
            }
            return next;
        }

        @Override
        public ChatModel chat() {
            return new ChatModel() {
                @Override
                public ChatResponse doChat(ChatRequest request) {
                    return ChatResponse.builder().aiMessage(next()).build();
                }
            };
        }

        @Override
        public StreamingChatModel streaming() {
            return null;
        }

        @Override
        public AiMessage stream(List<ChatMessage> messages, List<ToolSpecification> tools,
                                Consumer<String> onDelta) {
            return next();
        }

        @Override
        public AiMessage invoke(List<ChatMessage> messages) {
            return next();
        }
    }

    private static AiMessage handoff(String target, String callId) {
        return new AiMessage("", List.of(ToolExecutionRequest.builder()
                .id(callId)
                .name(SupervisorGraphs.HANDOFF_TOOL_PREFIX + target)
                .arguments("{}")
                .build()));
    }

    private static AgentModule.ModuleContext ctx(ModelRuntime model, int maxPerRun) {
        Settings settings = Settings.load(Map.of(
                "AGENT_HANDOFF_MAX_PER_RUN", String.valueOf(maxPerRun)), null);
        return new AgentModule.ModuleContext(settings, model, new InMemoryCheckpointer(),
                List.of(), null, null);
    }

    @Test
    void budgetExhaustionForcesConvergenceWithoutRunningSubgraph() {
        // 上限 1：第一次 handoff 正常执行（0→1）；supervisor 再要求交接时
        // 预算已耗尽 → 结构化提示回填，sub-graph 不再跑（脚本里没有给
        // sub-agent 的位置——若仍执行会 poll 出 final 消息导致断言失败）。
        ScriptedModel model = new ScriptedModel(
                handoff("chat", "h1"),
                AiMessage.from("draft: 子图回答"),
                handoff("chat", "h2"),
                AiMessage.from("final: 基于已有信息收尾"));
        CompiledGraph graph = SupervisorGraphs.buildSupervisorGraph(
                ctx(model, 1), Map.of("chat", new ChatModule()));
        GraphState result = graph.stream(
                new GraphState(List.of(UserMessage.from("做点事")), Map.of()),
                new RunConfig("sup:b1", "u1", 25),
                event -> {
                });

        // 第一次 handoff 正常执行
        assertTrue(result.messages().stream().anyMatch(
                m -> m instanceof ToolExecutionResultMessage tool
                        && tool.text().contains("draft: 子图回答")));
        // 第二次 handoff 被预算拦下：ToolMessage 是收敛提示而非子图输出
        assertTrue(result.messages().stream().anyMatch(
                m -> m instanceof ToolExecutionResultMessage tool
                        && tool.text().contains("handoff 预算耗尽")
                        && tool.text().contains("上限 1")));
        // supervisor 收敛出最终回答（不是超限裸错误）
        List<ChatMessage> messages = result.messages();
        assertEquals("final: 基于已有信息收尾",
                io.github.paopaonb666.agentbase.core.context.ContextEngine
                        .contentText(messages.get(messages.size() - 1)).strip());
        // 计数器状态：本轮共执行 1 次
        assertEquals(1, ((Number) result.get(SupervisorGraphs.HANDOFF_COUNT)).intValue());
    }

    @Test
    void unlimitedWhenConfiguredZero() {
        // 0 = 不限：三次交接都执行（sub-agent 固定回复，脚本按序弹出）
        ScriptedModel model = new ScriptedModel(
                handoff("chat", "h1"),
                AiMessage.from("第一次"),
                handoff("chat", "h2"),
                AiMessage.from("第二次"),
                handoff("chat", "h3"),
                AiMessage.from("第三次"),
                AiMessage.from("final: 完成"));
        CompiledGraph graph = SupervisorGraphs.buildSupervisorGraph(
                ctx(model, 0), Map.of("chat", new ChatModule()));
        GraphState result = graph.stream(
                new GraphState(List.of(UserMessage.from("做点事")), Map.of()),
                new RunConfig("sup:b2", "u1", 25),
                event -> {
                });
        assertEquals(3, ((Number) result.get(SupervisorGraphs.HANDOFF_COUNT)).intValue());
        assertTrue(result.messages().stream().noneMatch(
                m -> m instanceof ToolExecutionResultMessage tool
                        && tool.text().contains("预算耗尽")));
    }
}
