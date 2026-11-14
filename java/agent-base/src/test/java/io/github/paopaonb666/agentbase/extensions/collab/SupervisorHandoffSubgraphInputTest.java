package io.github.paopaonb666.agentbase.extensions.collab;

import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
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
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * GUI 验收二轮 D1 回归锁：handoff 传入子图的 messages 不得携带尾部悬空的
 * handoff tool_calls AiMessage——父图的 ToolMessage 要等 handoff 返回后才
 * 回填，原样透传会让子图首轮 LLM 调用带着「assistant 带 tool_calls 却无
 * 后续 tool 消息」的非法序列，被 provider 以 invalid_request_error 拒绝
 * （生产实测：supervisor 首次 handoff 即 400，三连重试全败，run 异常终止）。
 */
class SupervisorHandoffSubgraphInputTest {

    /** 脚本化替身：额外记录每次模型调用收到的完整消息序列。 */
    private static final class RecordingModel implements ModelRuntime {
        private final Deque<AiMessage> script = new ArrayDeque<>();
        /** 每次 stream 调用收到的消息序列（按调用顺序）。 */
        final List<List<ChatMessage>> receivedMessages = new ArrayList<>();

        RecordingModel(AiMessage... responses) {
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
            receivedMessages.add(List.copyOf(messages));
            return next();
        }

        @Override
        public AiMessage invoke(List<ChatMessage> messages) {
            receivedMessages.add(List.copyOf(messages));
            return next();
        }
    }

    private static AgentModule.ModuleContext ctx(ModelRuntime model) {
        return new AgentModule.ModuleContext(new Settings(), model, new InMemoryCheckpointer(),
                List.of(), null, null);
    }

    private static AiMessage handoff(String target, String callId) {
        return new AiMessage("", List.of(ToolExecutionRequest.builder()
                .id(callId)
                .name(SupervisorGraphs.HANDOFF_TOOL_PREFIX + target)
                .arguments("{}")
                .build()));
    }

    @Test
    void subgraphInputCarriesNoDanglingHandoffToolCalls() {
        // 脚本：supervisor 决策交接 → chat 子图作答 → supervisor 收尾。
        RecordingModel model = new RecordingModel(
                handoff("chat", "h1"),
                AiMessage.from("draft: 子图回答"),
                AiMessage.from("final: 收尾"));
        CompiledGraph graph = SupervisorGraphs.buildSupervisorGraph(
                ctx(model), Map.of("chat", new ChatModule()));
        GraphState result = graph.stream(
                new GraphState(List.of(UserMessage.from("写一篇文章")), Map.of()),
                new RunConfig("sup:d1", "u1", 25),
                event -> {
                });

        // 子图首轮模型输入（第 2 次调用）不得含任何带 tool_calls 的 AiMessage
        // ——修复前这里会携带 supervisor 的 transfer_to_chat 悬空调用。
        assertTrue(model.receivedMessages.size() >= 2,
                "应有 supervisor 与子图两次以上模型调用，实际 " + model.receivedMessages.size());
        List<ChatMessage> subInput = model.receivedMessages.get(1);
        assertTrue(subInput.stream().noneMatch(m -> m instanceof AiMessage ai
                        && ai.hasToolExecutionRequests()),
                "子图输入泄漏了悬空的 handoff tool_calls: " + subInput);
        // 子图仍能看到用户请求（剥离只去掉调用壳，不丢上下文）。
        assertTrue(subInput.stream().anyMatch(m -> m instanceof UserMessage
                && ((UserMessage) m).singleText().contains("写一篇文章")));

        // 父图历史配对完整：h1 的 ToolMessage 以子图输出回填，supervisor 的
        // 收尾输入（第 3 次调用）能看到这条配对结果。
        assertTrue(result.messages().stream().anyMatch(
                m -> m instanceof ToolExecutionResultMessage tool
                        && "transfer_to_chat".equals(tool.toolName())
                        && tool.text().contains("draft: 子图回答")));
        List<ChatMessage> synthInput = model.receivedMessages.get(model.receivedMessages.size() - 1);
        assertTrue(synthInput.stream().anyMatch(m -> m instanceof ToolExecutionResultMessage tool
                        && "transfer_to_chat".equals(tool.toolName())
                        && tool.text().contains("draft: 子图回答")),
                "supervisor 收尾输入丢失了 handoff 配对结果");
        List<ChatMessage> messages = result.messages();
        assertEquals("final: 收尾",
                io.github.paopaonb666.agentbase.core.context.ContextEngine
                        .contentText(messages.get(messages.size() - 1)).strip());
    }

    @Test
    void messagesForSubgraphStripsOnlyTrailingDanglingCall() {
        // 单元面：只剥尾部悬空调用；配对完整的历史（tool_calls+tool 结果）
        // 原样保留；末尾非 tool_calls 时零拷贝返回。
        ToolExecutionRequest request = ToolExecutionRequest.builder()
                .id("h9").name("transfer_to_chat").arguments("{}").build();
        ToolExecutionResultMessage paired = ToolExecutionResultMessage.from("h9", "transfer_to_chat", "ok");
        List<ChatMessage> history = List.of(
                UserMessage.from("hi"),
                new AiMessage("", List.of(request)),
                paired,
                UserMessage.from("继续"));
        assertTrue(SupervisorGraphs.messagesForSubgraph(history) == history);

        List<ChatMessage> dangling = List.of(
                UserMessage.from("hi"),
                new AiMessage("", List.of(request)));
        List<ChatMessage> stripped = SupervisorGraphs.messagesForSubgraph(dangling);
        assertEquals(1, stripped.size());
        assertTrue(stripped.get(0) instanceof UserMessage);
    }
}
