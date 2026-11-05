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
import io.github.paopaonb666.agentbase.core.context.ContextEngine;
import io.github.paopaonb666.agentbase.core.graph.CompiledGraph;
import io.github.paopaonb666.agentbase.core.graph.GraphState;
import io.github.paopaonb666.agentbase.core.graph.InMemoryCheckpointer;
import io.github.paopaonb666.agentbase.core.graph.RunConfig;
import io.github.paopaonb666.agentbase.core.llm.ModelRuntime;
import io.github.paopaonb666.agentbase.extensions.events.Events;
import io.github.paopaonb666.agentbase.modules.chat.ChatModule;
import org.junit.jupiter.api.Test;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * supervisor 模板的测试——对齐 Python tests/test_collab.py 的断言面。
 *
 * <p>脚本化模型同时扮演两个角色：supervisor 节点（它的第一条响应是一次
 * 交给 {@code chat} 的 handoff 工具调用）以及 chat sub-agent（它的第二条
 * 响应是答案）。因此路由可以在没有真实 LLM 的情况下端到端地断言。
 *
 * <p>脚本化模型替身在<b>本测试文件内部</b>定义（不动 core/llm 包——那里
 * 有并行的 ScriptedModels）；chat 模块用真实的 {@link ChatModule}，但
 * ModuleContext 直接构造（llm 换 fake）。
 */
class SupervisorGraphTest {

    /** 脚本化测试替身：每次 stream 调用弹出一条脚本化 AiMessage。 */
    private static final class ScriptedModel implements ModelRuntime {
        private final Deque<AiMessage> script = new ArrayDeque<>();
        /** 每次模型调用收到的工具规格（handoff 绑定断言用）。 */
        final List<List<ToolSpecification>> receivedSpecs = new ArrayList<>();

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
            receivedSpecs.add(tools == null ? List.of() : List.copyOf(tools));
            AiMessage reply = next();
            String text = reply.text() == null ? "" : reply.text();
            for (int i = 0; i < text.length(); i += 4) {
                onDelta.accept(text.substring(i, Math.min(text.length(), i + 4)));
            }
            return reply;
        }

        @Override
        public AiMessage invoke(List<ChatMessage> messages) {
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
    void supervisorRoutesToChatAndSynthesizes() {
        // 第 1 次调用 = supervisor 决策：transfer_to_chat 交接；
        // 第 2 次调用 = chat sub-agent 起草答案；
        // 第 3 次调用 = supervisor 合成最终回复。
        ScriptedModel model = new ScriptedModel(
                handoff("chat", "handoff-1"),
                AiMessage.from("draft: the essay body"),
                AiMessage.from("final: your essay is ready"));
        CompiledGraph graph = SupervisorGraphs.buildSupervisorGraph(
                ctx(model), Map.of("chat", new ChatModule()));
        List<Object> events = new ArrayList<>();
        GraphState result = graph.stream(
                new GraphState(List.of(UserMessage.from("write me an essay")), Map.of()),
                new RunConfig("sup:t1", "u1", 25),
                events::add);

        // 对话经过了 chat sub-agent（它的回复出现在最终状态里），且
        // supervisor 产出了一条收尾消息。
        List<String> contents = new ArrayList<>();
        for (ChatMessage message : result.messages()) {
            contents.add(ContextEngine.contentText(message));
        }
        assertTrue(contents.contains("draft: the essay body"));
        assertEquals("final: your essay is ready", contents.get(contents.size() - 1));
        // 子 agent 的答案以 handoff 工具结果（ToolMessage）回填历史。
        assertTrue(result.messages().stream().anyMatch(
                m -> m instanceof ToolExecutionResultMessage tool
                        && "transfer_to_chat".equals(tool.toolName())
                        && tool.text().contains("draft: the essay body")));

        // supervisor 的模型输入只绑定 handoff 工具：名字与 description 来自模块契约。
        List<ToolSpecification> specs = model.receivedSpecs.get(0);
        assertEquals(1, specs.size());
        assertEquals("transfer_to_chat", specs.get(0).name());
        assertEquals(new ChatModule().description(), specs.get(0).description());
        // 第二次绑定是 chat 子图自己的（无工具：ctx.tools 为空）。
        assertTrue(model.receivedSpecs.get(1).isEmpty());

        // 事件序列：supervisor / handoffs 的 step + 子图节点事件透传 + delta。
        Set<String> stepNames = new HashSet<>();
        int deltas = 0;
        for (Object event : events) {
            if (event instanceof Events.StepEvent step) {
                stepNames.add(step.name());
            } else if (event instanceof Events.DeltaEvent) {
                deltas++;
            }
        }
        assertTrue(stepNames.contains("supervisor"));
        assertTrue(stepNames.contains("handoffs"));
        assertTrue(stepNames.contains("agent")); // 子图的 LLM 节点事件被转发到父 sink
        assertTrue(deltas > 0); // 子 agent 与 supervisor 的 token 增量都直达流
    }

    @Test
    void unknownHandoffTargetIsNormalizedToolResult() {
        // 无匹配模块 → 工具结果文本说明未知目标（模型自愈重路由），图不崩。
        ScriptedModel model = new ScriptedModel(
                handoff("nobody", "handoff-x"),
                AiMessage.from("明白了，目标不存在，我直接回答"));
        CompiledGraph graph = SupervisorGraphs.buildSupervisorGraph(
                ctx(model), Map.of("chat", new ChatModule()));
        GraphState result = graph.stream(
                new GraphState(List.of(UserMessage.from("hello")), Map.of()),
                new RunConfig("sup:t1", "u1", 25),
                event -> { });

        assertTrue(result.messages().stream().anyMatch(m -> m instanceof ToolExecutionResultMessage tool
                && tool.text().contains("unknown handoff target 'nobody'")
                && tool.text().contains("chat"))); // 可用目标列在错误文本里
        List<ChatMessage> messages = result.messages();
        assertEquals("明白了，目标不存在，我直接回答",
                ContextEngine.contentText(messages.get(messages.size() - 1)).strip());
    }

    @Test
    void supervisorRequiresModules() {
        // 对模块的要求在这里作为早期失败强制执行。
        IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> SupervisorGraphs.buildSupervisorGraph(ctx(new ScriptedModel()), Map.of()));
        assertTrue(error.getMessage().contains("at least one module"));
    }

    @Test
    void supervisorPromptAllowsMultiStepDispatch() {
        // D1：prompt 必须允许多步顺序派发，不再含"不要多调"的束缚。
        String prompt = SupervisorGraphs.DEFAULT_PROMPT;
        assertTrue(prompt.contains("steps"));
        assertTrue(prompt.contains("sequence"));
        assertTrue(prompt.contains("as many agents as the task requires"));
        assertTrue(!prompt.contains("Do not call an agent more than necessary"));
    }
}
