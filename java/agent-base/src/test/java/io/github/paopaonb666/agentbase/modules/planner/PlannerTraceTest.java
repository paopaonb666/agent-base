package io.github.paopaonb666.agentbase.modules.planner;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import io.github.paopaonb666.agentbase.core.AgentModule;
import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.core.graph.*;
import io.github.paopaonb666.agentbase.core.llm.ModelRuntime;
import io.github.paopaonb666.agentbase.extensions.events.Events;
import io.github.paopaonb666.agentbase.modules.planner.graph.PlannerGraphs;
import io.github.paopaonb666.agentbase.modules.planner.graph.PlannerState;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

/** 临时 trace：planner cursor 流转。 */
class PlannerTraceTest {

    static final class Fake implements ModelRuntime {
        final List<AiMessage> script;
        final AtomicInteger idx = new AtomicInteger();

        Fake(List<AiMessage> script) { this.script = script; }

        @Override public ChatModel chat() { throw new UnsupportedOperationException(); }
        @Override public StreamingChatModel streaming() { return null; }

        @Override
        public AiMessage stream(List<ChatMessage> messages, List<dev.langchain4j.agent.tool.ToolSpecification> tools,
                                java.util.function.Consumer<String> onDelta) {
            int i = idx.getAndIncrement();
            AiMessage m = i < script.size() ? script.get(i) : AiMessage.from("fallback");
            onDelta.accept(m.text());
            return m;
        }

        @Override
        public AiMessage invoke(List<ChatMessage> messages) {
            return script.get(idx.getAndIncrement());
        }
    }

    @Test
    void trace() {
        Fake model = new Fake(List.of(
                AiMessage.from("[{\"id\": 1, \"goal\": \"A\"}, {\"id\": 2, \"goal\": \"B\"}]"),
                AiMessage.from("A 的结果"),
                AiMessage.from("B 的结果"),
                AiMessage.from("综合答复：A 与 B 都完成了")));
        Settings settings = new Settings();
        // 与失败测试同构：带 RecordingPlanStore。
        List<Object> saves = new ArrayList<>();
        AgentModule.PlanStorePort planStore = new AgentModule.PlanStorePort() {
            @Override public void savePlan(String threadId, String module,
                    List<Map<String, Object>> tasks, int cursor, int replans) {
                System.out.println("SAVE cursor=" + cursor + " replans=" + replans);
            }
            @Override public Map<String, Object> getPlan(String threadId) { return null; }
            @Override public void deletePlan(String threadId) { }
        };
        AgentModule.ModuleContext ctx = new AgentModule.ModuleContext(
                settings, model, new InMemoryCheckpointer(), List.of(), null, planStore);
        CompiledGraph graph = PlannerGraphs.buildPlannerGraph(ctx);
        List<Object> events = new ArrayList<>();
        GraphState result = graph.stream(
                new GraphState(List.of(UserMessage.from("做 A 和 B")), Map.of()),
                new RunConfig("planner:trace", "u1", 25),
                events::add);
        System.out.println("CURSOR=" + result.getOrDefault(PlannerState.CURSOR, -1));
        System.out.println("TASKS=" + result.getOrDefault(PlannerState.TASKS, List.of()));
        for (Object e : events) {
            if (e instanceof Events.StepEvent s) {
                System.out.println("STEP " + s.name() + " " + s.status());
            } else if (e instanceof Events.PlanEvent p) {
                System.out.println("PLAN " + p.status() + " " + p.detail());
            }
        }
    }
}
