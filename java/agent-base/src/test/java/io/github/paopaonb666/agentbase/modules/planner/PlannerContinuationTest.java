package io.github.paopaonb666.agentbase.modules.planner;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import io.github.paopaonb666.agentbase.core.AgentModule;
import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.core.graph.CompiledGraph;
import io.github.paopaonb666.agentbase.core.graph.GraphState;
import io.github.paopaonb666.agentbase.core.graph.InMemoryCheckpointer;
import io.github.paopaonb666.agentbase.core.graph.RunConfig;
import io.github.paopaonb666.agentbase.core.llm.ModelRuntime;
import io.github.paopaonb666.agentbase.extensions.events.Events;
import io.github.paopaonb666.agentbase.modules.planner.graph.PlannerGraphs;
import io.github.paopaonb666.agentbase.modules.planner.graph.PlannerState;
import org.junit.jupiter.api.Test;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 跨轮 plan 延续（#66）：continue_plan 注入快照后 plan 节点跳过重拆、
 * 从 cursor 续跑、plan 事件 status=progress。
 */
class PlannerContinuationTest {

    /** 脚本化模型替身（plan 被调用即意味着脚本错位——poll 顺序暴露）。 */
    private static final class ScriptedModel implements ModelRuntime {
        private final ArrayDeque<AiMessage> script = new ArrayDeque<>();
        final List<List<ChatMessage>> received = new ArrayList<>();

        ScriptedModel(String... responses) {
            for (String response : responses) {
                script.add(AiMessage.from(response));
            }
        }

        @Override
        public ChatModel chat() {
            return new ChatModel() {
                @Override
                public ChatResponse doChat(ChatRequest request) {
                    return ChatResponse.builder().aiMessage(script.poll()).build();
                }
            };
        }

        @Override
        public dev.langchain4j.model.chat.StreamingChatModel streaming() {
            return null;
        }

        @Override
        public AiMessage stream(List<ChatMessage> messages, List<dev.langchain4j.agent.tool.ToolSpecification> tools,
                                Consumer<String> onDelta) {
            return invoke(messages);
        }

        @Override
        public AiMessage invoke(List<ChatMessage> messages) {
            received.add(List.copyOf(messages));
            AiMessage next = script.poll();
            if (next == null) {
                throw new IllegalStateException("scripted responses exhausted");
            }
            return next;
        }
    }

    private static Map<String, Object> task(int id, String status, String goal, String summary) {
        Map<String, Object> t = new LinkedHashMap<>();
        t.put(PlannerState.ID, id);
        t.put(PlannerState.GOAL, goal);
        t.put(PlannerState.STATUS, status);
        t.put(PlannerState.SUMMARY, summary);
        t.put(PlannerState.ATTEMPTS, "done".equals(status) ? 1 : 0);
        return t;
    }

    @Test
    void seededPlanContinuesFromCursorWithoutReplanning() {
        // 脚本从 execute 的输出开始——plan 节点若发起 LLM 调用就会吃掉这
        // 条消息，后续错位必炸（脚本即断言）。
        ScriptedModel model = new ScriptedModel(
                "任务 2 的结果",
                "综合：两步都完成");
        List<Map<String, Object>> snapshotTasks = List.of(
                task(1, "done", "第一步", "第一步的结果"),
                task(2, "pending", "第二步", ""));
        AgentModule.ModuleContext ctx = new AgentModule.ModuleContext(
                new Settings(), model, new InMemoryCheckpointer(), List.of(), null, null);
        CompiledGraph graph = PlannerGraphs.buildPlannerGraph(ctx);
        Map<String, Object> extra = new LinkedHashMap<>();
        extra.put(PlannerState.TASKS, snapshotTasks);
        extra.put(PlannerState.CURSOR, 1);
        extra.put(PlannerState.REPLANS, 0);
        List<Object> events = new ArrayList<>();
        GraphState result = graph.stream(
                new GraphState(List.of(UserMessage.from("继续")), extra),
                new RunConfig("planner:c1", "u1", 25),
                events::add);

        // 一次模型调用 = execute（任务 2）；plan/replan 都没碰 LLM
        assertEquals(2, model.received.size());
        // 任务 2 完成，任务 1 保留 done
        List<Map<String, Object>> tasks = result.getOrDefault(PlannerState.TASKS, List.of());
        assertEquals("done", tasks.get(0).get(PlannerState.STATUS));
        assertEquals("第一步的结果", tasks.get(0).get(PlannerState.SUMMARY));
        assertEquals("done", tasks.get(1).get(PlannerState.STATUS));
        assertEquals("任务 2 的结果", tasks.get(1).get(PlannerState.SUMMARY));
        assertEquals(0, result.getOrDefault(PlannerState.REPLANS, -1));
        // plan 事件：progress（延续）→ done（synthesize 收口）
        List<Events.PlanEvent> planEvents = new ArrayList<>();
        for (Object event : events) {
            if (event instanceof Events.PlanEvent p) {
                planEvents.add(p);
            }
        }
        assertTrue(planEvents.size() >= 2);
        assertEquals("progress", planEvents.get(0).status());
        assertEquals("done", planEvents.get(planEvents.size() - 1).status());
        // 最终综合回复进消息流
        List<ChatMessage> messages = result.messages();
        assertEquals("综合：两步都完成",
                io.github.paopaonb666.agentbase.core.context.ContextEngine
                        .contentText(messages.get(messages.size() - 1)).strip());
    }
}
