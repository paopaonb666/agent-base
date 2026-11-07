package io.github.paopaonb666.agentbase.modules.planner;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.agent.tool.ToolSpecification;
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
import io.github.paopaonb666.agentbase.core.context.ContextEngine;
import io.github.paopaonb666.agentbase.core.llm.ModelRuntime;
import io.github.paopaonb666.agentbase.extensions.events.Events;
import io.github.paopaonb666.agentbase.modules.planner.graph.PlannerGraphs;
import io.github.paopaonb666.agentbase.modules.planner.graph.PlannerState;
import org.junit.jupiter.api.Test;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * planner 图端到端测试（B4）——对齐 Python tests/test_planner_graph.py 的
 * 四条执行路径（本文件覆盖 1/2/3/4 + 最短摘要门）：
 *
 * <ol>
 *   <li>全成功：plan → 逐条 execute → synthesize；</li>
 *   <li>失败 → replan → 成功（replans=1）；</li>
 *   <li>replan 预算耗尽：仍走完 → best-effort synthesize；</li>
 *   <li>空输出 → 空输出门改判 failed → replan（评审修正 3 守护）。</li>
 * </ol>
 *
 * <p>脚本化模型替身在<b>本测试文件内部</b>定义（不动 core/llm 包——那里
 * 有并行的 ScriptedModels）：plan/replan 走非流式 {@code invoke}，execute/
 * synthesize 走流式 {@code stream}（按 ~4 字符切片回调 onDelta），共用
 * 一份脚本队列——与生产 ModelRuntime 的两条调用路径同构。
 */
class PlannerGraphTest {

    /** 脚本化测试替身：每次调用弹出一条脚本化 AiMessage；耗尽即抛。 */
    private static final class ScriptedModel implements ModelRuntime {
        private final Deque<AiMessage> script = new ArrayDeque<>();
        final List<List<ChatMessage>> received = new ArrayList<>();

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
            received.add(List.copyOf(messages));
            AiMessage reply = next();
            String text = reply.text() == null ? "" : reply.text();
            for (int i = 0; i < text.length(); i += 4) {
                onDelta.accept(text.substring(i, Math.min(text.length(), i + 4)));
            }
            return reply;
        }

        @Override
        public AiMessage invoke(List<ChatMessage> messages) {
            received.add(List.copyOf(messages));
            return next();
        }
    }

    /** 捕获 save_plan 调用的测试替身（对应 Python _CapturePlanStore）。 */
    private static final class RecordingPlanStore implements AgentModule.PlanStorePort {
        record Save(String threadId, String module, List<Map<String, Object>> tasks,
                    int cursor, int replans) {}
        final List<Save> saves = new ArrayList<>();

        @Override
        public void savePlan(String threadId, String module, List<Map<String, Object>> tasks,
                             int cursor, int replans) {
            saves.add(new Save(threadId, module, PlannerState.copyTasks(tasks), cursor, replans));
        }

        @Override
        public Map<String, Object> getPlan(String threadId) {
            return null;
        }

        @Override
        public void deletePlan(String threadId) {
        }
    }

    private static AgentModule.ModuleContext ctx(ModelRuntime model, InMemoryCheckpointer checkpointer,
                                                 AgentModule.PlanStorePort planStore) {
        return new AgentModule.ModuleContext(new Settings(), model, checkpointer, List.of(), null, planStore);
    }

    private static List<Events.PlanEvent> planEvents(List<Object> events) {
        List<Events.PlanEvent> plan = new ArrayList<>();
        for (Object event : events) {
            if (event instanceof Events.PlanEvent p) {
                plan.add(p);
            }
        }
        return plan;
    }

    /** 1. 全成功：plan → 逐条 execute → synthesize。 */
    @Test
    void allSuccessPath() {
        ScriptedModel model = new ScriptedModel(
                AiMessage.from("[{\"id\": 1, \"goal\": \"A\"}, {\"id\": 2, \"goal\": \"B\"}]"),
                AiMessage.from("A 的结果"),
                AiMessage.from("B 的结果"),
                AiMessage.from("综合答复：A 与 B 都完成了"));
        RecordingPlanStore planStore = new RecordingPlanStore();
        CompiledGraph graph = PlannerGraphs.buildPlannerGraph(
                ctx(model, new InMemoryCheckpointer(), planStore));
        List<Object> events = new ArrayList<>();
        GraphState result = graph.stream(
                new GraphState(List.of(UserMessage.from("做 A 和 B")), Map.of()),
                new RunConfig("planner:t1", "u1", 25),
                events::add);

        // tasks/cursor/replans 状态断言。
        List<Map<String, Object>> tasks = result.getOrDefault(PlannerState.TASKS, List.of());
        assertEquals(2, tasks.size());
        assertEquals("done", tasks.get(0).get(PlannerState.STATUS));
        assertEquals("done", tasks.get(1).get(PlannerState.STATUS));
        assertEquals("A 的结果", tasks.get(0).get(PlannerState.SUMMARY));
        assertEquals(1, tasks.get(0).get(PlannerState.ATTEMPTS));
        assertEquals(2, result.getOrDefault(PlannerState.CURSOR, -1));
        assertEquals(0, result.getOrDefault(PlannerState.REPLANS, -1));
        // 最后一条消息是 synthesize 的综合答复。
        List<ChatMessage> messages = result.messages();
        assertEquals("综合答复：A 与 B 都完成了",
                ContextEngine.contentText(messages.get(messages.size() - 1)).strip());

        // PlanEvent 序列：created（拆解出 2 个子任务）→ progress×4 → done；
        // 每次携带全量清单（created 时全部 pending）。
        List<Events.PlanEvent> plan = planEvents(events);
        assertEquals("created", plan.get(0).status());
        assertEquals("拆解出 2 个子任务", plan.get(0).detail());
        assertEquals(2, plan.get(0).plan().size());
        assertEquals("pending", plan.get(0).plan().get(0).status());
        assertEquals("done", plan.get(plan.size() - 1).status());
        // 最终全量清单与 Python 一致保持原始顺序（无轮转）：#1 done、#2 done。
        assertEquals(1, plan.get(plan.size() - 1).plan().get(0).id());
        assertEquals("done", plan.get(plan.size() - 1).plan().get(0).status());
        assertEquals(2, plan.get(plan.size() - 1).plan().get(1).id());
        assertEquals("done", plan.get(plan.size() - 1).plan().get(1).status());

        // planStore 假实现收到 savePlan（plan/check×2/synthesize 共 4 次）。
        assertEquals(4, planStore.saves.size());
        assertEquals("planner:t1", planStore.saves.get(0).threadId());
        assertEquals("planner", planStore.saves.get(0).module());
        assertEquals(0, planStore.saves.get(0).cursor());

        // execute / synthesize 的 token 增量以 DeltaEvent 透传。
        assertTrue(events.stream().anyMatch(e -> e instanceof Events.DeltaEvent));

        // plan 节点的模型输入是单条 human 消息（PLAN_PROMPT）。
        assertTrue(model.received.get(0).get(0) instanceof UserMessage);
        assertTrue(ContextEngine.contentText(model.received.get(0).get(0))
                .startsWith("你是任务规划器"));
    }

    /** 2/4. 空输出 → 空输出门改判 failed → replan → 成功（replans=1）。 */
    @Test
    void emptyOutputGateTriggersReplan() {
        ScriptedModel model = new ScriptedModel(
                AiMessage.from("[{\"id\": 1, \"goal\": \"A\"}]"),
                AiMessage.from(""), // execute 停止但 summary 为空 → 门改判 failed
                AiMessage.from("[{\"id\": 1, \"goal\": \"重做 A\"}]"),
                AiMessage.from("这次有产出"),
                AiMessage.from("最终答复"));
        RecordingPlanStore planStore = new RecordingPlanStore();
        CompiledGraph graph = PlannerGraphs.buildPlannerGraph(
                ctx(model, new InMemoryCheckpointer(), planStore));
        List<Object> events = new ArrayList<>();
        GraphState result = graph.stream(
                new GraphState(List.of(UserMessage.from("做 A")), Map.of()),
                new RunConfig("planner:t1", "u1", 25),
                events::add);

        assertEquals(1, result.getOrDefault(PlannerState.REPLANS, -1));
        List<Map<String, Object>> tasks = result.getOrDefault(PlannerState.TASKS, List.of());
        assertEquals(1, tasks.size());
        assertEquals("done", tasks.get(0).get(PlannerState.STATUS));
        List<ChatMessage> messages = result.messages();
        assertEquals("最终答复", ContextEngine.contentText(messages.get(messages.size() - 1)).strip());

        // 事件序列含 replanned；首次 check 的快照里任务已被门改判 failed。
        List<String> statuses = planEvents(events).stream().map(Events.PlanEvent::status).toList();
        assertTrue(statuses.contains("replanned"));
        assertEquals("created", statuses.get(0));
        assertEquals("done", statuses.get(statuses.size() - 1));
        RecordingPlanStore.Save firstCheck = planStore.saves.get(1);
        assertEquals("failed", firstCheck.tasks().get(0).get(PlannerState.STATUS));
        assertEquals("输出为空", firstCheck.tasks().get(0).get(PlannerState.SUMMARY));
    }

    /** 3. replan 预算耗尽：仍走完 → best-effort synthesize（如实汇报失败）。 */
    @Test
    void replanBudgetExhaustedSynthesizesBestEffort() {
        ScriptedModel model = new ScriptedModel(
                AiMessage.from("[{\"id\": 1, \"goal\": \"A\"}]"),
                AiMessage.from(""), // execute：空输出 → failed
                AiMessage.from("不是 JSON"), // replan 1：解析失败 → failed 目标兜底单任务
                AiMessage.from(""), // execute：又空输出 → failed
                AiMessage.from("还不是 JSON"), // replan 2（预算最后一次）
                AiMessage.from(""), // execute：仍空输出 → failed
                AiMessage.from("如实汇报：没有完成任何子任务")); // replans=2 == max → synthesize
        CompiledGraph graph = PlannerGraphs.buildPlannerGraph(
                ctx(model, new InMemoryCheckpointer(), null));
        List<Object> events = new ArrayList<>();
        GraphState result = graph.stream(
                new GraphState(List.of(UserMessage.from("做 A")), Map.of()),
                new RunConfig("planner:t1", "u1", 25),
                events::add);

        assertEquals(2, result.getOrDefault(PlannerState.REPLANS, -1));
        List<ChatMessage> messages = result.messages();
        assertEquals("如实汇报：没有完成任何子任务",
                ContextEngine.contentText(messages.get(messages.size() - 1)).strip());
        List<Map<String, Object>> tasks = result.getOrDefault(PlannerState.TASKS, List.of());
        assertTrue(tasks.stream().anyMatch(t -> "failed".equals(t.get(PlannerState.STATUS))));
        // replanned 事件恰好两次（预算内）。
        assertEquals(2, planEvents(events).stream().filter(p -> "replanned".equals(p.status())).count());
    }

    /** 最小摘要长度门（PLANNER_MIN_SUMMARY_CHARS）：过短输出改判 failed。 */
    @Test
    void minSummaryCharsGate() {
        ScriptedModel model = new ScriptedModel(
                AiMessage.from("[{\"id\": 1, \"goal\": \"A\"}]"),
                AiMessage.from("太短"),
                AiMessage.from("[{\"id\": 1, \"goal\": \"重做 A\"}]"),
                AiMessage.from("这次是一段足够长的产出摘要"),
                AiMessage.from("最终答复"));
        Settings settings = new Settings();
        settings.planner.minSummaryChars = 10;
        RecordingPlanStore planStore = new RecordingPlanStore();
        CompiledGraph graph = PlannerGraphs.buildPlannerGraph(new AgentModule.ModuleContext(
                settings, model, new InMemoryCheckpointer(), List.of(), null, planStore));
        GraphState result = graph.stream(
                new GraphState(List.of(UserMessage.from("做 A")), Map.of()),
                new RunConfig("planner:t1", "u1", 25),
                event -> { /* 事件旁路：本测试只看落库快照 */ });

        assertEquals(1, result.getOrDefault(PlannerState.REPLANS, -1));
        // 首次 check 的快照：过短输出被改判 failed，摘要说明门槛。
        RecordingPlanStore.Save firstCheck = planStore.saves.get(1);
        assertEquals("failed", firstCheck.tasks().get(0).get(PlannerState.STATUS));
        assertEquals("输出不足 10 字", firstCheck.tasks().get(0).get(PlannerState.SUMMARY));
    }

    /** #259：done 占位后重拆产出超限 → replanned detail 标注 dropped_tasks，截断行为不动。 */
    @Test
    void replanTruncationAnnotatesDroppedTasks() {
        ScriptedModel model = new ScriptedModel(
                AiMessage.from("[{\"id\": 1, \"goal\": \"A\"}, {\"id\": 2, \"goal\": \"B\"}]"),
                AiMessage.from("A 的结果"),
                AiMessage.from(""), // execute B：空输出 → 门改判 failed
                AiMessage.from("[{\"id\": 1, \"goal\": \"重做 A\"}, {\"id\": 2, \"goal\": \"重做 B\"}]"),
                AiMessage.from("重做的结果"),
                AiMessage.from("最终答复"));
        Settings settings = new Settings();
        settings.planner.maxSubtasks = 2;
        RecordingPlanStore planStore = new RecordingPlanStore();
        CompiledGraph graph = PlannerGraphs.buildPlannerGraph(new AgentModule.ModuleContext(
                settings, model, new InMemoryCheckpointer(), List.of(), null, planStore));
        List<Object> events = new ArrayList<>();
        GraphState result = graph.stream(
                new GraphState(List.of(UserMessage.from("做 A 和 B")), Map.of()),
                new RunConfig("planner:t1", "u1", 25),
                events::add);

        assertEquals(1, result.getOrDefault(PlannerState.REPLANS, -1));
        // done(1) + 重拆(2) = 3 > max_subtasks=2：清单截到 2，丢弃 1。
        List<Map<String, Object>> tasks = result.getOrDefault(PlannerState.TASKS, List.of());
        assertEquals(2, tasks.size());
        Events.PlanEvent replanned = planEvents(events).stream()
                .filter(p -> "replanned".equals(p.status())).findFirst().orElseThrow();
        assertEquals(2, replanned.plan().size());
        assertTrue(replanned.detail().contains("dropped_tasks=1"));
    }

    /** 解析兜底：模型输出非 JSON → 以目标兜底单任务，规划不失败。 */
    @Test
    void unparseablePlanFallsBackToSingleTask() {
        ScriptedModel model = new ScriptedModel(
                AiMessage.from("我觉得应该分两步：先查再写（不是 JSON）"),
                AiMessage.from("兜底执行的结果"),
                AiMessage.from("综合完成"));
        CompiledGraph graph = PlannerGraphs.buildPlannerGraph(
                ctx(model, new InMemoryCheckpointer(), null));
        List<Object> events = new ArrayList<>();
        GraphState result = graph.stream(
                new GraphState(List.of(UserMessage.from("做个复杂的事")), Map.of()),
                new RunConfig("planner:t1", "u1", 25),
                events::add);

        List<Map<String, Object>> tasks = result.getOrDefault(PlannerState.TASKS, List.of());
        assertEquals(1, tasks.size());
        assertEquals(1, tasks.get(0).get(PlannerState.ID));
        assertEquals("做个复杂的事", tasks.get(0).get(PlannerState.GOAL));
        assertEquals("created", planEvents(events).get(0).status());
        assertEquals("拆解出 1 个子任务", planEvents(events).get(0).detail());
        assertFalse(planEvents(events).isEmpty());
    }
}
