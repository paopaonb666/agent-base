package io.github.paopaonb666.agentbase.core.graph;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.UserMessage;
import io.github.paopaonb666.agentbase.extensions.events.Events;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * #7 steering 注入矩阵与事件序 golden：插话经 {@link RunRegistry} 队列，
 * 图执行线程在步边界（循环顶取消检查之后）以 UserMessage 注入状态、落盘
 * 并发一条 {@code steer} step 事件（wire additive——复用既有 StepEvent，
 * 无新事件类型）。三窗口矩阵：节点前（早于首节点）、节点后（同测试窗口
 * ——节点体执行期间入队，下一边界注入）、取消竞态（取消检查先行，插话
 * 随取消丢弃不注入）。
 */
class SteeringTest {

    /** 录制 sink（可中途置取消）。 */
    private static final class RecordingSink implements EventSink {
        final List<Object> events = new ArrayList<>();
        volatile boolean cancelled;

        @Override
        public void accept(Object event) {
            events.add(event);
        }

        @Override
        public boolean cancelled() {
            return cancelled;
        }
    }

    /** 两步图：a → b → END，各节点追加一条自己的消息；onNodeA 在 a 执行期回调。 */
    private static CompiledGraph twoStepGraph(Runnable onNodeA) {
        return GraphBuilder.begin()
                .addNode("a", (state, ctx) -> {
                    if (onNodeA != null) {
                        onNodeA.run();
                    }
                    return Map.of(GraphState.MESSAGES, List.of(AiMessage.from("a输出")));
                })
                .addNode("b", (state, ctx) ->
                        Map.of(GraphState.MESSAGES, List.of(AiMessage.from("b输出"))))
                .entry("a")
                .addEdge("a", "b")
                .addEdge("b", GraphBuilder.END)
                .compile(null, "steer");
    }

    @Test
    void midRunSteerInjectsAtNextBoundaryBeforeNodeB() {
        CompiledGraph graph = twoStepGraph(() ->
                RunRegistry.lookup("chat:t1").steer("user", "插话内容"));
        GraphState out = graph.invoke(new GraphState(), RunConfig.of("chat:t1", "u1"));
        // 注入发生在 a 之后、b 之前：messages = [a输出, 插话, b输出]。
        assertEquals(3, out.messages().size());
        assertEquals("a输出", ((AiMessage) out.messages().get(0)).text());
        assertEquals("插话内容", ((UserMessage) out.messages().get(1)).singleText());
        assertEquals("b输出", ((AiMessage) out.messages().get(2)).text());
    }

    @Test
    void steerStepEventOrderGoldenWithDetailCount() {
        RecordingSink sink = new RecordingSink();
        CompiledGraph graph = twoStepGraph(() ->
                RunRegistry.lookup("chat:t1").steer("user", "插话内容"));
        graph.stream(new GraphState(), RunConfig.of("chat:t1", "u1"), sink);
        List<String> shape = new ArrayList<>();
        for (Object event : sink.events) {
            if (event instanceof Events.StepEvent step) {
                shape.add(step.name() + ":" + step.status());
            }
        }
        assertEquals(List.of("a:running", "a:completed", "steer:completed",
                        "b:running", "b:completed"),
                shape, "事件序 golden：steer 夹在 a-completed 与 b-running 之间");
        Events.StepEvent steer = sink.events.stream()
                .filter(e -> e instanceof Events.StepEvent s && "steer".equals(s.name()))
                .map(e -> (Events.StepEvent) e)
                .findFirst().orElseThrow();
        assertEquals("injected_messages=1", steer.detail());
    }

    @Test
    void cancelRaceDropsPendingSteerWithoutInjection() {
        // 取消竞态窗口（矩阵 c）：节点 a 执行期间插话入队且 sink 置取消——
        // 循环顶取消检查先于 drain → 插话不注入、随 run 终止丢弃。
        RecordingSink sink = new RecordingSink();
        CompiledGraph graph = twoStepGraph(() -> {
            RunRegistry.lookup("chat:t1").steer("user", "迟到的插话");
            sink.cancelled = true;
        });
        CompiledGraph.RunOutcome outcome =
                graph.streamWithTermination(new GraphState(), RunConfig.of("chat:t1", "u1"), sink);
        assertEquals(Events.RunTermination.CANCELLED, outcome.termination());
        // 状态里只有 a 的半步产出，没有插话。
        assertEquals(1, outcome.state().messages().size());
        assertEquals("a输出", ((AiMessage) outcome.state().messages().get(0)).text());
    }

    @Test
    void runRegistryLifecycleCoversRunAndCleansUp() {
        // run 在途 = lookup 命中且属主正确；终态后注销（端点 404 的依据）。
        assertNull(RunRegistry.lookup("chat:t2"), "空闲线程无句柄");
        RecordingSink sink = new RecordingSink();
        CompiledGraph graph = twoStepGraph(() -> {
            RunRegistry.RunHandle handle = RunRegistry.lookup("chat:t2");
            assertEquals("u1", handle.userId(), "句柄携带属主");
        });
        graph.stream(new GraphState(), RunConfig.of("chat:t2", "u1"), sink);
        assertNull(RunRegistry.lookup("chat:t2"), "终态后注销");
    }

    @Test
    void steerQueueDrainsExactlyOnceAndRespectsCapacity() {
        RunRegistry.RunHandle handle = new RunRegistry.RunHandle("u1");
        for (int i = 0; i < RunRegistry.STEER_QUEUE_CAPACITY; i++) {
            assertTrue(handle.steer("user", "m" + i), "容量内投递成功");
        }
        assertFalse(handle.steer("user", "溢出"), "队列满投递失败（端点 429 依据）");
        assertEquals(RunRegistry.STEER_QUEUE_CAPACITY, handle.pendingSteers());
        assertEquals(RunRegistry.STEER_QUEUE_CAPACITY, handle.drain().size());
        assertEquals(0, handle.pendingSteers());
        assertTrue(handle.drain().isEmpty(), "二次 drain 为空（恰好一次）");
    }

    @Test
    void registerRejectsDuplicateConcurrentRun() {
        RunRegistry.RunHandle handle = RunRegistry.register("chat:t3", "u1");
        try {
            org.junit.jupiter.api.Assertions.assertThrows(IllegalStateException.class,
                    () -> RunRegistry.register("chat:t3", "u2"));
        } finally {
            RunRegistry.unregister("chat:t3", handle);
        }
    }

    @Test
    void systemKindInjectsReminderPrefixedSystemMessage() {
        // #14：kind=system → SystemMessage（SYSTEM_REMINDER_PREFIX 包裹），
        // 事件 detail 追加 injected_kind=system（wire additive）。
        RecordingSink sink = new RecordingSink();
        CompiledGraph graph = twoStepGraph(() ->
                RunRegistry.lookup("chat:t1").steer("system", "预算即将耗尽"));
        GraphState out = graph.stream(new GraphState(), RunConfig.of("chat:t1", "u1"), sink);
        assertEquals(3, out.messages().size());
        assertEquals("a输出", ((AiMessage) out.messages().get(0)).text());
        dev.langchain4j.data.message.SystemMessage reminder =
                (dev.langchain4j.data.message.SystemMessage) out.messages().get(1);
        assertEquals(CompiledGraph.SYSTEM_REMINDER_PREFIX + "预算即将耗尽", reminder.text());
        assertEquals("b输出", ((AiMessage) out.messages().get(2)).text());
        Events.StepEvent steer = sink.events.stream()
                .filter(e -> e instanceof Events.StepEvent s && "steer".equals(s.name()))
                .map(e -> (Events.StepEvent) e)
                .findFirst().orElseThrow();
        assertEquals("injected_messages=1;injected_kind=system", steer.detail());
    }

    @Test
    void systemReminderPrefixDeliberatelyDiffersFromCompactionPrefix() {
        // #14 与 #328 语义不冲突（计划裁决）：注入前缀与压缩摘要前缀字面量
        // 刻意不同——压缩转写折叠只认摘要前缀，注入提醒永不被当摘要折叠。
        assertFalse(CompiledGraph.SYSTEM_REMINDER_PREFIX.equals(
                io.github.paopaonb666.agentbase.core.context.ContextEngine.COMPACTION_SUMMARY_PREFIX));
    }

    @Test
    void steerItemKindIsClosedVocabulary() {
        org.junit.jupiter.api.Assertions.assertThrows(IllegalArgumentException.class,
                () -> new RunRegistry.SteerItem("bogus", "x"));
        assertEquals("user", new RunRegistry.SteerItem("user", "x").kind());
        assertEquals("system", new RunRegistry.SteerItem("system", "x").kind());
    }

    @Test
    void lastRunRecordedAtTerminalWithWireLabel() {
        // 正常完成 → done；取消 → cancelled；记录带 runId（run- 前缀）与属主。
        RecordingSink sink = new RecordingSink();
        CompiledGraph graph = twoStepGraph(null);
        graph.stream(new GraphState(), RunConfig.of("chat:t6", "u1"), sink);
        RunRegistry.LastRun last = RunRegistry.lastRun("chat:t6");
        assertEquals("done", last.termination());
        assertEquals("u1", last.userId());
        assertTrue(last.runId().startsWith("run-"), last.runId());
        assertTrue(last.endedAtMillis() > 0);

        RecordingSink cancelSink = new RecordingSink();
        CompiledGraph cancelGraph = twoStepGraph(() -> cancelSink.cancelled = true);
        cancelGraph.streamWithTermination(new GraphState(), RunConfig.of("chat:t7", "u1"), cancelSink);
        assertEquals("cancelled", RunRegistry.lastRun("chat:t7").termination(),
                "取消路径的终态标注与 streamWithTermination 同口径");
    }

    @Test
    void cancelledRunResumesFromSnapshotAndCompletes() {
        // #8 e2e：第一轮取消（a 的半步产出带标记落盘）→ 空输入重入（resume
        // 语义）从入口续跑 → done；快照完整性=先前消息保留、终态记录翻 done。
        io.github.paopaonb666.agentbase.core.graph.InMemoryCheckpointer checkpointer =
                new io.github.paopaonb666.agentbase.core.graph.InMemoryCheckpointer();
        RecordingSink cancelSink = new RecordingSink();
        CompiledGraph graph = GraphBuilder.begin()
                .addNode("a", (state, ctx) -> {
                    cancelSink.cancelled = true;
                    return Map.of(GraphState.MESSAGES, List.of(AiMessage.from("a输出")));
                })
                .addNode("b", (state, ctx) ->
                        Map.of(GraphState.MESSAGES, List.of(AiMessage.from("b输出"))))
                .entry("a")
                .addEdge("a", "b")
                .addEdge("b", GraphBuilder.END)
                .compile(checkpointer, "resumable");
        graph.streamWithTermination(new GraphState(), RunConfig.of("chat:t8", "u1"), cancelSink);
        assertEquals("cancelled", RunRegistry.lastRun("chat:t8").termination());

        GraphState resumed = graph.stream(new GraphState(), RunConfig.of("chat:t8", "u1"),
                new RecordingSink());
        assertEquals("done", RunRegistry.lastRun("chat:t8").termination(), "续跑翻 done");
        // 快照完整性：取消轮的 a输出 保留，重入按入口再跑 a、b。
        assertEquals(3, resumed.messages().size());
        assertEquals("a输出", ((AiMessage) resumed.messages().get(0)).text());
        assertEquals("a输出", ((AiMessage) resumed.messages().get(1)).text());
        assertEquals("b输出", ((AiMessage) resumed.messages().get(2)).text());
    }
}
