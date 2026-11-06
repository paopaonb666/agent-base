package io.github.paopaonb666.agentbase.core.graph;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.UserMessage;
import io.github.paopaonb666.agentbase.extensions.events.Events;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 图引擎执行语义测试（对应 LangGraph ainvoke/astream/get_state 的受控子集）。
 *
 * <p>覆盖路线图 #285 圈定的悬空语义：递归触顶、取消丢步、put 失败传播、
 * 条件边未知节点、并发同 thread 快照互踩。其中"取消丢步 / put 失败 /
 * 快照互踩"是 #199/#198/#197 的改造前基线——语义若变，本组测试同步改写。
 */
class CompiledGraphTest {

    /** 录制全部事件的 sink；见到标记事件后进入取消态（模拟客户端断开）。 */
    private static final class RecordingSink implements EventSink {
        static final Object CANCEL_MARKER = new Object();
        final List<Object> events = new ArrayList<>();
        volatile boolean cancelSeen = false;

        @Override
        public void accept(Object event) {
            if (event == CANCEL_MARKER) {
                cancelSeen = true;
                return;
            }
            events.add(event);
        }

        @Override
        public boolean cancelled() {
            return cancelSeen;
        }
    }

    @Test
    void singleNodeGraphRunsToEndAndPersists() {
        InMemoryCheckpointer cp = new InMemoryCheckpointer();
        CompiledGraph graph = GraphBuilder.singleNode("echo",
                (state, ctx) -> Map.of(GraphState.MESSAGES, List.of(AiMessage.from("hi"))),
                cp);

        GraphState out = graph.invoke(
                new GraphState(List.of(UserMessage.from("hello")), Map.of()),
                RunConfig.of("chat:t1", "u1"));

        assertEquals(2, out.messages().size());
        assertEquals("hi", ((AiMessage) out.messages().get(1)).text());
        // 快照落盘：checkpointer 里能看到同形状的历史。
        GraphState persisted = cp.get("chat:t1");
        assertNotNull(persisted);
        assertEquals(2, persisted.messages().size());
    }

    @Test
    void recursionLimitThrowsGraphRecursionError() {
        // 自环图：每步回到自身，必然触顶。
        CompiledGraph graph = GraphBuilder.begin()
                .addNode("loop", (state, ctx) -> Map.of())
                .entry("loop")
                .addEdge("loop", "loop")
                .compile(new InMemoryCheckpointer(), "loop");

        CompiledGraph.GraphRecursionError error = assertThrows(
                CompiledGraph.GraphRecursionError.class,
                () -> graph.invoke(new GraphState(), new RunConfig("chat:t1", "u1", 3)));
        assertEquals("Recursion limit of 3 reached", error.getMessage());
    }

    @Test
    void stepEventsFollowRunningBeforeCompleted() {
        RecordingSink sink = new RecordingSink();
        CompiledGraph graph = GraphBuilder.begin()
                .addNode("agent", (state, ctx) -> {
                    ctx.emit(new Events.DeltaEvent("tok"));
                    return Map.of();
                })
                .entry("agent")
                .addEdge("agent", GraphBuilder.END)
                .compile(null, "chat");

        graph.stream(new GraphState(), RunConfig.of("chat:t1", "u1"), sink);

        assertEquals(3, sink.events.size());
        assertEquals(Events.StepEvent.running("agent"), sink.events.get(0));
        assertEquals(new Events.DeltaEvent("tok"), sink.events.get(1));
        assertEquals(Events.StepEvent.completed("agent"), sink.events.get(2));
    }

    @Test
    void cancelAfterNodeExecutionDropsStepFromCheckpointer() {
        // #199 基线：节点已在途执行完（无法回滚），但取消后不再落盘——
        // 中断的 superstep 不进 checkpoint（恢复后看不到这半步）。
        InMemoryCheckpointer cp = new InMemoryCheckpointer();
        RecordingSink sink = new RecordingSink();
        AtomicBoolean nodeRan = new AtomicBoolean(false);
        CompiledGraph graph = GraphBuilder.begin()
                .addNode("agent", (state, ctx) -> {
                    nodeRan.set(true);
                    // 节点在途期间客户端断开：emit 仍尽力送达，取消态在
                    // 节点返回后由步间检查发现。
                    ctx.emit(RecordingSink.CANCEL_MARKER);
                    return Map.of(GraphState.MESSAGES, List.of(AiMessage.from("partial")));
                })
                .entry("agent")
                .addEdge("agent", GraphBuilder.END)
                .compile(cp, "chat");

        GraphState out = graph.stream(new GraphState(), RunConfig.of("chat:t1", "u1"), sink);

        assertTrue(nodeRan.get());
        // 取消后的步间检查先于 state.apply：这一步的产出既不落盘、也不进
        // 返回值（对应 Python await 点被取消的 superstep——历史里凭空消失，
        // #199 的改造基线）。
        assertEquals(0, out.messages().size());
        assertNull(cp.get("chat:t1"));
    }

    @Test
    void routerReturningUnknownNodeFailsOnNextIteration() {
        // #202 基线：条件边返回未注册节点名，编译期不拦，首次命中该分支才炸。
        CompiledGraph graph = GraphBuilder.begin()
                .addNode("router", (state, ctx) -> Map.of())
                .entry("router")
                .addConditionalEdges("router", state -> "missing_node")
                .compile(null, "chat");

        IllegalStateException error = assertThrows(IllegalStateException.class,
                () -> graph.invoke(new GraphState(), RunConfig.of("chat:t1", "u1")));
        assertTrue(error.getMessage().contains("missing_node"), error.getMessage());
    }

    @Test
    void checkpointerPutFailureRetriesOnceThenContinuesInMemory() {
        // #198：put 失败先重试 1 次，仍失败则内存续跑——整轮已产出的
        // token 不再因快照欠账报废；欠账计入 /health 可见的进程级计数。
        long debtBefore = CompiledGraph.persistDebtSteps();
        InMemoryCheckpointer good = new InMemoryCheckpointer();
        good.put("chat:t1", new GraphState(List.of(UserMessage.from("old")), Map.of()));
        AtomicInteger putAttempts = new AtomicInteger();
        Checkpointer flaky = new Checkpointer() {
            @Override
            public void put(String threadId, GraphState state) {
                // 恒失败（磁盘满不因重试恢复）——验证重试只发生一次。
                putAttempts.incrementAndGet();
                throw new IllegalStateException("checkpoint put failed: disk full");
            }

            @Override
            public GraphState get(String threadId) {
                return good.get(threadId);
            }

            @Override
            public void deleteThread(String threadId) {
                good.deleteThread(threadId);
            }

            @Override
            public List<String> listThreadIds() {
                return good.listThreadIds();
            }

            @Override
            public boolean healthy() {
                return false;
            }

            @Override
            public void close() {
            }
        };
        CompiledGraph graph = GraphBuilder.singleNode("echo",
                (state, ctx) -> Map.of(GraphState.MESSAGES, List.of(AiMessage.from("new"))),
                flaky);

        // run 正常完成（内存续跑）——不再抛错中断整轮；本轮消息在内存态完整。
        GraphState out = graph.invoke(new GraphState(), RunConfig.of("chat:t1", "u1"));
        assertEquals(2, out.messages().size());
        assertEquals("new", ((AiMessage) out.messages().get(1)).text());
        // 3 个落盘点（终态 1 步……单节点图 1 步 = 1 次 persist × 首次+重试 = 2 次尝试）。
        assertEquals(2, putAttempts.get(), "每步落盘恰一次重试");
        // 欠账计数对 /health 可见。
        assertTrue(CompiledGraph.persistInDebt());
        assertEquals(debtBefore + 1, CompiledGraph.persistDebtSteps());
        // 上一轮快照保持原样（不会写坏）。
        GraphState persisted = flaky.get("chat:t1");
        assertNotNull(persisted);
        assertEquals("old", ((UserMessage) persisted.messages().get(0)).singleText());
    }

    @Test
    void checkpointerPutRecoversOnRetry() {
        // 抖动语义：第一次 put 失败、重试成功——run 正常、欠账不累积。
        InMemoryCheckpointer good = new InMemoryCheckpointer();
        AtomicInteger putAttempts = new AtomicInteger();
        Checkpointer flaky = new Checkpointer() {
            @Override
            public void put(String threadId, GraphState state) {
                if (putAttempts.incrementAndGet() == 1) {
                    throw new IllegalStateException("transient lock contention");
                }
                good.put(threadId, state);
            }

            @Override
            public GraphState get(String threadId) {
                return good.get(threadId);
            }

            @Override
            public void deleteThread(String threadId) {
                good.deleteThread(threadId);
            }

            @Override
            public List<String> listThreadIds() {
                return good.listThreadIds();
            }

            @Override
            public boolean healthy() {
                return true;
            }

            @Override
            public void close() {
            }
        };
        CompiledGraph graph = GraphBuilder.singleNode("echo",
                (state, ctx) -> Map.of(GraphState.MESSAGES, List.of(AiMessage.from("new"))),
                flaky);
        GraphState out = graph.invoke(new GraphState(), RunConfig.of("chat:t1", "u1"));
        assertEquals(1, out.messages().size());
        assertEquals(2, putAttempts.get());
        // 重试成功后快照在库。
        assertNotNull(flaky.get("chat:t1"));
    }

    @Test
    void concurrentRunsOnSameThreadClobberSnapshot() throws Exception {
        // #197 基线：同 thread 两个 run 无互斥——双方都在对方落盘前
        // loadState，各自整行覆盖写，最后落盘的一方胜出，另一方消息丢失
        // （last-writer-wins）。栅栏保证确定性复现。
        InMemoryCheckpointer cp = new InMemoryCheckpointer();
        CyclicBarrier barrier = new CyclicBarrier(2);
        CompiledGraph graph = GraphBuilder.begin()
                .addNode("agent", (state, ctx) -> {
                    String tag = (String) state.get("tag");
                    // 双方都进节点后才返回：确保两次 loadState 都发生在首次
                    // persistState 之前。
                    barrier.await();
                    return Map.of(GraphState.MESSAGES,
                            List.of(AiMessage.from("from-" + tag)));
                })
                .entry("agent")
                .addEdge("agent", GraphBuilder.END)
                .compile(cp, "chat");

        List<Throwable> failures = new ArrayList<>();
        Thread a = new Thread(() -> runQuietly(graph, "a", barrier, failures));
        Thread b = new Thread(() -> runQuietly(graph, "b", barrier, failures));
        a.start();
        b.start();
        a.join(5000);
        b.join(5000);
        assertEquals(List.of(), failures);

        GraphState finalState = cp.get("chat:t1");
        assertNotNull(finalState);
        // 整行覆盖：最终快照只剩最后写入方的一条消息（另一条被踩掉）。
        assertEquals(1, finalState.messages().size());
        String survivor = ((AiMessage) finalState.messages().get(0)).text();
        assertTrue(survivor.startsWith("from-"), survivor);
    }

    private void runQuietly(CompiledGraph graph, String tag, CyclicBarrier barrier,
                            List<Throwable> failures) {
        try {
            GraphState input = new GraphState(List.of(), Map.of("tag", tag));
            graph.invoke(input, RunConfig.of("chat:t1", "u1"));
        } catch (RuntimeException e) {
            synchronized (failures) {
                failures.add(e);
            }
        }
    }

    @Test
    void stateRestoresFromCheckpointerAcrossInvocations() {
        InMemoryCheckpointer cp = new InMemoryCheckpointer();
        CompiledGraph graph = GraphBuilder.singleNode("echo",
                (state, ctx) -> Map.of(GraphState.MESSAGES, List.of(AiMessage.from("r2"))),
                cp);

        graph.invoke(new GraphState(List.of(UserMessage.from("u1")), Map.of()),
                RunConfig.of("chat:t1", "u1"));
        // 第二次 invoke：从快照恢复历史再追加。
        GraphState out = graph.invoke(new GraphState(List.of(UserMessage.from("u2")), Map.of()),
                RunConfig.of("chat:t1", "u1"));

        assertEquals(4, out.messages().size());
        assertEquals("u1", ((UserMessage) out.messages().get(0)).singleText());
        assertEquals("u2", ((UserMessage) out.messages().get(2)).singleText());
    }

    @Test
    void deleteThreadAndListThreadIdsRoundTrip() {
        InMemoryCheckpointer cp = new InMemoryCheckpointer();
        CompiledGraph graph = GraphBuilder.singleNode("echo",
                (state, ctx) -> Map.of(), cp);
        RunConfig config = RunConfig.of("chat:t1", "u1");
        graph.invoke(new GraphState(), config);

        assertTrue(cp.listThreadIds().contains("chat:t1"));
        graph.deleteThread(config);
        assertFalse(cp.listThreadIds().contains("chat:t1"));
        assertNull(cp.get("chat:t1"));
    }

    @Test
    void compileRejectsBrokenTopologies() {
        // 编译期校验（静态边目标在 compile 拦截——#202 的编译侧保证）。
        GraphBuilder builder = GraphBuilder.begin();
        assertThrows(IllegalStateException.class, () -> builder.compile(null, "x"));

        GraphBuilder missingEntry = GraphBuilder.begin()
                .addNode("a", (s, c) -> Map.of())
                .addEdge("a", GraphBuilder.END);
        assertThrows(IllegalStateException.class, () -> missingEntry.compile(null, "x"));

        GraphBuilder unknownEntry = GraphBuilder.begin()
                .addNode("a", (s, c) -> Map.of())
                .entry("nope")
                .addEdge("a", GraphBuilder.END);
        assertThrows(IllegalStateException.class, () -> unknownEntry.compile(null, "x"));

        GraphBuilder unknownTarget = GraphBuilder.begin()
                .addNode("a", (s, c) -> Map.of())
                .entry("a")
                .addEdge("a", "ghost");
        assertThrows(IllegalStateException.class, () -> unknownTarget.compile(null, "x"));

        GraphBuilder noOutEdge = GraphBuilder.begin()
                .addNode("a", (s, c) -> Map.of())
                .addNode("b", (s, c) -> Map.of())
                .entry("a")
                .addEdge("a", "b");
        assertThrows(IllegalStateException.class, () -> noOutEdge.compile(null, "x"));

        GraphBuilder duplicateNode = GraphBuilder.begin()
                .addNode("a", (s, c) -> Map.of());
        assertThrows(IllegalArgumentException.class,
                () -> duplicateNode.addNode("a", (s, c) -> Map.of()));

        GraphBuilder duplicateOutEdge = GraphBuilder.begin()
                .addNode("a", (s, c) -> Map.of())
                .entry("a")
                .addEdge("a", GraphBuilder.END);
        assertThrows(IllegalArgumentException.class,
                () -> duplicateOutEdge.addEdge("a", GraphBuilder.END));
    }

    @Test
    void routerReturningNullTerminatesRun() {
        CompiledGraph graph = GraphBuilder.begin()
                .addNode("agent", (state, ctx) -> Map.of())
                .entry("agent")
                .addConditionalEdges("agent", state -> null)
                .compile(null, "chat");

        GraphState out = graph.invoke(new GraphState(), RunConfig.of("chat:t1", "u1"));
        assertNotNull(out);
    }
}
