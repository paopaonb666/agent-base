package io.github.paopaonb666.agentbase.core.graph;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.extensions.events.Events;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 图引擎执行语义测试（对应 LangGraph ainvoke/astream/get_state 的受控子集）。
 *
 * <p>覆盖路线图 #285 圈定的悬空语义：递归触顶、取消半步落盘、put 失败传播、
 * 条件边未知节点、并发同 thread 快照互踩。其中"取消丢步 / put 失败 /
 * 快照互踩"原是 #199/#198/#197 的改造前基线——#199 已翻转（半步带
 * cancelled 标记落盘）、#198/#197 已落地，语义若再变，本组测试同步改写。
 *
 * <p>另覆盖 #9 的统一终止标注：streamWithTermination 对每个终止点产出
 * 封闭枚举（DONE/CANCELLED/RECURSION_LIMIT/THREAD_BUSY/INTERNAL_ERROR），
 * invoke/stream 的既有抛出契约不变。
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
        // #229：错误带图名/节点数，可直接定位。
        assertTrue(error.getMessage().contains("Recursion limit of 3 reached"),
                error.getMessage());
        assertTrue(error.getMessage().contains("graph 'loop'"), error.getMessage());
        assertTrue(error.getMessage().contains("1 nodes"), error.getMessage());
        // 非 planner 图不带 planner 公式。
        assertFalse(error.getMessage().contains("PLANNER_MAX_SUBTASKS"), error.getMessage());
    }

    @Test
    void plannerRecursionErrorCarriesTheBudgetFormula() {
        // #229：planner 图触顶时错误信息附 H5 公式与建议配置。
        CompiledGraph plannerGraph = GraphBuilder.begin()
                .addNode("plan", (state, ctx) -> Map.of())
                .entry("plan")
                .addEdge("plan", "plan")
                .compile(new InMemoryCheckpointer(), "planner");
        CompiledGraph.GraphRecursionError error = assertThrows(
                CompiledGraph.GraphRecursionError.class,
                () -> plannerGraph.invoke(new GraphState(), new RunConfig("chat:t1", "u1", 3)));
        assertTrue(error.getMessage().contains("PLANNER_MAX_SUBTASKS"), error.getMessage());
        assertTrue(error.getMessage().contains("AGENT_RECURSION_LIMIT"), error.getMessage());
    }

    @Test
    void plannerWorstStepsFormulaMatchesH5() {
        // H5 口径：8 + 2*5*(8+1) = 98（默认配置）；配置推导跟随 PLANNER_*。
        Settings defaults = Settings.load(Map.of("ENV", "development"), null);
        assertEquals(98, io.github.paopaonb666.agentbase.core.graphs.Graphs
                .plannerWorstSteps(defaults));
        Settings tuned = Settings.load(Map.of(
                "ENV", "development",
                "PLANNER_MAX_SUBTASKS", "3",
                "PLANNER_SUBTASK_TOOL_ROUNDS", "2"), null);
        assertEquals(8 + 2 * 3 * 3, io.github.paopaonb666.agentbase.core.graphs.Graphs
                .plannerWorstSteps(tuned));
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
    void cancelAfterNodeExecutionPersistsPartialStateWithCancelledMarker() {
        // #199 翻转基线：节点已在途执行完（无法回滚），取消后半步产出带
        // cancelled 标记落盘——「取消的轮次」不再从历史里凭空消失（对齐
        // Python asyncio 取消语义的显式偏离，差异登记簿承载）。
        InMemoryCheckpointer cp = new InMemoryCheckpointer();
        RecordingSink sink = new RecordingSink();
        AtomicBoolean nodeRan = new AtomicBoolean(false);
        CompiledGraph graph = GraphBuilder.begin()
                .addNode("agent", (state, ctx) -> {
                    nodeRan.set(true);
                    // 节点在途期间客户端断开：emit 仍尽力送达，取消态在
                    // 节点返回后由步后检查发现。
                    ctx.emit(RecordingSink.CANCEL_MARKER);
                    return Map.of(GraphState.MESSAGES, List.of(AiMessage.from("partial")));
                })
                .entry("agent")
                .addEdge("agent", GraphBuilder.END)
                .compile(cp, "chat");

        GraphState out = graph.stream(new GraphState(), RunConfig.of("chat:t1", "u1"), sink);

        assertTrue(nodeRan.get());
        // 半步产出既进返回值、也落盘（基线翻转：原先两处皆为空）。
        assertEquals(1, out.messages().size());
        assertEquals("partial", ((AiMessage) out.messages().get(0)).text());
        GraphState persisted = cp.get("chat:t1");
        assertNotNull(persisted);
        assertEquals(1, persisted.messages().size());
        assertTrue(persisted.cancelledRound(), "快照行应携带取消标记");
    }

    @Test
    void cancelledRoundStripsDanglingToolCallsFromSnapshot() {
        // #199 兜底：取消落在「chat 已产出 tool_calls、tools 未执行」的窗口
        // 时，快照剥掉未配对 tool_calls——下一轮模型输入不再以悬挂
        // assistant(tool_calls) 结尾被 OpenAI 兼容 provider 拒收（400）。
        InMemoryCheckpointer cp = new InMemoryCheckpointer();
        RecordingSink sink = new RecordingSink();
        CompiledGraph graph = GraphBuilder.begin()
                .addNode("chat", (state, ctx) -> {
                    ctx.emit(RecordingSink.CANCEL_MARKER);
                    return Map.of(GraphState.MESSAGES, List.of(AiMessage.from(
                            "让我查一下",
                            List.of(dev.langchain4j.agent.tool.ToolExecutionRequest.builder()
                                    .id("call_1").name("web_search")
                                    .arguments("{\"query\":\"x\"}").build()))));
                })
                .entry("chat")
                .addEdge("chat", GraphBuilder.END)
                .compile(cp, "chat");

        graph.stream(new GraphState(), RunConfig.of("chat:t1", "u1"), sink);

        GraphState persisted = cp.get("chat:t1");
        assertNotNull(persisted);
        assertEquals(1, persisted.messages().size());
        AiMessage stripped = (AiMessage) persisted.messages().get(0);
        assertFalse(stripped.hasToolExecutionRequests(), "悬挂 tool_calls 必须剥除");
        assertEquals("让我查一下", stripped.text());
    }

    @Test
    void loopTopCancellationHealsDanglingToolCallsPersistedByPreviousStep() {
        // 审查 A-P1 步间窗：取消落在「上一步完成态（含 tool_calls）已落盘、
        // tools 节点未执行」的窗口时，loop-top 检查取消返回——修复前快照
        // 保持悬挂 AiMessage(tool_calls)，该线程下一轮起每轮被 provider 400。
        // 修复后与节点中途取消同款善后：剥除 + cancelled 标记 + 重落盘。
        InMemoryCheckpointer cp = new InMemoryCheckpointer();
        // sink：见到 chat 的 completed 事件即进入取消态（模拟断连落在步间）。
        EventSink sink = new EventSink() {
            volatile boolean cancelSeen = false;

            @Override public void accept(Object event) {
                if (event instanceof Events.StepEvent step
                        && "completed".equals(step.status()) && "chat".equals(step.name())) {
                    cancelSeen = true;
                }
            }

            @Override public boolean cancelled() {
                return cancelSeen;
            }
        };
        CompiledGraph graph = GraphBuilder.begin()
                .addNode("chat", (state, ctx) -> Map.of(GraphState.MESSAGES, List.of(AiMessage.from(
                        "让我查一下",
                        List.of(dev.langchain4j.agent.tool.ToolExecutionRequest.builder()
                                .id("call_1").name("web_search")
                                .arguments("{\"query\":\"x\"}").build())))))
                .addNode("tools", (state, ctx) -> {
                    throw new AssertionError("取消后不应再执行任何节点");
                })
                .entry("chat")
                .addEdge("chat", "tools")
                .addEdge("tools", GraphBuilder.END)
                .compile(cp, "chat");

        CompiledGraph.RunOutcome outcome = graph.streamWithTermination(
                new GraphState(), RunConfig.of("chat:t1", "u1"), sink);
        assertEquals(Events.RunTermination.CANCELLED, outcome.termination());

        GraphState persisted = cp.get("chat:t1");
        assertNotNull(persisted);
        assertTrue(persisted.cancelledRound(), "快照行应携带取消标记");
        ChatMessage last = persisted.messages().get(persisted.messages().size() - 1);
        assertTrue(last instanceof AiMessage ai && !ai.hasToolExecutionRequests(),
                "loop-top 取消必须剥除上步悬挂 tool_calls：" + last);
        assertEquals("让我查一下", ((AiMessage) last).text());
    }

    @Test
    void cancelledRoundPersistFailureStillCountsAsDebtAndStaysCancelled() {
        // 取消落盘复用 #198 的重试+欠账语义：put 恒失败时半步留在内存续跑
        // （返回值完整），终态仍是 CANCELLED——不因快照欠账把取消变成异常。
        long debtBefore = CompiledGraph.persistDebtSteps();
        InMemoryCheckpointer good = new InMemoryCheckpointer();
        Checkpointer flaky = new Checkpointer() {
            @Override
            public void put(String threadId, GraphState state) {
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
        RecordingSink sink = new RecordingSink();
        CompiledGraph graph = GraphBuilder.begin()
                .addNode("agent", (state, ctx) -> {
                    ctx.emit(RecordingSink.CANCEL_MARKER);
                    return Map.of(GraphState.MESSAGES, List.of(AiMessage.from("partial")));
                })
                .entry("agent")
                .addEdge("agent", GraphBuilder.END)
                .compile(flaky, "chat");

        CompiledGraph.RunOutcome outcome = graph.streamWithTermination(
                new GraphState(), RunConfig.of("chat:t1", "u1"), sink);

        // 终态仍是 CANCELLED（不因快照欠账变成异常/其他终止点），半步在
        // 内存态完整（#198 内存续跑语义对取消落盘同样成立）。
        assertEquals(Events.RunTermination.CANCELLED, outcome.termination());
        assertNull(outcome.error());
        assertEquals(1, outcome.state().messages().size());
        assertTrue(CompiledGraph.persistInDebt());
        assertEquals(debtBefore + 1, CompiledGraph.persistDebtSteps());
    }

    @Test
    void cancelledPartialRoundIsVisibleToNextRoundHistory() {
        // #199 主诉闭环：半截轮落盘后同 thread 再 invoke——新 round 的节点
        // 看到的历史含半截消息（模型输入组装兼容），且新 round 首次正常
        // 落盘后快照行的取消标记被整行覆盖消失。
        InMemoryCheckpointer cp = new InMemoryCheckpointer();
        RecordingSink sink = new RecordingSink();
        CompiledGraph round1 = GraphBuilder.begin()
                .addNode("agent", (state, ctx) -> {
                    ctx.emit(RecordingSink.CANCEL_MARKER);
                    return Map.of(GraphState.MESSAGES, List.of(AiMessage.from("partial")));
                })
                .entry("agent")
                .addEdge("agent", GraphBuilder.END)
                .compile(cp, "chat");
        round1.stream(new GraphState(), RunConfig.of("chat:t1", "u1"), sink);
        assertTrue(cp.get("chat:t1").cancelledRound());

        AtomicReference<List<ChatMessage>> seen = new AtomicReference<>();
        CompiledGraph round2 = GraphBuilder.begin()
                .addNode("agent", (state, ctx) -> {
                    seen.set(List.copyOf(state.messages()));
                    return Map.of(GraphState.MESSAGES, List.of(AiMessage.from("round2")));
                })
                .entry("agent")
                .addEdge("agent", GraphBuilder.END)
                .compile(cp, "chat");
        round2.invoke(new GraphState(List.of(UserMessage.from("next")), Map.of()),
                RunConfig.of("chat:t1", "u1"));

        // 节点视角：半截 partial（上轮取消产出）+ 本轮输入 next。
        assertEquals(2, seen.get().size());
        assertEquals("partial", ((AiMessage) seen.get().get(0)).text());
        assertEquals("next", ((UserMessage) seen.get().get(1)).singleText());
        // 正常落盘后标记消失（行元数据，非状态通道）。
        GraphState after = cp.get("chat:t1");
        assertFalse(after.cancelledRound());
        assertEquals(3, after.messages().size());
    }

    @Test
    void routerReturningUnknownNodeStillFailsAtRuntimeWithoutTargets() {
        // #202 翻转：旧两参重载未声明目标集——编译期不拦，首次命中该分支
        // 仍在运行期炸（兜底语义保留）；错误带节点名 + 可用节点列表。
        CompiledGraph graph = GraphBuilder.begin()
                .addNode("router", (state, ctx) -> Map.of())
                .entry("router")
                .addConditionalEdges("router", state -> "missing_node")
                .compile(null, "chat");

        IllegalStateException error = assertThrows(IllegalStateException.class,
                () -> graph.invoke(new GraphState(), RunConfig.of("chat:t1", "u1")));
        assertTrue(error.getMessage().contains("missing_node"), error.getMessage());
        assertTrue(error.getMessage().contains("router"), error.getMessage());
    }

    @Test
    void compileRejectsConditionalEdgeWithUnknownDeclaredTarget() {
        // #202 翻转：条件边声明目标集后，router 返回值笔误在编译期炸——
        // 不再等运行期首次命中分支才暴露。
        GraphBuilder builder = GraphBuilder.begin()
                .addNode("router", (state, ctx) -> Map.of())
                .entry("router")
                .addConditionalEdges("router", state -> "missing_node",
                        Set.of("missing_node", GraphBuilder.END));
        IllegalStateException error = assertThrows(IllegalStateException.class,
                () -> builder.compile(null, "chat"));
        assertTrue(error.getMessage().contains("missing_node"), error.getMessage());
        // 文案带图名 + 出边节点 + 可用列表。
        assertTrue(error.getMessage().contains("chat"), error.getMessage());
        assertTrue(error.getMessage().contains("router"), error.getMessage());
        assertTrue(error.getMessage().contains("可用"), error.getMessage());
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
    void concurrentRunOnSameThreadFailsFastWithThreadBusy() throws Exception {
        // #197：同 thread 第二个 run 快速失败（ThreadBusyError）——快照
        // last-writer-wins 互踩从根上消除；先到的 run 正常完成。
        InMemoryCheckpointer cp = new InMemoryCheckpointer();
        CountDownLatch entered = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        CompiledGraph graph = GraphBuilder.begin()
                .addNode("agent", (state, ctx) -> {
                    entered.countDown();
                    // 持锁等放行：模拟在途长 run。
                    release.await();
                    return Map.of(GraphState.MESSAGES, List.of(AiMessage.from("from-a")));
                })
                .entry("agent")
                .addEdge("agent", GraphBuilder.END)
                .compile(cp, "chat");

        AtomicReference<Throwable> failureB = new AtomicReference<>();
        Thread a = new Thread(() -> graph.invoke(
                new GraphState(List.of(), Map.of("tag", "a")), RunConfig.of("chat:t1", "u1")));
        a.start();
        assertTrue(entered.await(5, java.util.concurrent.TimeUnit.SECONDS), "A 应先进入节点");
        // A 在途：探针为真；B 快速失败而不是交错执行。
        assertTrue(graph.isThreadBusy("chat:t1"));
        Thread b = new Thread(() -> {
            try {
                graph.invoke(new GraphState(List.of(), Map.of("tag", "b")),
                        RunConfig.of("chat:t1", "u1"));
            } catch (RuntimeException e) {
                failureB.set(e);
            }
        });
        b.start();
        b.join(5000);
        assertTrue(failureB.get() instanceof CompiledGraph.ThreadBusyError,
                "第二个 run 应 ThreadBusyError：" + failureB.get());
        // 释放 A 让它正常完成；不同 thread 的 run 不受影响。
        release.countDown();
        a.join(5000);
        assertFalse(graph.isThreadBusy("chat:t1"), "A 完成后锁应释放");
        GraphState finalState = cp.get("chat:t1");
        assertNotNull(finalState);
        assertEquals(1, finalState.messages().size());
        assertEquals("from-a", ((AiMessage) finalState.messages().get(0)).text());
    }

    @Test
    void concurrentRunsOnDifferentThreadsAreIndependent() throws Exception {
        // 互斥按 threadId 键控：不同 thread 并发 run 互不阻塞。
        InMemoryCheckpointer cp = new InMemoryCheckpointer();
        CyclicBarrier barrier = new CyclicBarrier(2);
        CompiledGraph graph = GraphBuilder.begin()
                .addNode("agent", (state, ctx) -> {
                    barrier.await();
                    return Map.of(GraphState.MESSAGES,
                            List.of(AiMessage.from("from-" + state.get("tag"))));
                })
                .entry("agent")
                .addEdge("agent", GraphBuilder.END)
                .compile(cp, "chat");
        List<Throwable> failures = new ArrayList<>();
        Thread a = new Thread(() -> runQuietly(graph, "a", barrier, failures, "chat:ta"));
        Thread b = new Thread(() -> runQuietly(graph, "b", barrier, failures, "chat:tb"));
        a.start();
        b.start();
        a.join(5000);
        b.join(5000);
        assertEquals(List.of(), failures);
        assertNotNull(cp.get("chat:ta"));
        assertNotNull(cp.get("chat:tb"));
    }

    private void runQuietly(CompiledGraph graph, String tag, CyclicBarrier barrier,
                            List<Throwable> failures, String threadId) {
        try {
            GraphState input = new GraphState(List.of(), Map.of("tag", tag));
            graph.invoke(input, new RunConfig(threadId, "u1", 25));
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

        // #202：条件边声明目标集后，未注册目标同样在编译期拦截。
        GraphBuilder unknownConditionalTarget = GraphBuilder.begin()
                .addNode("a", (s, c) -> Map.of())
                .entry("a")
                .addConditionalEdges("a", s -> "ghost", Set.of("ghost", GraphBuilder.END));
        assertThrows(IllegalStateException.class, () -> unknownConditionalTarget.compile(null, "x"));

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

    @Test
    void streamWithTerminationAnnotatesNormalCompletion() {
        // #9：正常走到 END（含 put 失败内存续跑后的收尾）→ DONE。
        CompiledGraph graph = GraphBuilder.singleNode("echo",
                (state, ctx) -> Map.of(GraphState.MESSAGES, List.of(AiMessage.from("hi"))),
                new InMemoryCheckpointer());

        CompiledGraph.RunOutcome outcome = graph.streamWithTermination(
                new GraphState(), RunConfig.of("chat:t1", "u1"), null);

        assertEquals(Events.RunTermination.DONE, outcome.termination());
        assertNull(outcome.error());
        assertEquals(1, outcome.state().messages().size());
    }

    @Test
    void streamWithTerminationAnnotatesRecursionLimit() {
        // #9：触顶不抛出——RECURSION_LIMIT 进 outcome，原始异常随 error 携带
        //（invoke/stream 的抛出契约由 recursionLimitThrowsGraphRecursionError 守住）。
        CompiledGraph graph = GraphBuilder.begin()
                .addNode("loop", (state, ctx) -> Map.of())
                .entry("loop")
                .addEdge("loop", "loop")
                .compile(new InMemoryCheckpointer(), "loop");

        CompiledGraph.RunOutcome outcome = graph.streamWithTermination(
                new GraphState(), new RunConfig("chat:t1", "u1", 3), null);

        assertEquals(Events.RunTermination.RECURSION_LIMIT, outcome.termination());
        assertTrue(outcome.error() instanceof CompiledGraph.GraphRecursionError,
                "error 应携带 GraphRecursionError：" + outcome.error());
        assertNull(outcome.state());
    }

    @Test
    void streamWithTerminationAnnotatesCancellation() {
        // #9：断连 break（#199：半步已带标记落盘）的出口归 CANCELLED，error 为 null。
        RecordingSink sink = new RecordingSink();
        CompiledGraph graph = GraphBuilder.begin()
                .addNode("agent", (state, ctx) -> {
                    ctx.emit(RecordingSink.CANCEL_MARKER);
                    return Map.of();
                })
                .entry("agent")
                .addEdge("agent", GraphBuilder.END)
                .compile(null, "chat");

        CompiledGraph.RunOutcome outcome = graph.streamWithTermination(
                new GraphState(), RunConfig.of("chat:t1", "u1"), sink);

        assertEquals(Events.RunTermination.CANCELLED, outcome.termination());
        assertNull(outcome.error());
    }

    @Test
    void streamWithTerminationAnnotatesThreadBusy() throws Exception {
        // #9：tryLock 失败 → THREAD_BUSY（#197 的快速失败语义投影到终态标注）。
        InMemoryCheckpointer cp = new InMemoryCheckpointer();
        CountDownLatch entered = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        CompiledGraph graph = GraphBuilder.begin()
                .addNode("agent", (state, ctx) -> {
                    entered.countDown();
                    release.await();
                    return Map.of(GraphState.MESSAGES, List.of(AiMessage.from("from-a")));
                })
                .entry("agent")
                .addEdge("agent", GraphBuilder.END)
                .compile(cp, "chat");

        AtomicReference<CompiledGraph.RunOutcome> outcomeB = new AtomicReference<>();
        Thread a = new Thread(() -> graph.invoke(new GraphState(), RunConfig.of("chat:t1", "u1")));
        a.start();
        assertTrue(entered.await(5, java.util.concurrent.TimeUnit.SECONDS), "A 应先进入节点");
        Thread b = new Thread(() -> outcomeB.set(graph.streamWithTermination(
                new GraphState(), RunConfig.of("chat:t1", "u1"), null)));
        b.start();
        b.join(5000);
        assertNotNull(outcomeB.get(), "B 应已返回 outcome");
        assertEquals(Events.RunTermination.THREAD_BUSY, outcomeB.get().termination());
        assertTrue(outcomeB.get().error() instanceof CompiledGraph.ThreadBusyError,
                "error 应携带 ThreadBusyError：" + outcomeB.get().error());
        release.countDown();
        a.join(5000);
    }

    @Test
    void streamWithTerminationAnnotatesInternalError() {
        // #9：节点异常/未知节点等其余异常 → INTERNAL_ERROR，异常随 error 携带。
        CompiledGraph graph = GraphBuilder.begin()
                .addNode("boom", (state, ctx) -> {
                    throw new IllegalStateException("boom");
                })
                .entry("boom")
                .addEdge("boom", GraphBuilder.END)
                .compile(null, "chat");

        CompiledGraph.RunOutcome outcome = graph.streamWithTermination(
                new GraphState(), RunConfig.of("chat:t1", "u1"), null);

        assertEquals(Events.RunTermination.INTERNAL_ERROR, outcome.termination());
        assertNotNull(outcome.error());
        assertNull(outcome.state());
    }

    // ── 压缩写回互斥入口（#330 方案 B） ──────────────────────────────────

    @Test
    void compactThreadFailsFastWhileRunHoldsLock() throws Exception {
        // #330：run 持锁期间 compactThread 快速失败（与 run 同语义不排队）；
        // 锁释放后写回成功且持久化。
        InMemoryCheckpointer cp = new InMemoryCheckpointer();
        CountDownLatch entered = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        CompiledGraph graph = GraphBuilder.begin()
                .addNode("agent", (state, ctx) -> {
                    entered.countDown();
                    release.await();
                    return Map.of(GraphState.MESSAGES, List.of(AiMessage.from("run")));
                })
                .entry("agent")
                .addEdge("agent", GraphBuilder.END)
                .compile(cp, "chat");

        Thread run = new Thread(() -> graph.invoke(
                new GraphState(List.of(UserMessage.from("u1")), Map.of()),
                RunConfig.of("chat:t1", "u1")));
        run.start();
        assertTrue(entered.await(5, java.util.concurrent.TimeUnit.SECONDS), "run 应先进入节点");
        // run 持锁：写回快速返回 false（不排队）。
        assertFalse(graph.compactThread("chat:t1", fresh -> java.util.Optional.of(fresh)));
        release.countDown();
        run.join(5000);
        // 锁释放：写回成功、快照被整行替换。
        boolean written = graph.compactThread("chat:t1", fresh -> java.util.Optional.of(
                new GraphState(List.of(SystemMessage.from("摘要")), Map.of())));
        assertTrue(written);
        GraphState persisted = cp.get("chat:t1");
        assertEquals(1, persisted.messages().size());
        assertEquals("摘要", ((SystemMessage) persisted.messages().get(0)).text());
    }

    @Test
    void compactThreadAbandonsOnEmptyRewriteWithoutHalfWrite() {
        // #330：rewriter 返回 empty = 基准不一致/放弃——快照保持原样
        // （放弃路径不产生半写）。
        InMemoryCheckpointer cp = new InMemoryCheckpointer();
        cp.put("chat:t1", new GraphState(
                List.of(UserMessage.from("u1"), AiMessage.from("a1")), Map.of()));
        CompiledGraph graph = GraphBuilder.singleNode("echo", (state, ctx) -> Map.of(), cp);

        assertFalse(graph.compactThread("chat:t1", fresh -> java.util.Optional.empty()));
        GraphState persisted = cp.get("chat:t1");
        assertEquals(2, persisted.messages().size());
        assertEquals("u1", ((UserMessage) persisted.messages().get(0)).singleText());
        assertEquals("a1", ((AiMessage) persisted.messages().get(1)).text());
    }

    @Test
    void compactThreadHandsFreshStateAndNullCheckpointerIsNoop() {
        // #330：rewriter 拿到的是锁内 re-read 的最新快照（不是调用方缓存）；
        // checkpointer 缺席（测试直构图）为无操作。
        InMemoryCheckpointer cp = new InMemoryCheckpointer();
        cp.put("chat:t1", new GraphState(List.of(UserMessage.from("u1")), Map.of()));
        CompiledGraph graph = GraphBuilder.singleNode("echo", (state, ctx) -> Map.of(), cp);
        AtomicReference<Integer> seenCount = new AtomicReference<>();
        assertTrue(graph.compactThread("chat:t1", fresh -> {
            seenCount.set(fresh.messages().size());
            return java.util.Optional.of(fresh);
        }));
        assertEquals(1, seenCount.get());

        CompiledGraph hookless = GraphBuilder.singleNode("echo", (state, ctx) -> Map.of(), null);
        assertFalse(hookless.compactThread("chat:t2", fresh -> java.util.Optional.of(fresh)));
    }

    @Test
    void compactionHookFiresOnDoneAndSkipsCancelled() {
        // #328：非 CANCELLED 终态触发钩子（锁释放后）；仅 CANCELLED 跳过。
        InMemoryCheckpointer cp = new InMemoryCheckpointer();
        List<String> fired = new java.util.concurrent.CopyOnWriteArrayList<>();
        CompiledGraph.CompactionHook hook = (config, finalState) -> fired.add(config.threadId());

        CompiledGraph doneGraph = GraphBuilder.begin()
                .addNode("agent", (state, ctx) ->
                        Map.of(GraphState.MESSAGES, List.of(AiMessage.from("hi"))))
                .entry("agent")
                .addEdge("agent", GraphBuilder.END)
                .compile(cp, "chat", null, hook);
        doneGraph.invoke(new GraphState(), RunConfig.of("chat:t1", "u1"));
        assertEquals(List.of("chat:t1"), fired);

        // 取消轮：半步落盘（#199）但钩子不触发。
        RecordingSink sink = new RecordingSink();
        CompiledGraph cancelGraph = GraphBuilder.begin()
                .addNode("agent", (state, ctx) -> {
                    ctx.emit(RecordingSink.CANCEL_MARKER);
                    return Map.of();
                })
                .entry("agent")
                .addEdge("agent", GraphBuilder.END)
                .compile(cp, "chat", null, hook);
        cancelGraph.stream(new GraphState(), RunConfig.of("chat:t2", "u1"), sink);
        assertEquals(List.of("chat:t1"), fired, "CANCELLED 终态不得触发钩子");
    }

    @Test
    void compactionHookFiresOnFailureAndThreadBusyDoesNotFire() throws Exception {
        // #328：failed（节点异常）/触顶轮的快照同样会无界增长——failed 也触发；
        // THREAD_BUSY 的 run 根本没执行，不触发。钩子异常被吞掉记 WARN，
        // 绝不影响 invoke 的抛出契约。
        List<String> fired = new java.util.concurrent.CopyOnWriteArrayList<>();
        CompiledGraph.CompactionHook hook = (config, finalState) -> {
            fired.add(config.threadId());
            throw new IllegalStateException("hook boom");
        };
        CompiledGraph boomGraph = GraphBuilder.begin()
                .addNode("boom", (state, ctx) -> {
                    throw new IllegalStateException("node boom");
                })
                .entry("boom")
                .addEdge("boom", GraphBuilder.END)
                .compile(null, "chat", null, hook);
        assertThrows(IllegalStateException.class, () -> boomGraph.invoke(
                new GraphState(), RunConfig.of("chat:t1", "u1")));
        assertEquals(List.of("chat:t1"), fired, "failed 轮也要触发（异常被钩子内抛出吞掉）");

        // THREAD_BUSY：第二个 run 快速失败，不触发钩子。
        InMemoryCheckpointer cp = new InMemoryCheckpointer();
        CountDownLatch entered = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        CompiledGraph graph = GraphBuilder.begin()
                .addNode("agent", (state, ctx) -> {
                    entered.countDown();
                    release.await();
                    return Map.of();
                })
                .entry("agent")
                .addEdge("agent", GraphBuilder.END)
                .compile(cp, "chat", null, hook);
        Thread a = new Thread(() -> graph.invoke(new GraphState(), RunConfig.of("chat:t9", "u1")));
        a.start();
        assertTrue(entered.await(5, java.util.concurrent.TimeUnit.SECONDS));
        assertThrows(CompiledGraph.ThreadBusyError.class, () -> graph.invoke(
                new GraphState(), RunConfig.of("chat:t9", "u1")));
        assertEquals(List.of("chat:t1"), fired, "THREAD_BUSY 不触发");
        release.countDown();
        a.join(5000);
    }

    @Test
    void compactionHookSkipsNodeThrownCancellationWithoutSinkCancel() {
        // R4-a F10①：节点抛 CancellationException 且 sink 未取消——
        // streamWithTermination 已把它归 CANCELLED（#9 既有口径），runExit 的
        // failure 路径按同口径归类后，钩子契约「仅 CANCELLED 不调用」不再有缝。
        List<String> fired = new java.util.concurrent.CopyOnWriteArrayList<>();
        CompiledGraph.CompactionHook hook = (config, finalState) -> fired.add(config.threadId());
        CompiledGraph graph = GraphBuilder.begin()
                .addNode("cancel", (state, ctx) -> {
                    throw new java.util.concurrent.CancellationException("node cancelled");
                })
                .entry("cancel")
                .addEdge("cancel", GraphBuilder.END)
                .compile(null, "chat", null, hook);
        // invoke 的抛出契约不变：CancellationException 原样穿出（RuntimeException
        // 归一分支不再包装）。
        assertThrows(java.util.concurrent.CancellationException.class,
                () -> graph.invoke(new GraphState(), RunConfig.of("chat:t1", "u1")));
        assertTrue(fired.isEmpty(), "节点抛 CancellationException = CANCELLED 终态，钩子不触发");

        // 对照：同形态的 run 经 streamWithTermination 归 CANCELLED（口径一致）。
        CompiledGraph annotated = GraphBuilder.begin()
                .addNode("cancel", (state, ctx) -> {
                    throw new java.util.concurrent.CancellationException("node cancelled");
                })
                .entry("cancel")
                .addEdge("cancel", GraphBuilder.END)
                .compile(null, "chat");
        CompiledGraph.RunOutcome outcome = annotated.streamWithTermination(
                new GraphState(), RunConfig.of("chat:t2", "u1"), null);
        assertEquals(Events.RunTermination.CANCELLED, outcome.termination());
    }
}
