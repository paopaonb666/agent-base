package io.github.paopaonb666.agentbase.entrypoints.server;

import io.github.paopaonb666.agentbase.core.AgentModule;
import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.core.graph.CompiledGraph;
import io.github.paopaonb666.agentbase.core.graph.GraphBuilder;
import io.github.paopaonb666.agentbase.core.graph.GraphState;
import io.github.paopaonb666.agentbase.core.graph.InMemoryCheckpointer;
import io.github.paopaonb666.agentbase.core.graph.NodeContext;
import io.github.paopaonb666.agentbase.core.graph.RunConfig;
import io.github.paopaonb666.agentbase.extensions.events.Events;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.verify;

/**
 * SSE 流式契约（#292）：事件序（ping 首帧 → 图事件 → done → complete）、
 * step running 首见去重、图异常转 ErrorEvent + done 补统一终止标注（#9，
 * URL 脱敏在 safeErrorText）、客户端断连取消传播（sink.cancelled 可见、
 * 连接已死时 done 无从送达）。断言用 send 计数 + data 行收集桩
 * （SseEventBuilder 经 build() 反解出 event/data 内容）。
 */
class SseStreamsTest {

    private static RunConfig config() {
        return new RunConfig("chat:sse", "u1", 25);
    }

    /** 单节点图：按脚本 emit 事件后返回。 */
    private static CompiledGraph scriptGraph(java.util.List<Object> eventsToEmit) {
        AgentModule.ModuleContext ctx = new AgentModule.ModuleContext(
                new Settings(), null, new InMemoryCheckpointer(), List.of(), null, null);
        return GraphBuilder.begin()
                .addNode("n", (GraphState state, NodeContext nodeCtx) -> {
                    for (Object event : eventsToEmit) {
                        nodeCtx.emit(event);
                    }
                    return Map.of();
                })
                .entry("n")
                .addEdge("n", GraphBuilder.END)
                .compile(ctx.checkpointer(), "chat");
    }

    private static boolean await(java.util.function.BooleanSupplier condition) {
        // 10s：全量套件下 JVM 负载会让「中断→归类→入队→消费」链路显著
        // 慢于单类跑（实测 5s 预算在满套件偶发超时），条件本身是终态、
        // 提前满足即提前返回，预算只影响失败路径耗时。
        long deadline = System.currentTimeMillis() + 10_000;
        while (System.currentTimeMillis() < deadline) {
            if (condition.getAsBoolean()) {
                return true;
            }
            try {
                Thread.sleep(20);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return false;
            }
        }
        return condition.getAsBoolean();
    }

    /**
     * 事件收集桩：把 send 捕获的 SseEventBuilder 反解为 data 行
     * （SseEventBuilderImpl.build() → DataWithMediaType.getData()）。
     */
    private static List<String> collectData(List<Object> builders) {
        try {
            List<String> dataLines = new ArrayList<>();
            for (Object builder : builders) {
                java.lang.reflect.Method build = builder.getClass().getDeclaredMethod("build");
                build.setAccessible(true);
                for (Object item : (java.util.Set<?>) build.invoke(builder)) {
                    Object data = item.getClass().getMethod("getData").invoke(item);
                    if (data instanceof String line) {
                        dataLines.add(line);
                    }
                }
            }
            return dataLines;
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException("SseEventBuilder 反解失败", e);
        }
    }

    @Test
    void streamsPingFirstThenEventsThenDoneAndCompletes() throws Exception {
        SseEmitter emitter = Mockito.mock(SseEmitter.class);
        AtomicInteger sends = new AtomicInteger();
        List<Object> builders = new CopyOnWriteArrayList<>();
        doAnswer(inv -> {
            sends.incrementAndGet();
            builders.add(inv.getArgument(0));
            return null;
        }).when(emitter).send(any(SseEmitter.SseEventBuilder.class));
        CountDownLatch completed = new CountDownLatch(1);
        Mockito.doAnswer(inv -> {
            completed.countDown();
            return null;
        }).when(emitter).complete();
        CountDownLatch hook = new CountDownLatch(1);
        SseEmitter opened = SseStreams.open(scriptGraph(List.of(
                new Events.StepEvent("n", "running", null),
                new Events.DeltaEvent("你好"))),
                List.of(new dev.langchain4j.data.message.UserMessage("hi")),
                Map.of(), config(), "sse-t1", hook::countDown, emitter);

        assertTrue(completed.await(5, TimeUnit.SECONDS), "complete() 应被调用");
        assertTrue(hook.await(5, TimeUnit.SECONDS), "onComplete 钩子应被触发");
        // ping + 图自身的 running（我发的 running 被首见去重吞掉）+
        // 图 completed + delta + done = 5 次 send
        assertEquals(5, sends.get());
        verify(emitter).complete();
        // #9：正常完成 → done 带统一终止标注 termination=done。
        assertTrue(collectData(builders).stream().anyMatch(
                        d -> d.startsWith("{\"type\":\"done\"") && d.contains("\"termination\":\"done\"")),
                "done 应带 termination=done：" + collectData(builders));
    }

    @Test
    void runningEventsAreDeduplicatedByFirstSight() throws Exception {
        SseEmitter emitter = Mockito.mock(SseEmitter.class);
        AtomicInteger sends = new AtomicInteger();
        doAnswer(inv -> {
            sends.incrementAndGet();
            return null;
        }).when(emitter).send(any(SseEmitter.SseEventBuilder.class));
        CountDownLatch completed = new CountDownLatch(1);
        Mockito.doAnswer(inv -> {
            completed.countDown();
            return null;
        }).when(emitter).complete();
        SseStreams.open(scriptGraph(List.of(
                new Events.StepEvent("n", "running", null),
                new Events.StepEvent("n", "running", null),
                new Events.StepEvent("n", "running", null))),
                List.of(new dev.langchain4j.data.message.UserMessage("hi")),
                Map.of(), config(), "sse-t2", null, emitter);
        assertTrue(completed.await(5, TimeUnit.SECONDS));
        // ping + running（图与我发的多条去重成 1 条）+ completed + done = 4
        assertEquals(4, sends.get());
    }

    @Test
    void graphFailureBecomesErrorEventAndCompletesWithError() throws Exception {
        AgentModule.ModuleContext ctx = new AgentModule.ModuleContext(
                new Settings(), null, new InMemoryCheckpointer(), List.of(), null, null);
        CompiledGraph failing = GraphBuilder.begin()
                .addNode("boom", (GraphState state, NodeContext nodeCtx) -> {
                    throw new IllegalStateException(
                            "上游不可达：https://api.example.com/v1/chat 503");
                })
                .entry("boom")
                .addEdge("boom", GraphBuilder.END)
                .compile(ctx.checkpointer(), "chat");
        SseEmitter emitter = Mockito.mock(SseEmitter.class);
        AtomicInteger sends = new AtomicInteger();
        List<Object> builders = new CopyOnWriteArrayList<>();
        doAnswer(inv -> {
            sends.incrementAndGet();
            builders.add(inv.getArgument(0));
            return null;
        }).when(emitter).send(any(SseEmitter.SseEventBuilder.class));
        // 错误路径：ErrorEvent + done(termination) 经队列发出，consume 收到
        // 哨兵后走 complete()（completeWithError 只在消费端自身的 send
        // 失败/断连时触发）
        CountDownLatch normalCompleted = new CountDownLatch(1);
        Mockito.doAnswer(inv -> {
            normalCompleted.countDown();
            return null;
        }).when(emitter).complete();
        SseStreams.open(failing, List.of(new dev.langchain4j.data.message.UserMessage("hi")),
                Map.of(), config(), "sse-t3", null, emitter);
        assertTrue(normalCompleted.await(5, TimeUnit.SECONDS));
        // ping + 图的 running（节点执行前发出）+ error + done = 4；#9 后异常
        // 路径以 done(termination) 收尾（旧形状为 3 且 done 永不发出）。
        assertEquals(4, sends.get());
        Mockito.verify(emitter, Mockito.never()).completeWithError(any(Throwable.class));
        // ErrorEvent（既有形状不动）与 done 的统一终止标注各就各位。
        List<String> data = collectData(builders);
        assertTrue(data.stream().anyMatch(d -> d.startsWith("{\"type\":\"error\"")),
                "error 事件应在场：" + data);
        assertTrue(data.stream().anyMatch(
                        d -> d.startsWith("{\"type\":\"done\"") && d.contains("\"termination\":\"internal_error\"")),
                "done 应带 termination=internal_error：" + data);
    }

    @Test
    void clientDisconnectCancelsProducerAndSkipsDone() throws Exception {
        // 节点自旋直到 sink 报告取消（模拟在途 LLM 工作）；第 2 次 send 抛
        // IOException 触发 shutdown → 取消传播。
        AgentModule.ModuleContext ctx = new AgentModule.ModuleContext(
                new Settings(), null, new InMemoryCheckpointer(), List.of(), null, null);
        CompiledGraph spinner = GraphBuilder.begin()
                .addNode("spin", (GraphState state, NodeContext nodeCtx) -> {
                    long deadline = System.currentTimeMillis() + 8000;
                    while (!nodeCtx.cancelled() && System.currentTimeMillis() < deadline) {
                        nodeCtx.emit(new Events.StepEvent("spin", "running", null));
                        Thread.sleep(10);
                    }
                    return Map.of();
                })
                .entry("spin")
                .addEdge("spin", GraphBuilder.END)
                .compile(ctx.checkpointer(), "chat");
        SseEmitter emitter = Mockito.mock(SseEmitter.class);
        AtomicInteger sends = new AtomicInteger();
        doAnswer(inv -> {
            if (sends.incrementAndGet() > 1) { // 首帧 ping 后模拟断连
                throw new java.io.IOException("broken pipe");
            }
            return null;
        }).when(emitter).send(any(SseEmitter.SseEventBuilder.class));
        AtomicReference<Throwable> errorRef = new AtomicReference<>();
        CountDownLatch errorCompleted = new CountDownLatch(1);
        Mockito.doAnswer(inv -> {
            errorRef.set(inv.getArgument(0));
            errorCompleted.countDown();
            return null;
        }).when(emitter).completeWithError(any(Throwable.class));
        SseStreams.open(spinner, List.of(new dev.langchain4j.data.message.UserMessage("hi")),
                Map.of(), config(), "sse-t4", null, emitter);
        assertTrue(errorCompleted.await(5, TimeUnit.SECONDS), "断连应 completeWithError");
        // done(termination=cancelled) 会被生产者尽力入队（#9），但连接已死
        // 无从送达——wire 上停发（send 总数停在断连时刻，且不再稳定递增）。
        int sendsAtError = sends.get();
        Thread.sleep(150);
        assertEquals(sendsAtError, sends.get(), "取消后生产者应停发");
        assertTrue(sendsAtError >= 2, "ping 与首个事件都已尝试发送");
    }

    @Test
    void clientCancellationEmitsDoneWithCancelledTermination() throws Exception {
        // #9：取消态的标注可观测路径——经 onCompletion 回调触发 shutdown
        //（置取消态 + 中断生产者），消费者存活；节点被中断后图层在取消态
        // 归 CANCELLED（而非 INTERNAL_ERROR），done(termination=cancelled)
        // 送达 wire。
        AgentModule.ModuleContext ctx = new AgentModule.ModuleContext(
                new Settings(), null, new InMemoryCheckpointer(), List.of(), null, null);
        CompiledGraph spinner = GraphBuilder.begin()
                .addNode("spin", (GraphState state, NodeContext nodeCtx) -> {
                    long deadline = System.currentTimeMillis() + 8000;
                    while (!nodeCtx.cancelled() && System.currentTimeMillis() < deadline) {
                        Thread.sleep(20);
                    }
                    return Map.of();
                })
                .entry("spin")
                .addEdge("spin", GraphBuilder.END)
                .compile(ctx.checkpointer(), "chat");
        SseEmitter emitter = Mockito.mock(SseEmitter.class);
        List<Object> builders = new CopyOnWriteArrayList<>();
        doAnswer(inv -> {
            builders.add(inv.getArgument(0));
            return null;
        }).when(emitter).send(any(SseEmitter.SseEventBuilder.class));
        AtomicReference<Runnable> shutdown = new AtomicReference<>();
        Mockito.doAnswer(inv -> {
            shutdown.set(inv.getArgument(0));
            return null;
        }).when(emitter).onCompletion(any(Runnable.class));
        SseStreams.open(spinner, List.of(new dev.langchain4j.data.message.UserMessage("hi")),
                Map.of(), config(), "sse-t5", null, emitter);

        assertNotNull(shutdown.get(), "onCompletion 回调应已注册");
        shutdown.get().run(); // 模拟容器层取消：置取消态 + 中断生产者
        assertTrue(await(() -> collectData(builders).stream().anyMatch(
                        d -> d.startsWith("{\"type\":\"done\"") && d.contains("\"termination\":\"cancelled\""))),
                "done 应带 termination=cancelled 送达：" + collectData(builders));
    }

    @Test
    void safeErrorTextRedactsUrlsAndCapsLength() {
        String redacted = SseStreams.safeErrorText(
                new IllegalStateException("connect https://user:secret@internal.corp/x?token=1 failed"));
        assertTrue(redacted.contains("<redacted-url>"));
        assertTrue(!redacted.contains("secret"));
        String huge = SseStreams.safeErrorText(
                new IllegalStateException("x".repeat(1000)));
        assertEquals(300, huge.length());
        assertEquals("RuntimeException: ", SseStreams.safeErrorText(
                new RuntimeException("")).substring(0, "RuntimeException: ".length()));
    }

    @Test
    void putUninterruptiblyBailsOutOfFullQueueWhenCancelled() throws Exception {
        // review R1-F1（P0）回归：取消态 + 满队列 + 消费者死亡（再无新中断）
        // 时，入队必须有限等待后放弃——否则生产者永久阻塞、终态钩子（#79
        // 闸门释放）永不执行，32 次慢断连后全站 invoke 永久 429。
        java.util.concurrent.BlockingQueue<Object> queue =
                new java.util.concurrent.ArrayBlockingQueue<>(2);
        queue.put(new Object());
        queue.put(new Object()); // 满
        java.util.concurrent.atomic.AtomicBoolean cancelled =
                new java.util.concurrent.atomic.AtomicBoolean(true);
        long start = System.nanoTime();
        org.junit.jupiter.api.Assertions.assertThrows(
                java.util.concurrent.CancellationException.class,
                () -> SseStreams.putUninterruptibly(queue, new Object(), cancelled));
        long elapsedMillis = (System.nanoTime() - start) / 1_000_000;
        org.junit.jupiter.api.Assertions.assertTrue(elapsedMillis < 5_000,
                "取消态满队列应在有限时间放弃，实测 " + elapsedMillis + "ms");
        // 对照：队列有空位时取消态事件照常送达（终态标注尽力可达）。
        queue.take();
        SseStreams.putUninterruptibly(queue, new Object(), cancelled);
        assertEquals(2, queue.size());
    }
}
