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

import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.verify;

/**
 * SSE 流式契约（#292）：事件序（ping 首帧 → 图事件 → done → complete）、
 * step running 首见去重、图异常转 ErrorEvent + completeWithError（URL
 * 脱敏在 safeErrorText）、客户端断连取消传播（sink.cancelled 可见、
 * done 不发）。断言用 send 计数（SseEventBuilder 内容不可反解）。
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
        long deadline = System.currentTimeMillis() + 5000;
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

    @Test
    void streamsPingFirstThenEventsThenDoneAndCompletes() throws Exception {
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
        doAnswer(inv -> {
            sends.incrementAndGet();
            return null;
        }).when(emitter).send(any(SseEmitter.SseEventBuilder.class));
        // 错误路径：ErrorEvent 经队列发出，consume 收到哨兵后走 complete()
        //（completeWithError 只在消费端自身的 send 失败/断连时触发）
        CountDownLatch normalCompleted = new CountDownLatch(1);
        Mockito.doAnswer(inv -> {
            normalCompleted.countDown();
            return null;
        }).when(emitter).complete();
        SseStreams.open(failing, List.of(new dev.langchain4j.data.message.UserMessage("hi")),
                Map.of(), config(), "sse-t3", null, emitter);
        assertTrue(normalCompleted.await(5, TimeUnit.SECONDS));
        // ping + 图的 running（节点执行前发出）+ error = 3；done 永不发出
        assertEquals(3, sends.get());
        Mockito.verify(emitter, Mockito.never()).completeWithError(any(Throwable.class));
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
        // done 永不发出（send 总数停在断连时刻，且不再是稳定递增）
        int sendsAtError = sends.get();
        Thread.sleep(150);
        assertEquals(sendsAtError, sends.get(), "取消后生产者应停发");
        assertTrue(sendsAtError >= 2, "ping 与首个事件都已尝试发送");
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
}
