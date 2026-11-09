package io.github.paopaonb666.agentbase.core.llm;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.StreamingChatModel;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** #72：自研熔断器——连续失败开路、快速失败、冷却后单发探测、禁用直通。 */
class CircuitBreakedModelRuntimeTest {

    private static final List<ChatMessage> MESSAGES = List.of(new UserMessage("hi"));

    /** 计数桩：failures>0 时每次调用抛指定异常，否则成功。 */
    private static final class CountingRuntime implements ModelRuntime {
        final AtomicInteger calls = new AtomicInteger();
        private final AtomicInteger remainingFailures;
        private final RuntimeException failure;

        CountingRuntime(int remainingFailures, RuntimeException failure) {
            this.remainingFailures = new AtomicInteger(remainingFailures);
            this.failure = failure;
        }

        int invocations() { return calls.get(); }

        @Override public ChatModel chat() { throw new UnsupportedOperationException(); }
        @Override public StreamingChatModel streaming() { return null; }
        @Override public AiMessage stream(List<ChatMessage> messages,
                List<dev.langchain4j.agent.tool.ToolSpecification> tools,
                java.util.function.Consumer<String> onDelta) { return invoke(messages); }
        @Override public AiMessage invoke(List<ChatMessage> messages) {
            calls.incrementAndGet();
            if (remainingFailures.getAndDecrement() > 0) {
                throw failure;
            }
            return AiMessage.from("ok");
        }
    }

    @Test
    void consecutiveFailuresTripOpenAndFastFail() {
        CountingRuntime inner = new CountingRuntime(3, new IllegalStateException("provider down"));
        AtomicLong clock = new AtomicLong(0);
        CircuitBreakedModelRuntime breaker =
                new CircuitBreakedModelRuntime(inner, 3, 30, clock::get);
        for (int i = 0; i < 3; i++) {
            assertThrows(IllegalStateException.class, () -> breaker.invoke(MESSAGES));
        }
        assertEquals(3, inner.invocations(), "开路前三次真实到达 inner");
        // open：第 4 次不再到达 inner，快速失败带剩余冷却秒数。
        CircuitBreakedModelRuntime.CircuitOpenException open = assertThrows(
                CircuitBreakedModelRuntime.CircuitOpenException.class,
                () -> breaker.invoke(MESSAGES));
        assertTrue(open.retryAfterSeconds > 0 && open.retryAfterSeconds <= 30,
                "Retry-After 素材应为剩余冷却秒数：" + open.retryAfterSeconds);
        assertEquals(3, inner.invocations());
        assertEquals(1, breaker.openCount());
        assertEquals("open(30s)", breaker.stateDescription());
    }

    @Test
    void cooldownExpiryProbesOnceThenClosesOrReopens() {
        CountingRuntime inner = new CountingRuntime(3, new IllegalStateException("down"));
        AtomicLong clock = new AtomicLong(0);
        CircuitBreakedModelRuntime breaker =
                new CircuitBreakedModelRuntime(inner, 3, 30, clock::get);
        for (int i = 0; i < 3; i++) {
            assertThrows(IllegalStateException.class, () -> breaker.invoke(MESSAGES));
        }
        // 冷却到期：单发探测——探测成功闭路，后续调用恢复直通。
        clock.addAndGet(31_000_000_000L);
        assertEquals("ok", breaker.invoke(MESSAGES).text());
        assertEquals(4, inner.invocations(), "half-open 恰好放行一个探测");
        assertEquals("ok", breaker.invoke(MESSAGES).text());
        assertEquals(5, inner.invocations(), "闭路后直通");

        // 探测失败 → 重开：再次快速失败（不达 inner）。
        CountingRuntime alwaysFailing = new CountingRuntime(Integer.MAX_VALUE,
                new IllegalStateException("still down"));
        CircuitBreakedModelRuntime reopening =
                new CircuitBreakedModelRuntime(alwaysFailing, 2, 30, clock::get);
        for (int i = 0; i < 2; i++) {
            assertThrows(IllegalStateException.class, () -> reopening.invoke(MESSAGES));
        }
        clock.addAndGet(31_000_000_000L);
        assertThrows(IllegalStateException.class, () -> reopening.invoke(MESSAGES)); // 探测，失败
        CircuitBreakedModelRuntime.CircuitOpenException reopened = assertThrows(
                CircuitBreakedModelRuntime.CircuitOpenException.class,
                () -> reopening.invoke(MESSAGES));
        assertTrue(reopened.retryAfterSeconds > 0, "探测失败应重开");
    }

    @Test
    void disabledThresholdPassesThrough() {
        CountingRuntime inner = new CountingRuntime(Integer.MAX_VALUE,
                new IllegalStateException("down"));
        CircuitBreakedModelRuntime breaker = new CircuitBreakedModelRuntime(inner, 0, 30);
        for (int i = 0; i < 10; i++) {
            assertThrows(IllegalStateException.class, () -> breaker.invoke(MESSAGES));
        }
        assertEquals(10, inner.invocations(), "阈值 0 = 禁用，全部直通");
        assertEquals(0, breaker.openCount());
    }

    @Test
    void openPrimaryBreakerLetsChainReachFallback() {
        // #72 × #71 组合：主力熔断 open 抛 CircuitOpenException → 链把它当
        // 普通失败处理，自动跳到备用模型（备用熔断独立状态机）。
        CountingRuntime primary = new CountingRuntime(Integer.MAX_VALUE,
                new IllegalStateException("primary down"));
        CountingRuntime backup = new CountingRuntime(0, null);
        AtomicLong clock = new AtomicLong(0);
        ResilientModelRuntime chain = new ResilientModelRuntime(
                new CircuitBreakedModelRuntime(primary, 2, 30, clock::get),
                List.of(new CircuitBreakedModelRuntime(backup, 2, 30, clock::get)),
                0, 4, 0.0, 60, 20);
        for (int i = 0; i < 2; i++) {
            chain.invoke(MESSAGES); // 前两次走主力失败 → fallback 成功
        }
        assertEquals(2, primary.invocations());
        assertEquals(2, backup.invocations());
        // 第 3 次：主力熔断 open（连续 2 败）→ 链直接跳 fallback，主力不再被碰。
        assertEquals("ok", chain.invoke(MESSAGES).text());
        assertEquals(2, primary.invocations(), "主力熔断 open 后链不再触碰主力");
        assertEquals(3, backup.invocations());
    }
}
