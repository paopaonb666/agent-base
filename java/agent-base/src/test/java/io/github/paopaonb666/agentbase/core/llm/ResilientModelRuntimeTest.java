package io.github.paopaonb666.agentbase.core.llm;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.output.TokenUsage;
import io.github.paopaonb666.agentbase.core.config.Settings;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 主路径弹性包装测试（roadmap #1/#10/#73/#75）：瞬时错误指数退避重试、
 * 流式重试边界（发过 delta 不重试）、滑窗重试预算（重试风暴防护）、
 * 并发信号量与 attempt 计量口径。
 */
class ResilientModelRuntimeTest {

    /** 计数桩：前 failTimes 次抛错，之后成功返回固定 AiMessage。 */
    private static class FlakyRuntime implements ModelRuntime {
        final AtomicInteger invocations = new AtomicInteger();
        final AtomicInteger failures;
        final RuntimeException failure;

        FlakyRuntime(int failures, RuntimeException failure) {
            this.failures = new AtomicInteger(failures);
            this.failure = failure;
        }

        private AiMessage ok() {
            return AiMessage.from("ok");
        }

        @Override
        public ChatModel chat() {
            throw new UnsupportedOperationException();
        }

        @Override
        public StreamingChatModel streaming() {
            return null;
        }

        @Override
        public AiMessage stream(List<ChatMessage> messages, List<dev.langchain4j.agent.tool.ToolSpecification> tools,
                                java.util.function.Consumer<String> onDelta) {
            invocations.incrementAndGet();
            if (failures.getAndDecrement() > 0) {
                // 先发一个 delta 再失败（模拟半截流）——是否重试由调用方守卫。
                onDelta.accept("partial-");
                throw failure;
            }
            onDelta.accept("done");
            return ok();
        }

        @Override
        public AiMessage invoke(List<ChatMessage> messages) {
            invocations.incrementAndGet();
            if (failures.getAndDecrement() > 0) {
                throw failure;
            }
            return ok();
        }
    }

    private static List<ChatMessage> oneMessage() {
        return List.of(UserMessage.from("hi"));
    }

    @Test
    void transientInvokeFailuresAreRetriedWithBackoff() {
        FlakyRuntime inner = new FlakyRuntime(2, new IllegalStateException("503"));
        ResilientModelRuntime wrapper = new ResilientModelRuntime(inner, 2, 4, 0.0, 60, 20);
        assertEquals("ok", wrapper.invoke(oneMessage()).text());
        // 2 次失败 + 1 次成功 = 3 次真实调用（usage 每次各记一条的结构前提）。
        assertEquals(3, inner.invocations.get());
    }

    @Test
    void retriesExhaustedRethrowsTheLastError() {
        FlakyRuntime inner = new FlakyRuntime(99, new IllegalStateException("429"));
        ResilientModelRuntime wrapper = new ResilientModelRuntime(inner, 1, 4, 0.0, 60, 20);
        IllegalStateException error = assertThrows(IllegalStateException.class,
                () -> wrapper.invoke(oneMessage()));
        assertEquals("429", error.getMessage());
        // 1 次重试 + 1 次首发 = 2 次真实调用。
        assertEquals(2, inner.invocations.get());
    }

    @Test
    void streamWithEmittedDeltasIsNeverRetried() {
        // 半截流（已发 delta）中断 → 立即上抛，不重试——重试会重复输出。
        FlakyRuntime inner = new FlakyRuntime(1, new IllegalStateException("stream reset"));
        ResilientModelRuntime wrapper = new ResilientModelRuntime(inner, 3, 4, 0.0, 60, 20);
        List<String> deltas = new java.util.ArrayList<>();
        assertThrows(IllegalStateException.class,
                () -> wrapper.stream(oneMessage(), null, deltas::add));
        assertEquals(1, inner.invocations.get(), "已发 delta 的流中断不重试");
        assertEquals(List.of("partial-"), deltas);
    }

    @Test
    void streamBeforeFirstDeltaIsRetried() {
        // 首个 delta 之前的失败（连接/建流阶段）→ 可重试。
        FlakyRuntime inner = new FlakyRuntime(1, new IllegalStateException("connect timeout")) {
            @Override
            public AiMessage stream(List<ChatMessage> messages,
                                    List<dev.langchain4j.agent.tool.ToolSpecification> tools,
                                    java.util.function.Consumer<String> onDelta) {
                invocations.incrementAndGet();
                if (failures.getAndDecrement() > 0) {
                    throw failure; // 未发任何 delta
                }
                onDelta.accept("done");
                return AiMessage.from("done");
            }
        };
        ResilientModelRuntime wrapper = new ResilientModelRuntime(inner, 2, 4, 0.0, 60, 20);
        assertEquals("done", wrapper.stream(oneMessage(), null, d -> {}).text());
        assertEquals(2, inner.invocations.get());
    }

    @Test
    void retryBudgetStopsTheRetryStorm() {
        // 滑窗预算：窗口内重试次数封顶——预算耗尽后最后一次错误原样上抛。
        FlakyRuntime inner = new FlakyRuntime(99, new IllegalStateException("429"));
        ResilientModelRuntime wrapper = new ResilientModelRuntime(inner, 50, 4, 0.0, 60, 2);
        assertThrows(IllegalStateException.class, () -> wrapper.invoke(oneMessage()));
        // 首发 1 + 预算内重试 2 = 3 次真实调用（第 3 次重试被预算拒绝）。
        assertEquals(3, inner.invocations.get());
    }

    @Test
    void semaphoreBoundsConcurrentCalls() throws Exception {
        // 并发上限：maxConcurrency=1 时两个并发调用必须串行。
        AtomicInteger inFlight = new AtomicInteger();
        AtomicInteger maxInFlight = new AtomicInteger();
        ModelRuntime slow = new ModelRuntime() {
            @Override public ChatModel chat() { throw new UnsupportedOperationException(); }
            @Override public StreamingChatModel streaming() { return null; }
            @Override public AiMessage stream(List<ChatMessage> m,
                    List<dev.langchain4j.agent.tool.ToolSpecification> t,
                    java.util.function.Consumer<String> onDelta) {
                throw new UnsupportedOperationException();
            }
            @Override public AiMessage invoke(List<ChatMessage> messages) {
                int now = inFlight.incrementAndGet();
                maxInFlight.accumulateAndGet(now, Math::max);
                try {
                    Thread.sleep(50);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
                inFlight.decrementAndGet();
                return AiMessage.from("ok");
            }
        };
        ResilientModelRuntime wrapper = new ResilientModelRuntime(slow, 0, 1);
        Thread a = new Thread(() -> wrapper.invoke(oneMessage()));
        Thread b = new Thread(() -> wrapper.invoke(oneMessage()));
        a.start();
        b.start();
        a.join(5_000);
        b.join(5_000);
        assertEquals(1, maxInFlight.get(), "并发信号量应把在飞调用数钉在 1");
    }

    @Test
    void settingsCarryTheResilienceKnobs() {
        // 配置面（#1/#73）：LLM_MAIN_TIMEOUT/LLM_MAIN_RETRIES/LLM_MAIN_CONCURRENCY。
        Settings settings = Settings.load(Map.of(
                "ENV", "development",
                "LLM_MAIN_TIMEOUT", "120.5",
                "LLM_MAIN_RETRIES", "3",
                "LLM_MAIN_CONCURRENCY", "4"), null);
        assertEquals(120.5, settings.llm.mainTimeoutSeconds);
        assertEquals(3, settings.llm.mainRetries);
        assertEquals(4, settings.llm.mainConcurrency);
        // 默认值：600s / 2 / 8。
        Settings defaults = Settings.load(Map.of("ENV", "development"), null);
        assertEquals(600.0, defaults.llm.mainTimeoutSeconds);
        assertEquals(2, defaults.llm.mainRetries);
        assertEquals(8, defaults.llm.mainConcurrency);
        // 非法值快速失败。
        assertThrows(Settings.SettingsError.class,
                () -> Settings.load(Map.of("ENV", "development", "LLM_MAIN_RETRIES", "-1"), null));
        assertThrows(Settings.SettingsError.class,
                () -> Settings.load(Map.of("ENV", "development", "LLM_MAIN_CONCURRENCY", "0"), null));
        assertTrue(true);
    }

    @Test
    void transparentLayerDelegatesAuditFaces() {
        // review R2-F5：弹性包装是透明层——toolSafeStreaming（#128 可观测面）
        // 与 contextWindowTokens（#4）都必须穿透到内层运行时，否则快档将来
        // 被包装时 tool-safe 回退静默失效、窗口推导拿 0。
        ModelRuntime inner = new ModelRuntime() {
            @Override public ChatModel chat() { throw new UnsupportedOperationException(); }
            @Override public StreamingChatModel streaming() { return null; }
            @Override public AiMessage stream(List<ChatMessage> messages,
                    List<dev.langchain4j.agent.tool.ToolSpecification> tools,
                    java.util.function.Consumer<String> onDelta) { throw new UnsupportedOperationException(); }
            @Override public AiMessage invoke(List<ChatMessage> messages) { throw new UnsupportedOperationException(); }
            @Override public boolean toolSafeStreaming() { return true; }
            @Override public int contextWindowTokens() { return 8192; }
        };
        ModelRuntime wrapped = new ResilientModelRuntime(inner, 2, 4);
        assertTrue(wrapped.toolSafeStreaming());
        assertEquals(8192, wrapped.contextWindowTokens());
    }

    // ── #71 fallback 链 ──────────────────────────────────────────────

    /** 恒成功/恒败的计数桩（fallback 语义断言用）。 */
    private static class CountingRuntime implements ModelRuntime {
        final java.util.concurrent.atomic.AtomicInteger calls = new java.util.concurrent.atomic.AtomicInteger();
        private final AiMessage reply;
        private final RuntimeException failure;

        CountingRuntime(AiMessage reply, RuntimeException failure) {
            this.reply = reply;
            this.failure = failure;
        }

        int invocations() { return calls.get(); }

        @Override public ChatModel chat() { throw new UnsupportedOperationException(); }
        @Override public StreamingChatModel streaming() { return null; }
        @Override public AiMessage stream(List<ChatMessage> messages,
                List<dev.langchain4j.agent.tool.ToolSpecification> tools,
                java.util.function.Consumer<String> onDelta) {
            calls.incrementAndGet();
            if (failure != null) {
                throw failure;
            }
            onDelta.accept("ok");
            return reply;
        }
        @Override public AiMessage invoke(List<ChatMessage> messages) {
            calls.incrementAndGet();
            if (failure != null) {
                throw failure;
            }
            return reply;
        }
    }

    @Test
    void invokeFallsThroughChainWhenPrimaryExhausts() {
        CountingRuntime primary = new CountingRuntime(null, new IllegalStateException("primary down"));
        CountingRuntime backup1 = new CountingRuntime(null, new IllegalStateException("backup1 down"));
        CountingRuntime backup2 = new CountingRuntime(AiMessage.from("ok"), null);
        ResilientModelRuntime chain = new ResilientModelRuntime(primary,
                List.of(backup1, backup2), 1, 4, 0.0, 60, 20);
        assertEquals("ok", chain.invoke(oneMessage()).text());
        // 主力打满重试（1+1），两个 fallback 各恰一次真实调用。
        assertEquals(2, primary.invocations());
        assertEquals(1, backup1.invocations());
        assertEquals(1, backup2.invocations());
    }

    @Test
    void allChainMembersFailThrowsLastFallbackError() {
        CountingRuntime primary = new CountingRuntime(null, new IllegalStateException("primary down"));
        CountingRuntime backup = new CountingRuntime(null, new IllegalStateException("backup down"));
        ResilientModelRuntime chain = new ResilientModelRuntime(primary,
                List.of(backup), 0, 4, 0.0, 60, 20);
        IllegalStateException error = assertThrows(IllegalStateException.class,
                () -> chain.invoke(oneMessage()));
        assertEquals("backup down", error.getMessage(), "全链失败抛最后一个 fallback 的错误");
        assertEquals(1, error.getSuppressed().length, "主力故障以 suppressed 保留对账");
    }

    @Test
    void streamSwitchesToFallbackOnlyBeforeFirstDelta() {
        // delta 已发 → 不切 fallback（重放=重复输出）；未发 delta → 正常切换。
        CountingRuntime emittingThenFailing = new CountingRuntime(null, null) {
            @Override public AiMessage stream(List<ChatMessage> messages,
                    List<dev.langchain4j.agent.tool.ToolSpecification> tools,
                    java.util.function.Consumer<String> onDelta) {
                calls.incrementAndGet();
                onDelta.accept("半截");
                throw new IllegalStateException("stalled after delta");
            }
        };
        CountingRuntime backup = new CountingRuntime(AiMessage.from("ok"), null);
        ResilientModelRuntime chain = new ResilientModelRuntime(emittingThenFailing,
                List.of(backup), 0, 4, 0.0, 60, 20);
        StringBuilder deltas = new StringBuilder();
        IllegalStateException error = assertThrows(IllegalStateException.class,
                () -> chain.stream(oneMessage(), List.of(), deltas::append));
        assertEquals("stalled after delta", error.getMessage());
        assertEquals("半截", deltas.toString());
        assertEquals(0, backup.invocations(), "已发 delta 不得切 fallback");

        // 对照：主力未发 delta 即败 → fallback 正常接手。
        CountingRuntime silentFailure = new CountingRuntime(null, new IllegalStateException("primary down"));
        ResilientModelRuntime chain2 = new ResilientModelRuntime(silentFailure,
                List.of(backup), 0, 4, 0.0, 60, 20);
        assertEquals("ok", chain2.stream(oneMessage(), List.of(), delta -> {}).text());
        assertEquals(1, backup.invocations());
    }

    @Test
    void tokenFactorDelegatesThroughTransparencyContract() {
        // #340：ResilientModelRuntime/CircuitBreakedModelRuntime 是透明层
        // ——tokenFactor 必须委托内层（R2-F5 同款教训：漏委托=静默回 1.0）。
        io.github.paopaonb666.agentbase.core.llm.ModelRuntime inner =
                new io.github.paopaonb666.agentbase.core.llm.ModelRuntime() {
                    @Override public dev.langchain4j.model.chat.ChatModel chat() {
                        throw new UnsupportedOperationException();
                    }
                    @Override public dev.langchain4j.model.chat.StreamingChatModel streaming() {
                        throw new UnsupportedOperationException();
                    }
                    @Override public dev.langchain4j.data.message.AiMessage stream(
                            java.util.List<dev.langchain4j.data.message.ChatMessage> m,
                            java.util.List<dev.langchain4j.agent.tool.ToolSpecification> t,
                            java.util.function.Consumer<String> c) {
                        throw new UnsupportedOperationException();
                    }
                    @Override public dev.langchain4j.data.message.AiMessage invoke(
                            java.util.List<dev.langchain4j.data.message.ChatMessage> m) {
                        throw new UnsupportedOperationException();
                    }
                    @Override public double tokenFactor() { return 1.75; }
                };
        var resilient = new io.github.paopaonb666.agentbase.core.llm.ResilientModelRuntime(
                inner, java.util.List.of(), 2, 4);
        assertEquals(1.75, resilient.tokenFactor(), 1e-12);
        var breaked = new io.github.paopaonb666.agentbase.core.llm.CircuitBreakedModelRuntime(
                resilient, 3, 30);
        assertEquals(1.75, breaked.tokenFactor(), 1e-12);
    }
}
