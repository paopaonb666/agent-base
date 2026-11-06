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
}
