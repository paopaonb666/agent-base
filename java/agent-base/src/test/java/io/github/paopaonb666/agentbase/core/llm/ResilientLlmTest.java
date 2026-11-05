package io.github.paopaonb666.agentbase.core.llm;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.output.TokenUsage;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.BiConsumer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@link ResilientLlm} 的分派/限流/重试/回退测试。
 *
 * <p>重试语义与 Python core/llm.py 的 {@code ResilientLLM} 对齐：attempts =
 * max_retries + 1 = 3（默认），指数退避 1s/2s——测试里用 baseDelay=0 的
 * 全参构造避免真睡；默认值与 Python 的一致性单独断言。
 */
class ResilientLlmTest {

    private static final List<ChatMessage> MESSAGES = List.of(new UserMessage("hi"));

    /** 429/超时等瞬时错误：Python except Exception 全兜，这里同样用任意 RuntimeException。 */
    private static ScriptedModels.ScriptedResponse rateLimited() {
        return ScriptedModels.ScriptedResponse.failing(new RuntimeException("HTTP 429 Too Many Requests"));
    }

    private static ResilientLlm resilient(ScriptedModels.ScriptedRuntime fast,
                                          ScriptedModels.ScriptedRuntime main,
                                          String profile) {
        // 全参构造：maxConcurrency=1（默认构造的语义等价，只是避免真睡 1s/2s）。
        return new ResilientLlm(fast, main, 1, profile, ResilientLlm.DEFAULT_MAX_RETRIES, 0.0);
    }

    @Test
    void pythonRetryDefaultsAreLocked() {
        // 重试参数对齐 Python llm.py：max_retries=2、base_delay=1.0。
        assertEquals(2, ResilientLlm.DEFAULT_MAX_RETRIES);
        assertEquals(1.0, ResilientLlm.DEFAULT_BASE_DELAY_SECONDS);
    }

    @Test
    void invalidConstructorArgsThrow() {
        ScriptedModels.ScriptedRuntime main = ScriptedModels.ScriptedRuntime.main(
                List.of(ScriptedModels.ScriptedResponse.of("ok")));
        assertThrows(IllegalArgumentException.class, () -> new ResilientLlm(null, main, 0));
        assertThrows(IllegalArgumentException.class,
                () -> new ResilientLlm(null, main, 1, "split", -1, 1.0));
        assertThrows(IllegalArgumentException.class,
                () -> new ResilientLlm(null, main, 1, "split", 2, -0.5));
        assertThrows(IllegalArgumentException.class,
                () -> new ResilientLlm(null, main, 1, "nope", 2, 1.0));
        assertThrows(NullPointerException.class, () -> new ResilientLlm(null, null, 1));
    }

    @Test
    void splitDispatchesToFast() {
        ScriptedModels.ScriptedRuntime fast = ScriptedModels.ScriptedRuntime.fast(
                List.of(ScriptedModels.ScriptedResponse.of("fast answer")));
        ScriptedModels.ScriptedRuntime main = ScriptedModels.ScriptedRuntime.main(
                List.of(ScriptedModels.ScriptedResponse.of("main answer")));
        ResilientLlm llm = resilient(fast, main, "split");
        assertEquals("fast answer", llm.invoke(MESSAGES));
        assertEquals(1, fast.chatModel.received().size());
        assertEquals(0, main.chatModel.received().size(), "主力未被消耗：快档可用时不回退");
    }

    @Test
    void allFastDispatchesToFast() {
        ScriptedModels.ScriptedRuntime fast = ScriptedModels.ScriptedRuntime.fast(
                List.of(ScriptedModels.ScriptedResponse.of("fast answer")));
        ScriptedModels.ScriptedRuntime main = ScriptedModels.ScriptedRuntime.main(
                List.of(ScriptedModels.ScriptedResponse.of("main answer")));
        assertEquals("fast answer", resilient(fast, main, "all_fast").invoke(MESSAGES));
        assertEquals(0, main.chatModel.received().size());
    }

    @Test
    void allMainSkipsFast() {
        ScriptedModels.ScriptedRuntime fast = ScriptedModels.ScriptedRuntime.fast(
                List.of(ScriptedModels.ScriptedResponse.of("fast answer")));
        ScriptedModels.ScriptedRuntime main = ScriptedModels.ScriptedRuntime.main(
                List.of(ScriptedModels.ScriptedResponse.of("main answer")));
        assertEquals("main answer", resilient(fast, main, "all_main").invoke(MESSAGES));
        assertEquals(0, fast.chatModel.received().size(), "all_main 不经快档");
    }

    @Test
    void splitWithFastNullUsesMain() {
        ScriptedModels.ScriptedRuntime main = ScriptedModels.ScriptedRuntime.main(
                List.of(ScriptedModels.ScriptedResponse.of("main answer")));
        // 快档未配置（组合根传 null）：全部流量走主力，不报错。
        ResilientLlm llm = new ResilientLlm(null, main, 2);
        assertEquals("main answer", llm.invoke(MESSAGES));
    }

    @Test
    void splitRetriesFastBeforeFallingBack() {
        ScriptedModels.ScriptedRuntime fast = ScriptedModels.ScriptedRuntime.fast(
                List.of(rateLimited(), ScriptedModels.ScriptedResponse.of("fast answer")));
        ScriptedModels.ScriptedRuntime main = ScriptedModels.ScriptedRuntime.main(
                List.of(ScriptedModels.ScriptedResponse.of("main answer")));
        assertEquals("fast answer", resilient(fast, main, "split").invoke(MESSAGES));
        assertEquals(2, fast.chatModel.received().size(), "第一次 429、第二次成功");
        assertEquals(0, main.chatModel.received().size(), "重试成功就不回退");
    }

    @Test
    void splitFallsBackToMainAfterRetriesExhausted() {
        // 默认 max_retries=2 → 快档最多试 3 次，全部 429 后回退主力。
        ScriptedModels.ScriptedRuntime fast = ScriptedModels.ScriptedRuntime.fast(
                List.of(rateLimited(), rateLimited(), rateLimited()));
        ScriptedModels.ScriptedRuntime main = ScriptedModels.ScriptedRuntime.main(
                List.of(ScriptedModels.ScriptedResponse.of("main answer")));
        assertEquals("main answer", resilient(fast, main, "split").invoke(MESSAGES));
        assertEquals(3, fast.chatModel.received().size());
        assertEquals(1, main.chatModel.received().size(), "回退恰好一次");
    }

    @Test
    void fallbackExceptionPropagatesRaw() {
        // 回退调用本身不再重试；其异常原样上抛。
        ScriptedModels.ScriptedRuntime fast = ScriptedModels.ScriptedRuntime.fast(
                List.of(rateLimited(), rateLimited(), rateLimited()));
        ScriptedModels.ScriptedRuntime main = ScriptedModels.ScriptedRuntime.main(
                List.of(ScriptedModels.ScriptedResponse.failing(new RuntimeException("main down"))));
        RuntimeException exc = assertThrows(RuntimeException.class,
                () -> resilient(fast, main, "split").invoke(MESSAGES));
        assertEquals("main down", exc.getMessage());
    }

    @Test
    void allMainThrowsLastScriptedErrorAfterExhaustion() {
        // 无回退（all_main）：重试耗尽后原样抛最后一次错误。
        ScriptedModels.ScriptedRuntime main = ScriptedModels.ScriptedRuntime.main(
                List.of(ScriptedModels.ScriptedResponse.failing(new RuntimeException("first")),
                        ScriptedModels.ScriptedResponse.failing(new RuntimeException("second")),
                        ScriptedModels.ScriptedResponse.failing(new RuntimeException("last"))));
        RuntimeException exc = assertThrows(RuntimeException.class,
                () -> resilient(null, main, "all_main").invoke(MESSAGES));
        assertEquals("last", exc.getMessage());
        assertEquals(3, main.chatModel.received().size());
    }

    @Test
    void interruptDuringBackoffAbortsRetryAndPreservesFlag() throws Exception {
        // 重试等待要响应线程中断（对应 asyncio.sleep 可被取消）：
        // baseDelay=10s 的退避中收到 interrupt → 中止并恢复中断标志。
        ScriptedModels.ScriptedRuntime fast = ScriptedModels.ScriptedRuntime.fast(
                List.of(rateLimited()));
        ScriptedModels.ScriptedRuntime main = ScriptedModels.ScriptedRuntime.main(
                List.of(ScriptedModels.ScriptedResponse.of("main answer")));
        ResilientLlm llm = new ResilientLlm(fast, main, 1, "split",
                ResilientLlm.DEFAULT_MAX_RETRIES, 10.0);
        AtomicReference<Throwable> caught = new AtomicReference<>();
        AtomicBoolean interruptPreserved = new AtomicBoolean(false);
        Thread worker = new Thread(() -> {
            try {
                llm.invoke(MESSAGES);
            } catch (Throwable e) {
                caught.set(e);
                interruptPreserved.set(Thread.currentThread().isInterrupted());
            }
        });
        worker.start();
        long deadline = System.currentTimeMillis() + 5000;
        while (fast.chatModel.received().isEmpty() && System.currentTimeMillis() < deadline) {
            Thread.sleep(10);
        }
        worker.interrupt();
        worker.join(5000);
        assertTrue(caught.get() instanceof RuntimeException, "退避中被打断应中止重试并抛出，实际: " + caught.get());
        assertTrue(interruptPreserved.get(), "中断标志必须被恢复");
        assertTrue(caught.get().getMessage().contains("中断"));
        assertEquals(1, fast.chatModel.received().size(), "打断后不再发起下一次尝试");
        assertEquals(0, main.chatModel.received().size());
    }

    @Test
    void usageSinkReceivesModelAndUsage() {
        // usage 钩子（对应 CostMeterHandler.on_llm_end 的唯一入口）：
        // invoke 消费脚本回复里的 TokenUsage 并投递 (model, usage)。
        List<String> models = new java.util.ArrayList<>();
        List<TokenUsage> usages = new java.util.ArrayList<>();
        BiConsumer<String, TokenUsage> sink = (model, usage) -> {
            models.add(model);
            usages.add(usage);
        };
        ScriptedModels.ScriptedRuntime fast = new ScriptedModels.ScriptedRuntime(
                List.of(ScriptedModels.ScriptedResponse.of("hey", new TokenUsage(11, 7))),
                "scripted-fast", true, sink);
        AiMessage reply = fast.invoke(MESSAGES);
        assertEquals("hey", reply.text());
        assertEquals(List.of("scripted-fast"), models);
        assertEquals(1, usages.size());
        assertEquals(11, usages.get(0).inputTokenCount());
        assertEquals(7, usages.get(0).outputTokenCount());
        assertEquals(18, usages.get(0).totalTokenCount());
    }

    @Test
    void usageSinkDeliversFallbackCallToo() {
        // 回退到主力的那次成功调用同样投递 usage（成本计量不漏主力账）；
        // 失败的快档调用抛在 sink 之前，不产生条目。
        List<String> models = new java.util.ArrayList<>();
        BiConsumer<String, TokenUsage> sink = (model, usage) -> models.add(model + ":" + usage);
        ScriptedModels.ScriptedRuntime fast = new ScriptedModels.ScriptedRuntime(
                List.of(rateLimited(), rateLimited(), rateLimited()),
                "scripted-fast", true, sink);
        ScriptedModels.ScriptedRuntime main = new ScriptedModels.ScriptedRuntime(
                List.of(ScriptedModels.ScriptedResponse.of("main answer", new TokenUsage(1, 2))),
                "scripted-main", false, sink);
        assertEquals("main answer", resilient(fast, main, "split").invoke(MESSAGES));
        assertEquals(List.of("scripted-main:" + new TokenUsage(1, 2)), models);
    }
}
