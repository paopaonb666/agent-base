package io.github.paopaonb666.agentbase.core.llm;

import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.output.TokenUsage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;

/**
 * 主对话路径的弹性包装（#1/#10/#73/#75）：并发信号量 + 瞬时错误指数
 * 退避重试 + 滑窗重试预算 + attempt 级可观测。图执行节点经
 * {@code ModuleContext.llm()} 拿到的就是包装后的运行时——重试/限流对
 * 节点透明（对应 Python 组合根把 ResilientLLM 包住主力模型的等价物）。
 *
 * <p>与 {@link ResilientLlm}（快档管线包装）的分工：那边面向后台管线的
 * 单消息直答（ChatInvokerPort），这边面向图执行的双通道（stream/invoke）。
 * 主力没有更低档的回退目标（快档不是主力的降级），重试耗尽后原样上抛，
 * 由顶层归一为用户可见的 error 事件。
 *
 * <ul>
 *   <li><b>并发限流（#73）</b>：{@code LLM_MAIN_CONCURRENCY} 全局信号量
 *       包住每次真实调用——突发流量下 LLM 并发有界，provider 限流不再
 *       连锁放大；快档独立信号量（ResilientLlm）不受影响。</li>
 *   <li><b>重试（#1）</b>：指数退避（base 0.5s × 2^attempt，LLM_MAIN_RETRIES
 *       次重试）。<b>流式重试边界</b>：只在「尚未发出任何 delta」时重试
 *       ——delta 一旦下发客户端，重试会造成重复输出；已开始的流中断按
 *       失败上抛。重试等待在信号量之外（不占并发额度）。</li>
 *   <li><b>重试预算（#10）</b>：滚动窗口内全局重试次数封顶（默认
 *       60s/20 次）——重试 × 多步图的放大效应在窗口边界被截断，超预算
 *       抛 {@link RetryBudgetExhausted}（终止性异常，当轮以错误收口）。
 *       成本维度由 CostGovernor 独立把守。</li>
 *   <li><b>usage 全路径（#75）</b>：每次真实 API 调用由内层运行时的
 *       usageSink 记账（重试每次各记一条，结构上无漏记）；包装层按
 *       attempt 序号写日志供账本对账。</li>
 * </ul>
 */
public final class ResilientModelRuntime implements ModelRuntime {

    /** Python ResilientLLM 的退避节奏在这个档位减半：主循环延迟敏感。 */
    public static final double DEFAULT_BASE_DELAY_SECONDS = 0.5;
    /** 重试预算默认窗口（秒）。 */
    public static final double DEFAULT_RETRY_BUDGET_WINDOW_SECONDS = 60.0;
    /** 重试预算默认窗口内重试次数上限。 */
    public static final int DEFAULT_RETRY_BUDGET_PER_WINDOW = 20;

    /** 重试预算耗尽：终止性异常（当轮应立即失败，不再放大）。 */
    public static final class RetryBudgetExhausted extends IllegalStateException {
        public RetryBudgetExhausted(String message) {
            super(message);
        }
    }

    private static final Logger LOG = LoggerFactory.getLogger(ResilientModelRuntime.class);

    private final ModelRuntime delegate;
    private final int maxRetries;
    private final long baseDelayMillis;
    private final Semaphore semaphore;
    private final long budgetWindowNanos;
    private final int budgetPerWindow;
    private final AtomicLong budgetWindowStart = new AtomicLong(System.nanoTime());
    private final AtomicInteger budgetSpent = new AtomicInteger();

    public ResilientModelRuntime(ModelRuntime delegate, int maxRetries, int maxConcurrency) {
        this(delegate, maxRetries, maxConcurrency, DEFAULT_BASE_DELAY_SECONDS,
                DEFAULT_RETRY_BUDGET_WINDOW_SECONDS, DEFAULT_RETRY_BUDGET_PER_WINDOW);
    }

    /** 全参构造（测试注入零退避/小预算用）。 */
    public ResilientModelRuntime(ModelRuntime delegate, int maxRetries, int maxConcurrency,
                                 double baseDelaySeconds, double budgetWindowSeconds,
                                 int budgetPerWindow) {
        this.delegate = delegate;
        if (maxRetries < 0) {
            throw new IllegalArgumentException("max_retries must be >= 0, got " + maxRetries);
        }
        if (maxConcurrency < 1) {
            throw new IllegalArgumentException("max_concurrency must be >= 1, got " + maxConcurrency);
        }
        this.maxRetries = maxRetries;
        this.baseDelayMillis = (long) (baseDelaySeconds * 1000);
        this.semaphore = new Semaphore(maxConcurrency);
        this.budgetWindowNanos = (long) (budgetWindowSeconds * 1_000_000_000L);
        this.budgetPerWindow = budgetPerWindow;
    }

    @Override
    public ChatModel chat() {
        return delegate.chat();
    }

    @Override
    public StreamingChatModel streaming() {
        return delegate.streaming();
    }

    @Override
    public AiMessage stream(List<ChatMessage> messages, List<ToolSpecification> tools,
                            Consumer<String> onDelta) {
        // 流式重试边界：包装 onDelta，一旦发出第一个 delta 就不可重试——
        // 重试会把已下发的内容重复输出给客户端。
        java.util.concurrent.atomic.AtomicBoolean deltaEmitted = new java.util.concurrent.atomic.AtomicBoolean();
        Consumer<String> guarded = delta -> {
            deltaEmitted.set(true);
            onDelta.accept(delta);
        };
        RuntimeException lastError = null;
        for (int attempt = 0; attempt <= maxRetries; attempt++) {
            deltaEmitted.set(false);
            try {
                AiMessage message = callWithLimit(() -> delegate.stream(messages, tools, guarded),
                        "stream", attempt);
                if (attempt > 0) {
                    LOG.info("llm: stream 第 {} 次尝试成功（重试恢复）", attempt + 1);
                }
                return message;
            } catch (RuntimeException exc) {
                lastError = exc;
                if (deltaEmitted.get() || attempt == maxRetries || !acquireRetryBudget()) {
                    throw exc;
                }
                sleepBeforeRetry(attempt, exc);
            }
        }
        throw lastError; // 不可达（循环内必 return/throw）
    }

    @Override
    public AiMessage invoke(List<ChatMessage> messages) {
        RuntimeException lastError = null;
        for (int attempt = 0; attempt <= maxRetries; attempt++) {
            try {
                AiMessage message = callWithLimit(() -> delegate.invoke(messages), "invoke", attempt);
                if (attempt > 0) {
                    LOG.info("llm: invoke 第 {} 次尝试成功（重试恢复）", attempt + 1);
                }
                return message;
            } catch (RuntimeException exc) {
                lastError = exc;
                if (attempt == maxRetries || !acquireRetryBudget()) {
                    throw exc;
                }
                sleepBeforeRetry(attempt, exc);
            }
        }
        throw lastError; // 不可达
    }

    /** 信号量内执行单次真实调用（并发上限 + attempt 日志）。 */
    private <T> T callWithLimit(java.util.function.Supplier<T> body, String kind, int attempt) {
        try {
            semaphore.acquire();
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("llm 并发等待被中断", interrupted);
        }
        try {
            if (attempt > 0) {
                LOG.warn("llm: {} 第 {}/{} 次重试（真实调用，usage 由内层 sink 记账）",
                        kind, attempt + 1, maxRetries + 1);
            }
            return body.get();
        } finally {
            semaphore.release();
        }
    }

    /**
     * 滑窗重试预算（#10）：窗口过期则重置；预算耗尽返回 false——调用方
     * 把最后一次错误原样上抛（终止当轮）而不是继续重试放大。
     */
    private boolean acquireRetryBudget() {
        long now = System.nanoTime();
        long windowStart = budgetWindowStart.get();
        if (now - windowStart > budgetWindowNanos
                && budgetWindowStart.compareAndSet(windowStart, now)) {
            budgetSpent.set(0);
        }
        for (;;) {
            int spent = budgetSpent.get();
            if (spent >= budgetPerWindow) {
                LOG.warn("llm: 重试预算耗尽（{}s 窗口内已重试 {} 次）——重试风暴防护触发",
                        budgetWindowNanos / 1_000_000_000L, spent);
                return false;
            }
            if (budgetSpent.compareAndSet(spent, spent + 1)) {
                return true;
            }
        }
    }

    /** 退避等待（信号量之外）；被中断时恢复标志并中止重试。 */
    private void sleepBeforeRetry(int attempt, RuntimeException failure) {
        long delayMillis = baseDelayMillis * (1L << attempt);
        LOG.warn("llm: 调用失败（第 {}/{} 次尝试），{}s 后重试: {}",
                attempt + 1, maxRetries + 1, delayMillis / 1000.0, failure.toString());
        try {
            Thread.sleep(delayMillis);
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            RuntimeException abort = new IllegalStateException("llm 重试等待被中断", interrupted);
            abort.addSuppressed(failure);
            throw abort;
        }
    }
}
