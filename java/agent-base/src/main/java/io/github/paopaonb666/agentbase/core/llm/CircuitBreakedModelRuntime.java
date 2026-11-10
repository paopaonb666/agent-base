package io.github.paopaonb666.agentbase.core.llm;

import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.StreamingChatModel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;
import java.util.function.LongSupplier;

/**
 * provider 级熔断器（#72，自研最小实现——与 ResilientModelRuntime/
 * RateLimiters/RunGate 同风格，不引 resilience4j）：连续 N 次调用失败 →
 * open（快速失败，带剩余冷却秒数）→ 冷却到期 half-open 单发探测 → 成功
 * 闭路 / 失败重开。惰性时间检查（无定时线程，与滑窗重试预算同风格）。
 *
 * <p><b>与 #71 fallback 链的组合方式</b>：逐链元素各包一个熔断器（每个
 * (baseUrl,model) 独立状态机）——主力熔断 open 时抛 {@link CircuitOpenException}，
 * 链逻辑把它当普通失败处理、自动跳到下一个备用模型，无需改链代码。</p>
 *
 * <p><b>失败判定口径（v1）</b>：run 级最终失败（ResilientModelRuntime 重试
 * 与重试预算耗尽后的那次异常）都计入——provider 持续 429 时开路同样是
 * 期望行为（停止锤击）；更细的异常分类（超时/连接 vs 业务错）留待 #91
 * 错误码落地后按类型细分。流式半途失败与 invoke 失败同口径。</p>
 */
public final class CircuitBreakedModelRuntime implements ModelRuntime {

    private static final int CLOSED = 0;
    private static final int OPEN = 1;
    private static final int HALF_OPEN = 2;

    /** 熔断 open 期间的快速失败异常（retryAfterSeconds 供 Retry-After 素材）。 */
    public static final class CircuitOpenException extends IllegalStateException {
        public final long retryAfterSeconds;

        public CircuitOpenException(String message, long retryAfterSeconds) {
            super(message);
            this.retryAfterSeconds = retryAfterSeconds;
        }
    }

    private static final Logger LOG = LoggerFactory.getLogger(CircuitBreakedModelRuntime.class);

    private final ModelRuntime delegate;
    private final int failureThreshold;
    private final long cooldownNanos;
    private final LongSupplier nanoClock;
    private final AtomicInteger state = new AtomicInteger(CLOSED);
    private final AtomicInteger consecutiveFailures = new AtomicInteger();
    private final AtomicLong openedAt = new AtomicLong();
    private final AtomicLong openCount = new AtomicLong();

    public CircuitBreakedModelRuntime(ModelRuntime delegate, int failureThreshold,
                                      long cooldownSeconds) {
        this(delegate, failureThreshold, cooldownSeconds, System::nanoTime);
    }

    /** 全参构造（测试注入时钟用）。threshold < 1 = 禁用（直通）。 */
    public CircuitBreakedModelRuntime(ModelRuntime delegate, int failureThreshold,
                                      long cooldownSeconds, LongSupplier nanoClock) {
        this.delegate = delegate;
        this.failureThreshold = failureThreshold;
        this.cooldownNanos = cooldownSeconds * 1_000_000_000L;
        this.nanoClock = nanoClock;
    }

    /** 熔断器开启累计次数（可观测面）。 */
    public long openCount() {
        return openCount.get();
    }

    /** 当前状态描述（closed/open/half-open，/health 素材）。 */
    public String stateDescription() {
        int s = state.get();
        if (s == OPEN) {
            long remaining = cooldownRemainingNanos(nanoClock.getAsLong()) / 1_000_000_000L;
            return "open(" + Math.max(0, remaining) + "s)";
        }
        return s == HALF_OPEN ? "half-open" : "closed";
    }

    @Override
    public AiMessage invoke(List<ChatMessage> messages) {
        guard();
        try {
            AiMessage message = delegate.invoke(messages);
            onSuccess();
            return message;
        } catch (RuntimeException exc) {
            onFailure();
            throw exc;
        }
    }

    @Override
    public AiMessage stream(List<ChatMessage> messages, List<ToolSpecification> tools,
                            Consumer<String> onDelta) {
        guard();
        try {
            AiMessage message = delegate.stream(messages, tools, onDelta);
            onSuccess();
            return message;
        } catch (RuntimeException exc) {
            onFailure();
            throw exc;
        }
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
    public boolean toolSafeStreaming() {
        // 透明层（review R2-F5 同口径）：审计面穿透。
        return delegate.toolSafeStreaming();
    }

    @Override
    public double tokenFactor() {
        return delegate.tokenFactor();
    }

    @Override
    public int contextWindowTokens() {
        return delegate.contextWindowTokens();
    }

    /** open/half-open 拒绝调用（half-open 已有探测在途）；closed 放行。 */
    private void guard() {
        if (failureThreshold < 1) {
            return; // 禁用
        }
        long now = nanoClock.getAsLong();
        for (;;) {
            int s = state.get();
            if (s == CLOSED) {
                return;
            }
            if (s == OPEN) {
                if (cooldownRemainingNanos(now) > 0) {
                    long remainingSeconds = cooldownRemainingNanos(now) / 1_000_000_000L;
                    throw new CircuitOpenException(
                            "llm 熔断开启：provider 连续失败 %d 次，%ds 后半开探测"
                                    .formatted(failureThreshold, Math.max(0, remainingSeconds)),
                            Math.max(1, remainingSeconds));
                }
                if (state.compareAndSet(OPEN, HALF_OPEN)) {
                    LOG.warn("llm: 熔断冷却到期，放行单发探测（half-open）");
                    return; // 本调用即探测
                }
                continue; // 另一线程已抢到探测权
            }
            // HALF_OPEN：探测已在途，其余调用快速失败。
            throw new CircuitOpenException("llm 熔断半开探测中", 1);
        }
    }

    private void onSuccess() {
        if (failureThreshold < 1) {
            return;
        }
        consecutiveFailures.set(0);
        if (state.get() != CLOSED && state.compareAndSet(HALF_OPEN, CLOSED)) {
            LOG.info("llm: 熔断探测成功，恢复 closed");
        }
    }

    private void onFailure() {
        if (failureThreshold < 1) {
            return;
        }
        int failures = consecutiveFailures.incrementAndGet();
        if (state.get() == HALF_OPEN) {
            openCircuit();
            return;
        }
        if (failures >= failureThreshold && state.compareAndSet(CLOSED, OPEN)) {
            openCircuit();
        }
    }

    private void openCircuit() {
        openedAt.set(nanoClock.getAsLong());
        openCount.incrementAndGet();
        LOG.warn("llm: 熔断开启（连续失败 {} 次达到阈值 {}），{}s 后半开探测",
                consecutiveFailures.get(), failureThreshold, cooldownNanos / 1_000_000_000L);
    }

    private long cooldownRemainingNanos(long now) {
        return cooldownNanos - (now - openedAt.get());
    }
}
