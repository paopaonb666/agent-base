package io.github.paopaonb666.agentbase.entrypoints.server;

import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * 全局 run 并发闸门（#79）：同时在途的 SSE 对话轮数上限（
 * AGENT_RUN_CONCURRENCY，0 = 不限）。饱和快速失败（调用方渲染
 * 429 + Retry-After）——排队只把超时推迟到客户端断连之后。
 *
 * <p>进入/退出必须配对：AgentsRoutes 在限流检查后 {@link #tryEnter()}，
 * 退出经 SseStreams 的终态钩子（produce 的 finally，正常/失败/取消全
 * 路径恰好一次）。</p>
 */
public final class RunGate {

    /**
     * 进程级活跃闸门（A-#214 /health sse 水位的读取缝）：AgentsRoutes 构造
     * 时登记——单实例假设与 THREAD_LOCKS/RunRegistry 进程级静态同口径
     * （多 AgentsRoutes 实例的测试直构以最后构造者为准，只影响观测面）。
     */
    private static volatile RunGate active;

    private final Semaphore permits;
    private final int configured;
    private final AtomicInteger inFlight = new AtomicInteger();
    private final AtomicLong rejected = new AtomicLong();

    public RunGate(int maxConcurrent) {
        if (maxConcurrent < 0) {
            throw new IllegalArgumentException("maxConcurrent must be >= 0, got " + maxConcurrent);
        }
        this.permits = maxConcurrent <= 0 ? null : new Semaphore(maxConcurrent);
        this.configured = maxConcurrent;
        active = this;
    }

    /** 活跃闸门（观测面读取缝）；尚无 AgentsRoutes 装配时返回 null。 */
    public static RunGate active() {
        return active;
    }

    /** 配置的并发上限（0 = 不限）——429 文案引用，不随负载漂移（review R3）。 */
    public int maxConcurrent() {
        return configured;
    }

    /** 在途 run 数（可观测面）。 */
    public int inFlight() {
        return inFlight.get();
    }

    /** 累计饱和拒绝次数（限流/并发闸门的可观测面）。 */
    public long rejectionCount() {
        return rejected.get();
    }

    /** 尝试进入一次 run；饱和返回 false。0 = 不限恒真。 */
    public boolean tryEnter() {
        if (permits == null) {
            return true;
        }
        if (permits.tryAcquire()) {
            inFlight.incrementAndGet();
            return true;
        }
        rejected.incrementAndGet();
        return false;
    }

    /** 退出一次 run（与 tryEnter 配对；不限时 no-op）。 */
    public void exit() {
        if (permits != null) {
            inFlight.decrementAndGet();
            permits.release();
        }
    }
}
