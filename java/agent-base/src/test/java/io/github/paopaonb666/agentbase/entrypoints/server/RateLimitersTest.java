package io.github.paopaonb666.agentbase.entrypoints.server;

import org.junit.jupiter.api.Test;

import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** #78：per-user 令牌桶——突发上限、按时间回填、用户隔离、拒绝计数。 */
class RateLimitersTest {

    private static final long MINUTE_NANOS = 60L * 1_000_000_000L;

    @Test
    void burstUpToLimitThenRejectsAndRefills() {
        RateLimiters.PerUserRateLimiter limiter = new RateLimiters.PerUserRateLimiter(3);
        AtomicLong now = new AtomicLong(0);
        assertTrue(limiter.tryAcquire("u", now.get()));
        assertTrue(limiter.tryAcquire("u", now.get()));
        assertTrue(limiter.tryAcquire("u", now.get()));
        // 桶空：同刻拒绝。
        assertFalse(limiter.tryAcquire("u", now.get()));
        assertEquals(1, limiter.rejectionCount());
        // 半分钟后回填 1.5 个令牌 → 可再取 1 个。
        now.addAndGet(MINUTE_NANOS / 2);
        assertTrue(limiter.tryAcquire("u", now.get()));
        assertFalse(limiter.tryAcquire("u", now.get()));
        assertEquals(2, limiter.rejectionCount());
        // 整分钟过去 → 桶重新装满（上限截断，不累积超额）。
        now.addAndGet(MINUTE_NANOS);
        assertTrue(limiter.tryAcquire("u", now.get()));
        assertTrue(limiter.tryAcquire("u", now.get()));
        assertTrue(limiter.tryAcquire("u", now.get()));
        assertFalse(limiter.tryAcquire("u", now.get()));
    }

    @Test
    void bucketsAreIsolatedPerUser() {
        RateLimiters.PerUserRateLimiter limiter = new RateLimiters.PerUserRateLimiter(1);
        long now = 0;
        assertTrue(limiter.tryAcquire("alice", now));
        assertFalse(limiter.tryAcquire("alice", now));
        // bob 的桶独立满仓。
        assertTrue(limiter.tryAcquire("bob", now));
        assertFalse(limiter.tryAcquire("bob", now));
        assertEquals(2, limiter.rejectionCount());
    }

    @Test
    void zeroMeansUnlimited() {
        RateLimiters.PerUserRateLimiter limiter = new RateLimiters.PerUserRateLimiter(0);
        for (int i = 0; i < 100; i++) {
            assertTrue(limiter.tryAcquire("u", i));
        }
        assertEquals(0, limiter.rejectionCount());
    }
}
