package io.github.paopaonb666.agentbase.entrypoints.server;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

/**
 * per-user 请求限流（#78 多用户硬前提）：令牌桶按 user 维度限 invoke
 * 频率（RATE_LIMIT_PER_MINUTE，0 = 不限——本机开发语义）。桶满即拒，
 * 429 + Retry-After 由调用方（AgentsRoutes）渲染；无第三方依赖
 * （Guava/Bucket4j 的最小自研替代，CAS 无锁）。
 *
 * <p>用户表有界防护：容量上限 {@value #MAX_TRACKED_USERS}，插入超限时
 * 先清一轮空闲超过 {@value #IDLE_EVICT_MINUTES} 分钟的条目（限流状态
 * 是可丢弃的——清掉后用户重新拿满桶，等效于一次自然的窗口重置）。</p>
 */
public final class RateLimiters {

    private static final int MAX_TRACKED_USERS = 10_000;
    private static final long IDLE_EVICT_MINUTES = 10L;

    /** 单用户令牌桶。 */
    public static final class PerUserRateLimiter {

        private final int perMinute;
        private final ConcurrentHashMap<String, AtomicReference<BucketState>> buckets =
                new ConcurrentHashMap<>();
        private final AtomicLong rejections = new AtomicLong();

        /** 桶状态（不可变，CAS 替换）：tokens 为当前余量，lastNanos 为上次结算点。 */
        private record BucketState(double tokens, long lastNanos) {}

        public PerUserRateLimiter(int perMinute) {
            if (perMinute < 0) {
                throw new IllegalArgumentException("perMinute must be >= 0, got " + perMinute);
            }
            this.perMinute = perMinute;
        }

        /** 配置的每分钟限额（0 = 不限）。 */
        public int perMinute() {
            return perMinute;
        }

        /** 累计拒绝次数（限流可观测面）。 */
        public long rejectionCount() {
            return rejections.get();
        }

        /**
         * 尝试取一个令牌。{@code nowNanos} 可注入（测试确定性；生产用
         * System.nanoTime）。不限配置恒真。
         */
        public boolean tryAcquire(String userId, long nowNanos) {
            if (perMinute <= 0) {
                return true;
            }
            evictIfOverflow(nowNanos);
            AtomicReference<BucketState> ref =
                    buckets.computeIfAbsent(userId, k -> new AtomicReference<>(new BucketState(perMinute, nowNanos)));
            while (true) {
                BucketState cur = ref.get();
                double refill = (nowNanos - cur.lastNanos()) / 60_000_000_000.0 * perMinute;
                double tokens = Math.min(perMinute, cur.tokens() + refill);
                if (tokens >= 1.0) {
                    if (ref.compareAndSet(cur, new BucketState(tokens - 1.0, nowNanos))) {
                        return true;
                    }
                } else {
                    if (ref.compareAndSet(cur, new BucketState(tokens, nowNanos))) {
                        rejections.incrementAndGet();
                        return false;
                    }
                }
            }
        }

        /** 容量防护：超限时清一轮空闲条目（限流状态可丢弃语义）。 */
        private void evictIfOverflow(long nowNanos) {
            if (buckets.size() < MAX_TRACKED_USERS) {
                return;
            }
            long idleCutoff = nowNanos - IDLE_EVICT_MINUTES * 60L * 1_000_000_000L;
            buckets.entrySet().removeIf(e -> e.getValue().get().lastNanos() < idleCutoff);
        }
    }

    private RateLimiters() {
    }
}
