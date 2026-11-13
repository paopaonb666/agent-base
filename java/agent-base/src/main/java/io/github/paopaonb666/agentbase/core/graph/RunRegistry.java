package io.github.paopaonb666.agentbase.core.graph;

import dev.langchain4j.data.message.UserMessage;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.LinkedBlockingQueue;

/**
 * 在途 run 生命周期注册表（#7 steering / #8 resume 的共享地基）：线程 →
 * 活跃运行句柄。句柄承载 steering 通道（用户中途插话的注入队列）与属主
 * ——注册/注销都发生在 per-thread 锁内（#197 同线程互斥保证同一线程至多
 * 一个在途 run），lookup 天然无歧义。
 *
 * <p>steering 语义（v1，有意边界）：{@code steer()} 是尽力投递——run 在
 * 最后一个步边界之后结束前到达的插话不被消费、随 run 终止丢弃（端点已
 * 返回 queued=true）。at-least-once 不承诺；#8 resume 落地后生命周期记录
 * 会把未消费插话入账。
 */
public final class RunRegistry {

    /** steering 队列上限：超过即满（端点 429）—— runaway 插话防线。 */
    public static final int STEER_QUEUE_CAPACITY = 16;

    /** 单个在途 run 的句柄：run id + 属主 + steering 通道。 */
    public static final class RunHandle {
        private final String runId;
        private final String userId;
        private final LinkedBlockingQueue<SteerItem> steering =
                new LinkedBlockingQueue<>(STEER_QUEUE_CAPACITY);

        RunHandle(String userId) {
            this.runId = "run-"
                    + io.github.paopaonb666.agentbase.extensions.Observability.newRequestId();
            this.userId = userId;
        }

        /**
         * 尽力投递一条插话；队列满返回 false（端点 429）。kind ∈
         * {@code user|system}（#14 system-reminder 形态）。
         */
        public boolean steer(String kind, String content) {
            return steering.offer(new SteerItem(kind, content));
        }

        /** 属主（端点掩蔽校验：非属主 404）。 */
        public String userId() {
            return userId;
        }

        /** run 标识（生命周期记录/对账锚点）。 */
        public String runId() {
            return runId;
        }

        /** 待消费插话数（观测/测试）。 */
        public int pendingSteers() {
            return steering.size();
        }

        /** 取走全部待注入插话（步边界由图执行线程调用；测试断言面）。 */
        public List<SteerItem> drain() {
            if (steering.isEmpty()) {
                return List.of();
            }
            List<SteerItem> drained = new ArrayList<>(steering.size());
            steering.drainTo(drained);
            return List.copyOf(drained);
        }
    }

    /**
     * 一条待注入插话（#7/#14）：kind=user → UserMessage 原样注入；
     * kind=system → system-reminder 形态（前缀见 CompiledGraph）。
     */
    public record SteerItem(String kind, String content) {
        public SteerItem {
            if (!"user".equals(kind) && !"system".equals(kind)) {
                throw new IllegalArgumentException("steer kind 必须是 user|system: " + kind);
            }
        }
    }

    private static final ConcurrentHashMap<String, RunHandle> ACTIVE = new ConcurrentHashMap<>();

    /**
     * 末次终态记录（#8 resume）：threadId → 最近一次 run 的生命周期快照。
     * 与 THREAD_LOCKS 同款「每线程一条、常驻不回收」取舍（进程级线程数即
     * 上界，javadoc 登记）；resume 端点据此判定「有无可恢复的运行」与属主。
     */
    public record LastRun(String runId, String userId, String termination, long endedAtMillis) {
    }

    private static final ConcurrentHashMap<String, LastRun> LAST = new ConcurrentHashMap<>();

    private RunRegistry() {
    }

    /**
     * 注册在途 run（图执行线程在 per-thread 锁内调用——putIfAbsent 冲突即
     * 编程错误，防御性抛出）。
     */
    public static RunHandle register(String threadId, String userId) {
        RunHandle handle = new RunHandle(userId);
        if (ACTIVE.putIfAbsent(threadId, handle) != null) {
            throw new IllegalStateException(
                    "run already registered for thread " + threadId);
        }
        return handle;
    }

    /** 注销（终态 finally；identity 语义防误删后继 run 的句柄）。 */
    public static void unregister(String threadId, RunHandle handle) {
        ACTIVE.remove(threadId, handle);
    }

    /** 查在途 run 句柄；线程空闲返回 null。 */
    public static RunHandle lookup(String threadId) {
        return ACTIVE.get(threadId);
    }

    /**
     * 记录末次终态（#8：图执行线程在注销句柄后调用——termination 用
     * {@link Events.RunTermination} 的 wire 小写形态）。
     */
    public static void recordLastRun(String threadId, String userId, String runId,
                                     String termination) {
        LAST.put(threadId, new LastRun(runId, userId, termination, System.currentTimeMillis()));
    }

    /** 查末次终态记录；该线程从未运行过返回 null。 */
    public static LastRun lastRun(String threadId) {
        return LAST.get(threadId);
    }
}
