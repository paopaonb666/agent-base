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

    /** 单个在途 run 的句柄：steering 通道 + 属主。 */
    public static final class RunHandle {
        private final String userId;
        private final LinkedBlockingQueue<UserMessage> steering =
                new LinkedBlockingQueue<>(STEER_QUEUE_CAPACITY);

        RunHandle(String userId) {
            this.userId = userId;
        }

        /** 属主（端点掩蔽校验：非属主 404）。 */
        public String userId() {
            return userId;
        }

        /** 尽力投递一条插话；队列满返回 false（端点转 429）。 */
        public boolean steer(UserMessage message) {
            return steering.offer(message);
        }

        /** 待消费插话数（观测/测试）。 */
        public int pendingSteers() {
            return steering.size();
        }

        /** 取走全部待注入插话（步边界由图执行线程调用；测试断言面）。 */
        public List<UserMessage> drain() {
            if (steering.isEmpty()) {
                return List.of();
            }
            List<UserMessage> drained = new ArrayList<>(steering.size());
            steering.drainTo(drained);
            return List.copyOf(drained);
        }
    }

    private static final ConcurrentHashMap<String, RunHandle> ACTIVE = new ConcurrentHashMap<>();

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
}
