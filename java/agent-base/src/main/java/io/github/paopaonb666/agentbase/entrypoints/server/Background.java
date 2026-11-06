package io.github.paopaonb666.agentbase.entrypoints.server;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import io.github.paopaonb666.agentbase.extensions.Observability;

/**
 * 后台任务调度（对应 Python 侧 {@code entrypoints/server/background.py}）：
 * 异常必落日志；任务计数在飞+排队（测试断言用）。
 *
 * <p><b>有界池（#212/旧 #81）</b>：capture/摄取任务洪峰不再线程爆炸——
 * 固定 4 线程 + 100 深度队列，池满即拒绝（任务丢弃并 WARN，绝不上抛：
 * 后台任务是尽力而为的旁路，拒绝不能打断用户请求线程）。MDC 上下文
 * （request_id / 记忆作用域）随任务携带。
 */
public final class Background {

    private static final Logger LOG = LoggerFactory.getLogger(Background.class);

    /** 有界池：4 线程 + 100 深度队列（capture/摄取洪峰的上限）。 */
    static final int POOL_SIZE = 4;
    static final int QUEUE_CAPACITY = 100;

    /** 在飞 + 排队任务数（测试断言用）。 */
    private static final AtomicInteger PENDING = new AtomicInteger();

    private static final ThreadPoolExecutor EXECUTOR = new ThreadPoolExecutor(
            2, POOL_SIZE, 60L, TimeUnit.SECONDS,
            new ArrayBlockingQueue<>(QUEUE_CAPACITY),
            r -> {
                Thread t = new Thread(r, "agent-background");
                t.setDaemon(true);
                return t;
            },
            // AbortPolicy（默认）：spawn 捕获后带任务名落日志。
            new ThreadPoolExecutor.AbortPolicy());

    private Background() {}

    /**
     * 调度一个后台任务：异常落日志、完成自动减计。池满/队列满时任务
     * 被丢弃（WARN 带任务名）——后台任务是旁路，拒绝不向上传播。
     */
    public static void spawn(Runnable body, String name) {
        PENDING.incrementAndGet();
        try {
            EXECUTOR.execute(Observability.wrap(() -> {
                try {
                    body.run();
                } catch (Exception e) {
                    LOG.warn("background task '{}' failed", name, e);
                } finally {
                    PENDING.decrementAndGet();
                }
            }));
        } catch (RuntimeException rejected) {
            PENDING.decrementAndGet();
            LOG.warn("background: 任务 '{}' 被有界池拒绝（线程 {} + 队列 {} 已满）——已丢弃",
                    name, POOL_SIZE, QUEUE_CAPACITY, rejected);
        }
    }

    /** 仍在运行/排队中的后台任务数（测试断言用）。 */
    public static int pending() {
        return PENDING.get();
    }
}
