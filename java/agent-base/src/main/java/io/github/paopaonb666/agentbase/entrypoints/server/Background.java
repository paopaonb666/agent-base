package io.github.paopaonb666.agentbase.entrypoints.server;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import io.github.paopaonb666.agentbase.extensions.Observability;

/**
 * 后台任务调度（对应 Python 侧 {@code entrypoints/server/background.py}）：
 * 进程级强引用注册表防任务被 GC + 异常必落日志。
 */
public final class Background {

    private static final Logger LOG = LoggerFactory.getLogger(Background.class);

    /** 进程级强引用集合（对应 _TASKS：防 GC 官方约定）。 */
    private static final Set<Thread> TASKS = ConcurrentHashMap.newKeySet();

    private static final ExecutorService EXECUTOR = Executors.newCachedThreadPool(r -> {
        Thread t = new Thread(r, "agent-background");
        t.setDaemon(true);
        return t;
    });

    private Background() {}

    /**
     * 调度一个后台任务：持强引用、异常落日志、完成自动清理。
     * MDC 上下文（request_id / 记忆作用域）随任务携带。
     */
    public static void spawn(Runnable body, String name) {
        Thread thread = new Thread(Observability.wrap(() -> {
            try {
                body.run();
            } catch (Exception e) {
                LOG.warn("background task '{}' failed", name, e);
            } finally {
                TASKS.remove(Thread.currentThread());
            }
        }), "agent-background-" + name);
        TASKS.add(thread);
        thread.setDaemon(true);
        thread.start();
    }

    /** 仍在运行的后台任务数（测试断言用）。 */
    public static int pending() {
        return TASKS.size();
    }
}
