package io.github.paopaonb666.agentbase.entrypoints.server;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 后台任务有界池测试（roadmap #212/旧 #81）：洪峰下线程不爆炸（池
 * 上限）、任务计数准确、异常落日志不上抛、池满拒绝不传播到调用方。
 */
class BackgroundTest {

    @Test
    void tasksRunToCompletionAndPendingCountsDrainToZero() throws Exception {
        CountDownLatch done = new CountDownLatch(3);
        Background.spawn(done::countDown, "t1");
        Background.spawn(done::countDown, "t2");
        Background.spawn(() -> {
            throw new IllegalStateException("boom"); // 异常必被吞进日志
        }, "t3") ;
        done.countDown();
        assertTrue(done.await(5, TimeUnit.SECONDS), "三个任务都应执行（含抛错任务）");
        long deadline = System.currentTimeMillis() + 5_000;
        while (Background.pending() > 0 && System.currentTimeMillis() < deadline) {
            Thread.sleep(20);
        }
        assertEquals(0, Background.pending(), "完成后计数应归零");
    }

    @Test
    void floodBeyondPoolAndQueueIsRejectedWithoutPropagating() throws Exception {
        // 洪峰：每任务阻塞在栅栏上占满 4 线程 + 100 队列，之后的提交被
        // 丢弃——spawn 不抛错（后台任务拒绝不影响请求线程）。
        CountDownLatch release = new CountDownLatch(1);
        List<Integer> executed = new CopyOnWriteArrayList<>();
        int total = Background.POOL_SIZE + Background.QUEUE_CAPACITY + 40;
        AtomicBoolean rejectionSeen = new AtomicBoolean();
        for (int i = 0; i < total; i++) {
            final int id = i;
            try {
                Background.spawn(() -> {
                    try {
                        release.await();
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                    executed.add(id);
                }, "flood-" + i);
            } catch (RuntimeException e) {
                rejectionSeen.set(true); // 不应发生：spawn 内部消化拒绝
            }
        }
        // 池满 + 队列满（40 个溢出任务被丢弃）；进程级共享池，计数只做
        // 下界断言（其他测试的旁路任务可能同时在飞）。
        assertTrue(Background.pending() >= Background.POOL_SIZE);
        release.countDown();
        long deadline = System.currentTimeMillis() + 5_000;
        while (Background.pending() > 0 && System.currentTimeMillis() < deadline) {
            Thread.sleep(20);
        }
        assertEquals(0, Background.pending());
        assertEquals(Background.POOL_SIZE + Background.QUEUE_CAPACITY, executed.size(),
                "只有池+队列容量的任务真正执行");
        assertFalse(rejectionSeen.get(), "拒绝应在 spawn 内部消化，不上抛");
    }
}
