package io.github.paopaonb666.agentbase.extensions.costmeter;

import java.util.List;
import java.util.Map;

/**
 * 账本协议：pending 行缓冲 + 批量落库 + 按时间点累计 + 按日聚合（重启
 * 恢复用）。对应 Python 侧 {@code costmeter.CostLedger} Protocol 与其
 * 三个实现（Memory / Sqlite / Mysql）。
 *
 * <p>落库节奏与 Python 一致：server lifespan 的周期 flush 任务从
 * {@link CostMeter#takePending()} 取走明细，经 {@link #append} 入账本
 * pending 队列，{@link #flush()} 批量写入 {@code cost_usage} 表；关闭前
 * 再 flush 一次防丢尾。后端跟随 checkpointer（{@code CostLedgers.build}）。
 */
public interface CostLedger extends AutoCloseable {

    /** 入 pending 队列（不直接落库，等 {@link #flush()} 批量写）。 */
    void append(UsageRecord record);

    /** 把 pending 队列批量落库（对应 Python {@code insert(rows)} 的批量语义）；空队列直接返回。 */
    void flush();

    /** 某时间点以来的累计花费（元）。 */
    double totalsSince(double ts);

    /** day-iso → 当日累计（元）；进程重启恢复的数据源。 */
    Map<String, Double> totalsByDay();

    /** sqlite 实现关闭连接；memory 无资源；mysql 短连接无需关闭。 */
    @Override
    default void close() {}
}
