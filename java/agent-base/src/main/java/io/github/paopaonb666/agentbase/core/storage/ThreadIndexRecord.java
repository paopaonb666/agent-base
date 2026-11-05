package io.github.paopaonb666.agentbase.core.storage;

/**
 * 一行会话索引（S1 线程作用域的数据模型基础，对应 Python 侧
 * {@code models.py} 的 {@code ThreadIndex} frozen dataclass）。
 *
 * <p>{@code threadId} 是带模块前缀的完整 id（{@code module:user_thread_id}）；
 * 属主（userId）在 invoke 时写入，线程端点的列表/属主校验据此执行。
 *
 * @param threadId  带模块前缀的完整线程 id
 * @param userId    属主（invoke 时写入）
 * @param module    所属模块名
 * @param title     会话标题
 * @param createdAt 创建时刻（epoch 秒，{@code time.time()} 语义）
 * @param updatedAt 最近更新时刻（epoch 秒）
 */
public record ThreadIndexRecord(
        String threadId,
        String userId,
        String module,
        String title,
        double createdAt,
        double updatedAt) {

    /** 便捷构造：时间戳缺省取当前时刻（对应 Python 的 default_factory=time.time）。 */
    public static ThreadIndexRecord create(String threadId, String userId, String module, String title) {
        double now = System.currentTimeMillis() / 1000.0;
        return new ThreadIndexRecord(threadId, userId, module, title, now, now);
    }
}
