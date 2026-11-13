package io.github.paopaonb666.agentbase.extensions.costmeter;

/**
 * 一次模型调用的用量行（对应 Python 侧 {@code costmeter.UsageRecord}）。
 *
 * <p>由 {@link CostMeter#recordUsage} 在累计的同时压入待落库队列，账本
 * {@code flush()} 时批量写入 {@code cost_usage} 表。字段与建表列一一
 * 对应；{@code ts} 是 Unix 秒（浮点，对应 Python {@code time.time()}）。
 *
 * @param ts               记录时刻（Unix 秒）
 * @param requestId        请求 ID（MDC 里的 request_id，缺省空串）
 * @param model            模型名（价目表键）
 * @param profile          模型档位（main / fast，落库默认 'main'）
 * @param promptTokens     输入 token 数
 * @param completionTokens 输出 token 数
 * @param cacheReadTokens  命中缓存的输入 token 数（未上报按 0）
 * @param costCny          本次调用计价（元）
 * @param attempt          弹性包尝试位次（A-L79+#76；无弹性包装/旧行 NULL）
 * @param fallback         备用候选标记（"fallback#序号"；主档尝试 NULL）
 */
public record UsageRecord(
        double ts,
        String requestId,
        String model,
        String profile,
        long promptTokens,
        long completionTokens,
        long cacheReadTokens,
        double costCny,
        Integer attempt,
        String fallback) {
}
