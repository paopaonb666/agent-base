package io.github.paopaonb666.agentbase.extensions.costmeter;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Instant;
import java.time.LocalDate;
import java.time.ZoneId;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import io.github.paopaonb666.agentbase.extensions.Metrics;
import io.github.paopaonb666.agentbase.extensions.Observability;

/**
 * 进程内用量累计器：按日/月聚合 + 价目表计价 + 待落库行队列。
 *
 * <p>对应 Python 侧 {@code costmeter.CostMeter}。设计说明（移植自
 * costmeter.py 模块注释）：
 * <ul>
 *   <li><b>采集 → 累计 → 账本</b> 三件套里的"累计"环节：模型调用方在
 *       响应结束后把 usage 交给 {@link #recordUsage}，这里按价目表
 *       （元/百万 tokens，键 {@code input_miss} / {@code input_hit} /
 *       {@code output}，缺失键按 0）计价并累计到当日/当月桶。</li>
 *   <li><b>同步更新</b>：累计值就在内存里，预算门（{@link CostGovernor}）
 *       读它不需要任何异步等待。</li>
 *   <li><b>未知模型计 0 元并告警一次</b>（每个模型只提醒一次，避免刷屏）；
 *       补齐价目后账单才完整。</li>
 *   <li><b>重启恢复</b>：进程重启后由装配方从账本回放
 *       {@link #restoreDay}；明细落库由调用方（server lifespan 的周期
 *       flush 任务）通过 {@link #takePending} 驱动，本类不直接碰存储。</li>
 * </ul>
 *
 * <p>已知局限：单进程口径（与 /metrics 同款）；价目手工维护。
 * 全部方法线程安全（Python 靠事件循环串行，这里显式加锁）。
 */
public final class CostMeter {

    private static final Logger LOG = LoggerFactory.getLogger(CostMeter.class);

    /** 价目表：model → {input_miss, input_hit, output}（元/百万 tokens）。 */
    private Map<String, Map<String, Double>> prices = Map.of();

    /** 待落库行队列（账本 flush 时由 {@link #takePending} 取走）。 */
    private final Deque<UsageRecord> pending = new ArrayDeque<>();

    /** day-iso → 当日累计（元）。 */
    private final Map<String, Double> dayTotals = new HashMap<>();
    /** month-iso → 当月累计（元）。 */
    private final Map<String, Double> monthTotals = new HashMap<>();
    /** 已告警过的未知模型（每模型只告警一次）。 */
    private final Set<String> warnedModels = new HashSet<>();

    private final Object lock = new Object();

    /**
     * 服务装配时注入价目表（覆盖单例的空价目）；重复配置会重置未知模型
     * 告警去重（补齐价目后，之前按 0 计的模型要重新提醒到账）。
     */
    public void configure(Map<String, Map<String, Double>> prices) {
        synchronized (lock) {
            this.prices = prices == null ? Map.of() : Map.copyOf(prices);
            this.warnedModels.clear();
        }
    }

    /** 记录一次模型调用（main 档、无缓存上报）。 */
    public void recordUsage(String model, long inputTokens, long outputTokens) {
        recordUsage(model, "main", inputTokens, outputTokens, 0);
    }

    /** 记录一次模型调用（指定档位与缓存命中数）。 */
    public void recordUsage(String model, String profile, long inputTokens, long outputTokens,
            long cacheReadTokens) {
        observe(model, profile, inputTokens, outputTokens, cacheReadTokens);
    }

    /**
     * 记录一次模型调用（对应 Python 侧 {@code observe}）。
     *
     * <p>未知模型告警一次、按 0 元计量；同步更新 Prometheus 用量指标。
     * 计价公式：{@code miss/1e6*input_miss + hit/1e6*input_hit +
     * completion/1e6*output}，其中 {@code hit = min(cache_read, prompt)}。
     */
    public void observe(String model, String profile, long prompt, long completion, long cacheRead) {
        double cost;
        synchronized (lock) {
            if (!prices.containsKey(model) && !warnedModels.contains(model)) {
                warnedModels.add(model);
                LOG.warn(
                        "cost: 模型 '{}' 不在 COST_PRICES_JSON 价目表中，按 0 元计量（补齐价目后账单才完整）",
                        model);
            }
            cost = priceOf(prices, model, prompt, completion, cacheRead);
            double now = System.currentTimeMillis() / 1000.0;
            // ts 是 Unix 秒；日键由同一时刻取本地日期（注意秒→毫秒换算）。
            String day = LocalDate.ofInstant(Instant.ofEpochMilli((long) Math.floor(now * 1000.0)),
                    ZoneId.systemDefault()).toString();
            String month = day.substring(0, 7);
            dayTotals.merge(day, cost, Double::sum);
            monthTotals.merge(month, cost, Double::sum);
            pending.add(new UsageRecord(
                    now, Observability.requestId(), model, profile,
                    prompt, completion, cacheRead, cost));
        }
        Metrics.cost().observeTokens(model, profile, "input", prompt);
        Metrics.cost().observeTokens(model, profile, "output", completion);
        Metrics.cost().observeCost(cost);
    }

    /**
     * 按价目表计价（元/百万 tokens）；未知模型 0 元（调用方告警）。
     * 表内缺失键按 0；缓存命中数不会超过 prompt（超出部分按 miss 计）。
     */
    static double priceOf(
            Map<String, Map<String, Double>> prices, String model,
            long prompt, long completion, long cacheRead) {
        Map<String, Double> table = prices.get(model);
        if (table == null || table.isEmpty()) {
            return 0.0;
        }
        long hit = Math.min(cacheRead, prompt);
        long miss = prompt - hit;
        return miss / 1e6 * table.getOrDefault("input_miss", 0.0)
                + hit / 1e6 * table.getOrDefault("input_hit", 0.0)
                + completion / 1e6 * table.getOrDefault("output", 0.0);
    }

    /** 取走待落库行（账本 flush 任务调用；之后队列为空）。 */
    public List<UsageRecord> takePending() {
        synchronized (lock) {
            List<UsageRecord> rows = new ArrayList<>(pending);
            pending.clear();
            return rows;
        }
    }

    /** 当日累计（元）。 */
    public double dayCost() {
        return dayCost(today());
    }

    /** 指定日（day-iso，如 {@code 2026-11-04}）的累计（元）。 */
    public double dayCost(String day) {
        synchronized (lock) {
            return dayTotals.getOrDefault(day, 0.0);
        }
    }

    /** 当月累计（元）。 */
    public double monthCost() {
        return monthCost(today().substring(0, 7));
    }

    /** 指定月（month-iso，如 {@code 2026-11}）的累计（元）。 */
    public double monthCost(String month) {
        synchronized (lock) {
            return monthTotals.getOrDefault(month, 0.0);
        }
    }

    /**
     * 进程重启后从账本恢复累计（对应 Python 侧 {@code restore_day}，
     * 叠加语义：当日 + 当月各加一次）。{@code inputTokens} /
     * {@code outputTokens} 供调用方按明细行形状传参，但 per-周期 token
     * 聚合与 Python 一致不回放——token 指标是进程本地口径，重启后从零计。
     */
    public void restoreDay(String day, long inputTokens, long outputTokens, double cost) {
        synchronized (lock) {
            dayTotals.merge(day, cost, Double::sum);
            monthTotals.merge(day.substring(0, 7), cost, Double::sum);
        }
    }

    /** 清空全部累计与队列（测试隔离用）。 */
    public void reset() {
        synchronized (lock) {
            dayTotals.clear();
            monthTotals.clear();
            pending.clear();
            warnedModels.clear();
        }
    }

    private static String today() {
        return LocalDate.now(ZoneId.systemDefault()).toString();
    }
}
