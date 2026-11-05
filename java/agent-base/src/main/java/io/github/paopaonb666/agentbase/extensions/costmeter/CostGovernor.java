package io.github.paopaonb666.agentbase.extensions.costmeter;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.LocalDate;
import java.time.ZoneId;
import java.util.Locale;

/**
 * 预算门（T4.2）：日/月阈值 + warn/block 两档行为。
 *
 * <p>对应 Python 侧 {@code costmeter.CostGovernor}：
 * <ul>
 *   <li><b>block</b>（默认）：超限后 {@link #check()} 抛
 *       {@link CostBudgetExceededException}——invoke 路由转 429，只挡
 *       <b>新的 LLM 消耗</b>；检索、历史、记忆管理等非 LLM 端点照常
 *       （熔断的是电表，不是整个房子）；</li>
 *   <li><b>warn</b>：只告警放行，{@link #status()} 报 {@code warn}；</li>
 *   <li>阈值为 0 表示对应周期不限额。</li>
 * </ul>
 *
 * <p>{@code check()} 里还带 80% 预警（每日至多提醒一次）：让"快没钱了"
 * 先于"没钱"出现——达阈值 80% 时打一条 WARN，同一天内不重复。
 */
public final class CostGovernor {

    private static final Logger LOG = LoggerFactory.getLogger(CostGovernor.class);

    private final double daily;
    private final double monthly;
    private final String action;
    private final CostMeter meter;
    private String warnedDay = "";

    /**
     * 以进程单例计量器（{@link CostMeters#cost()}）构造，供一般装配
     * （对应 Python 侧 {@code CostGovernor(..., meter=COST_METER)}）。
     */
    public CostGovernor(double dailyCny, double monthlyCny, String action) {
        this(dailyCny, monthlyCny, action, CostMeters.cost());
    }

    /** 显式注入计量器（测试 / 多表装配用）。 */
    public CostGovernor(double dailyCny, double monthlyCny, String action, CostMeter meter) {
        this.daily = dailyCny;
        this.monthly = monthlyCny;
        this.action = action;
        this.meter = meter;
    }

    private boolean exceeded() {
        return (daily > 0 && meter.dayCost() >= daily)
                || (monthly > 0 && meter.monthCost() >= monthly);
    }

    /** 预算门前置检查；超限且 action=block 时抛出。 */
    public void check() {
        if (exceeded()) {
            if ("block".equals(action)) {
                throw new CostBudgetExceededException(
                        "cost budget exceeded: 今日 " + fmt4(meter.dayCost()) + " 元 / 本月 "
                                + fmt4(meter.monthCost()) + " 元（阈值 " + daily + "/" + monthly + "）");
            }
            LOG.warn("cost: 超出预算阈值仍在运行（action=warn）：今日 {} / 本月 {} 元",
                    fmt4(meter.dayCost()), fmt4(meter.monthCost()));
            return;
        }
        // 80% 预警（每日至多提醒一次）：让"快没钱了"先于"没钱"出现。
        String day = LocalDate.now(ZoneId.systemDefault()).toString();
        boolean near = (daily > 0 && meter.dayCost() >= daily * 0.8)
                || (monthly > 0 && meter.monthCost() >= monthly * 0.8);
        if (near && !warnedDay.equals(day)) {
            warnedDay = day;
            LOG.warn("cost: 已达预算阈值的 80%（今日 {} / 本月 {} 元）",
                    fmt4(meter.dayCost()), fmt4(meter.monthCost()));
        }
    }

    /** health 组件状态：ok | warn | blocked（/health 的 cost 组件用）。 */
    public String status() {
        if (exceeded()) {
            return "block".equals(action) ? "blocked" : "warn";
        }
        return "ok";
    }

    /** 四位小数金额（Locale.ROOT：小数点不随运行环境变逗号）。 */
    private static String fmt4(double value) {
        return String.format(Locale.ROOT, "%.4f", value);
    }
}
