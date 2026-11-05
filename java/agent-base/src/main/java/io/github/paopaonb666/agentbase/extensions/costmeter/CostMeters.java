package io.github.paopaonb666.agentbase.extensions.costmeter;

/**
 * 进程级成本计量单例持有者（对应 Python 侧 {@code costmeter.COST_METER}）。
 *
 * <p>与 {@code TOOL_METRICS} 同理：全部模型客户端共享一份累计，预算门
 * （{@link CostGovernor}）与周期落库任务都读同一个实例。单例以空价目表
 * 构造，服务装配时由 server lifespan 调 {@code configure(prices)} 注入
 * 价目表（对应 Python 侧 {@code COST_METER.configure(cost.prices_json)}）。
 */
public final class CostMeters {

    private static final CostMeter COST = new CostMeter();

    private CostMeters() {}

    /** 进程唯一的成本计量器（对应 COST_METER）。 */
    public static CostMeter cost() {
        return COST;
    }
}
