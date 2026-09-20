"""演示：成本预算门——电表累计、80% 预警、超限熔断（invoke 转 429）。

演示什么
    `COST_ENABLED=true` 时基座给全部 LLM 调用挂 usage 采集 callback
    （含流式），按 `COST_PRICES_JSON` 价目表（元/百万 tokens）计价累计；
    `CostGovernor` 在每次 invoke 前置检查：超限且 `COST_ACTION=block` 时
    抛 `CostBudgetExceeded`（HTTP 路由转 429，只挡新的 LLM 消耗）。
    本脚本用两次 `meter.observe` 代替真实模型调用，完整走一遍
    「计量 → 80% 预警 → 熔断 → health 状态」。

怎么跑
    python examples/budget_gate_demo.py
    无需 API key、无需 server——计量与预算门是纯进程内组件。

预期输出
    WARNING cost: 已达预算阈值的 80% ...              ← 第 1 次就过 80% 线
    第 1 次「调用」后: 今日 0.0080 元 / 日阈值 0.0100 元 → check 通过
    第 2 次「调用」后: CostBudgetExceeded（HTTP 层转 429）
    cost budget exceeded: 今日 0.0160 元 / 本月 0.0160 元（阈值 0.01/0）
    /health 的 cost 组件状态: blocked
"""

from __future__ import annotations

import logging

from agent_base.extensions.costmeter import CostBudgetExceeded, CostGovernor, CostMeter

# 价目表：demo-model 输入未命中 2 元/百万 tokens、命中 0.2、输出 8 元。
PRICES_JSON = '{"demo-model": {"input_miss": 2.0, "input_hit": 0.2, "output": 8.0}}'
DAILY_BUDGET_CNY = 0.01  # 每次「调用」约 0.008 元：第 1 次过 80% 线，第 2 次熔断


def invoke_once(meter: CostMeter) -> None:
    """一次「模型调用」：真实链路里由 build_llm 挂载的 callback 采集 usage。"""
    meter.observe(model="demo-model", profile="main", prompt=2000, completion=500, cache_read=0)


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    meter = CostMeter(PRICES_JSON)
    governor = CostGovernor(daily_cny=DAILY_BUDGET_CNY, monthly_cny=0, action="block", meter=meter)

    invoke_once(meter)
    governor.check()  # 0.0080 < 0.0100 → 放行（但已触发 80% 预警）
    print(
        f"第 1 次「调用」后: 今日 {meter.day_total_cny():.4f} 元"
        f" / 日阈值 {DAILY_BUDGET_CNY:.4f} 元 → check 通过"
    )

    invoke_once(meter)
    try:
        governor.check()
    except CostBudgetExceeded as exc:
        print(f"第 2 次「调用」后: {type(exc).__name__}（HTTP 层转 429）")
        print(exc)

    print(f"/health 的 cost 组件状态: {governor.status()}")


if __name__ == "__main__":
    main()
