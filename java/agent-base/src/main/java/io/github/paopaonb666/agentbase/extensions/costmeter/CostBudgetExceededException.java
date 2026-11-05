package io.github.paopaonb666.agentbase.extensions.costmeter;

/**
 * 预算熔断（{@code action="block"}）时抛出；HTTP 层（invoke 路由）把它
 * 转 429。对应 Python 侧 {@code costmeter.CostBudgetExceeded}。
 *
 * <p>是 RuntimeException：熔断是流控信号而不是受检的业务失败，让调用方
 * 在不声明 throws 的前提下按需捕获。
 */
public class CostBudgetExceededException extends RuntimeException {

    public CostBudgetExceededException(String message) {
        super(message);
    }
}
