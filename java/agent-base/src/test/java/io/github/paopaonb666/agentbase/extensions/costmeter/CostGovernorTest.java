package io.github.paopaonb666.agentbase.extensions.costmeter;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;
import ch.qos.logback.core.read.ListAppender;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.slf4j.LoggerFactory;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** 预算门测试：80% 每日预警一次、超限 block/warn 两档行为与 status 三态。 */
class CostGovernorTest {

    /** 价目表：input_miss 10 元/百万 → 1M tokens 计 10 元。 */
    private static final Map<String, Map<String, Double>> PRICES = Map.of(
            "m", Map.of("input_miss", 10.0));

    private final CostMeter meter = new CostMeter();

    private final ListAppender<ch.qos.logback.classic.spi.ILoggingEvent> logAppender =
            new ListAppender<>();

    private Logger governorLogger;

    @BeforeEach
    void setUp() {
        meter.reset();
        meter.configure(PRICES);
        governorLogger = (Logger) LoggerFactory.getLogger(CostGovernor.class);
        logAppender.start();
        governorLogger.addAppender(logAppender);
    }

    @AfterEach
    void tearDown() {
        governorLogger.detachAppender(logAppender);
        logAppender.stop();
    }

    private long warnCount(String contains) {
        return logAppender.list.stream()
                .filter(e -> e.getLevel() == Level.WARN)
                .filter(e -> e.getFormattedMessage().contains(contains))
                .count();
    }

    @Test
    void underBudgetIsOk() {
        CostGovernor governor = new CostGovernor(10.0, 20.0, "block", meter);
        meter.recordUsage("m", 500_000, 0); // 5 元 = 50%
        governor.check();
        assertEquals("ok", governor.status());
        assertEquals(0, warnCount("80%"));
    }

    @Test
    void eightyPercentWarnsOncePerDay() {
        CostGovernor governor = new CostGovernor(10.0, 0.0, "block", meter);
        meter.recordUsage("m", 800_000, 0); // 8 元 = 80%
        governor.check();
        governor.check();
        governor.check();
        assertEquals(1, warnCount("已达预算阈值的 80%"), "80% 预警每日至多一次");
        assertEquals("ok", governor.status(), "80% 不是超限，状态仍 ok");
    }

    @Test
    void monthlyThresholdAlsoTriggersNearWarning() {
        CostGovernor governor = new CostGovernor(0.0, 10.0, "block", meter);
        meter.recordUsage("m", 850_000, 0); // 8.5 元 = 当月 85%
        governor.check();
        assertEquals(1, warnCount("已达预算阈值的 80%"));
    }

    @Test
    void blockActionThrowsAndReportsBlocked() {
        CostGovernor governor = new CostGovernor(5.0, 0.0, "block", meter);
        meter.recordUsage("m", 600_000, 0); // 6 元 > 5 元
        CostBudgetExceededException ex = assertThrows(
                CostBudgetExceededException.class, governor::check);
        assertTrue(ex.getMessage().startsWith("cost budget exceeded"));
        assertTrue(ex.getMessage().contains("6.0000"));
        assertEquals("blocked", governor.status());
    }

    @Test
    void exactThresholdCountsAsExceeded() {
        CostGovernor governor = new CostGovernor(5.0, 0.0, "block", meter);
        meter.recordUsage("m", 500_000, 0); // 恰好 5 元（>= 判定）
        assertThrows(CostBudgetExceededException.class, governor::check);
        assertEquals("blocked", governor.status());
    }

    @Test
    void warnActionLetsRequestsThrough() {
        CostGovernor governor = new CostGovernor(5.0, 0.0, "warn", meter);
        meter.recordUsage("m", 600_000, 0); // 6 元 > 5 元
        assertDoesNotThrow(governor::check);
        assertEquals(1, warnCount("超出预算阈值仍在运行"));
        assertEquals("warn", governor.status());
    }

    @Test
    void zeroThresholdMeansNoLimit() {
        CostGovernor governor = new CostGovernor(0.0, 0.0, "block", meter);
        meter.recordUsage("m", 9_999_999, 0);
        assertDoesNotThrow(governor::check);
        assertEquals("ok", governor.status());
    }

    @Test
    void singletonConstructorUsesProcessMeter() {
        // 三参构造走 COST_METER 单例（对应 Python meter=COST_METER）。
        CostGovernor governor = new CostGovernor(0.0, 0.0, "block");
        assertEquals("ok", governor.status());
    }
}
