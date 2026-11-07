package io.github.paopaonb666.agentbase.extensions.costmeter;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;
import ch.qos.logback.core.read.ListAppender;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.slf4j.LoggerFactory;

import java.nio.file.Path;
import java.time.LocalDate;
import java.util.List;
import java.util.Map;

import io.github.paopaonb666.agentbase.core.config.Settings;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** 成本计量（计量器 + 账本）测试：计价、告警去重、pending 队列与 sqlite 往返。 */
class CostMeterTest {

    /** 价目表（元/百万 tokens）：miss 2 / hit 0.5 / output 8。 */
    private static final Map<String, Map<String, Double>> PRICES = Map.of(
            "m", Map.of("input_miss", 2.0, "input_hit", 0.5, "output", 8.0));

    private final CostMeter meter = new CostMeter();

    private final ListAppender<ch.qos.logback.classic.spi.ILoggingEvent> logAppender =
            new ListAppender<>();

    private Logger costLogger;

    @BeforeEach
    void setUp() {
        meter.reset();
        meter.configure(PRICES);
        costLogger = (Logger) LoggerFactory.getLogger(CostMeter.class);
        logAppender.start();
        costLogger.addAppender(logAppender);
    }

    @AfterEach
    void tearDown() {
        costLogger.detachAppender(logAppender);
        logAppender.stop();
    }

    private long warnCount(String contains) {
        return logAppender.list.stream()
                .filter(e -> e.getLevel() == Level.WARN)
                .filter(e -> e.getFormattedMessage().contains(contains))
                .count();
    }

    @Test
    void pricesByMillionTokens() {
        // 1M input miss（2 元） + 0.5M output（8 元/百万 → 4 元） = 6 元。
        meter.recordUsage("m", 1_000_000, 500_000);
        assertEquals(6.0, meter.dayCost(), 1e-9);
        assertEquals(6.0, meter.monthCost(), 1e-9);
    }

    @Test
    void cacheHitPricedLowerAndClampedToPrompt() {
        // 600k miss（1.2） + 400k hit（0.2） + 1M output（8） = 9.4 元。
        meter.recordUsage("m", "main", 1_000_000, 1_000_000, 400_000);
        assertEquals(9.4, meter.dayCost(), 1e-9);
        // cache_read 超出 prompt 时按 prompt 封顶：全部按 hit（0.5 元/百万）。
        meter.recordUsage("m", "main", 1_000_000, 0, 2_000_000);
        assertEquals(9.4 + 0.5, meter.dayCost(), 1e-9);
    }

    @Test
    void missingTableKeysAreZero() {
        // 空价目子表：全 0。
        meter.configure(Map.of("bare", Map.of()));
        meter.recordUsage("bare", 1_000_000, 1_000_000);
        assertEquals(0.0, meter.dayCost(), 1e-9);
        // 只报 input_miss：output 不计价。
        meter.configure(Map.of("half", Map.of("input_miss", 3.0)));
        meter.recordUsage("half", 1_000_000, 1_000_000);
        assertEquals(3.0, meter.dayCost(), 1e-9);
    }

    @Test
    void unknownModelCostsZeroAndWarnsOnce() {
        meter.recordUsage("ghost", 1_000_000, 1_000_000);
        meter.recordUsage("ghost", 1_000_000, 1_000_000);
        assertEquals(0.0, meter.dayCost(), 1e-9);
        assertEquals(1, warnCount("ghost"), "未知模型只告警一次");
        // 已知模型不产生未知告警。
        meter.recordUsage("m", 1_000_000, 0);
        assertEquals(1, warnCount("不在 COST_PRICES_JSON 价目表中"));
    }

    @Test
    void configureResetsWarnDedup() {
        meter.recordUsage("ghost", 0, 0);
        meter.configure(PRICES); // 重新装配（覆盖价目）会重置告警去重。
        meter.recordUsage("ghost", 0, 0);
        assertEquals(2, warnCount("ghost"));
    }

    @Test
    void pendingQueueDrainsOnce() {
        meter.recordUsage("m", 1_000, 500);
        List<UsageRecord> rows = meter.takePending();
        assertEquals(1, rows.size());
        UsageRecord row = rows.get(0);
        assertEquals("m", row.model());
        assertEquals("main", row.profile());
        assertEquals(1_000, row.promptTokens());
        assertEquals(500, row.completionTokens());
        assertEquals(0, row.cacheReadTokens());
        assertEquals(0.006, row.costCny(), 1e-12); // 1000 miss×2 + 500 output×8（均 /1e6）
        assertTrue(row.ts() > 0);
        assertEquals("", row.requestId(), "requestId 缺省为空串（MDC 未设置）");
        assertTrue(meter.takePending().isEmpty(), "取走后队列为空");
    }

    @Test
    void pendingQueueIsBoundedAndDropsOldestOnOverflow() {
        // #263：灌满上限后再 observe → 丢最老并计数；队头是最老未丢的；
        // takePending 正常取走；后续 observe 恢复正常追加。
        for (int i = 0; i <= CostMeter.MAX_PENDING; i++) { // 上限 + 1 次
            meter.recordUsage("m", i, 0); // prompt=i 作行的身份标记
        }
        assertEquals(1, meter.droppedRecordCount(), "溢出恰好丢 1 行（最老的 prompt=0）");

        List<UsageRecord> rows = meter.takePending();
        assertEquals(CostMeter.MAX_PENDING, rows.size());
        assertEquals(1, rows.get(0).promptTokens(), "队头应是最老未丢的行");
        assertEquals(CostMeter.MAX_PENDING, rows.get(rows.size() - 1).promptTokens(),
                "队尾应是最新追加的行");
        assertTrue(meter.takePending().isEmpty(), "取走后队列为空");

        // 计数是累计口径：不随 takePending 回零。
        assertEquals(1, meter.droppedRecordCount());

        // 后续 observe 恢复正常（容量已腾出）。
        meter.recordUsage("m", 42, 0);
        List<UsageRecord> again = meter.takePending();
        assertEquals(1, again.size());
        assertEquals(42, again.get(0).promptTokens());
        assertEquals(1, meter.droppedRecordCount(), "未再溢出，计数保持");
    }

    @Test
    void restoreDayAccumulatesIntoDayAndMonth() {
        meter.recordUsage("m", 1_000_000, 0); // 2 元
        String today = LocalDate.now().toString();
        meter.restoreDay(today, 0, 0, 3.5);
        assertEquals(5.5, meter.dayCost(today), 1e-9);
        assertEquals(5.5, meter.monthCost(today.substring(0, 7)), 1e-9);
    }

    @Test
    void costMeterSingletonIsStable() {
        // 对应 COST_METER：进程内同一实例。
        assertTrue(CostMeters.cost() == CostMeters.cost());
    }

    // ── 账本 ─────────────────────────────────────────────────────────

    @TempDir
    Path tempDir;

    @Test
    void sqliteLedgerRoundtrip() {
        Path db = tempDir.resolve("cost_usage_test.db");
        CostLedgers.SqliteCostLedger ledger = new CostLedgers.SqliteCostLedger(db.toString());
        try {
            CostMeter local = new CostMeter();
            local.configure(PRICES);
            local.recordUsage("m", 1_000_000, 500_000); // 6 元
            for (UsageRecord row : local.takePending()) {
                ledger.append(row);
            }
            ledger.flush();
            String today = LocalDate.now().toString();
            Map<String, Double> byDay = ledger.totalsByDay();
            assertEquals(6.0, byDay.get(today), 1e-9);
            assertEquals(6.0, ledger.totalsSince(0), 1e-9);
            assertEquals(0.0, ledger.totalsSince(Double.MAX_VALUE / 2), 1e-9);
            // 重启恢复：新计量器从账本按日回放。
            CostMeter rebooted = new CostMeter();
            for (Map.Entry<String, Double> e : byDay.entrySet()) {
                rebooted.restoreDay(e.getKey(), 0, 0, e.getValue());
            }
            assertEquals(6.0, rebooted.dayCost(today), 1e-9);
            assertEquals(6.0, rebooted.monthCost(today.substring(0, 7)), 1e-9);
            // 空队列 flush 不落行也不报错。
            ledger.flush();
            assertEquals(6.0, ledger.totalsByDay().get(today), 1e-9);
        } finally {
            ledger.close();
        }
    }

    @Test
    void memoryLedgerRoundtrip() {
        CostLedgers.MemoryCostLedger ledger = new CostLedgers.MemoryCostLedger();
        UsageRecord row = new UsageRecord(
                System.currentTimeMillis() / 1000.0, "", "m", "main", 10, 5, 0, 1.25);
        ledger.append(row);
        assertTrue(ledger.totalsByDay().isEmpty(), "flush 前不落账");
        ledger.flush();
        String today = LocalDate.now().toString();
        assertEquals(1.25, ledger.totalsByDay().get(today), 1e-9);
        assertEquals(1.25, ledger.totalsSince(0), 1e-9);
    }

    @Test
    void factoryDispatchesOnCheckpointerBackend() {
        Settings memory = Settings.load(
                Map.of("CHECKPOINTER_BACKEND", "memory"), null);
        assertInstanceOf(CostLedgers.MemoryCostLedger.class, CostLedgers.build(memory));
        Settings sqlite = Settings.load(
                Map.of("CHECKPOINTER_BACKEND", "sqlite",
                        "CHECKPOINTER_SQLITE_PATH", tempDir.resolve("factory.db").toString()),
                null);
        CostLedger sqliteLedger = CostLedgers.build(sqlite);
        assertInstanceOf(CostLedgers.SqliteCostLedger.class, sqliteLedger);
        sqliteLedger.close();
        // 未知后端返回 null（成本治理不启用），与 toollog 工厂同款约定。
        // Settings.load 会在校验时拒绝未知后端，这里直接改字段绕过装配校验。
        Settings bogus = new Settings();
        bogus.checkpointer.backend = "redis";
        assertNull(CostLedgers.build(bogus));
    }
}
