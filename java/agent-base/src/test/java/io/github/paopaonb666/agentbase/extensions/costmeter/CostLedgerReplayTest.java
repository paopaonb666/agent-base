package io.github.paopaonb666.agentbase.extensions.costmeter;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * 账本 flush 失败重放测试（roadmap #262）：存储抖动窗口内的用量批次
 * 还回队列头部、下轮 flush 重放——之前 executeBatch 失败仅落日志，
 * takePending 已清空的明细永久丢失（少记 → 预算门漏熔断）。
 */
class CostLedgerReplayTest {

    /** 首次落库失败的桩账本（模拟 MySQL/磁盘抖动窗口）。 */
    private static final class FlakyLedger extends CostLedgers.BufferedCostLedger {
        int attempts = 0;
        final List<UsageRecord> stored = new ArrayList<>();

        @Override
        protected boolean insertRows(List<UsageRecord> rows) {
            attempts++;
            if (attempts == 1) {
                return false;
            }
            stored.addAll(rows);
            return true;
        }

        @Override
        public double totalsSince(double ts) {
            return 0.0;
        }

        @Override
        public java.util.Map<String, Double> totalsByDay() {
            return java.util.Map.of();
        }
    }

    private static UsageRecord row(String model) {
        return new UsageRecord(1000.0, "req-1", model, "main", 10, 5, 0, 0.01);
    }

    @Test
    void flushFailureRequeuesTheBatchForReplay() {
        FlakyLedger ledger = new FlakyLedger();
        ledger.append(row("m-a"));
        ledger.append(row("m-b"));

        ledger.flush(); // 第一次失败 → 批次还回队列头部
        assertEquals(0, ledger.stored.size());

        ledger.append(row("m-c"));
        ledger.flush(); // 第二次成功 → 重放批次 + 新行全部落库
        assertEquals(3, ledger.stored.size());
        assertEquals("m-a", ledger.stored.get(0).model(), "重放批次保序在队头");
        assertEquals("m-b", ledger.stored.get(1).model());
        assertEquals("m-c", ledger.stored.get(2).model());
        // 之后再 flush 不重复落库。
        ledger.flush();
        assertEquals(3, ledger.stored.size());
    }
}
