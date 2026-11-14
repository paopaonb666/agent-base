package io.github.paopaonb666.agentbase.extensions.costmeter;

import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.core.llm.UsageSinks;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A-L79+#76 账本列演进测试：cost_usage 的 attempt/fallback 列经
 * SchemaMigrations 迁移补列（新库与旧库单一路径）、写读对称、旧行 NULL
 * 兼容。mysql 无活体（迁移清单与 sqlite 侧同一份，Executor 方言适配）。
 */
class CostLedgerMigrationTest {

    @Test
    void migrationAddsColumnsAndRoundtripsAttemptFallback(@TempDir Path tempDir) throws Exception {
        Path db = tempDir.resolve("cost.db");
        CostLedgers.SqliteCostLedger ledger = new CostLedgers.SqliteCostLedger(db.toString());
        // 弹性上下文形态：attempt=2、fallback#1 → 随行落账。
        ledger.append(new UsageRecord(1000.0, "r1", "m", "main", 10, 5, 0, 0.02, 2, "fallback#1"));
        // 无上下文形态（直连运行时/旧行为）→ NULL。
        ledger.append(new UsageRecord(1000.0, "r2", "m", "main", 10, 5, 0, 0.02, null, null));
        ledger.flush();

        try (var conn = DriverManager.getConnection("jdbc:sqlite:" + db);
             PreparedStatement ps = conn.prepareStatement(
                     "SELECT attempt, fallback FROM cost_usage ORDER BY id")) {
            try (ResultSet rs = ps.executeQuery()) {
                assertTrue(rs.next());
                assertEquals(2, rs.getInt(1));
                assertEquals("fallback#1", rs.getString(2));
                assertTrue(rs.next());
                assertEquals(false, rs.getBoolean(1) && rs.wasNull() == false,
                        "无上下文行 attempt 为 NULL");
                assertTrue(rs.getString(2) == null, "无上下文行 fallback 为 NULL");
            }
        }
        ledger.close();
    }

    @Test
    void migrationIsIdempotentAndVersioned(@TempDir Path tempDir) throws Exception {
        Path db = tempDir.resolve("cost.db");
        CostLedgers.SqliteCostLedger ledger = new CostLedgers.SqliteCostLedger(db.toString());
        try (var conn = DriverManager.getConnection("jdbc:sqlite:" + db);
             var rs = conn.createStatement().executeQuery(
                     "SELECT id FROM cost_schema_versions")) {
            assertTrue(rs.next());
            assertEquals("001_cost_attempt_fallback", rs.getString(1), "版本表记录迁移 id");
            assertFalse(rs.next(), "仅一条迁移");
        }
        // 重复打开同一库：迁移已应用即跳过（幂等）——列不重复、不报错。
        CostLedgers.SqliteCostLedger reopened = new CostLedgers.SqliteCostLedger(db.toString());
        reopened.append(new UsageRecord(1000.0, "r3", "m", "main", 10, 5, 0, 0.02, 0, null));
        reopened.flush();
        try (var conn = DriverManager.getConnection("jdbc:sqlite:" + db);
             var rs = conn.createStatement().executeQuery("SELECT COUNT(*) FROM cost_usage")) {
            rs.next();
            assertEquals(1, rs.getInt(1), "重开库不重放迁移（首库零行 + 重开库追加一行）");
        }
        ledger.close();
        reopened.close();
    }

    @Test
    void oldFormatRowsStayReadableAfterMigration(@TempDir Path tempDir) throws Exception {
        // 旧行 NULL 兼容：迁移前手工插入的 8 列行（无 attempt/fallback）在
        // 迁移后照常参与总额（totalsSince 不触新列）。
        Path db = tempDir.resolve("cost.db");
        // 先造一个"旧库"：原 8 列 DDL + 一行旧数据（无迁移）。
        try (var conn = DriverManager.getConnection("jdbc:sqlite:" + db);
             var st = conn.createStatement()) {
            st.execute("""
                    CREATE TABLE cost_usage (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ts REAL NOT NULL,
                        request_id TEXT NOT NULL DEFAULT '',
                        model TEXT NOT NULL,
                        profile TEXT NOT NULL DEFAULT 'main',
                        prompt_tokens INTEGER NOT NULL DEFAULT 0,
                        completion_tokens INTEGER NOT NULL DEFAULT 0,
                        cache_read_tokens INTEGER NOT NULL DEFAULT 0,
                        cost_cny REAL NOT NULL DEFAULT 0
                    )""");
            try (PreparedStatement ps = conn.prepareStatement(
                    "INSERT INTO cost_usage (ts, request_id, model, profile, prompt_tokens,"
                            + " completion_tokens, cache_read_tokens, cost_cny)"
                            + " VALUES (1000.0, 'old', 'm', 'main', 10, 5, 0, 0.5)")) {
                ps.executeUpdate();
            }
        }
        // 打开账本：迁移补列 + 旧行保留（attempt/fallback 为 NULL）。
        CostLedgers.SqliteCostLedger ledger = new CostLedgers.SqliteCostLedger(db.toString());
        assertEquals(0.5, ledger.totalsSince(0), 1e-9, "旧行参与总额");
        ledger.close();
        try (var conn = DriverManager.getConnection("jdbc:sqlite:" + db);
             var rs = conn.createStatement().executeQuery(
                     "SELECT attempt, fallback FROM cost_usage WHERE request_id = 'old'")) {
            assertTrue(rs.next());
            assertTrue(rs.getObject(1) == null && rs.getObject(2) == null,
                    "存量行新列为 NULL");
        }
        // sink 形态端到端：AttemptContext 经 costMeterSink 进 UsageRecord
        // 队列（全局单例——增量断言）。
        var sink = UsageSinks.costMeterSink("main");
        int before = CostMeters.cost().pendingRows();
        UsageSinks.setAttemptContext(new UsageSinks.AttemptContext(1, -1));
        try {
            sink.accept("m", null);
        } finally {
            UsageSinks.clearAttemptContext();
        }
        assertEquals(before + 1, CostMeters.cost().pendingRows(), "sink 形态入队一行");
    }

    @Test
    void readPathDoesNotHoldWriteLockForOtherConnections(@TempDir Path tempDir) throws Exception {
        // GUI 验收缺陷 1（阻塞级）：账本连接常驻 autocommit(false)，读路径
        // （/health 的 cost governor 每次调 totalsSince/totalsByDay）执行
        // SELECT 后不结束事务 → SHARED 锁永久持有 → 同库所有其他连接的写
        // （thread index claim 等）SQLITE_BUSY → invoke 全量 500。
        // 回归锁：读路径之后，另一个连接必须能立即写入（修复前必红）。
        Path db = tempDir.resolve("cost.db");
        CostLedgers.SqliteCostLedger ledger = new CostLedgers.SqliteCostLedger(db.toString());
        ledger.append(new UsageRecord(1000.0, "r1", "m", "main", 10, 5, 0, 0.02, null, null));
        ledger.flush();
        ledger.totalsSince(0);
        ledger.totalsByDay();
        // 第二个连接（短 busy_timeout，失败要快而不是等默认 3s）
        org.sqlite.SQLiteConfig probeConfig = new org.sqlite.SQLiteConfig();
        probeConfig.setBusyTimeout(300);
        try (var probe = probeConfig.createConnection("jdbc:sqlite:" + db)) {
            try (var ps = probe.prepareStatement(
                    "INSERT INTO cost_usage (ts, request_id, model, profile, prompt_tokens,"
                            + " completion_tokens, cache_read_tokens, cost_cny)"
                            + " VALUES (1.0, 'probe', 'm', 'main', 0, 0, 0, 0)")) {
                ps.executeUpdate();
            }
            // probe 默认 autocommit=true：语句即提交，无需显式 commit。
        }
        ledger.close();
    }
}
