package io.github.paopaonb666.agentbase.extensions.costmeter;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.time.Instant;
import java.time.LocalDate;
import java.time.ZoneId;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import io.github.paopaonb666.agentbase.core.config.Settings;

/**
 * 账本三后端实现 + 装配工厂（对应 Python 侧 {@code costmeter} 的
 * {@code MemoryCostLedger} / {@code SqliteCostLedger} / {@code MysqlCostLedger}
 * 与 server lifespan 里的后端分派）。
 *
 * <p>后端跟随 {@code CHECKPOINTER_BACKEND}：
 * <ul>
 *   <li><b>memory</b>：进程内 Map，重启清零——方向安全：只会少记不会
 *       误熔断（预算门宁可漏熔断也不可无中生有地熔断正常用户）；</li>
 *   <li><b>sqlite</b>：JDBC 常驻连接，{@code cost_usage} 表与对话状态
 *       同一文件（路径来自 {@code CHECKPOINTER_SQLITE_PATH}），DDL 与
 *       Python 侧 {@code _COST_USAGE_DDL} 逐条一致（幂等自举）；</li>
 *   <li><b>mysql</b>：DriverManager 短连接（账本是低频批量写，参数来自
 *       checkpointer 节），{@code _COST_USAGE_MYSQL_DDL} 幂等建表
 *       （5.7 没有 CREATE INDEX IF NOT EXISTS，索引进 CREATE TABLE）。</li>
 * </ul>
 * 后端未知时 {@link #build} 返回 null（成本治理不启用）。
 */
public final class CostLedgers {

    private static final Logger LOG = LoggerFactory.getLogger(CostLedgers.class);

    /**
     * sqlite 账本 DDL（幂等自举），逐条复制 Python 侧 {@code _COST_USAGE_DDL}
     * （账本表 DDL 的单一来源是 Python 代码——cost_usage 无专门 alembic 迁移）。
     */
    private static final String SQLITE_DDL = """
            CREATE TABLE IF NOT EXISTS cost_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                request_id TEXT NOT NULL DEFAULT '',
                model TEXT NOT NULL,
                profile TEXT NOT NULL DEFAULT 'main',
                prompt_tokens INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                cache_read_tokens INTEGER NOT NULL DEFAULT 0,
                cost_cny REAL NOT NULL DEFAULT 0
            )
            """;

    /** 对应 Python 侧 {@code _COST_USAGE_INDEX_DDL}。 */
    private static final String SQLITE_INDEX_DDL =
            "CREATE INDEX IF NOT EXISTS idx_cost_usage_ts ON cost_usage (ts)";

    /**
     * MySQL 账本 DDL，逐条复制 Python 侧 {@code _COST_USAGE_MYSQL_DDL}。
     */
    private static final String MYSQL_DDL = """
            CREATE TABLE IF NOT EXISTS cost_usage (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                ts DOUBLE NOT NULL,
                request_id VARCHAR(64) NOT NULL DEFAULT '',
                model VARCHAR(128) NOT NULL,
                profile VARCHAR(16) NOT NULL DEFAULT 'main',
                prompt_tokens BIGINT NOT NULL DEFAULT 0,
                completion_tokens BIGINT NOT NULL DEFAULT 0,
                cache_read_tokens BIGINT NOT NULL DEFAULT 0,
                cost_cny DOUBLE NOT NULL DEFAULT 0,
                INDEX idx_cost_usage_ts (ts)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """;

    private static final String INSERT_SQL =
            "INSERT INTO cost_usage (ts, request_id, model, profile, prompt_tokens,"
                    + " completion_tokens, cache_read_tokens, cost_cny)"
                    + " VALUES (?, ?, ?, ?, ?, ?, ?, ?)";

    private static final String TOTALS_SINCE_SQL =
            "SELECT COALESCE(SUM(cost_cny), 0) FROM cost_usage WHERE ts >= ?";

    private static final String SQLITE_BY_DAY_SQL =
            "SELECT date(ts, 'unixepoch', 'localtime'), SUM(cost_cny) FROM cost_usage GROUP BY 1";

    // MySQL 的 DATE_FORMAT(FROM_UNIXTIME(...)) 用会话时区，与 sqlite 侧的
    // 'localtime' 语义对齐（Python 侧注释原文）。
    private static final String MYSQL_BY_DAY_SQL =
            "SELECT DATE_FORMAT(FROM_UNIXTIME(ts), '%Y-%m-%d'), SUM(cost_cny)"
                    + " FROM cost_usage GROUP BY 1";

    private CostLedgers() {}

    /** 按 settings 装配账本；后端未知时返回 null（成本治理不启用）。 */
    public static CostLedger build(Settings settings) {
        String backend = settings.checkpointer.backend;
        return switch (backend) {
            case "memory" -> new MemoryCostLedger();
            case "sqlite" -> new SqliteCostLedger(settings.checkpointer.sqlitePath);
            case "mysql" -> new MySqlCostLedger(settings);
            default -> {
                LOG.warn("cost: unknown checkpointer backend '{}'; cost ledger disabled", backend);
                yield null;
            }
        };
    }

    /** day-iso 键（本地时区；对应 Python datetime.fromtimestamp(ts).date().isoformat()）。 */
    static String dayOf(double ts) {
        return LocalDate.ofInstant(
                Instant.ofEpochMilli((long) Math.floor(ts * 1000.0)), ZoneId.systemDefault()).toString();
    }

    /** pending 队列的公共骨架：append 缓冲、flush 批量交给后端落库。 */
    abstract static class BufferedCostLedger implements CostLedger {

        /** 失败重放缓冲上限（#262）：存储长期不可用时防爆内存，超限丢最老。 */
        static final int RETRY_BUFFER_LIMIT = 10_000;

        private final List<UsageRecord> pending = new ArrayList<>();

        @Override
        public final synchronized void append(UsageRecord record) {
            pending.add(record);
        }

        @Override
        public final synchronized void flush() {
            if (pending.isEmpty()) {
                return; // 对应 Python insert(rows) 的 `if not rows: return`
            }
            List<UsageRecord> rows = new ArrayList<>(pending);
            pending.clear();
            if (insertRows(rows)) {
                return;
            }
            // 落库失败（#262）：批次还回队列头部，下轮 flush 重放——存储
            // 抖动窗口内的用量计量不再永久丢失（少记→漏熔断的账面失真）。
            pending.addAll(0, rows);
            if (pending.size() > RETRY_BUFFER_LIMIT) {
                int dropped = pending.size() - RETRY_BUFFER_LIMIT;
                LOG.error("cost: 账本重放缓冲超限（{} > {}）——丢弃最老 {} 行",
                        pending.size(), RETRY_BUFFER_LIMIT, dropped);
                pending.subList(0, dropped).clear();
            }
        }

        /**
         * 批量落库一批明细；返回是否成功（失败时调用方重放该批次）。
         * 实现内部吞掉异常（失败语义经返回值传达，不向上抛）。
         */
        protected abstract boolean insertRows(List<UsageRecord> rows);
    }

    /** 内存账本（测试与 memory 后端：重启清零，方向安全——只会少记不会误熔断）。 */
    public static final class MemoryCostLedger extends BufferedCostLedger {
        private final List<UsageRecord> rows = new ArrayList<>();

        @Override
        protected synchronized boolean insertRows(List<UsageRecord> newRows) {
            rows.addAll(newRows);
            return true;
        }

        @Override
        public synchronized double totalsSince(double ts) {
            double total = 0.0;
            for (UsageRecord row : rows) {
                if (row.ts() >= ts) {
                    total += row.costCny();
                }
            }
            return total;
        }

        @Override
        public synchronized Map<String, Double> totalsByDay() {
            Map<String, Double> totals = new LinkedHashMap<>();
            for (UsageRecord row : rows) {
                totals.merge(dayOf(row.ts()), row.costCny(), Double::sum);
            }
            return totals;
        }
    }

    /** sqlite 账本：跟随 checkpointer 的 sqlite 库文件，独立连接。 */
    public static final class SqliteCostLedger extends BufferedCostLedger {
        private final Connection conn;

        public SqliteCostLedger(String dbPath) {
            try {
                this.conn = DriverManager.getConnection("jdbc:sqlite:" + dbPath);
                this.conn.setAutoCommit(false);
                try (Statement st = conn.createStatement()) {
                    st.execute(SQLITE_DDL);
                    st.execute(SQLITE_INDEX_DDL);
                }
                conn.commit(); // 对应 Python aiosqlite 的显式 commit
            } catch (SQLException e) {
                throw new IllegalStateException("cost: sqlite 账本初始化失败: " + e.getMessage(), e);
            }
        }

        @Override
        protected boolean insertRows(List<UsageRecord> rows) {
            try {
                try (PreparedStatement ps = conn.prepareStatement(INSERT_SQL)) {
                    for (UsageRecord r : rows) {
                        fill(ps, r);
                        ps.addBatch();
                    }
                    ps.executeBatch();
                }
                conn.commit();
                return true;
            } catch (SQLException e) {
                // 回滚半批（autocommit=false），失败批次由骨架重放（#262）。
                try {
                    conn.rollback();
                } catch (SQLException rollbackError) {
                    LOG.warn("cost: sqlite rollback failed", rollbackError);
                }
                LOG.error("cost: sqlite insert failed——批次将重放", e);
                return false;
            }
        }

        @Override
        public synchronized double totalsSince(double ts) {
            try (PreparedStatement ps = conn.prepareStatement(TOTALS_SINCE_SQL)) {
                ps.setDouble(1, ts);
                try (ResultSet rs = ps.executeQuery()) {
                    return rs.next() ? rs.getDouble(1) : 0.0;
                }
            } catch (SQLException e) {
                LOG.error("cost: sqlite totals_since failed", e);
                return 0.0;
            }
        }

        @Override
        public synchronized Map<String, Double> totalsByDay() {
            Map<String, Double> out = new LinkedHashMap<>();
            try (PreparedStatement ps = conn.prepareStatement(SQLITE_BY_DAY_SQL);
                 ResultSet rs = ps.executeQuery()) {
                while (rs.next()) {
                    out.put(rs.getString(1), rs.getDouble(2)); // NULL SUM → 0.0（float(cost or 0.0)）
                }
            } catch (SQLException e) {
                LOG.error("cost: sqlite totals_by_day failed", e);
            }
            return out;
        }

        @Override
        public void close() {
            try {
                conn.close();
            } catch (SQLException e) {
                LOG.debug("cost: sqlite close failed", e);
            }
        }
    }

    /** MySQL 账本：DriverManager 短连接（账本是低频批量写，无跨语句事务需求）。 */
    public static final class MySqlCostLedger extends BufferedCostLedger {
        private final Settings settings;
        private volatile boolean tableReady = false;

        public MySqlCostLedger(Settings settings) {
            this.settings = settings;
        }

        private Connection connect() throws SQLException {
            var s = settings.checkpointer;
            return DriverManager.getConnection(
                    "jdbc:mysql://%s:%d/%s".formatted(s.mysqlHost, s.mysqlPort, s.mysqlDatabase),
                    s.mysqlUser, s.mysqlPassword);
        }

        /** 幂等建表（每连接首次操作时执行一次；对应 Python _ensure_table）。 */
        private void ensureTable(Connection conn) throws SQLException {
            if (!tableReady) {
                try (Statement st = conn.createStatement()) {
                    st.execute(MYSQL_DDL);
                }
                tableReady = true;
            }
        }

        @Override
        protected boolean insertRows(List<UsageRecord> rows) {
            try (Connection conn = connect()) {
                ensureTable(conn);
                try (PreparedStatement ps = conn.prepareStatement(INSERT_SQL)) {
                    for (UsageRecord r : rows) {
                        fill(ps, r);
                        ps.addBatch();
                    }
                    ps.executeBatch();
                }
                return true;
            } catch (SQLException e) {
                LOG.error("cost: mysql insert failed——批次将重放", e);
                return false;
            }
        }

        @Override
        public double totalsSince(double ts) {
            try (Connection conn = connect()) {
                ensureTable(conn);
                try (PreparedStatement ps = conn.prepareStatement(TOTALS_SINCE_SQL)) {
                    ps.setDouble(1, ts);
                    try (ResultSet rs = ps.executeQuery()) {
                        return rs.next() ? rs.getDouble(1) : 0.0;
                    }
                }
            } catch (SQLException e) {
                LOG.error("cost: mysql totals_since failed", e);
                return 0.0;
            }
        }

        @Override
        public Map<String, Double> totalsByDay() {
            Map<String, Double> out = new LinkedHashMap<>();
            try (Connection conn = connect()) {
                ensureTable(conn);
                try (PreparedStatement ps = conn.prepareStatement(MYSQL_BY_DAY_SQL);
                     ResultSet rs = ps.executeQuery()) {
                    while (rs.next()) {
                        out.put(rs.getString(1), rs.getDouble(2));
                    }
                }
            } catch (SQLException e) {
                LOG.error("cost: mysql totals_by_day failed", e);
            }
            return out;
        }
    }

    /** 明细行 → INSERT 参数（列序与 Python 侧 executemany 完全一致）。 */
    private static void fill(PreparedStatement ps, UsageRecord r) throws SQLException {
        ps.setDouble(1, r.ts());
        ps.setString(2, r.requestId());
        ps.setString(3, r.model());
        ps.setString(4, r.profile());
        ps.setLong(5, r.promptTokens());
        ps.setLong(6, r.completionTokens());
        ps.setLong(7, r.cacheReadTokens());
        ps.setDouble(8, r.costCny());
    }
}
