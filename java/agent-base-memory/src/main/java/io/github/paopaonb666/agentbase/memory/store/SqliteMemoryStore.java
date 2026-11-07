package io.github.paopaonb666.agentbase.memory.store;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.time.LocalDateTime;
import java.time.format.DateTimeParseException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * sqlite 记忆表后端：与对话状态同一文件，JDBC 持久单连接。
 *
 * <p>对应 Python 侧 {@code agent_base_memory/store/sqlite_backend.py}：
 * {@code SqliteMemoryStore.create} 的自举（幂等建表 + 旧库补列 + 索引）在
 * Java 侧放在构造期完成。并发模型与 {@code core.graph.SqliteCheckpointer}
 * 一致：单连接 + 全操作串行（sqlite 文件库同一时刻只允许一个写事务，
 * 对应 Python aiosqlite 单连接的串行语义）。
 */
public final class SqliteMemoryStore extends AbstractSqlMemoryStore {

    private static final Logger LOG = LoggerFactory.getLogger(SqliteMemoryStore.class);

    private final Connection conn;
    private final Object lock = new Object();

    /** 打开（或创建）sqlite 记忆库并自举建表；失败抛出，由装配方决定降级。 */
    public SqliteMemoryStore(String path) {
        super("INSERT OR REPLACE");
        Connection connection;
        try {
            connection = DriverManager.getConnection("jdbc:sqlite:" + path);
        } catch (SQLException e) {
            throw new IllegalStateException(
                    "sqlite memory store: 无法打开数据库 '%s'".formatted(path), e);
        }
        this.conn = connection;
        try {
            for (String ddl : MemoryDdl.SQLITE_DDLS) {
                try (Statement st = conn.createStatement()) {
                    st.execute(ddl);
                }
            }
            migrateLegacyColumns();
            for (String index : MemoryDdl.SQLITE_INDEXES) {
                try (Statement st = conn.createStatement()) {
                    st.execute(index);
                }
            }
            // #275：声明式 schema 迁移（版本表 + 按序应用，fail-fast）。
            io.github.paopaonb666.agentbase.core.storage.SchemaMigrations.apply(
                    "memory", new SqliteMigrationExecutor(), MemoryDdl.SQLITE_MIGRATIONS, LOG);
        } catch (SQLException e) {
            closeQuietly();
            throw new IllegalStateException("sqlite memory store: 建表失败", e);
        }
    }

    /** sqlite 方言的迁移执行通道（连接为本 store 私有，构造期单线程）。 */
    private final class SqliteMigrationExecutor
            implements io.github.paopaonb666.agentbase.core.storage.SchemaMigrations.Executor {

        @Override
        public void ensureVersionsTable(String table) throws SQLException {
            try (Statement st = conn.createStatement()) {
                st.execute("CREATE TABLE IF NOT EXISTS " + table
                        + " (id TEXT PRIMARY KEY, applied_at TEXT NOT NULL)");
            }
        }

        @Override
        public void recordVersion(String table, String id) throws SQLException {
            try (Statement st = conn.createStatement()) {
                st.execute("INSERT INTO " + table + " (id, applied_at) VALUES ('"
                        + id.replace("'", "''") + "', '" + LocalDateTime.now() + "')");
            }
        }

        @Override
        public List<String> appliedVersions(String table) throws SQLException {
            List<String> ids = new ArrayList<>();
            try (Statement st = conn.createStatement();
                 ResultSet rs = st.executeQuery("SELECT id FROM " + table)) {
                while (rs.next()) {
                    ids.add(rs.getString(1));
                }
            }
            return ids;
        }

        @Override
        public void execute(String sql) throws SQLException {
            try (Statement st = conn.createStatement()) {
                st.execute(sql);
            }
        }
    }

    /**
     * 旧库补列（sqlite 无 alembic，运行时自迁移）：offsets_json 是切片
     * 可视化（M7）后加的。用 {@code PRAGMA table_info} 检查后
     * {@code ALTER TABLE ADD COLUMN}（对应 Python create 的补列分支）。
     */
    private void migrateLegacyColumns() throws SQLException {
        Set<String> columns = new HashSet<>();
        try (Statement st = conn.createStatement();
             ResultSet rs = st.executeQuery("PRAGMA table_info(doc_chunks)")) {
            // PRAGMA table_info 的第 2 列（下标 1）是列名——对应 Python row[1]。
            while (rs.next()) {
                columns.add(rs.getString(2));
            }
        }
        if (!columns.contains("offsets_json")) {
            try (Statement st = conn.createStatement()) {
                st.execute("ALTER TABLE doc_chunks ADD COLUMN offsets_json TEXT");
            }
        }
    }

    @Override
    protected int execute(String sql, Object... params) {
        synchronized (lock) {
            try (PreparedStatement ps = conn.prepareStatement(sql)) {
                bind(ps, params);
                int rowcount = ps.executeUpdate();
                return Math.max(rowcount, 0);
            } catch (SQLException e) {
                throw new IllegalStateException("sqlite memory store: 语句执行失败", e);
            }
        }
    }

    @Override
    protected List<Object[]> fetchAll(String sql, Object... params) {
        synchronized (lock) {
            try (PreparedStatement ps = conn.prepareStatement(sql)) {
                bind(ps, params);
                try (ResultSet rs = ps.executeQuery()) {
                    int columns = rs.getMetaData().getColumnCount();
                    List<Object[]> rows = new ArrayList<>();
                    while (rs.next()) {
                        Object[] row = new Object[columns];
                        for (int i = 0; i < columns; i++) {
                            row[i] = rs.getObject(i + 1);
                        }
                        rows.add(row);
                    }
                    return rows;
                }
            } catch (SQLException e) {
                throw new IllegalStateException("sqlite memory store: 查询失败", e);
            }
        }
    }

    @Override
    protected Object toSqlTs(double epochSeconds) {
        // 微秒精度：memory_ops 的同秒并列依赖它稳定排序（对应 Python
        // datetime.fromtimestamp(...).isoformat(timespec="microseconds")）。
        return toLocalDateTime(epochSeconds).format(TS_WRITE);
    }

    @Override
    protected double fromSqlTs(Object value) {
        try {
            return toEpochSeconds(LocalDateTime.parse(String.valueOf(value), FLEX_TS));
        } catch (DateTimeParseException e) {
            // 坏时间戳静默伪装成"刚刚更新"会污染时间衰减排序：显式告警
            // 而不是无声兜底（L3）。
            LOG.warn("sqlite memory store: 无法解析时间戳 {}，按当前时间兜底", value);
            return epochNow();
        }
    }

    @Override
    protected String opsTiebreak() {
        return "rowid DESC";
    }

    @Override
    protected String versionsTiebreak() {
        return "rowid DESC";
    }

    /** 尽力而为地关闭连接（对应 Python aclose）。 */
    @Override
    public void close() {
        closeQuietly();
    }

    private void closeQuietly() {
        try {
            conn.close();
        } catch (SQLException e) {
            LOG.warn("sqlite memory store: 关闭连接失败: {}", e.getMessage());
        }
    }
}
