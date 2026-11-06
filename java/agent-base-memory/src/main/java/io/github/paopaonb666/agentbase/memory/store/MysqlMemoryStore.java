package io.github.paopaonb666.agentbase.memory.store;

import io.github.paopaonb666.agentbase.core.config.Settings;
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
import java.util.List;

/**
 * MySQL 记忆表后端：每操作一条 {@code DriverManager} 短连接。
 *
 * <p>对应 Python 侧 {@code agent_base_memory/store/mysql_backend.py}（aiomysql
 * 连接池，锐评 #2）。连接模型（与 Python 侧的差异及依据，与基座
 * {@code core.storage.MysqlCoreStore} 同一取舍）：Python 用 aiomysql 小连接池
 * （maxsize = memory_mysql_pool_size），Java 侧暂不建池——记忆读写是每轮
 * 一到两次的稳定低频流量，短连接用完即关开销等价，且免去连接池的维护；
 * {@code MEMORY_MYSQL_POOL_SIZE} 暂不消费，将来确需池化时再引入。
 *
 * <p>建表惰性执行（{@code CREATE TABLE IF NOT EXISTS} 幂等，并发重复执行
 * 无害）：volatile 标记保证首次操作建一次，对应 Python {@code _ensure_tables}。
 * MySQL 5.7 兼容的 DDL 见 {@link MemoryDdl#MYSQL_DDLS}。
 */
public final class MysqlMemoryStore extends AbstractSqlMemoryStore {

    private static final Logger LOG = LoggerFactory.getLogger(MysqlMemoryStore.class);

    private final String url;
    private final String user;
    private final String password;
    /** 建表是否已就绪（首次操作执行一次）。 */
    private volatile boolean tableReady = false;

    /** 连接参数全部来自配置（CHECKPOINTER_MYSQL_*；与对话状态同一服务器）。 */
    public MysqlMemoryStore(Settings settings) {
        super("REPLACE");
        Settings.CheckpointerSettings s = settings.checkpointer;
        this.url = ("jdbc:mysql://%s:%d/%s?useUnicode=true&characterEncoding=utf8"
                + "&allowPublicKeyRetrieval=true&useSSL=false")
                .formatted(s.mysqlHost, s.mysqlPort, s.mysqlDatabase);
        this.user = s.mysqlUser;
        this.password = s.mysqlPassword;
    }

    private Connection open() throws SQLException {
        return DriverManager.getConnection(url, user, password);
    }

    private void ensureTables(Connection conn) throws SQLException {
        if (tableReady) {
            return;
        }
        try (Statement st = conn.createStatement()) {
            for (String ddl : MemoryDdl.MYSQL_DDLS) {
                st.execute(ddl);
            }
        }
        ensureIndex(conn, "memories", "idx_mem_user_updated",
                MemoryDdl.MYSQL_INDEX_MEM_USER_UPDATED);
        tableReady = true;
    }

    /**
     * 存量表补索引（#159）：内联 KEY 只对新建表生效，MySQL 又没有
     * CREATE INDEX IF NOT EXISTS——查 information_schema 后按需 ALTER。
     * 索引缺失只影响性能，ALTER 失败按 WARN 降级（不挡存储可用）。
     */
    private void ensureIndex(Connection conn, String table, String indexName, String alter) {
        try (PreparedStatement ps = conn.prepareStatement(
                "SELECT COUNT(*) FROM information_schema.statistics "
                        + "WHERE table_schema = DATABASE() AND table_name = ? AND index_name = ?")) {
            ps.setString(1, table);
            ps.setString(2, indexName);
            try (ResultSet rs = ps.executeQuery()) {
                if (rs.next() && rs.getInt(1) > 0) {
                    return;
                }
            }
            try (Statement st = conn.createStatement()) {
                st.execute(alter);
                LOG.info("mysql memory store: 索引 {}.{} 已补齐（#159）", table, indexName);
            }
        } catch (SQLException e) {
            LOG.warn("mysql memory store: 补建索引 {}.{} 失败（不影响存储可用）：{}",
                    table, indexName, e.toString());
        }
    }

    @Override
    protected int execute(String sql, Object... params) {
        try (Connection conn = open()) {
            ensureTables(conn);
            try (PreparedStatement ps = conn.prepareStatement(sql)) {
                bind(ps, params);
                return Math.max(ps.executeUpdate(), 0);
            }
        } catch (SQLException e) {
            throw new IllegalStateException("mysql memory store: 语句执行失败", e);
        }
    }

    @Override
    protected List<Object[]> fetchAll(String sql, Object... params) {
        try (Connection conn = open()) {
            ensureTables(conn);
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
            }
        } catch (SQLException e) {
            throw new IllegalStateException("mysql memory store: 查询失败", e);
        }
    }

    @Override
    protected Object toSqlTs(double epochSeconds) {
        return toLocalDateTime(epochSeconds);
    }

    @Override
    protected double fromSqlTs(Object value) {
        if (value instanceof LocalDateTime dateTime) {
            return toEpochSeconds(dateTime);
        }
        if (value instanceof java.sql.Timestamp timestamp) {
            return timestamp.getTime() / 1000.0;
        }
        try {
            return toEpochSeconds(LocalDateTime.parse(String.valueOf(value), FLEX_TS));
        } catch (DateTimeParseException e) {
            LOG.warn("mysql memory store: 无法解析时间戳 {}，按当前时间兜底", value);
            return epochNow();
        }
    }

    @Override
    protected String opsTiebreak() {
        // 同一秒内多条 op 的稳定次序：MySQL 用自增 seq（sqlite 用 rowid）。
        return "seq DESC";
    }

    @Override
    protected String versionsTiebreak() {
        return "seq DESC";
    }

    /** 无常驻资源可释放（每操作短连接用完即关；对应 Python aclose 关池）。 */
    @Override
    public void close() {
        LOG.debug("mysql memory store: close（无常驻连接，空操作）");
    }
}
