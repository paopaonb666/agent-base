package io.github.paopaonb666.agentbase.core.storage;

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
 * MySQL 基座表后端：每操作一条 {@code DriverManager} 短连接。
 *
 * <p>连接模型（与 Python 侧的差异及依据）：Python 用 aiomysql 小连接池
 * （size=2——线程索引/计划快照是低频读写，每轮一到两次，与记忆池分开是
 * 为了让"关掉记忆"的部署不必为记忆流量付费）。Java 侧改为每操作开一条
 * 短连接、用完即关：低频语义下开销等价，且免去连接池的维护。
 *
 * <p>建表惰性执行一次（{@code CREATE TABLE IF NOT EXISTS} 幂等，并发重复
 * 执行无害）；构造期做一次建连+建表探测，让"连不上/建不了表"在装配期
 * 就暴露——降级（返回 null）由 {@code CoreStores.build} 统一实现。
 */
public final class MysqlCoreStore extends AbstractSqlCoreStore {

    private static final Logger LOG = LoggerFactory.getLogger(MysqlCoreStore.class);

    private final String url;
    private final String user;
    private final String password;
    /** 建表是否已就绪（只执行一次）。 */
    private volatile boolean tableReady = false;

    /** 复用 CheckpointerSettings 的 MySQL 参数（与对话状态同一服务器）。 */
    public MysqlCoreStore(Settings.CheckpointerSettings s) {
        super("REPLACE");
        this.url = ("jdbc:mysql://%s:%d/%s?useUnicode=true&characterEncoding=utf8"
                + "&allowPublicKeyRetrieval=true&useSSL=false")
                .formatted(s.mysqlHost, s.mysqlPort, s.mysqlDatabase);
        this.user = s.mysqlUser;
        this.password = s.mysqlPassword;
        probeAndEnsureTables();
    }

    /** 装配期探测：建连 + 惰性建表，失败抛出（CoreStores.build 捕获后降级）。 */
    private void probeAndEnsureTables() {
        try (Connection conn = open()) {
            ensureTables(conn);
        } catch (SQLException e) {
            throw new IllegalStateException("mysql core store: 建连或建表失败", e);
        }
    }

    private Connection open() throws SQLException {
        return DriverManager.getConnection(url, user, password);
    }

    private void ensureTables(Connection conn) throws SQLException {
        if (tableReady) {
            return;
        }
        try (Statement st = conn.createStatement()) {
            for (String ddl : CoreStoreDdl.MYSQL_DDLS) {
                st.execute(ddl);
            }
        }
        tableReady = true;
    }

    @Override
    protected int execute(String sql, Object... params) {
        try (Connection conn = open()) {
            ensureTables(conn);
            try (PreparedStatement ps = conn.prepareStatement(sql)) {
                bind(ps, params);
                return ps.executeUpdate();
            }
        } catch (SQLException e) {
            throw new IllegalStateException("mysql core store: 语句执行失败", e);
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
            throw new IllegalStateException("mysql core store: 查询失败", e);
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
            LOG.warn("mysql core store: 无法解析时间戳 {}，按当前时间兜底", value);
            return epochNow();
        }
    }

    /** 无常驻资源可释放（每操作短连接用完即关；对应 Python aclose 关池）。 */
    @Override
    public void close() {
        LOG.debug("mysql core store: close（无常驻连接，空操作）");
    }
}
