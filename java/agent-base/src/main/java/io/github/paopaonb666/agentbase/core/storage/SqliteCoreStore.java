package io.github.paopaonb666.agentbase.core.storage;

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
 * sqlite 基座表后端：与对话状态同一文件（即
 * {@code CHECKPOINTER_SQLITE_PATH}），JDBC 持久单连接。
 *
 * <p>并发模型与 {@code core.graph.SqliteCheckpointer} 一致：单连接 +
 * 全操作串行（sqlite 文件库同一时刻只允许一个写事务）。建表在构造期
 * 自举（对应 Python {@code CoreSqliteStore.create}）。
 */
public final class SqliteCoreStore extends AbstractSqlCoreStore {

    private static final Logger LOG = LoggerFactory.getLogger(SqliteCoreStore.class);

    private final Connection conn;
    private final Object lock = new Object();

    /** 打开（或创建）sqlite 库并自举建表；失败抛出，由装配方决定降级。 */
    public SqliteCoreStore(String path) {
        super("INSERT OR REPLACE");
        Connection connection;
        try {
            connection = DriverManager.getConnection("jdbc:sqlite:" + path);
        } catch (SQLException e) {
            throw new IllegalStateException(
                    "sqlite core store: 无法打开数据库 '%s'".formatted(path), e);
        }
        this.conn = connection;
        try {
            for (String ddl : CoreStoreDdl.SQLITE_DDLS) {
                try (Statement st = conn.createStatement()) {
                    st.execute(ddl);
                }
            }
            for (String index : CoreStoreDdl.SQLITE_INDEXES) {
                try (Statement st = conn.createStatement()) {
                    st.execute(index);
                }
            }
        } catch (SQLException e) {
            closeQuietly();
            throw new IllegalStateException("sqlite core store: 建表失败", e);
        }
    }

    @Override
    protected int execute(String sql, Object... params) {
        synchronized (lock) {
            try (PreparedStatement ps = conn.prepareStatement(sql)) {
                bind(ps, params);
                return ps.executeUpdate();
            } catch (SQLException e) {
                throw new IllegalStateException("sqlite core store: 语句执行失败", e);
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
                throw new IllegalStateException("sqlite core store: 查询失败", e);
            }
        }
    }

    @Override
    protected Object toSqlTs(double epochSeconds) {
        // 微秒精度：线程列表的同秒并列依赖它稳定排序（对应 Python _to_sql_ts）。
        return toLocalDateTime(epochSeconds).format(TS_WRITE);
    }

    @Override
    protected double fromSqlTs(Object value) {
        try {
            return toEpochSeconds(LocalDateTime.parse(String.valueOf(value), FLEX_TS));
        } catch (DateTimeParseException e) {
            LOG.warn("sqlite core store: 无法解析时间戳 {}，按当前时间兜底", value);
            return epochNow();
        }
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
            LOG.warn("sqlite core store: 关闭连接失败: {}", e.getMessage());
        }
    }
}
