package io.github.paopaonb666.agentbase.core.graph;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

/**
 * sqlite 后端的对话状态存储：JDBC 单连接 + 单连接互斥锁。
 *
 * <p>存储表为 Java 版自有 schema {@code ab_thread_states}（不与 langgraph
 * 的 msgpack checkpoints 表互通——已批准的内部差异）：每线程一行，状态以
 * {@link GraphStateJson} 的 JSON 形态整行覆盖写入（{@code INSERT OR
 * REPLACE}，与 Python 侧"遇冲突删旧插新"的原子 upsert 语义一致）。
 *
 * <p>并发模型：sqlite 文件库同一时刻只允许一个写事务，因此所有操作都在
 * 同一把锁上串行执行（对应 Python AsyncSqliteSaver 的单连接串行语义）。
 */
public final class SqliteCheckpointer implements Checkpointer {

    private static final Logger LOG = LoggerFactory.getLogger(SqliteCheckpointer.class);

    private static final String DDL = """
            CREATE TABLE IF NOT EXISTS ab_thread_states (
              thread_id TEXT PRIMARY KEY,
              state_json TEXT NOT NULL,
              updated_at TEXT NOT NULL
            )""";

    private static final String UPSERT = """
            INSERT OR REPLACE INTO ab_thread_states (thread_id, state_json, updated_at)
            VALUES (?, ?, ?)""";

    private final Connection conn;
    private final Object lock = new Object();

    /** 打开（或创建）sqlite 状态库并自举建表；失败抛出由装配方决定降级与否。 */
    public SqliteCheckpointer(String path) {
        Connection connection;
        try {
            connection = DriverManager.getConnection("jdbc:sqlite:" + path);
        } catch (SQLException e) {
            throw new IllegalStateException(
                    "sqlite checkpointer: 无法打开状态库 '%s'".formatted(path), e);
        }
        this.conn = connection;
        try (Statement st = conn.createStatement()) {
            st.execute(DDL);
        } catch (SQLException e) {
            closeQuietly();
            throw new IllegalStateException("sqlite checkpointer: 建表失败", e);
        }
    }

    @Override
    public void put(String threadId, GraphState state) {
        String json = GraphStateJson.toJson(state);
        String updatedAt = LocalDateTime.now().toString();
        synchronized (lock) {
            try (PreparedStatement ps = conn.prepareStatement(UPSERT)) {
                ps.setString(1, threadId);
                ps.setString(2, json);
                ps.setString(3, updatedAt);
                ps.executeUpdate();
            } catch (SQLException e) {
                throw new IllegalStateException("sqlite checkpointer: 状态写入失败", e);
            }
        }
    }

    @Override
    public GraphState get(String threadId) {
        synchronized (lock) {
            try (PreparedStatement ps = conn.prepareStatement(
                    "SELECT state_json FROM ab_thread_states WHERE thread_id = ?")) {
                ps.setString(1, threadId);
                try (ResultSet rs = ps.executeQuery()) {
                    if (!rs.next()) {
                        return null;
                    }
                    // 损坏行（非法 JSON）返回 null：旧行当作缺失处理。
                    return GraphStateJson.fromJson(rs.getString(1));
                }
            } catch (SQLException e) {
                throw new IllegalStateException("sqlite checkpointer: 状态读取失败", e);
            }
        }
    }

    @Override
    public void deleteThread(String threadId) {
        synchronized (lock) {
            try (PreparedStatement ps = conn.prepareStatement(
                    "DELETE FROM ab_thread_states WHERE thread_id = ?")) {
                ps.setString(1, threadId);
                ps.executeUpdate();
            } catch (SQLException e) {
                throw new IllegalStateException("sqlite checkpointer: 线程删除失败", e);
            }
        }
    }

    @Override
    public List<String> listThreadIds() {
        synchronized (lock) {
            try (Statement st = conn.createStatement();
                 ResultSet rs = st.executeQuery("SELECT thread_id FROM ab_thread_states")) {
                List<String> ids = new ArrayList<>();
                while (rs.next()) {
                    ids.add(rs.getString(1));
                }
                return ids;
            } catch (SQLException e) {
                throw new IllegalStateException("sqlite checkpointer: 线程列举失败", e);
            }
        }
    }

    @Override
    public boolean healthy() {
        // 探一次读：任何 SQLException 都按不可读处理（health degraded，不崩溃）。
        synchronized (lock) {
            try (Statement st = conn.createStatement();
                 ResultSet rs = st.executeQuery("SELECT 1")) {
                return rs.next();
            } catch (SQLException e) {
                return false;
            }
        }
    }

    /** 尽力而为地拆除（对应 Python close_checkpointer 的尽力而为语义）。 */
    @Override
    public void close() {
        closeQuietly();
    }

    private void closeQuietly() {
        try {
            conn.close();
        } catch (SQLException e) {
            LOG.warn("sqlite checkpointer: 关闭连接失败: {}", e.getMessage());
        }
    }
}
