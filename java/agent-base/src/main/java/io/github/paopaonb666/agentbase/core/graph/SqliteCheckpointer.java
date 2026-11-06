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

    /**
     * 损坏快照隔离表（#235）：损坏行移出主表（不再每次 get 都重新炸）
     * 但保留原文供人工核查/工具恢复——静默当「无历史」会让用户会话无声
     * 丢失。unknown 版本的行不进隔离表（拒载保留原行，等迁移函数）。
     */
    private static final String QUARANTINE_DDL = """
            CREATE TABLE IF NOT EXISTS ab_thread_states_quarantine (
              thread_id TEXT NOT NULL,
              state_json TEXT NOT NULL,
              reason TEXT NOT NULL,
              detected_at TEXT NOT NULL
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
            st.execute(QUARANTINE_DDL);
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
                    String raw = rs.getString(1);
                    GraphStateJson.ParseResult parsed = GraphStateJson.fromJsonDetail(raw);
                    switch (parsed.kind()) {
                        case GraphStateJson.ParseResult.OK -> {
                            return parsed.state();
                        }
                        case GraphStateJson.ParseResult.VERSION_REFUSED -> {
                            // 未知版本：拒载但保留原行（等迁移函数处理），ERROR 可见。
                            LOG.error("sqlite checkpointer: 快照版本拒载（thread={}，{}）"
                                    + "——原行保留，等待迁移", threadId, parsed.detail());
                            return null;
                        }
                        default -> {
                            // 损坏行（非法 JSON）：隔离到 quarantine 表并从主表
                            // 移除（#235）——不再每次 get 都命中，原文可查。
                            quarantine(threadId, raw, parsed.detail());
                            return null;
                        }
                    }
                }
            } catch (SQLException e) {
                throw new IllegalStateException("sqlite checkpointer: 状态读取失败", e);
            }
        }
    }

    /** 损坏行隔离（#235）：入 quarantine 表 + 从主表删除；失败仅告警（读路径不炸）。 */
    private void quarantine(String threadId, String raw, String reason) {
        LOG.error("sqlite checkpointer: 快照损坏已隔离（thread={}，reason={}）"
                + "——原文保留在 ab_thread_states_quarantine", threadId, reason);
        try (PreparedStatement insert = conn.prepareStatement(
                "INSERT INTO ab_thread_states_quarantine (thread_id, state_json, reason, detected_at)"
                        + " VALUES (?, ?, ?, ?)");
             PreparedStatement delete = conn.prepareStatement(
                     "DELETE FROM ab_thread_states WHERE thread_id = ?")) {
            insert.setString(1, threadId);
            insert.setString(2, raw);
            insert.setString(3, reason);
            insert.setString(4, LocalDateTime.now().toString());
            insert.executeUpdate();
            delete.setString(1, threadId);
            delete.executeUpdate();
        } catch (SQLException e) {
            LOG.error("sqlite checkpointer: 损坏行隔离失败（主表行保留）：{}", e.toString());
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

    /**
     * updated_at 列是 ISO 文本（{@code LocalDateTime.now().toString()}）：
     * 解析回 epoch 毫秒（系统时区，对齐 Python naive fromisoformat 的
     * timestamp() 语义）。解析失败/缺席返回 0（时间未知），不报错。
     */
    @Override
    public long updatedAt(String threadId) {
        synchronized (lock) {
            try (PreparedStatement ps = conn.prepareStatement(
                    "SELECT updated_at FROM ab_thread_states WHERE thread_id = ?")) {
                ps.setString(1, threadId);
                try (ResultSet rs = ps.executeQuery()) {
                    if (!rs.next()) {
                        return 0L;
                    }
                    return LocalDateTime.parse(rs.getString(1))
                            .atZone(java.time.ZoneId.systemDefault())
                            .toInstant().toEpochMilli();
                }
            } catch (SQLException | java.time.format.DateTimeParseException e) {
                return 0L;
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
