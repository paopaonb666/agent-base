package io.github.paopaonb666.agentbase.core.graph;

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
import java.util.ArrayList;
import java.util.List;

/**
 * MySQL 后端的对话状态存储：每操作一条 {@code DriverManager} 短连接。
 *
 * <p>连接模型（与 Python 侧的差异及依据）：Python 用 aiomysql 持有一条
 * 长驻连接，因此需要周期性 keepalive ping 规避 {@code wait_timeout} 掐断
 * （extensions/memory.py 的 {@code _mysql_keepalive}，30 分钟一次）。基座
 * 对状态表的访问是低频的（每轮一到两次 upsert/get），Java 侧改为每操作
 * 开一条短连接、用完即关——连接根本不会存活到 {@code wait_timeout}，
 * keepalive 机制随之不需要。自有表 {@code ab_thread_states}，不与官方
 * langgraph saver 的表互通。
 */
public final class MysqlCheckpointer implements Checkpointer {

    private static final Logger LOG = LoggerFactory.getLogger(MysqlCheckpointer.class);

    private static final String DDL = """
            CREATE TABLE IF NOT EXISTS ab_thread_states (
              thread_id VARCHAR(190) PRIMARY KEY,
              state_json LONGTEXT,
              updated_at DATETIME
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""";

    private static final String UPSERT = """
            REPLACE INTO ab_thread_states (thread_id, state_json, updated_at)
            VALUES (?, ?, ?)""";

    private final String url;
    private final String user;
    private final String password;
    /** 建表是否已就绪（惰性执行一次；DDL 幂等，并发重复执行无害）。 */
    private volatile boolean tableReady = false;

    /** 连接参数全部来自配置节 {@code CHECKPOINTER_MYSQL_*}。 */
    public MysqlCheckpointer(Settings.CheckpointerSettings s) {
        this.url = ("jdbc:mysql://%s:%d/%s?useUnicode=true&characterEncoding=utf8"
                + "&allowPublicKeyRetrieval=true&useSSL=false")
                .formatted(s.mysqlHost, s.mysqlPort, s.mysqlDatabase);
        this.user = s.mysqlUser;
        this.password = s.mysqlPassword.value();
    }

    /**
     * 启动期自检：建连 + 建表，失败立即抛出（对应 Python 构建期的
     * {@code connect + saver.setup()}——坏配置在启动时快速失败，而不是
     * 会话中途以原始的 SQLException 爆出）。
     */
    public void init() {
        try (Connection conn = open()) {
            ensureTable(conn);
        } catch (SQLException e) {
            throw new IllegalStateException(
                    "mysql checkpointer: 启动自检失败（建连或建表不可用）", e);
        }
    }

    private Connection open() throws SQLException {
        // #161：短连接换共享 HikariCP 池（与 core store/记忆 store/工具
        // 日志同批合并——连接风暴与 max_connections 打爆的根因）。
        return io.github.paopaonb666.agentbase.core.storage.MySqlPools
                .pool(url, user, password).getConnection();
    }

    private void ensureTable(Connection conn) throws SQLException {
        if (tableReady) {
            return;
        }
        try (Statement st = conn.createStatement()) {
            st.execute(DDL);
        }
        tableReady = true;
    }

    @Override
    public void put(String threadId, GraphState state) {
        String json = GraphStateJson.toJson(state);
        LocalDateTime updatedAt = LocalDateTime.now();
        try (Connection conn = open()) {
            ensureTable(conn);
            try (PreparedStatement ps = conn.prepareStatement(UPSERT)) {
                ps.setString(1, threadId);
                ps.setString(2, json);
                ps.setObject(3, updatedAt);
                ps.executeUpdate();
            }
        } catch (SQLException e) {
            throw new IllegalStateException("mysql checkpointer: 状态写入失败", e);
        }
    }

    @Override
    public GraphState get(String threadId) {
        try (Connection conn = open()) {
            ensureTable(conn);
            try (PreparedStatement ps = conn.prepareStatement(
                    "SELECT state_json FROM ab_thread_states WHERE thread_id = ?")) {
                ps.setString(1, threadId);
                try (ResultSet rs = ps.executeQuery()) {
                    if (!rs.next()) {
                        return null;
                    }
                    // 分类解析（#200/#235）：损坏/未知版本返回 null 当缺失，
                    // 但 ERROR 可见（MySQL 侧隔离表随后续迁移机制一起做）。
                    String raw = rs.getString(1);
                    GraphStateJson.ParseResult parsed = GraphStateJson.fromJsonDetail(raw);
                    if (GraphStateJson.ParseResult.OK.equals(parsed.kind())) {
                        return parsed.state();
                    }
                    LOG.error("mysql checkpointer: 快照无法解析（thread={}，kind={}，{}）"
                            + "——原行保留", threadId, parsed.kind(), parsed.detail());
                    return null;
                }
            }
        } catch (SQLException e) {
            throw new IllegalStateException("mysql checkpointer: 状态读取失败", e);
        }
    }

    @Override
    public void deleteThread(String threadId) {
        try (Connection conn = open()) {
            ensureTable(conn);
            try (PreparedStatement ps = conn.prepareStatement(
                    "DELETE FROM ab_thread_states WHERE thread_id = ?")) {
                ps.setString(1, threadId);
                ps.executeUpdate();
            }
        } catch (SQLException e) {
            throw new IllegalStateException("mysql checkpointer: 线程删除失败", e);
        }
    }

    @Override
    public List<String> listThreadIds() {
        try (Connection conn = open()) {
            ensureTable(conn);
            try (Statement st = conn.createStatement();
                 ResultSet rs = st.executeQuery("SELECT thread_id FROM ab_thread_states")) {
                List<String> ids = new ArrayList<>();
                while (rs.next()) {
                    ids.add(rs.getString(1));
                }
                return ids;
            }
        } catch (SQLException e) {
            throw new IllegalStateException("mysql checkpointer: 线程列举失败", e);
        }
    }

    /**
     * updated_at 是 DATETIME 列（put 写 LocalDateTime）：读回转 epoch
     * 毫秒。缺席/读失败返回 0（时间未知），不报错——降级列表路径按
     * 「时间未知」降级而不是把半截错误冒给用户。
     */
    @Override
    public long updatedAt(String threadId) {
        try (Connection conn = open()) {
            ensureTable(conn);
            try (PreparedStatement ps = conn.prepareStatement(
                    "SELECT updated_at FROM ab_thread_states WHERE thread_id = ?")) {
                ps.setString(1, threadId);
                try (ResultSet rs = ps.executeQuery()) {
                    if (!rs.next()) {
                        return 0L;
                    }
                    java.time.LocalDateTime ts = rs.getObject("updated_at",
                            java.time.LocalDateTime.class);
                    return ts == null ? 0L
                            : ts.atZone(java.time.ZoneId.systemDefault())
                                    .toInstant().toEpochMilli();
                }
            }
        } catch (SQLException e) {
            return 0L;
        }
    }

    @Override
    public boolean healthy() {
        // 探一次读：任何 SQLException 都按不可读处理（health degraded，不崩溃）。
        try (Connection conn = open();
             Statement st = conn.createStatement();
             ResultSet rs = st.executeQuery("SELECT 1")) {
            return rs.next();
        } catch (SQLException e) {
            return false;
        }
    }

    /** 无常驻资源可释放（每操作短连接用完即关）。 */
    @Override
    public void close() {
        LOG.debug("mysql checkpointer: close（无常驻连接，空操作）");
    }
}
