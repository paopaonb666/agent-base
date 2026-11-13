package io.github.paopaonb666.agentbase.extensions.toollog;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.atomic.AtomicLong;

import io.github.paopaonb666.agentbase.core.config.Settings;

/**
 * 工具调用审计记录器的三后端实现 + 装配工厂（对应 Python 侧
 * {@code extensions/toollog.py} 的三个 Recorder 与
 * {@code build_tool_call_recorder}）。
 *
 * <p>memory → 进程内环形缓冲；sqlite → 专属写线程消化插入（写队列有界，
 * 满即丢样本计数——审计丢样本方向：宁可丢也不打断调用线程；与对话状态
 * 同一文件）；mysql → 每次写一条短连接（对应 Python 侧的每次连接语义，
 * 审计是低频旁路流量）。总开关关闭或后端未知时工厂返回 null。
 */
public final class ToolCallLoggers {

    private static final Logger LOG = LoggerFactory.getLogger(ToolCallLoggers.class);

    private static final String SQLITE_DDL = """
            CREATE TABLE IF NOT EXISTS tool_call_records (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              call_id VARCHAR(32) NOT NULL DEFAULT '',
              thread_id VARCHAR(190) NOT NULL DEFAULT '',
              module VARCHAR(64) NOT NULL DEFAULT '',
              request_id VARCHAR(64) NOT NULL DEFAULT '',
              tool VARCHAR(128) NOT NULL,
              args_json TEXT,
              result_text TEXT,
              status VARCHAR(16) NOT NULL,
              error_text TEXT,
              duration_ms INTEGER,
              created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """;

    private static final String MYSQL_DDL = """
            CREATE TABLE IF NOT EXISTS tool_call_records (
              id BIGINT NOT NULL AUTO_INCREMENT,
              call_id VARCHAR(32) NOT NULL DEFAULT '',
              thread_id VARCHAR(190) NOT NULL DEFAULT '',
              module VARCHAR(64) NOT NULL DEFAULT '',
              request_id VARCHAR(64) NOT NULL DEFAULT '',
              tool VARCHAR(128) NOT NULL,
              args_json TEXT,
              result_text TEXT,
              status VARCHAR(16) NOT NULL,
              error_text TEXT,
              duration_ms INT,
              created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
              PRIMARY KEY (id),
              KEY idx_tcr_thread (thread_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """;

    private static final String INSERT_SQL =
            "INSERT INTO tool_call_records (call_id, thread_id, module, request_id, tool,"
                    + " args_json, result_text, status, error_text, duration_ms)"
                    + " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)";

    private static final String SELECT_SQL =
            "SELECT call_id, thread_id, module, request_id, tool, args_json, result_text,"
                    + " status, error_text, duration_ms FROM tool_call_records"
                    + " WHERE thread_id = ? ORDER BY id DESC LIMIT ?";

    private ToolCallLoggers() {}

    /** 进程内环形缓冲；memory 后端（重启即丢）与测试用。 */
    public static final class MemoryToolCallRecorder implements ToolCallRecorder {
        private final Deque<ToolCallRecord> records = new ArrayDeque<>();
        private final Object lock = new Object();

        @Override
        public void record(ToolCallRecord record) {
            synchronized (lock) {
                records.addLast(record);
                while (records.size() > ToolCallRecord.MEMORY_MAX_RECORDS) {
                    records.removeFirst();
                }
            }
        }

        @Override
        public List<Map<String, Object>> listForThread(String threadId, int limit) {
            List<ToolCallRecord> matched = new ArrayList<>();
            synchronized (lock) {
                for (ToolCallRecord r : records) {
                    if (r.threadId().equals(threadId)) {
                        matched.add(r);
                    }
                }
            }
            return matched.subList(Math.max(0, matched.size() - limit), matched.size()).stream()
                    .map(ToolCallRecord::asDict)
                    .toList();
        }
    }

    /**
     * sqlite 审计表：与对话状态同一文件，专属写线程消化插入。
     *
     * <p>写队列有界（#266）：{@code record()} 只 {@code offer}，队列满即
     * 丢弃并计入 {@link #droppedRecordCount()}。审计是旁路数据，方向是
     * <b>丢样本优于打断调用线程</b>——绝不阻塞、绝不抛错拖垮对话主链路。
     */
    public static final class SqliteToolCallRecorder implements ToolCallRecorder {

        /** 写队列容量（#266 有界化）：超限即丢（审计丢样本，不打断调用线程）。 */
        public static final int QUEUE_CAPACITY = 1000;

        private final BlockingQueue<ToolCallRecord> queue = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
        private final AtomicLong droppedCount = new AtomicLong();
        private final Connection conn;
        private final Thread writer;

        public SqliteToolCallRecorder(String path) {
            try {
                this.conn = DriverManager.getConnection("jdbc:sqlite:" + path);
                try (Statement st = conn.createStatement()) {
                    st.execute(SQLITE_DDL);
                    st.execute("CREATE INDEX IF NOT EXISTS idx_tcr_thread ON tool_call_records (thread_id)");
                }
            } catch (SQLException e) {
                throw new IllegalStateException("toollog: sqlite 初始化失败: " + e.getMessage(), e);
            }
            this.writer = new Thread(this::writerLoop, "toollog-sqlite");
            this.writer.setDaemon(true);
            this.writer.start();
        }

        @Override
        public void record(ToolCallRecord record) {
            // offer 非阻塞：队列满即丢弃并计数（#266 审计丢样本方向），
            // 绝不拖垮调用线程。
            if (!queue.offer(record)) {
                droppedCount.incrementAndGet();
            }
        }

        /** 累计因队列满被丢弃的记录数（审计丢样本的可观测口径）。 */
        public long droppedRecordCount() {
            return droppedCount.get();
        }

        private void writerLoop() {
            while (true) {
                ToolCallRecord record;
                try {
                    record = queue.take();
                } catch (InterruptedException e) {
                    // close() 触发的退出：先尽力排空队列剩余再返回，
                    // close 前已 record() 的尾部记录不因关闭丢失。
                    drainRemaining();
                    return;
                }
                // 微批提交：单条自动提交 = 每行一次 fsync（实测慢盘上
                // 15s 仅落 772/1001，回归门被磁盘状态卡爆）。同一批连带
                // 排空队列再 commit——审计语义不变（未丢弃的记录最终全部
                // 落库），吞吐提一个量级。坏记录跳过不回滚整批（与原
                // 逐条 insertQuietly 的失败语义一致）。
                try {
                    conn.setAutoCommit(false);
                    try {
                        insert(record);
                        while ((record = queue.poll()) != null) {
                            insertQuietly(record);
                        }
                        conn.commit();
                    } catch (SQLException e) {
                        rollbackQuietly();
                        LOG.error("toollog: sqlite batch insert failed", e);
                    } finally {
                        conn.setAutoCommit(true);
                    }
                } catch (SQLException e) {
                    LOG.error("toollog: sqlite autocommit switch failed", e);
                }
            }
        }

        private void rollbackQuietly() {
            try {
                conn.rollback();
            } catch (SQLException e) {
                LOG.debug("toollog: rollback failed", e);
            }
        }

        /** 中断后的尽力排空：把队列剩余记录写完（写失败只记日志，不抛出）。 */
        private void drainRemaining() {
            ToolCallRecord record;
            if (queue.isEmpty()) {
                return;
            }
            try {
                conn.setAutoCommit(false);
            } catch (SQLException e) {
                LOG.error("toollog: sqlite autocommit switch failed", e);
                return;
            }
            try {
                while ((record = queue.poll()) != null) {
                    insertQuietly(record);
                }
                conn.commit();
            } catch (SQLException e) {
                rollbackQuietly();
                LOG.error("toollog: sqlite drain commit failed", e);
            } finally {
                try {
                    conn.setAutoCommit(true);
                } catch (SQLException e) {
                    LOG.debug("toollog: autocommit restore failed", e);
                }
            }
        }

        private void insertQuietly(ToolCallRecord record) {
            try {
                insert(record);
            } catch (SQLException e) {
                LOG.error("toollog: sqlite insert failed", e);
            }
        }

        private void insert(ToolCallRecord r) throws SQLException {
            try (PreparedStatement ps = conn.prepareStatement(INSERT_SQL)) {
                ps.setString(1, r.callId());
                ps.setString(2, r.threadId());
                ps.setString(3, r.module());
                ps.setString(4, r.requestId());
                ps.setString(5, r.tool());
                ps.setString(6, r.argsJson());
                ps.setString(7, r.resultText());
                ps.setString(8, r.status());
                ps.setString(9, r.errorText());
                ps.setLong(10, r.durationMs());
                ps.executeUpdate();
            }
        }

        @Override
        public List<Map<String, Object>> listForThread(String threadId, int limit) {
            List<Object[]> rows = new ArrayList<>();
            try (PreparedStatement ps = conn.prepareStatement(SELECT_SQL)) {
                ps.setString(1, threadId);
                ps.setInt(2, limit);
                try (ResultSet rs = ps.executeQuery()) {
                    while (rs.next()) {
                        rows.add(rowOf(rs));
                    }
                }
            } catch (SQLException e) {
                LOG.error("toollog: sqlite query failed", e);
                return List.of();
            }
            // DESC 查询 + 反转 = 时间升序（与 Python 侧一致）。
            List<Map<String, Object>> out = new ArrayList<>();
            for (int i = rows.size() - 1; i >= 0; i--) {
                out.add(ToolCallRecord.rowToDict(rows.get(i)));
            }
            return out;
        }

        private static Object[] rowOf(ResultSet rs) throws SQLException {
            return new Object[]{
                    rs.getString(1), rs.getString(2), rs.getString(3), rs.getString(4),
                    rs.getString(5), rs.getString(6), rs.getString(7), rs.getString(8),
                    rs.getString(9), rs.getObject(10)};
        }

        @Override
        public void close() {
            // 尽力排空再退（对应 Python 侧 aclose 的排空语义）：interrupt 让
            // 写线程从 take() 退出，写线程退出前把队列剩余记录写完
            // （见 writerLoop 的 drainRemaining）；join 限时 5s 兜底防卡死，
            // 超时则放弃排空直接关连接（残余记录丢弃——丢样本优于挂死）。
            writer.interrupt();
            try {
                writer.join(5000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            try {
                conn.close();
            } catch (SQLException e) {
                LOG.debug("toollog: sqlite close failed", e);
            }
        }
    }

    /** MySQL 审计表：每次写一条短连接（审计是低频旁路流量）。 */
    public static final class MySqlToolCallRecorder implements ToolCallRecorder {
        private final Settings settings;
        private volatile boolean tableReady = false;

        public MySqlToolCallRecorder(Settings settings) {
            this.settings = settings;
        }

        private Connection connect() throws SQLException {
            var s = settings.checkpointer;
            // #161：短连接换共享 HikariCP 池（与 checkpointer/core store/
            // 记忆 store 同批合并）；A-5：URL 规范串单源 MySqlPools.jdbcUrl
            // ——此前拼裸 URL（无 query 参数），同库同账号也 split 第二池。
            return io.github.paopaonb666.agentbase.core.storage.MySqlPools.pool(
                    io.github.paopaonb666.agentbase.core.storage.MySqlPools.jdbcUrl(
                            s.mysqlHost, s.mysqlPort, s.mysqlDatabase),
                    s.mysqlUser, s.mysqlPassword.value()).getConnection();
        }

        @Override
        public void record(ToolCallRecord record) {
            try (Connection conn = connect()) {
                if (!tableReady) {
                    try (Statement st = conn.createStatement()) {
                        st.execute(MYSQL_DDL);
                    }
                    tableReady = true;
                }
                try (PreparedStatement ps = conn.prepareStatement(INSERT_SQL)) {
                    ps.setString(1, record.callId());
                    ps.setString(2, record.threadId());
                    ps.setString(3, record.module());
                    ps.setString(4, record.requestId());
                    ps.setString(5, record.tool());
                    ps.setString(6, record.argsJson());
                    ps.setString(7, record.resultText());
                    ps.setString(8, record.status());
                    ps.setString(9, record.errorText());
                    ps.setLong(10, record.durationMs());
                    ps.executeUpdate();
                }
            } catch (SQLException e) {
                LOG.error("toollog: mysql insert failed", e);
            }
        }

        @Override
        public List<Map<String, Object>> listForThread(String threadId, int limit) {
            List<Object[]> rows = new ArrayList<>();
            try (Connection conn = connect();
                 PreparedStatement ps = conn.prepareStatement(SELECT_SQL)) {
                ps.setString(1, threadId);
                ps.setInt(2, limit);
                try (ResultSet rs = ps.executeQuery()) {
                    while (rs.next()) {
                        rows.add(new Object[]{
                                rs.getString(1), rs.getString(2), rs.getString(3), rs.getString(4),
                                rs.getString(5), rs.getString(6), rs.getString(7), rs.getString(8),
                                rs.getString(9), rs.getObject(10)});
                    }
                }
            } catch (SQLException e) {
                LOG.error("toollog: mysql query failed", e);
                return List.of();
            }
            List<Map<String, Object>> out = new ArrayList<>();
            for (int i = rows.size() - 1; i >= 0; i--) {
                out.add(ToolCallRecord.rowToDict(rows.get(i)));
            }
            return out;
        }
    }

    /** 按 settings 装配审计记录器；总开关关闭或后端未知时返回 null。 */
    public static ToolCallRecorder build(Settings settings) {
        if (!settings.toolkit.callLogEnabled) {
            return null;
        }
        String backend = settings.checkpointer.backend;
        return switch (backend) {
            case "memory" -> new MemoryToolCallRecorder();
            case "sqlite" -> new SqliteToolCallRecorder(settings.checkpointer.sqlitePath);
            case "mysql" -> new MySqlToolCallRecorder(settings);
            default -> {
                LOG.warn("toollog: unknown checkpointer backend '{}'; tool call audit disabled", backend);
                yield null;
            }
        };
    }
}
