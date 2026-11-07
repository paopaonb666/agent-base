package io.github.paopaonb666.agentbase.extensions.filestore;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.sql.Timestamp;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import io.github.paopaonb666.agentbase.core.config.Settings;

/**
 * 上传附件存储的三后端实现 + 装配工厂（对应 Python 侧
 * {@code extensions/filestore.py} 的三个 Store 与
 * {@code build_uploaded_file_store}）。
 *
 * <p>设计要点（继承自 Python 侧模块 docstring）：
 * <ul>
 *   <li><b>原始字节入库</b>（BLOB/LONGBLOB）：与提取文本并存——未来可以
 *       换更好的解析器重新解析、可以扩展下载端点；统一收进存储后端，
 *       零磁盘管理。</li>
 *   <li><b>存储后端跟随 checkpointer</b>（memory/sqlite/mysql），与
 *       toollog 同款装配语义；建表 DDL 与 alembic 0003 + 0006 的合并
 *       形态一致（含 {@code user_id} 属主列），未跑迁移的库也能自举。</li>
 *   <li><b>孤儿清理</b>：thread_id 为空且 created_at 超过
 *       {@link UploadedFileStore#ORPHAN_MAX_AGE_HOURS}（24h）的行由
 *       {@code purgeOrphans} 机会式删除。</li>
 * </ul>
 *
 * <p>memory → 进程内字典；sqlite → 单连接 + 方法级 synchronized（对应
 * Python aiosqlite 持久连接 + 事件循环串行化）；mysql → DriverManager
 * 短连接（对应 aiomysql 按操作连接，写入频率低，无需连接池）。
 * 后端未知时工厂返回 null（附件功能停用）。
 */
public final class UploadedFileStores {

    private static final Logger LOG = LoggerFactory.getLogger(UploadedFileStores.class);

    /** sqlite 落库的时间文本格式（对齐 Python {@code isoformat(sep=" ", timespec="seconds")}）。 */
    private static final DateTimeFormatter SECONDS_FORMAT =
            DateTimeFormatter.ofPattern("uuuu-MM-dd HH:mm:ss");

    /**
     * sqlite 建表 DDL（含运行时自迁移补的 user_id 列；与 alembic 0003
     * 的 mysql 形态同构，类型按 sqlite 习惯）。幂等可重复执行。
     */
    private static final String SQLITE_DDL = """
            CREATE TABLE IF NOT EXISTS uploaded_files (
              file_id VARCHAR(32) PRIMARY KEY,
              thread_id VARCHAR(190) NOT NULL DEFAULT '',
              module VARCHAR(64) NOT NULL DEFAULT '',
              user_id VARCHAR(64) NOT NULL DEFAULT '',
              filename VARCHAR(255) NOT NULL,
              format VARCHAR(16) NOT NULL,
              pages INTEGER,
              paragraphs INTEGER,
              truncated INTEGER NOT NULL DEFAULT 0,
              text_len INTEGER NOT NULL DEFAULT 0,
              warning VARCHAR(500) NOT NULL DEFAULT '',
              extracted_text TEXT,
              content BLOB,
              created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """;

    /**
     * sqlite 声明式迁移（#275）：001 是基线标记——user_id 列（#144 后置）
     * 由构造期的 PRAGMA 补列分支处理（sqlite 无 ADD COLUMN IF NOT EXISTS，
     * 裸 ALTER 会让新装库失败），记入版本表使演进审计有单一事实源；
     * 后续真正的新列/索引演进追加到本清单。
     */
    private static final java.util.List<io.github.paopaonb666.agentbase.core.storage.SchemaMigrations.Migration>
            SQLITE_MIGRATIONS = java.util.List.of(
            new io.github.paopaonb666.agentbase.core.storage.SchemaMigrations.Migration(
                    "001_uploaded_files_user_id",
                    "uploaded_files 属主列基线（实际补列走构造期 PRAGMA 分支）",
                    List.of("SELECT 1")));

    /** MySQL 建表 DDL：alembic 0003 + 0006 的合并形态（user_id 列位于
     * module 之后，与 0006 的 {@code AFTER `module`} 一致）。幂等可重复
     * 执行，未跑迁移的库首写时自举。 */
    private static final String MYSQL_DDL = """
            CREATE TABLE IF NOT EXISTS uploaded_files (
              file_id VARCHAR(32) PRIMARY KEY,
              thread_id VARCHAR(190) NOT NULL DEFAULT '',
              module VARCHAR(64) NOT NULL DEFAULT '',
              user_id VARCHAR(64) NOT NULL DEFAULT '',
              filename VARCHAR(255) NOT NULL,
              format VARCHAR(16) NOT NULL,
              pages INT NULL,
              paragraphs INT NULL,
              truncated SMALLINT NOT NULL DEFAULT 0,
              text_len INT NOT NULL DEFAULT 0,
              warning VARCHAR(500) NOT NULL DEFAULT '',
              extracted_text LONGTEXT,
              content LONGBLOB,
              created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """;

    /** SELECT 列清单（sqlite 形态，13 列，不含 created_at——对齐 Python 侧）。 */
    private static final String SQLITE_SELECT_COLUMNS =
            "file_id, thread_id, module, user_id, filename, format, pages, paragraphs,"
                    + " truncated, text_len, warning, extracted_text, content";

    /** SELECT 列清单（MySQL 形态，14 列，含 created_at——对齐 Python 侧）。 */
    private static final String MYSQL_SELECT_COLUMNS =
            "file_id, thread_id, module, user_id, filename, format, pages, paragraphs,"
                    + " truncated, text_len, warning, extracted_text, content, created_at";

    private static final String SQLITE_SAVE_SQL =
            "INSERT OR REPLACE INTO uploaded_files (file_id, thread_id, module, user_id,"
                    + " filename, format, pages, paragraphs, truncated, text_len, warning,"
                    + " extracted_text, content, created_at)"
                    + " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)";

    private static final String MYSQL_INSERT_SQL =
            "INSERT INTO uploaded_files (file_id, thread_id, module, user_id,"
                    + " filename, format, pages, paragraphs, truncated, text_len, warning,"
                    + " extracted_text, content)"
                    + " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)";

    private UploadedFileStores() {}

    /** 按 settings 装配附件存储（跟随 checkpointer 后端）；未知后端返回 null。 */
    public static UploadedFileStore build(Settings settings) {
        String backend = settings.checkpointer.backend;
        return switch (backend) {
            case "memory" -> new MemoryUploadedFileStore();
            case "sqlite" -> new SqliteUploadedFileStore(settings.checkpointer.sqlitePath);
            case "mysql" -> new MySqlUploadedFileStore(settings);
            default -> {
                LOG.warn("filestore: unknown checkpointer backend '{}'; attachments disabled", backend);
                yield null;
            }
        };
    }

    // ── 行映射与参数填充（三后端共用） ──────────────────────────────

    /** 文本列统一空串归一（对应 Python {@code row[i] or ""}）。 */
    private static String textOrEmpty(String value) {
        return value == null ? "" : value;
    }

    /** 截断列 SMALLINT/INTEGER → boolean。 */
    private static int boolToInt(boolean value) {
        return value ? 1 : 0;
    }

    /** epoch 秒 → 本地时间 "YYYY-MM-DD HH:MM:SS"（对齐 Python fromtimestamp + isoformat）。 */
    private static String secondsText(double epochSeconds) {
        return LocalDateTime.ofInstant(
                Instant.ofEpochMilli((long) Math.floor(epochSeconds * 1000.0)),
                ZoneId.systemDefault()).format(SECONDS_FORMAT);
    }

    /** 当前时间（epoch 秒，浮点；对应 Python {@code time.time()}）。 */
    private static double nowSeconds() {
        return System.currentTimeMillis() / 1000.0;
    }

    /** 填充 save 的占位符（不含 created_at：sqlite 14 列多一格由调用方补）。 */
    private static int fillColumns(PreparedStatement ps, UploadedFileInfo info, int startIndex)
            throws SQLException {
        int i = startIndex;
        ps.setString(i++, info.fileId());
        ps.setString(i++, info.threadId());
        ps.setString(i++, info.module());
        ps.setString(i++, info.userId());
        ps.setString(i++, info.filename());
        ps.setString(i++, info.format());
        if (info.pages() == null) {
            ps.setNull(i++, java.sql.Types.INTEGER);
        } else {
            ps.setInt(i++, info.pages());
        }
        if (info.paragraphs() == null) {
            ps.setNull(i++, java.sql.Types.INTEGER);
        } else {
            ps.setInt(i++, info.paragraphs());
        }
        ps.setInt(i++, boolToInt(info.truncated()));
        ps.setInt(i++, info.textLen());
        ps.setString(i++, info.warning());
        ps.setString(i++, info.extractedText());
        ps.setBytes(i++, info.content());
        return i;
    }

    /**
     * 进程内字典；memory 后端（重启即丢）与测试用。方法级 synchronized
     * 保证并发安全（对应 Python 事件循环的串行化语义）。
     */
    public static final class MemoryUploadedFileStore implements UploadedFileStore {

        private final Map<String, UploadedFileInfo> files = new LinkedHashMap<>();

        @Override
        public synchronized void save(UploadedFileInfo info) {
            files.put(info.fileId(), info);
        }

        @Override
        public synchronized List<UploadedFileInfo> getMany(List<String> fileIds) {
            // 按 get_many 的请求顺序返回，注入顺序与前端附件列表一致。
            List<UploadedFileInfo> found = new ArrayList<>();
            for (String fid : fileIds) {
                UploadedFileInfo info = files.get(fid);
                if (info != null) {
                    found.add(info);
                }
            }
            return found;
        }

        @Override
        public synchronized void bindThread(List<String> fileIds, String threadId) {
            for (String fid : fileIds) {
                UploadedFileInfo info = files.get(fid);
                if (info != null) {
                    // record 不可变：生成替换副本（对应 dataclasses.replace）。
                    files.put(fid, info.withThreadId(threadId));
                }
            }
        }

        @Override
        public synchronized List<UploadedFileInfo> listForUser(String userId, String module, int limit) {
            List<UploadedFileInfo> matched = new ArrayList<>();
            for (UploadedFileInfo info : files.values()) {
                if (info.userId().equals(userId) && (module == null || info.module().equals(module))) {
                    matched.add(info);
                }
            }
            // 创建时间倒序（稳定排序：并列时保持插入顺序，与 Python sort 一致）。
            matched.sort(Comparator.comparingDouble(UploadedFileInfo::createdAt).reversed());
            int capped = Math.min(matched.size(), Math.max(1, limit));
            return List.copyOf(matched.subList(0, capped));
        }

        @Override
        public synchronized void deleteForThread(String threadId) {
            files.values().removeIf(info -> info.threadId().equals(threadId));
        }

        @Override
        public synchronized boolean deleteFile(String fileId) {
            return files.remove(fileId) != null;
        }

        @Override
        public synchronized int purgeOrphans(int maxAgeHours) {
            double cutoff = nowSeconds() - maxAgeHours * 3600.0;
            List<String> stale = new ArrayList<>();
            for (Map.Entry<String, UploadedFileInfo> e : files.entrySet()) {
                if (e.getValue().threadId().isEmpty() && e.getValue().createdAt() < cutoff) {
                    stale.add(e.getKey());
                }
            }
            for (String fid : stale) {
                files.remove(fid);
            }
            return stale.size();
        }
    }

    /**
     * sqlite 附件表：与对话状态同一文件，单 JDBC 连接 + synchronized
     * （对应 Python aiosqlite 持久连接 + 事件循环串行化）。建表幂等，
     * 旧库缺 user_id 列时运行时自迁移（sqlite 无 alembic）。
     */
    public static final class SqliteUploadedFileStore implements UploadedFileStore {

        private final Connection conn;

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
                            + id.replace("'", "''") + "', '"
                            + java.time.LocalDateTime.now() + "')");
                }
            }

            @Override
            public java.util.List<String> appliedVersions(String table) throws SQLException {
                java.util.List<String> ids = new java.util.ArrayList<>();
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

        public SqliteUploadedFileStore(String path) {            Connection opened = null;
            try {
                opened = DriverManager.getConnection("jdbc:sqlite:" + path);
                try (Statement st = opened.createStatement()) {
                    st.execute(SQLITE_DDL);
                    // 旧库补列（sqlite 无 alembic，运行时自迁移）：user_id 是后加的。
                    boolean hasUserId = false;
                    try (ResultSet rs = st.executeQuery("PRAGMA table_info(uploaded_files)")) {
                        while (rs.next()) {
                            if ("user_id".equals(rs.getString("name"))) {
                                hasUserId = true;
                                break;
                            }
                        }
                    }
                    if (!hasUserId) {
                        st.execute("ALTER TABLE uploaded_files ADD COLUMN"
                                + " user_id VARCHAR(64) NOT NULL DEFAULT ''");
                    }
                    st.execute("CREATE INDEX IF NOT EXISTS idx_uf_thread"
                            + " ON uploaded_files (thread_id)");
                }
                this.conn = opened;
                // #275：声明式 schema 迁移（版本表 + 按序应用，fail-fast）。
                // 001 把「旧库补 user_id」收编为迁移项——新装库表里已有该列
                // （SQLITE_DDL），语句带存在性检查天然幂等。
                io.github.paopaonb666.agentbase.core.storage.SchemaMigrations.apply(
                        "core_files", new SqliteMigrationExecutor(), SQLITE_MIGRATIONS, LOG);
            } catch (SQLException e) {
                if (opened != null) {
                    try {
                        opened.close();
                    } catch (SQLException closeEx) {
                        LOG.debug("filestore: sqlite close failed", closeEx);
                    }
                }
                throw new IllegalStateException("filestore: sqlite 初始化失败: " + e.getMessage(), e);
            }
        }

        @Override
        public synchronized void save(UploadedFileInfo info) {
            try (PreparedStatement ps = conn.prepareStatement(SQLITE_SAVE_SQL)) {
                int next = fillColumns(ps, info, 1);
                ps.setString(next, secondsText(info.createdAt()));
                ps.executeUpdate();
            } catch (SQLException e) {
                throw new IllegalStateException("filestore: sqlite save failed: " + e.getMessage(), e);
            }
        }

        @Override
        public synchronized List<UploadedFileInfo> getMany(List<String> fileIds) {
            if (fileIds.isEmpty()) {
                return List.of();
            }
            String placeholders = String.join(",", Collections.nCopies(fileIds.size(), "?"));
            Map<String, UploadedFileInfo> infos = new LinkedHashMap<>();
            try (PreparedStatement ps = conn.prepareStatement(
                    "SELECT " + SQLITE_SELECT_COLUMNS + " FROM uploaded_files"
                            + " WHERE file_id IN (" + placeholders + ")")) {
                for (int i = 0; i < fileIds.size(); i++) {
                    ps.setString(i + 1, fileIds.get(i));
                }
                try (ResultSet rs = ps.executeQuery()) {
                    while (rs.next()) {
                        UploadedFileInfo info = toInfo(rs);
                        infos.put(info.fileId(), info);
                    }
                }
            } catch (SQLException e) {
                throw new IllegalStateException("filestore: sqlite get_many failed: " + e.getMessage(), e);
            }
            List<UploadedFileInfo> found = new ArrayList<>();
            for (String fid : fileIds) {
                UploadedFileInfo info = infos.get(fid);
                if (info != null) {
                    found.add(info);
                }
            }
            return found;
        }

        @Override
        public synchronized void bindThread(List<String> fileIds, String threadId) {
            if (fileIds.isEmpty()) {
                return;
            }
            String placeholders = String.join(",", Collections.nCopies(fileIds.size(), "?"));
            try (PreparedStatement ps = conn.prepareStatement(
                    "UPDATE uploaded_files SET thread_id = ? WHERE file_id IN (" + placeholders + ")")) {
                ps.setString(1, threadId);
                for (int i = 0; i < fileIds.size(); i++) {
                    ps.setString(i + 2, fileIds.get(i));
                }
                ps.executeUpdate();
            } catch (SQLException e) {
                throw new IllegalStateException("filestore: sqlite bind_thread failed: " + e.getMessage(), e);
            }
        }

        @Override
        public synchronized void deleteForThread(String threadId) {
            try (PreparedStatement ps = conn.prepareStatement(
                    "DELETE FROM uploaded_files WHERE thread_id = ?")) {
                ps.setString(1, threadId);
                ps.executeUpdate();
            } catch (SQLException e) {
                throw new IllegalStateException("filestore: sqlite delete_for_thread failed: " + e.getMessage(), e);
            }
        }

        @Override
        public synchronized int purgeOrphans(int maxAgeHours) {
            // 与 Python 一致：本地时间字符串比较（该格式字典序即时间序）。
            String cutoff = LocalDateTime.now().minusHours(maxAgeHours).format(SECONDS_FORMAT);
            try (PreparedStatement ps = conn.prepareStatement(
                    "DELETE FROM uploaded_files WHERE thread_id = '' AND created_at < ?")) {
                ps.setString(1, cutoff);
                return Math.max(0, ps.executeUpdate());
            } catch (SQLException e) {
                throw new IllegalStateException("filestore: sqlite purge_orphans failed: " + e.getMessage(), e);
            }
        }

        @Override
        public synchronized boolean deleteFile(String fileId) {
            try (PreparedStatement ps = conn.prepareStatement(
                    "DELETE FROM uploaded_files WHERE file_id = ?")) {
                ps.setString(1, fileId);
                return ps.executeUpdate() > 0;
            } catch (SQLException e) {
                throw new IllegalStateException("filestore: sqlite delete_file failed: " + e.getMessage(), e);
            }
        }

        @Override
        public synchronized List<UploadedFileInfo> listForUser(String userId, String module, int limit) {
            StringBuilder sql = new StringBuilder("SELECT " + SQLITE_SELECT_COLUMNS
                    + " FROM uploaded_files WHERE user_id = ?");
            if (module != null) {
                sql.append(" AND module = ?");
            }
            sql.append(" ORDER BY created_at DESC LIMIT ?");
            try (PreparedStatement ps = conn.prepareStatement(sql.toString())) {
                int i = 1;
                ps.setString(i++, userId);
                if (module != null) {
                    ps.setString(i++, module);
                }
                ps.setInt(i++, Math.max(1, limit));
                List<UploadedFileInfo> out = new ArrayList<>();
                try (ResultSet rs = ps.executeQuery()) {
                    while (rs.next()) {
                        out.add(toInfo(rs));
                    }
                }
                return out;
            } catch (SQLException e) {
                throw new IllegalStateException("filestore: sqlite list_for_user failed: " + e.getMessage(), e);
            }
        }

        /** 行 → record（列序即 SELECT 列序；created_at 未查询，缺省为当前时间，对齐 Python）。 */
        private static UploadedFileInfo toInfo(ResultSet rs) throws SQLException {
            byte[] content = rs.getBytes(13);
            return new UploadedFileInfo(
                    rs.getString(1),
                    rs.getString(2),
                    rs.getString(3),
                    textOrEmpty(rs.getString(4)),
                    rs.getString(5),
                    rs.getString(6),
                    (Integer) rs.getObject(7),
                    (Integer) rs.getObject(8),
                    rs.getInt(9) != 0,
                    rs.getInt(10),
                    rs.getString(11),
                    textOrEmpty(rs.getString(12)),
                    content == null ? new byte[0] : content);
        }

        @Override
        public synchronized void close() {
            closeQuietly();
        }

        private void closeQuietly() {
            try {
                if (conn != null) {
                    conn.close();
                }
            } catch (SQLException e) {
                LOG.debug("filestore: sqlite close failed", e);
            }
        }
    }

    /**
     * MySQL 附件表：DriverManager 短连接（对应 aiomysql 按操作连接——
     * 附件写入频率低，无需连接池）。建表 DDL 首写时执行（幂等，alembic
     * 0003 + 0006 的合并形态）。
     */
    public static final class MySqlUploadedFileStore implements UploadedFileStore {

        private final Settings settings;
        private volatile boolean tableReady = false;

        public MySqlUploadedFileStore(Settings settings) {
            this.settings = settings;
        }

        private Connection connect() throws SQLException {
            var s = settings.checkpointer;
            return DriverManager.getConnection(
                    "jdbc:mysql://%s:%d/%s".formatted(s.mysqlHost, s.mysqlPort, s.mysqlDatabase),
                    s.mysqlUser, s.mysqlPassword.value());
        }

        @Override
        public void save(UploadedFileInfo info) {
            try (Connection conn = connect()) {
                if (!tableReady) {
                    try (Statement st = conn.createStatement()) {
                        st.execute(MYSQL_DDL);
                    }
                    tableReady = true;
                }
                try (PreparedStatement ps = conn.prepareStatement(MYSQL_INSERT_SQL)) {
                    fillColumns(ps, info, 1);
                    ps.executeUpdate();
                }
            } catch (SQLException e) {
                throw new IllegalStateException("filestore: mysql save failed: " + e.getMessage(), e);
            }
        }

        @Override
        public List<UploadedFileInfo> getMany(List<String> fileIds) {
            if (fileIds.isEmpty()) {
                return List.of();
            }
            String placeholders = String.join(",", Collections.nCopies(fileIds.size(), "?"));
            try (Connection conn = connect();
                 PreparedStatement ps = conn.prepareStatement(
                         "SELECT " + MYSQL_SELECT_COLUMNS + " FROM uploaded_files"
                                 + " WHERE file_id IN (" + placeholders + ")")) {
                for (int i = 0; i < fileIds.size(); i++) {
                    ps.setString(i + 1, fileIds.get(i));
                }
                return readInRequestOrder(ps, fileIds);
            } catch (SQLException e) {
                throw new IllegalStateException("filestore: mysql get_many failed: " + e.getMessage(), e);
            }
        }

        @Override
        public void bindThread(List<String> fileIds, String threadId) {
            if (fileIds.isEmpty()) {
                return;
            }
            String placeholders = String.join(",", Collections.nCopies(fileIds.size(), "?"));
            try (Connection conn = connect();
                 PreparedStatement ps = conn.prepareStatement(
                         "UPDATE uploaded_files SET thread_id = ? WHERE file_id IN ("
                                 + placeholders + ")")) {
                ps.setString(1, threadId);
                for (int i = 0; i < fileIds.size(); i++) {
                    ps.setString(i + 2, fileIds.get(i));
                }
                ps.executeUpdate();
            } catch (SQLException e) {
                throw new IllegalStateException("filestore: mysql bind_thread failed: " + e.getMessage(), e);
            }
        }

        @Override
        public void deleteForThread(String threadId) {
            try (Connection conn = connect();
                 PreparedStatement ps = conn.prepareStatement(
                         "DELETE FROM uploaded_files WHERE thread_id = ?")) {
                ps.setString(1, threadId);
                ps.executeUpdate();
            } catch (SQLException e) {
                throw new IllegalStateException("filestore: mysql delete_for_thread failed: " + e.getMessage(), e);
            }
        }

        @Override
        public int purgeOrphans(int maxAgeHours) {
            try (Connection conn = connect();
                 PreparedStatement ps = conn.prepareStatement(
                         "DELETE FROM uploaded_files WHERE thread_id = '' AND created_at < ?")) {
                ps.setTimestamp(1, Timestamp.valueOf(LocalDateTime.now().minusHours(maxAgeHours)));
                return Math.max(0, ps.executeUpdate());
            } catch (SQLException e) {
                throw new IllegalStateException("filestore: mysql purge_orphans failed: " + e.getMessage(), e);
            }
        }

        @Override
        public boolean deleteFile(String fileId) {
            try (Connection conn = connect();
                 PreparedStatement ps = conn.prepareStatement(
                         "DELETE FROM uploaded_files WHERE file_id = ?")) {
                ps.setString(1, fileId);
                return ps.executeUpdate() > 0;
            } catch (SQLException e) {
                throw new IllegalStateException("filestore: mysql delete_file failed: " + e.getMessage(), e);
            }
        }

        @Override
        public List<UploadedFileInfo> listForUser(String userId, String module, int limit) {
            StringBuilder sql = new StringBuilder("SELECT " + MYSQL_SELECT_COLUMNS
                    + " FROM uploaded_files WHERE user_id = ?");
            if (module != null) {
                sql.append(" AND module = ?");
            }
            sql.append(" ORDER BY created_at DESC LIMIT ?");
            try (Connection conn = connect();
                 PreparedStatement ps = conn.prepareStatement(sql.toString())) {
                int i = 1;
                ps.setString(i++, userId);
                if (module != null) {
                    ps.setString(i++, module);
                }
                ps.setInt(i++, Math.max(1, limit));
                List<UploadedFileInfo> out = new ArrayList<>();
                try (ResultSet rs = ps.executeQuery()) {
                    while (rs.next()) {
                        out.add(toInfo(rs));
                    }
                }
                return out;
            } catch (SQLException e) {
                throw new IllegalStateException("filestore: mysql list_for_user failed: " + e.getMessage(), e);
            }
        }

        /** 按请求顺序整理命中行（get_many 的保序语义，与 Python 侧一致）。 */
        private static List<UploadedFileInfo> readInRequestOrder(PreparedStatement ps, List<String> fileIds)
                throws SQLException {
            Map<String, UploadedFileInfo> infos = new LinkedHashMap<>();
            try (ResultSet rs = ps.executeQuery()) {
                while (rs.next()) {
                    UploadedFileInfo info = toInfo(rs);
                    infos.put(info.fileId(), info);
                }
            }
            List<UploadedFileInfo> found = new ArrayList<>();
            for (String fid : fileIds) {
                UploadedFileInfo info = infos.get(fid);
                if (info != null) {
                    found.add(info);
                }
            }
            return found;
        }

        /** 行 → record（列序即 MYSQL_SELECT_COLUMNS 列序，created_at 由 TIMESTAMP 反推 epoch 秒）。 */
        private static UploadedFileInfo toInfo(ResultSet rs) throws SQLException {
            Timestamp created = rs.getTimestamp(14);
            byte[] content = rs.getBytes(13);
            return new UploadedFileInfo(
                    rs.getString(1),
                    rs.getString(2),
                    rs.getString(3),
                    textOrEmpty(rs.getString(4)),
                    rs.getString(5),
                    rs.getString(6),
                    (Integer) rs.getObject(7),
                    (Integer) rs.getObject(8),
                    rs.getInt(9) != 0,
                    rs.getInt(10),
                    textOrEmpty(rs.getString(11)),
                    textOrEmpty(rs.getString(12)),
                    content == null ? new byte[0] : content,
                    created != null ? created.getTime() / 1000.0 : nowSeconds());
        }
    }
}
