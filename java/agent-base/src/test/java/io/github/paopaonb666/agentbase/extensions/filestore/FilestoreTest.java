package io.github.paopaonb666.agentbase.extensions.filestore;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.sql.Statement;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.Map;

import io.github.paopaonb666.agentbase.core.config.Settings;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 上传附件存储测试：save→getMany 往返（含字节相等）、bindThread 与孤儿
 * 清理、deleteForThread/deleteFile、user_id 属主过滤与运行时自迁移。
 * sqlite 用例跑在临时库文件上；旧 created_at 行用 SQL 直插提供确定性的
 * "过期"时间源（对应可注入 clock 的效果）。
 */
class FilestoreTest {

    /** 含 0 与负字节的二进制载荷：证明 BLOB 往返不经过文本编解码。 */
    private static final byte[] BYTES = {0, 1, 2, -1, 127, -128, 42};

    /** alembic 0003 的原始形态（0006 加 user_id 列之前）：自迁移用例的旧表。 */
    private static final String OLD_SHAPE_DDL = """
            CREATE TABLE uploaded_files (
              file_id VARCHAR(32) PRIMARY KEY,
              thread_id VARCHAR(190) NOT NULL DEFAULT '',
              module VARCHAR(64) NOT NULL DEFAULT '',
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

    @TempDir
    Path tempDir;

    // ── 装配辅助 ─────────────────────────────────────────────────────

    private static Settings settings(String backend, String sqlitePath) {
        Settings s = new Settings();
        s.checkpointer.backend = backend;
        if (sqlitePath != null) {
            s.checkpointer.sqlitePath = sqlitePath;
        }
        return s;
    }

    private UploadedFileStores.SqliteUploadedFileStore sqliteStore(String fileName) {
        return new UploadedFileStores.SqliteUploadedFileStore(tempDir.resolve(fileName).toString());
    }

    /** 一条测试附件（threadId 传 "" 表示未绑定孤儿候选）。 */
    private static UploadedFileInfo info(String fileId, String threadId, String userId,
                                         String module, String warning) {
        return new UploadedFileInfo(fileId, threadId, module, userId, fileId + ".pdf", "pdf",
                3, 12, false, BYTES.length * 2, warning, "提取文本-" + fileId, BYTES);
    }

    /** 绕过存储层直插旧行（旧 created_at 直插——孤儿清理的确定性时间源）。 */
    private static void rawInsert(Path db, String fileId, String threadId, String createdAt)
            throws SQLException {
        try (Connection c = DriverManager.getConnection("jdbc:sqlite:" + db);
             PreparedStatement ps = c.prepareStatement(
                     "INSERT INTO uploaded_files (file_id, thread_id, module, user_id,"
                             + " filename, format, pages, paragraphs, truncated, text_len,"
                             + " warning, extracted_text, content, created_at)"
                             + " VALUES (?, ?, 'chat', '', 'old.txt', 'txt', NULL, NULL,"
                             + " 0, 0, '', '', NULL, ?)")) {
            ps.setString(1, fileId);
            ps.setString(2, threadId);
            ps.setString(3, createdAt);
            ps.executeUpdate();
        }
    }

    private static List<String> ids(List<UploadedFileInfo> infos) {
        return infos.stream().map(UploadedFileInfo::fileId).toList();
    }

    // ── 工厂分派 ─────────────────────────────────────────────────────

    @Test
    void factoryDispatchFollowsCheckpointerBackend() {
        try (UploadedFileStore memory = UploadedFileStores.build(settings("memory", null))) {
            assertInstanceOf(UploadedFileStores.MemoryUploadedFileStore.class, memory);
        }
        try (UploadedFileStore sqlite =
                     UploadedFileStores.build(settings("sqlite", tempDir.resolve("factory.db").toString()))) {
            assertInstanceOf(UploadedFileStores.SqliteUploadedFileStore.class, sqlite);
        }
        // mysql 构造不建连接（首写才建表），与 Python 侧 __init__ 一致。
        assertInstanceOf(UploadedFileStores.MySqlUploadedFileStore.class,
                UploadedFileStores.build(settings("mysql", null)));
        // 未知后端返回 null（附件功能停用；Settings 校验层通常先拦）。
        assertNull(UploadedFileStores.build(settings("redis", null)));
    }

    // ── save→getMany 往返 ────────────────────────────────────────────

    @Test
    void memoryRoundTripPreservesBytesAndRequestOrder() {
        try (UploadedFileStore store = UploadedFileStores.build(settings("memory", null))) {
            UploadedFileInfo a = info("f-a", "", "u1", "chat", "");
            UploadedFileInfo b = info("f-b", "t1", "u2", "writer", "");
            store.save(a);
            store.save(b);

            // 请求顺序（b 在前）即返回顺序；缺失 id 静默跳过。
            List<UploadedFileInfo> got = store.getMany(List.of("f-b", "missing", "f-a"));
            assertEquals(List.of("f-b", "f-a"), ids(got));
            assertArrayEquals(BYTES, got.get(0).content());
            assertEquals("提取文本-f-b", got.get(0).extractedText());
            assertEquals("u2", got.get(0).userId());
            assertEquals("t1", got.get(0).threadId());

            // memory 后端保留精确的 createdAt（epoch 秒）。
            assertEquals(a.createdAt(), store.getMany(List.of("f-a")).get(0).createdAt(), 1e-9);
        }
    }

    @Test
    void sqliteRoundTripPreservesBytesAndFields() {
        try (UploadedFileStores.SqliteUploadedFileStore store = sqliteStore("roundtrip.db")) {
            UploadedFileInfo warned = info("f-warn", "", "u1", "chat",
                    "未能从文件中提取到文本：可能是扫描件或纯图片 PDF");
            UploadedFileInfo bound = info("f-bound", "t1", "u2", "writer", "");
            store.save(warned);
            store.save(bound);

            List<UploadedFileInfo> got = store.getMany(List.of("f-warn", "missing", "f-bound"));
            assertEquals(List.of("f-warn", "f-bound"), ids(got));
            for (UploadedFileInfo g : got) {
                // 字节相等：原始字节入库（BLOB），往返不损。
                assertArrayEquals(BYTES, g.content());
                assertEquals(3, g.pages());
                assertEquals(12, g.paragraphs());
                assertFalse(g.truncated());
                assertEquals(BYTES.length * 2, g.textLen());
            }
            assertEquals("u1", got.get(0).userId());
            assertEquals("未能从文件中提取到文本：可能是扫描件或纯图片 PDF", got.get(0).warning());
            assertEquals("", got.get(0).threadId());
            assertEquals("t1", got.get(1).threadId());
            assertEquals("writer", got.get(1).module());

            // 空 id 清单直接返回空（不拼 IN ()）。
            assertTrue(store.getMany(List.of()).isEmpty());
            assertTrue(store.getMany(List.of("nope")).isEmpty());
        }
    }

    // ── meta() HTTP 视图 ─────────────────────────────────────────────

    @Test
    void metaKeysMatchPythonView() {
        UploadedFileInfo withWarning = new UploadedFileInfo("f1", "", "chat", "u1", "doc.pdf",
                "pdf", 5, 9, true, 123, "扫描件无文本", "全文不应出现在 meta", BYTES);
        Map<String, Object> meta = withWarning.meta();
        // 键集与顺序逐键对齐 Python meta()：无 content / extracted_text / thread_id。
        assertEquals(List.of("file_id", "filename", "format", "pages", "text_len",
                        "truncated", "user_id", "warning"),
                List.copyOf(meta.keySet()));
        assertEquals("f1", meta.get("file_id"));
        assertEquals("doc.pdf", meta.get("filename"));
        assertEquals("pdf", meta.get("format"));
        assertEquals(5, meta.get("pages"));
        assertEquals(123, meta.get("text_len"));
        assertEquals(Boolean.TRUE, meta.get("truncated"));
        assertEquals("u1", meta.get("user_id"));
        assertEquals("扫描件无文本", meta.get("warning"));

        // 空警告时不携带 warning 键（对应 Python 的条件展开）。
        UploadedFileInfo noWarning = new UploadedFileInfo("f2", "", "chat", "u1", "doc.pdf",
                "pdf", null, null, false, 0, "", "", new byte[0]);
        Map<String, Object> meta2 = noWarning.meta();
        assertEquals(List.of("file_id", "filename", "format", "pages", "text_len",
                        "truncated", "user_id"),
                List.copyOf(meta2.keySet()));
        assertNull(meta2.get("pages")); // 未知页数 → None 语义（键在、值为 null）
    }

    // ── 绑定与孤儿清理 ───────────────────────────────────────────────

    @Test
    void bindThreadRemovesFromOrphanSet() throws SQLException {
        Path db = tempDir.resolve("bind.db");
        try (UploadedFileStores.SqliteUploadedFileStore store = sqliteStore("bind.db")) {
            // 旧行直插：2020 年的未绑定孤儿（确定性过期）。
            rawInsert(db, "old-orphan", "", "2020-01-01 00:00:00");
            rawInsert(db, "old-bound", "t-old", "2020-01-01 00:00:00");
            // 绑定后 old-orphan 退出孤儿集合。
            store.bindThread(List.of("old-orphan"), "t-1");

            assertEquals(0, store.purgeOrphans());
            // 两个旧行都还在（一个已绑定、一个原本就有线程）。
            assertEquals(2, store.getMany(List.of("old-orphan", "old-bound")).size());
            // 空清单绑定是无操作（不拼 IN ()）。
            store.bindThread(List.of(), "t-2");
        }
    }

    @Test
    void purgeOrphansDeletesOnlyStaleUnbound() throws SQLException {
        Path db = tempDir.resolve("purge.db");
        try (UploadedFileStores.SqliteUploadedFileStore store = sqliteStore("purge.db")) {
            // 新鲜未绑定（store.save 写入，created_at = 现在）→ 保留。
            store.save(info("fresh-orphan", "", "u1", "chat", ""));
            // 旧的未绑定 → 删；旧的已绑定 → 保留。
            rawInsert(db, "stale-orphan", "", "2020-01-01 00:00:00");
            rawInsert(db, "stale-bound", "t-old", "2020-01-01 00:00:00");

            assertEquals(1, store.purgeOrphans());
            // stale-orphan 已删；fresh-orphan（新鲜未绑定）与 stale-bound（旧的已
            // 绑定）都保留。
            assertEquals(List.of("fresh-orphan", "stale-bound"),
                    ids(store.getMany(List.of("fresh-orphan", "stale-orphan", "stale-bound"))));
            // 重复清理：无更多孤儿，计数归零。
            assertEquals(0, store.purgeOrphans());
        }
    }

    // #268 后新增用例的时间源（镜像主类的 SECONDS_FORMAT——它私有）。
    private static final DateTimeFormatter TEST_SECONDS_FORMAT =
            DateTimeFormatter.ofPattern("uuuu-MM-dd HH:mm:ss");

    @Test
    void purgeOrphansSkipsMalformedCreatedAtRows() throws SQLException {
        Path db = tempDir.resolve("purge-broken.db");
        try (UploadedFileStores.SqliteUploadedFileStore store = sqliteStore("purge-broken.db")) {
            // 手工制造一行畸形 created_at（直插与 UPDATE 同效）：
            // #268 后判定走读侧解析，畸形行单行 WARN 跳过——purge 不抛、
            // 不误删该行；同批的合法旧行照常清理。
            rawInsert(db, "stale-orphan", "", "2020-01-01 00:00:00");
            rawInsert(db, "broken-orphan", "", "不是时间串");

            assertEquals(1, store.purgeOrphans());
            assertEquals(List.of("broken-orphan"),
                    ids(store.getMany(List.of("stale-orphan", "broken-orphan"))));
        }
    }

    @Test
    void purgeOrphansZeroMaxAgeKeepsFutureRows() throws SQLException {
        Path db = tempDir.resolve("purge-zero.db");
        try (UploadedFileStores.SqliteUploadedFileStore store = sqliteStore("purge-zero.db")) {
            // maxAge=0 边界照旧：严格早于 now 的未绑定行全清。用未来时间戳
            // 的保留行确定性地避开「同一秒写入/比较」的分辨率竞态。
            rawInsert(db, "past-orphan", "",
                    LocalDateTime.now().minusHours(2).format(TEST_SECONDS_FORMAT));
            rawInsert(db, "future-orphan", "",
                    LocalDateTime.now().plusHours(1).format(TEST_SECONDS_FORMAT));

            assertEquals(1, store.purgeOrphans(0));
            assertEquals(List.of("future-orphan"),
                    ids(store.getMany(List.of("past-orphan", "future-orphan"))));
        }
    }

    @Test
    void memoryPurgeUsesEpochSecondsCutoff() {
        try (UploadedFileStore store = UploadedFileStores.build(settings("memory", null))) {
            double now = System.currentTimeMillis() / 1000.0;
            // 直接以旧 createdAt 构造（memory 后端的确定性时间源）。
            store.save(new UploadedFileInfo("old-orphan", "", "chat", "u1", "a.pdf", "pdf",
                    null, null, false, 0, "", "", new byte[0], now - 25 * 3600));
            store.save(new UploadedFileInfo("old-bound", "t-old", "chat", "u1", "b.pdf", "pdf",
                    null, null, false, 0, "", "", new byte[0], now - 25 * 3600));
            store.save(new UploadedFileInfo("fresh-orphan", "", "chat", "u1", "c.pdf", "pdf",
                    null, null, false, 0, "", "", new byte[0], now));

            assertEquals(1, store.purgeOrphans());
            assertEquals(List.of("old-bound", "fresh-orphan"),
                    ids(store.getMany(List.of("old-orphan", "old-bound", "fresh-orphan"))));

            // 自定义窗口：maxAgeHours=0 连 1 秒前的孤儿也清（窗口可注入）。
            // 重新登记 fresh-orphan 为 2 秒前，使 0 小时窗口的清理范围
            // 确定性覆盖 fresh-orphan + fresh2（避免时钟分辨率竞态）。
            store.save(new UploadedFileInfo("fresh-orphan", "", "chat", "u1", "c.pdf", "pdf",
                    null, null, false, 0, "", "", new byte[0], now - 2));
            store.save(new UploadedFileInfo("fresh2", "", "chat", "u1", "d.pdf", "pdf",
                    null, null, false, 0, "", "", new byte[0], now - 1));
            assertEquals(2, store.purgeOrphans(0));
            assertTrue(store.getMany(List.of("fresh2", "fresh-orphan")).isEmpty());
        }
    }

    // ── 删除 ─────────────────────────────────────────────────────────

    @Test
    void deleteForThreadAndDeleteFile() {
        try (UploadedFileStores.SqliteUploadedFileStore store = sqliteStore("delete.db")) {
            store.save(info("f1", "", "u1", "chat", ""));
            store.save(info("f2", "", "u1", "chat", ""));
            store.save(info("f3", "", "u1", "chat", ""));
            store.bindThread(List.of("f1", "f2"), "t1");
            store.bindThread(List.of("f3"), "t2");

            // 级联删 t1：f1/f2 消失，f3 不受影响。
            store.deleteForThread("t1");
            assertEquals(List.of("f3"), ids(store.getMany(List.of("f1", "f2", "f3"))));
            store.deleteForThread("t-missing"); // 无命中也不报错

            // 单文件撤销：只有第一次返回 true。
            assertTrue(store.deleteFile("f3"));
            assertFalse(store.deleteFile("f3"));
            assertTrue(store.getMany(List.of("f3")).isEmpty());
        }
    }

    // ── 属主与模块过滤 ───────────────────────────────────────────────

    @Test
    void listForUserFiltersOwnerModuleAndLimit() throws SQLException {
        Path db = tempDir.resolve("list.db");
        try (UploadedFileStores.SqliteUploadedFileStore store = sqliteStore("list.db")) {
            store.save(info("u1-chat-1", "", "u1", "chat", ""));
            store.save(info("u1-chat-2", "", "u1", "chat", ""));
            store.save(info("u1-writer", "", "u1", "writer", ""));
            store.save(info("u2-chat", "", "u2", "chat", ""));

            // 属主过滤：别人的文件不可见。
            List<UploadedFileInfo> mine = store.listForUser("u1");
            assertEquals(3, mine.size());
            assertTrue(mine.stream().allMatch(g -> g.userId().equals("u1")));
            assertTrue(store.listForUser("ghost").isEmpty());

            // 模块过滤与 limit（limit 下限 1）。
            assertEquals(2, store.listForUser("u1", "chat", 200).size());
            assertEquals(1, store.listForUser("u1", "writer", 200).size());
            assertEquals(2, store.listForUser("u1", null, 2).size());
            assertEquals(1, store.listForUser("u1", null, 0).size());

            // 默认重载（对应 Python list_for_user(user_id) 的默认参数）。
            assertEquals(3, store.listForUser("u1").size());

            // 旧库"无主遗留"行（user_id 为空串）按空属主可见——撤销路径允许
            // 任意已验身份用户处理它们的前提是先可见。
            rawInsert(db, "ownerless", "", "2020-01-01 00:00:00");
            assertEquals(1, store.listForUser("").stream()
                    .filter(g -> g.fileId().equals("ownerless")).count());
        }
    }

    // ── 运行时自迁移（sqlite 无 alembic） ────────────────────────────

    @Test
    void sqliteSelfMigrationAddsUserIdColumn() throws SQLException {
        Path db = tempDir.resolve("migrate.db");
        try (Connection c = DriverManager.getConnection("jdbc:sqlite:" + db);
             Statement st = c.createStatement()) {
            st.execute(OLD_SHAPE_DDL);
            st.execute("INSERT INTO uploaded_files (file_id, thread_id, module, filename,"
                    + " format, pages, paragraphs, truncated, text_len, warning,"
                    + " extracted_text, content, created_at)"
                    + " VALUES ('legacy', 't-old', 'chat', 'old.txt', 'txt', NULL, NULL,"
                    + " 0, 0, '', '', NULL, '2020-01-01 00:00:00')");
        }
        // 打开即补列：0006 的 user_id 由 PRAGMA 检查 + ALTER TABLE 自迁移。
        try (UploadedFileStores.SqliteUploadedFileStore store = sqliteStore("migrate.db")) {
            List<UploadedFileInfo> rows = store.listForUser("");
            assertEquals(1, rows.size());
            assertEquals("legacy", rows.get(0).fileId());
            assertEquals("", rows.get(0).userId()); // 旧行缺省空串 = 无主遗留

            // 补列后新行正常往返（含 user_id）。
            store.save(info("new", "", "u9", "chat", ""));
            assertEquals("u9", store.getMany(List.of("new")).get(0).userId());
            assertEquals(1, store.listForUser("u9").size());
        }
    }

    // ── 可见清单（L92/R4-b）：tool_result 哨兵行排除 ─────────────────

    @Test
    void toolResultSentinelRowsExcludedFromVisibleList() {
        // 哨兵行（#6 外置）不是知识文件：listVisibleForUser 恒排除；
        // 三参 listForUser 按模块精确查询仍可见（级联/审计路径不回归）。
        try (UploadedFileStore store = UploadedFileStores.build(settings("memory", null))) {
            store.save(info("f-chat", "", "u1", "chat", ""));
            store.save(info("f-tool", "chat:t1", "u1", UploadedFileStore.TOOL_RESULT_MODULE, ""));
            store.save(info("f-writer", "", "u1", "writer", ""));
            List<String> visible = ids(store.listVisibleForUser("u1"));
            assertTrue(visible.contains("f-chat") && visible.contains("f-writer"),
                    "真实文件保留：" + visible);
            assertTrue(!visible.contains("f-tool"), "哨兵行必须被排除：" + visible);
            assertEquals(List.of("f-tool"),
                    ids(store.listForUser("u1", UploadedFileStore.TOOL_RESULT_MODULE, 200)),
                    "按模块查询哨兵行仍可见（删除走线程级联，查询面不回归）");
        }
        try (UploadedFileStores.SqliteUploadedFileStore store = sqliteStore("visible.db")) {
            // SQL 后端走查询级下推（AND module != ?）：LIMIT 预算花在真实文件上。
            store.save(info("s-chat", "", "u1", "chat", ""));
            store.save(info("s-tool", "chat:t1", "u1", UploadedFileStore.TOOL_RESULT_MODULE, ""));
            List<String> visible = ids(store.listVisibleForUser("u1"));
            assertTrue(visible.contains("s-chat"), "真实文件保留：" + visible);
            assertTrue(!visible.contains("s-tool"), "哨兵行必须被排除：" + visible);
            assertEquals(List.of("s-tool"),
                    ids(store.listForUser("u1", UploadedFileStore.TOOL_RESULT_MODULE, 200)));
        }
    }
}
