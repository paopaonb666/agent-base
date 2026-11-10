package io.github.paopaonb666.agentbase.memory.store;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.Statement;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 关键词下推的存储层语义（#326/#327）：FTS5 表建库即建、写路径四点同步
 * （add→命中 / REPLACE 覆盖→旧失新中 / delete→失 / 线程清理→失）、
 * 候选集与进程内关键词命中集一致、存量库回填幂等。SQL 侧分数不进混合
 * 打分（Retrieval.recallMemories 只用 id 集收窄），故一致口径是「集合」
 * 不是「分数」。MySQL 侧（ngram FULLTEXT）无本地服务器，SQL 形态由
 * MysqlMemoryStore 构造不建连的静态面覆盖（TimestampMatrix 模式）。
 */
class Bm25PushdownStoreTest {

    @TempDir
    Path tempDir;

    private static Models.MemoryRecord memory(String id, String content) {
        return new Models.MemoryRecord(id, "u1", "chat", "semantic", content,
                List.of(), null, null, 0.5, "active", "", List.of(), 1000.0, 1000.0, null, 0);
    }

    private static Models.MemoryRecord memoryInThread(String id, String content, String threadId) {
        return new Models.MemoryRecord(id, "u1", "chat", "semantic", content,
                List.of(), null, null, 0.5, "active", threadId, List.of(),
                1000.0, 1000.0, null, 0);
    }

    @Test
    void sqliteBuildsFtsTableOnCreate() throws Exception {
        Path db = tempDir.resolve("fts-" + System.nanoTime() + ".db");
        try (SqliteMemoryStore ignored = new SqliteMemoryStore(db.toString())) {
            assertTrue(tableExists(db, "memory_fts"), "003 迁移应建 FTS5 虚表");
        }
    }

    @Test
    void ftsStaysSyncedAcrossWritePaths() {
        Path db = tempDir.resolve("sync-" + System.nanoTime() + ".db");
        try (SqliteMemoryStore store = new SqliteMemoryStore(db.toString())) {
            // add → 命中。
            store.upsertMemory(memory("m1", "数据库每天凌晨三点全量备份"));
            assertSetEquals(Set.of("m1"), store.searchMemoryIds("u1", null, "数据库备份", 10));

            // REPLACE 同 id 覆盖（trigger 残影的回归锁）：旧词失、新词中，
            // 且同 id 不出现重复行（残影会让它命中两次/集合含脏匹配）。
            store.upsertMemory(memory("m1", "数据库改为每周全量快照"));
            assertSetEquals(Set.of(), store.searchMemoryIds("u1", null, "凌晨三点", 10));
            assertSetEquals(Set.of("m1"), store.searchMemoryIds("u1", null, "每周快照", 10));

            // delete → 失。
            store.deleteMemory("m1");
            assertSetEquals(Set.of(), store.searchMemoryIds("u1", null, "快照", 10));

            // 线程清理 → 失（#172 级联路径的索引同步）。
            store.upsertMemory(memoryInThread("m2", "发布走灰度流程", "chat:dz"));
            store.upsertMemory(memory("m3", "回滚保留三个版本"));
            assertSetEquals(Set.of("m2"), store.searchMemoryIds("u1", null, "灰度", 10));
            store.deleteMemoriesForThread("chat:dz");
            assertSetEquals(Set.of(), store.searchMemoryIds("u1", null, "灰度", 10));
            assertSetEquals(Set.of("m3"), store.searchMemoryIds("u1", null, "回滚", 10));
        }
    }

    @Test
    void candidateSetMatchesInProcessKeywordHits() {
        Path db = tempDir.resolve("cand-" + System.nanoTime() + ".db");
        try (SqliteMemoryStore store = new SqliteMemoryStore(db.toString())) {
            // 语料：3 条与查询相关、3 条无关——下推 top-N（N 宽松）应等于
            // 进程内关键词命中集（任一词项交集）。
            store.upsertMemory(memory("k1", "生产数据库部署在 Ubuntu 服务器"));
            store.upsertMemory(memory("k2", "数据库扩容先加只读副本"));
            store.upsertMemory(memory("k3", "数据库密码只放密钥管理服务"));
            store.upsertMemory(memory("n1", "团队周会固定在周一上午"));
            store.upsertMemory(memory("n2", "用户偏好 JetBrains 编辑器"));
            store.upsertMemory(memory("n3", "告警发到值班飞书群"));

            assertSetEquals(Set.of("k1", "k2", "k3"),
                    store.searchMemoryIds("u1", null, "数据库", 200));

            // user/status 过滤与召回同口径：其他用户/归档行不入候选。
            store.upsertMemory(new Models.MemoryRecord("k4", "u2", "chat", "semantic",
                    "数据库第二用户", List.of(), null, null, 0.5, "active", "", List.of(),
                    1000.0, 1000.0, null, 0));
            store.upsertMemory(new Models.MemoryRecord("k5", "u1", "chat", "semantic",
                    "数据库已归档行", List.of(), null, null, 0.5, "archived", "", List.of(),
                    1000.0, 1000.0, null, 0));
            assertSetEquals(Set.of("k1", "k2", "k3"),
                    store.searchMemoryIds("u1", null, "数据库", 200));
        }
    }

    @Test
    void legacyDbRebuildsFtsIndexOnOpen() throws Exception {
        // 存量库首启（FTS 行数 < memories）：构造期幂等重建。
        Path db = tempDir.resolve("rebuild-" + System.nanoTime() + ".db");
        try (SqliteMemoryStore store = new SqliteMemoryStore(db.toString())) {
            store.upsertMemory(memory("r1", "监控指标暴露给 Prometheus 抓取"));
        }
        // 模拟同步中断：外部清空 FTS 表。
        try (Connection conn = DriverManager.getConnection("jdbc:sqlite:" + db);
             Statement st = conn.createStatement()) {
            st.execute("DELETE FROM memory_fts");
        }
        try (SqliteMemoryStore store = new SqliteMemoryStore(db.toString())) {
            assertSetEquals(Set.of("r1"), store.searchMemoryIds("u1", null, "监控指标", 10));
        }
    }

    private static boolean tableExists(Path db, String name) throws Exception {
        try (Connection conn = DriverManager.getConnection("jdbc:sqlite:" + db);
             Statement st = conn.createStatement();
             var rs = st.executeQuery("SELECT COUNT(*) FROM sqlite_master "
                     + "WHERE type='table' AND name='" + name + "'")) {
            rs.next();
            return rs.getInt(1) > 0;
        }
    }

    private static void assertSetEquals(Set<String> expected, List<String> actual) {
        assertEquals(expected, new LinkedHashSet<>(actual),
                "候选 id 集合不一致：actual=" + actual);
    }
}
