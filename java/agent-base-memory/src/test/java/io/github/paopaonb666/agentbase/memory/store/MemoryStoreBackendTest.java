package io.github.paopaonb666.agentbase.memory.store;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.Statement;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 记忆存储层后端的全协议往返测试：临时 sqlite 文件库与进程内字典后端跑
 * 同一套断言（六张表核心路径：upsert/get/list 过滤/touch/delete/version/
 * ops/chunks 计数/deleteForThread 保留 memories），对应 Python 侧三个
 * 后端实现的行为契约。
 */
class MemoryStoreBackendTest {

    private static final String USER = "u1";

    @TempDir
    Path tempDir;

    // ── 后端 ×1：sqlite（临时文件库） ────────────────────────────────

    @Test
    void sqliteBackendFullProtocol() {
        Path db = tempDir.resolve("memory-" + System.nanoTime() + ".db");
        try (SqliteMemoryStore store = new SqliteMemoryStore(db.toString())) {
            runFullProtocol(store);
        }
    }

    @Test
    void sqliteCreatesRetrievalIndexOnMemories() throws Exception {
        // #159：检索主路径 WHERE user_id=? ORDER BY updated_at DESC 需要
        // (user_id, updated_at) 索引——建库即建，且重开幂等（存量库下次
        // 打开自动补齐）。
        Path db = tempDir.resolve("idx-" + System.nanoTime() + ".db");
        try (SqliteMemoryStore store = new SqliteMemoryStore(db.toString())) {
            assertTrue(indexExists(db, "idx_mem_user_updated"));
        }
        try (SqliteMemoryStore store = new SqliteMemoryStore(db.toString())) {
            assertTrue(indexExists(db, "idx_mem_user_updated"));
        }
    }

    @Test
    void sqliteCreatesOpsAuditIndexOnMemoryOps() throws Exception {
        // #160：审计主路径 WHERE user_id=? ORDER BY created_at DESC 需要
        // (user_id, created_at) 索引——索引清单与 002 迁移双路径幂等收敛，
        // 重开不重复建。
        Path db = tempDir.resolve("idx-ops-" + System.nanoTime() + ".db");
        try (SqliteMemoryStore store = new SqliteMemoryStore(db.toString())) {
            assertTrue(indexExists(db, "idx_ops_user_created"));
        }
        try (SqliteMemoryStore store = new SqliteMemoryStore(db.toString())) {
            assertTrue(indexExists(db, "idx_ops_user_created"));
        }
    }

    private static boolean indexExists(Path db, String name) throws Exception {
        try (Connection conn = DriverManager.getConnection("jdbc:sqlite:" + db);
             Statement st = conn.createStatement();
             var rs = st.executeQuery("SELECT COUNT(*) FROM sqlite_master "
                     + "WHERE type='index' AND name='" + name + "'")) {
            rs.next();
            return rs.getInt(1) > 0;
        }
    }

    @Test
    void corruptedTagsColumnDegradesToEmptyWithRowIntact() throws Exception {
        // #169：畸形 tags_json 按空处理但行不丢——之前静默退化无告警，
        // 现在存储层 WARN（含 memory_id，日志侧验证）。
        Path db = tempDir.resolve("tags-" + System.nanoTime() + ".db");
        try (SqliteMemoryStore store = new SqliteMemoryStore(db.toString())) {
            // 绕过写路径校验直接注入存量坏数据。
            try (Connection conn = DriverManager.getConnection("jdbc:sqlite:" + db);
                 Statement st = conn.createStatement()) {
                st.execute("""
                        INSERT INTO memories (memory_id, user_id, agent_id, kind, content,
                          tags_json, salience, status, source_thread_id, source_refs_json,
                          created_at, updated_at, access_count)
                        VALUES ('mbad', 'u1', 'chat', 'semantic', '坏标签行',
                          'not-json', 0.5, 'active', '', '[{"ref":"a"}]', 1.0, 1.0, 0)
                        """);
            }
            Models.MemoryRecord got = store.getMemory("mbad");
            assertEquals("坏标签行", got.content());
            assertEquals(List.of(), got.tags(), "畸形 tags 按空处理，行不丢");
            // 源引用列合法 JSON 不受影响（同口径解析）。
            assertEquals(1, got.sourceRefs().size());
        }
    }

    /** 旧库补列：缺 offsets_json 列的 doc_chunks 在打开时被补上（M7 自迁移）。 */
    @Test
    void sqliteLegacyDbAddsOffsetsColumn() throws Exception {
        Path db = tempDir.resolve("legacy-" + System.nanoTime() + ".db");
        try (Connection conn = DriverManager.getConnection("jdbc:sqlite:" + db);
             Statement st = conn.createStatement()) {
            // M7 之前的旧表形态：没有 offsets_json 列。
            st.execute("""
                    CREATE TABLE doc_chunks (
                      chunk_id VARCHAR(40) PRIMARY KEY,
                      file_id VARCHAR(32) NOT NULL,
                      thread_id VARCHAR(190) NOT NULL DEFAULT '',
                      user_id VARCHAR(64) NOT NULL,
                      agent_id VARCHAR(64) NOT NULL DEFAULT '',
                      ordinal INTEGER NOT NULL DEFAULT 0,
                      text TEXT NOT NULL,
                      embedding BLOB,
                      embedding_dim INTEGER,
                      created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )""");
        }
        try (SqliteMemoryStore store = new SqliteMemoryStore(db.toString())) {
            store.putChunks(List.of(new Models.DocChunk("c1", "f1", "t", USER, "chat", 0, "正文")));
            List<Models.DocChunk> chunks = store.listChunks(USER, null, null, 10);
            assertEquals(1, chunks.size());
            // 补列后旧行 offsets 为 NULL → 解码为空表（前端降级纯卡片视图）。
            assertTrue(chunks.get(0).offsets().isEmpty());
        }
    }

    // ── 后端 ×2：进程内字典 ──────────────────────────────────────────

    @Test
    void inMemoryBackendFullProtocol() {
        runFullProtocol(new InMemoryMemoryStore());
    }

    // ── 装配：按 checkpointer 后端分派 ───────────────────────────────

    @Test
    void buildDispatchesByCheckpointerBackend() {
        io.github.paopaonb666.agentbase.core.config.Settings settings =
                new io.github.paopaonb666.agentbase.core.config.Settings();
        settings.checkpointer.backend = "memory";
        assertInstanceOf(InMemoryMemoryStore.class, MemoryStores.build(settings));

        // 未知后端：不抛错，返回 null（memory store 禁用）。
        settings.checkpointer.backend = "does-not-exist";
        assertNull(MemoryStores.build(settings));
    }

    // ── 全协议：两个后端共用同一套断言 ───────────────────────────────

    private static void runFullProtocol(MemoryStore store) {
        memoriesRoundTrip(store);
        listMemoriesFilters(store);
        touchAndBackfill(store);
        blocksRoundTrip(store);
        summariesRoundTrip(store);
        chunksRoundTrip(store);
        opsAndVersionsRoundTrip(store);
        opsFilterAndCursorRoundTrip(store);
        deleteForThreadKeepsMemories(store);
        deleteMemory(store);
    }

    // ── memories：upsert/get 往返 + 写入边界校验 ─────────────────────

    private static void memoriesRoundTrip(MemoryStore store) {
        byte[] vector = Models.encodeEmbedding(List.of(0.1f, 0.2f));
        store.upsertMemory(new Models.MemoryRecord(
                "m1", USER, "chat", "semantic", "记住我喜欢咖啡", List.of("偏好"), vector, 2,
                0.8, "active", "chat:t1", List.of("doc:a"), 1000.0, 1000.5, null, 0));

        Models.MemoryRecord got = store.getMemory("m1");
        assertEquals("记住我喜欢咖啡", got.content());
        assertEquals(List.of("偏好"), got.tags());
        assertArrayEquals(vector, got.embedding());
        assertEquals(List.of(0.1f, 0.2f), Models.decodeEmbedding(got.embedding()));
        assertEquals(2, got.embeddingDim());
        assertEquals(0.8, got.salience());
        assertEquals("active", got.status());
        assertEquals("chat:t1", got.sourceThreadId());
        assertEquals(List.of("doc:a"), got.sourceRefs());
        assertEquals(1000.0, got.createdAt());
        assertEquals(1000.5, got.updatedAt());
        assertNull(got.lastAccessedAt());
        assertEquals(0, got.accessCount());

        // 写入边界快速失败：非法类别/状态被拒绝。
        assertThrows(Models.MemoryStoreError.class, () -> store.upsertMemory(
                new Models.MemoryRecord("mx", USER, "chat", "unknown_kind", "x")));
        assertThrows(Models.MemoryStoreError.class, () -> store.upsertMemory(
                new Models.MemoryRecord("my", USER, "chat", "semantic", "x", List.of(), null,
                        null, 0.5, "bogus", "", List.of(), 1.0, 1.0, null, 0)));
    }

    // ── memories：listMemories 动态 WHERE 过滤 ───────────────────────

    private static void listMemoriesFilters(MemoryStore store) {
        // 造数据：m2 全局已归档；profile:* 画像；m4/m5 活跃（m5 维度不符）。
        store.upsertMemory(new Models.MemoryRecord(
                "m2", USER, "*", "episodic", "全局事实", List.of(), null, null,
                0.5, "archived", "", List.of(), 1001.0, 1001.0, null, 0));
        store.upsertMemory(new Models.MemoryRecord(
                "profile:" + USER, USER, "*", "procedural", "画像", List.of("profile"), null,
                null, 0.5, "active", "", List.of(), 1002.0, 1002.0, null, 0));
        store.upsertMemory(new Models.MemoryRecord(
                "m4", USER, "chat", "semantic", "回填候选", List.of(), null, null,
                0.5, "active", "", List.of(), 1003.0, 1003.0, null, 0));
        store.upsertMemory(new Models.MemoryRecord(
                "m5", USER, "chat", "semantic", "维度不符", List.of(),
                Models.encodeEmbedding(List.of(0.1f, 0.2f, 0.3f)), 3,
                0.5, "active", "", List.of(), 1004.0, 1004.0, null, 0));

        // 默认（statuses=[active]）：按 updated_at 降序；archived 不出现。
        List<Models.MemoryRecord> all = store.listMemories(
                USER, null, true, null, List.of("active"), 2000, false);
        assertEquals(List.of("m5", "m4", "profile:" + USER, "m1"),
                all.stream().map(Models.MemoryRecord::memoryId).toList());

        // kinds 过滤（画像的 kind=procedural 被排除）。
        List<Models.MemoryRecord> semantic = store.listMemories(
                USER, null, true, List.of("semantic"), List.of("active"), 2000, false);
        assertEquals(List.of("m5", "m4", "m1"),
                semantic.stream().map(Models.MemoryRecord::memoryId).toList());

        // exclude_profile：画像混入列表是实现细节泄漏，默认列表可显式排除。
        List<Models.MemoryRecord> noProfile = store.listMemories(
                USER, null, true, null, List.of("active"), 2000, true);
        assertEquals(List.of("m5", "m4", "m1"),
                noProfile.stream().map(Models.MemoryRecord::memoryId).toList());

        // agent 过滤：includeGlobal=false 只看本模块；true 追加 agent_id='*'。
        List<Models.MemoryRecord> scoped = store.listMemories(
                USER, "chat", false, null, List.of("active"), 2000, true);
        assertEquals(List.of("m5", "m4", "m1"),
                scoped.stream().map(Models.MemoryRecord::memoryId).toList());
        List<Models.MemoryRecord> withGlobal = store.listMemories(
                USER, "chat", true, null, List.of("active"), 2000, true);
        assertEquals(List.of("m5", "m4", "m1"),
                withGlobal.stream().map(Models.MemoryRecord::memoryId).toList());

        // limit：取最近 N 条。
        List<Models.MemoryRecord> limited = store.listMemories(
                USER, null, true, null, List.of("active"), 2, false);
        assertEquals(List.of("m5", "m4"),
                limited.stream().map(Models.MemoryRecord::memoryId).toList());

        // 其他用户不可见（用户作用域隔离）。
        assertTrue(store.listMemories("u2", null, true, null, List.of("active"), 2000, false)
                .isEmpty());
    }

    // ── memories：touch 记账 + 向量回填查询 ──────────────────────────

    private static void touchAndBackfill(MemoryStore store) {
        store.touchMemories(List.of("m1", "ghost"));
        Models.MemoryRecord touched = store.getMemory("m1");
        assertEquals(1, touched.accessCount());
        assertNotEquals(null, touched.lastAccessedAt());

        // 回填：active 且向量缺失/维度不符的非画像记忆；画像与已向量的被排除。
        List<Models.MemoryRecord> needing = store.listMemoriesNeedingEmbedding(2, 100);
        assertEquals(List.of("m5", "m4"),
                needing.stream().map(Models.MemoryRecord::memoryId).toList());
    }

    // ── memory_blocks：块 upsert/get/list/delete ─────────────────────

    private static void blocksRoundTrip(MemoryStore store) {
        store.upsertBlock(new Models.MemoryBlock(USER, "chat", "human", "用户偏好简短回答",
                100, 3, 2000.0));
        store.upsertBlock(new Models.MemoryBlock(USER, "chat", "persona", "你是严谨助手",
                2000, 0, 2001.0));

        Models.MemoryBlock human = store.getBlock(USER, "chat", "human");
        assertEquals("用户偏好简短回答", human.content());
        assertEquals(100, human.charLimit());
        assertEquals(3, human.version());
        assertEquals(2000.0, human.updatedAt());
        assertNull(store.getBlock(USER, "chat", "missing"));

        // listBlocks 按 label 升序。
        assertEquals(List.of("human", "persona"),
                store.listBlocks(USER, "chat").stream().map(Models.MemoryBlock::label).toList());

        // 整行覆盖式 upsert：同键重写生效。
        store.upsertBlock(new Models.MemoryBlock(USER, "chat", "human", "内容已更新", 100, 4, 2002.0));
        assertEquals("内容已更新", store.getBlock(USER, "chat", "human").content());
        assertEquals(4, store.getBlock(USER, "chat", "human").version());

        assertTrue(store.deleteBlock(USER, "chat", "human"));
        assertFalse(store.deleteBlock(USER, "chat", "human"));
        assertEquals(List.of("persona"),
                store.listBlocks(USER, "chat").stream().map(Models.MemoryBlock::label).toList());
    }

    // ── session_summaries：摘要 upsert/get ───────────────────────────

    private static void summariesRoundTrip(MemoryStore store) {
        store.upsertSummary(new Models.SessionSummary(USER, "chat:t1", "此前讨论了咖啡",
                "chat", 5, 3000.0));
        Models.SessionSummary got = store.getSummary(USER, "chat:t1");
        assertEquals("此前讨论了咖啡", got.summary());
        assertEquals("chat", got.agentId());
        assertEquals(5, got.coveredMessageCount());
        assertEquals(3000.0, got.updatedAt());
        assertNull(store.getSummary(USER, "chat:t2"));
    }

    // ── doc_chunks：分块写入/过滤/计数/删除 ──────────────────────────

    private static void chunksRoundTrip(MemoryStore store) {
        byte[] vector = Models.encodeEmbedding(List.of(0.5f, -0.5f));
        store.putChunks(List.of(
                new Models.DocChunk("c1", "f1", "chat:t1", USER, "chat", 1, "文本一",
                        List.of(new long[]{0, 5}), vector, 2, 4000.0),
                new Models.DocChunk("c2", "f1", "chat:t1", USER, "*", 0, "文本零",
                        null, null, null, 4001.0),
                new Models.DocChunk("c3", "f2", "chat:t2", USER, "chat", 0, "另一文件",
                        null, null, null, 4002.0)));

        // list：file_id 升序、ordinal 升序。
        List<Models.DocChunk> all = store.listChunks(USER, null, null, 100);
        assertEquals(List.of("c2", "c1", "c3"),
                all.stream().map(Models.DocChunk::chunkId).toList());

        // agent 过滤：全局块（agent_id='*'）始终可见。
        List<Models.DocChunk> scoped = store.listChunks(USER, "chat", null, 100);
        assertEquals(List.of("c2", "c1", "c3"),
                scoped.stream().map(Models.DocChunk::chunkId).toList());

        // file 过滤。
        List<Models.DocChunk> f1 = store.listChunks(USER, null, "f1", 100);
        assertEquals(List.of("c2", "c1"), f1.stream().map(Models.DocChunk::chunkId).toList());

        // c1 往返：offsets 与向量 bit 级一致。
        Models.DocChunk c1 = all.stream().filter(c -> c.chunkId().equals("c1")).findFirst().orElseThrow();
        assertEquals(1, c1.offsets().size());
        assertArrayEquals(new long[]{0, 5}, c1.offsets().get(0));
        assertArrayEquals(vector, c1.embedding());
        assertEquals(2, c1.embeddingDim());
        assertEquals(4000.0, c1.createdAt());
        // c2 无 offsets（NULL 落库）→ 解码为空表。
        Models.DocChunk c2 = all.stream().filter(c -> c.chunkId().equals("c2")).findFirst().orElseThrow();
        assertTrue(c2.offsets().isEmpty());

        // 计数与分组计数。
        assertEquals(2, store.countChunksForFile("f1"));
        assertEquals(Map.of("f1", 2, "f2", 1), store.chunkCountsByFile(USER));

        // 按文件删除。
        assertEquals(1, store.deleteChunksForFile("f2"));
        assertEquals(0, store.countChunksForFile("f2"));
        assertEquals(Map.of("f1", 2), store.chunkCountsByFile(USER));

        // 回填查询：向量缺失的 c2 入列，已向量的 c1 排除。
        List<Models.DocChunk> needing = store.listChunksNeedingEmbedding(2, 100);
        assertEquals(List.of("c2"), needing.stream().map(Models.DocChunk::chunkId).toList());
    }

    // ── memory_ops + memory_versions：审计与版本史 ───────────────────

    private static void opsAndVersionsRoundTrip(MemoryStore store) {
        store.recordOp(new Models.MemoryOp("o1", "extract", USER, "chat", "chat:t1",
                Map.of("k", "v", "n", 1), "ok", "", 10, 5000.0));
        store.recordOp(new Models.MemoryOp("o2", "consolidate", USER, "chat", "",
                Map.of(), "error", "boom", 20, 5001.0));
        store.recordOp(new Models.MemoryOp("o3", "manual", "u2", "", "",
                Map.of(), "ok", "", 0, 4999.0));

        // 全量：created_at 降序。
        List<Models.MemoryOp> all = store.listOps(null, 100);
        assertEquals(List.of("o2", "o1", "o3"), all.stream().map(Models.MemoryOp::opId).toList());

        // 用户过滤 + limit。
        List<Models.MemoryOp> mine = store.listOps(USER, 100);
        assertEquals(List.of("o2", "o1"), mine.stream().map(Models.MemoryOp::opId).toList());
        assertEquals(List.of("o2"), store.listOps(USER, 1).stream()
                .map(Models.MemoryOp::opId).toList());

        // 往返：detail JSON、错误信息、空串归一。
        Models.MemoryOp o1 = all.stream().filter(o -> o.opId().equals("o1")).findFirst().orElseThrow();
        assertEquals(Map.of("k", "v", "n", 1), o1.detail());
        assertEquals(10, o1.durationMs());
        assertEquals(5000.0, o1.createdAt());
        Models.MemoryOp o2 = all.stream().filter(o -> o.opId().equals("o2")).findFirst().orElseThrow();
        assertEquals("error", o2.status());
        assertEquals("boom", o2.errorText());
        Models.MemoryOp o3 = all.stream().filter(o -> o.opId().equals("o3")).findFirst().orElseThrow();
        assertEquals("", o3.errorText());

        // 版本史：create（previous_content 空串归一 NULL → 读回空串）+ update。
        store.recordVersion(new Models.MemoryVersion("v1", "m1", USER, "create", "内容一",
                "", "active", 6000.0));
        store.recordVersion(new Models.MemoryVersion("v2", "m1", USER, "update", "内容二",
                "内容一", "active", 6001.0));

        List<Models.MemoryVersion> versions = store.listVersions("m1", 50);
        assertEquals(List.of("v2", "v1"), versions.stream()
                .map(Models.MemoryVersion::versionId).toList());
        assertEquals("内容二", versions.get(0).content());
        assertEquals("内容一", versions.get(0).previousContent());
        assertEquals("内容一", versions.get(1).content());
        assertEquals("", versions.get(1).previousContent());

        assertEquals("内容二", store.getVersion("v2").content());
        assertNull(store.getVersion("ghost"));
        assertEquals(List.of("v2"), store.listVersions("m1", 1).stream()
                .map(Models.MemoryVersion::versionId).toList());
    }

    // ── memory_ops：#160/#176 thread/since 过滤 + 游标翻页 ────────────

    private static void opsFilterAndCursorRoundTrip(MemoryStore store) {
        // 独立用户 u3 隔离前面段落的种子（o1/o2 是 USER=u1）。p3/p4 同刻
        // 并列（8002.0），验证 tiebreak 在过滤与游标下的稳定性。
        store.recordOp(new Models.MemoryOp("p1", "extract", "u3", "chat", "chat:t9",
                Map.of(), "ok", "", 1, 8000.0));
        store.recordOp(new Models.MemoryOp("p2", "consolidate", "u3", "chat", "chat:t8",
                Map.of(), "ok", "", 1, 8001.0));
        store.recordOp(new Models.MemoryOp("p3", "manual", "u3", "chat", "chat:t9",
                Map.of(), "ok", "", 1, 8002.0));
        store.recordOp(new Models.MemoryOp("p4", "manual", "u3", "chat", "chat:t9",
                Map.of(), "ok", "", 1, 8002.0));
        store.recordOp(new Models.MemoryOp("p5", "summary", "u3", "chat", "",
                Map.of(), "ok", "", 1, 8003.0));

        // thread 过滤：同刻并列 p4 在前（tiebreak DESC）。
        assertEquals(List.of("p4", "p3", "p1"),
                store.listOps("u3", "chat:t9", null, null, 100)
                        .stream().map(Models.MemoryOp::opId).toList());

        // since 过滤：含边界（>= 8001.0，p2 落在边界上）。
        assertEquals(List.of("p5", "p4", "p3", "p2"),
                store.listOps("u3", null, 8001.0, null, 100)
                        .stream().map(Models.MemoryOp::opId).toList());

        // thread + since 组合：8000.0 的 p1 被 since 排除。
        assertEquals(List.of("p4", "p3"),
                store.listOps("u3", "chat:t9", 8001.5, null, 100)
                        .stream().map(Models.MemoryOp::opId).toList());

        // 游标翻页（limit=2）：边界行含 op_id 时同刻并列稳定前移——无重无漏。
        List<Models.MemoryOp> page1 = store.listOps("u3", null, null, null, 2);
        assertEquals(List.of("p5", "p4"), ids(page1));
        MemoryStore.OpsCursor cursor = cursorOf(page1);
        List<Models.MemoryOp> page2 = store.listOps("u3", null, null, cursor, 2);
        assertEquals(List.of("p3", "p2"), ids(page2));
        cursor = cursorOf(page2);
        List<Models.MemoryOp> page3 = store.listOps("u3", null, null, cursor, 2);
        assertEquals(List.of("p1"), ids(page3));
        cursor = cursorOf(page3);
        assertEquals(List.of(), ids(store.listOps("u3", null, null, cursor, 2)));

        // since + 游标组合：游标只在 since 窗口内翻页。
        List<Models.MemoryOp> scopedPage = store.listOps("u3", null, 8001.0, null, 2);
        assertEquals(List.of("p5", "p4"), ids(scopedPage));
        assertEquals(List.of("p3", "p2"), ids(store.listOps(
                "u3", null, 8001.0, cursorOf(scopedPage), 2)));

        // 游标只带秒值（API 契约，无 op_id）：退化为严格早于该时刻——
        // 同刻并列的 p3 不再返回（其余量语义见 MemoryStore.OpsCursor）。
        assertEquals(List.of("p2", "p1"),
                store.listOps("u3", null, null, new MemoryStore.OpsCursor(8002.0, null), 100)
                        .stream().map(Models.MemoryOp::opId).toList());

        // limit 边界：0 视为 1（与旧签名同一钳制约定）。
        assertEquals(List.of("p5"),
                store.listOps("u3", null, null, null, 0)
                        .stream().map(Models.MemoryOp::opId).toList());
    }

    private static List<String> ids(List<Models.MemoryOp> ops) {
        return ops.stream().map(Models.MemoryOp::opId).toList();
    }

    /** 以上一页末行构造续页游标（created_at + op_id，同刻并列可稳定前移）。 */
    private static MemoryStore.OpsCursor cursorOf(List<Models.MemoryOp> page) {
        Models.MemoryOp last = page.get(page.size() - 1);
        return new MemoryStore.OpsCursor(last.createdAt(), last.opId());
    }

    // ── 线程级联：memories 刻意保留 ─────────────────────────────────

    private static void deleteForThreadKeepsMemories(MemoryStore store) {
        store.deleteForThread("chat:t1");

        // 随线程消亡：摘要与该线程的知识库分块。
        assertNull(store.getSummary(USER, "chat:t1"));
        assertTrue(store.listChunks(USER, null, null, 100).isEmpty());
        assertEquals(0, store.countChunksForFile("f1"));

        // 跨会话记忆独立于线程存活（出处仍在 source_thread_id）。
        assertNotEquals(null, store.getMemory("m1"));
    }

    // ── memories：显式删除 ───────────────────────────────────────────

    private static void deleteMemory(MemoryStore store) {
        assertTrue(store.deleteMemory("m1"));
        assertNull(store.getMemory("m1"));
        assertFalse(store.deleteMemory("m1"));
    }
}
