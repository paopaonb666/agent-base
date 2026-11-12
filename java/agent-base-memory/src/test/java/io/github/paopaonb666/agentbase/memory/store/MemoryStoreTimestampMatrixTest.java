package io.github.paopaonb666.agentbase.memory.store;

import io.github.paopaonb666.agentbase.core.config.Settings;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 时间戳矩阵 + 同秒并列 tiebreak + 跨后端语义矩阵（roadmap #301，#171
 * 并入）：两个 SQL 方言的 toSqlTs/fromSqlTs 往返与容错、同秒并列 ops 的
 * 稳定排序（sqlite rowid DESC / InMemory 插入序）、以及 InMemory 与
 * sqlite 在同一固定种子协议下产出一致的观察视图。MySQL 后端不在此跑
 * 活体协议（需服务器），其方言差异收敛在时间戳转换层单独锁定。
 */
class MemoryStoreTimestampMatrixTest {

    @TempDir
    Path tempDir;

    // ── sqlite 时间戳矩阵 ─────────────────────────────────────────────────

    @Test
    void sqliteTimestampsRoundTripAtMicrosecondPrecision() {
        try (SqliteMemoryStore store = new SqliteMemoryStore(
                tempDir.resolve("ts-" + System.nanoTime() + ".db").toString())) {
            for (double epoch : new double[]{0.0, 1000.5, 1700000000.123456, 32503680000.0}) {
                Object written = store.toSqlTs(epoch);
                assertTrue(String.valueOf(written).matches(
                        "\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}\\.\\d{6}"), String.valueOf(written));
                double back = store.fromSqlTs(written);
                assertEquals(epoch, back, 1e-6, "epoch=" + epoch);
            }
            // 亚微秒截断：纳秒位在写入格式（微秒）处舍入，往返误差 < 2µs。
            double back = store.fromSqlTs(store.toSqlTs(1700000000.1234567));
            assertEquals(1700000000.1234567, back, 2e-6);
            // 无小数位（旧数据/手工行）可解析。期望值经共享转换器计算
            // （本地时区相关，不硬编码 UTC 偏移）。
            assertEquals(AbstractSqlMemoryStore.toEpochSeconds(
                    LocalDateTime.of(2023, 11, 15, 6, 13, 20)),
                    store.fromSqlTs("2023-11-15 06:13:20"), 1e-9);
        }
    }

    @Test
    void sqliteMalformedTimestampFallsBackToNowWithWarning() {
        try (SqliteMemoryStore store = new SqliteMemoryStore(
                tempDir.resolve("ts-bad-" + System.nanoTime() + ".db").toString())) {
            double before = System.currentTimeMillis() / 1000.0;
            double fallback = store.fromSqlTs("not-a-timestamp");
            double after = System.currentTimeMillis() / 1000.0;
            // 坏时间戳不得伪装成 0（会污染时间衰减排序）：按当前时间兜底。
            assertTrue(fallback >= before && fallback <= after, String.valueOf(fallback));
        }
    }

    // ── mysql 时间戳矩阵（转换层；DATETIME 读取三种形态） ────────────────

    @Test
    void mysqlTimestampMatrixCoversAllThreeReadShapes() {
        // 构造不建连（连接懒开启），转换层测试无需 MySQL 服务器。
        Settings settings = Settings.load(Map.of(), null);
        try (MysqlMemoryStore store = new MysqlMemoryStore(settings)) {
            // 写入形态：LocalDateTime 原样（交给 JDBC 驱动）。
            LocalDateTime written = (LocalDateTime) store.toSqlTs(1700000000.123456);
            double back = store.fromSqlTs(written);
            assertEquals(1700000000.123456, back, 1e-6);

            // 读取形态 1：驱动返回 java.sql.Timestamp（期望经共享转换器算，
            // 本地时区相关）。
            java.sql.Timestamp ts = java.sql.Timestamp.valueOf("2023-11-15 06:13:20.123456");
            double expected = AbstractSqlMemoryStore.toEpochSeconds(
                    LocalDateTime.of(2023, 11, 15, 6, 13, 20, 123456000));
            assertEquals(expected, store.fromSqlTs(ts), 1e-6);

            // 读取形态 2：字符串（迁移/手工行）。
            assertEquals(AbstractSqlMemoryStore.toEpochSeconds(
                    LocalDateTime.of(2023, 11, 15, 6, 13, 20)),
                    store.fromSqlTs("2023-11-15 06:13:20"), 1e-9);

            // 读取形态 3：坏值兜底 now。
            double before = System.currentTimeMillis() / 1000.0;
            double fallback = store.fromSqlTs("garbage");
            assertTrue(fallback >= before, String.valueOf(fallback));
        }
    }

    // ── 同秒并列 tiebreak ─────────────────────────────────────────────────

    @Test
    void sameSecondOpsOrderNewestFirstOnBothBackends() {
        // 同一 createdAt 的两条 op：后写者在前（sqlite rowid DESC；
        // InMemory (op, 插入序) 二元组）——语义跨后端一致。
        Models.MemoryOp first = new Models.MemoryOp("o-first", "extract", "u", "chat",
                "chat:t", Map.of(), "ok", "", 1, 5000.0);
        Models.MemoryOp second = new Models.MemoryOp("o-second", "extract", "u", "chat",
                "chat:t", Map.of(), "ok", "", 1, 5000.0);

        InMemoryMemoryStore inMemory = new InMemoryMemoryStore();
        inMemory.recordOp(first);
        inMemory.recordOp(second);

        try (SqliteMemoryStore sqlite = new SqliteMemoryStore(
                tempDir.resolve("tie-" + System.nanoTime() + ".db").toString())) {
            sqlite.recordOp(first);
            sqlite.recordOp(second);
            assertEquals(List.of("o-second", "o-first"),
                    inMemory.listOps("u", 10).stream().map(Models.MemoryOp::opId).toList());
            assertEquals(List.of("o-second", "o-first"),
                    sqlite.listOps("u", 10).stream().map(Models.MemoryOp::opId).toList());
        }
    }

    @Test
    void sameMillisecondMemoryOrderStableAcrossSqliteReplace() {
        // review B-P3：memories 的 updated_at 同刻并列组必须有确定性次序键
        // ——sqlite 的 INSERT OR REPLACE 是删旧插新，重写即翻 rowid，修复前
        // 同刻并列组的返回序在两次查询间互换（RRF 通道排名/top-k 边界随之
        // 漂移，候选序是排序与并列判定的底序）。修复=ORDER BY 追加
        // memory_id DESC（与 ops 的 opsTiebreak 同款收口）。
        double same = 5000.0;
        Models.MemoryRecord a = new Models.MemoryRecord("m-aaa", "u", "chat", "semantic",
                "内容甲", List.of(), null, null, 0.5, "active", "", List.of(),
                same, same, null, 0);
        Models.MemoryRecord b = new Models.MemoryRecord("m-bbb", "u", "chat", "semantic",
                "内容乙", List.of(), null, null, 0.5, "active", "", List.of(),
                same, same, null, 0);
        try (SqliteMemoryStore sqlite = new SqliteMemoryStore(
                tempDir.resolve("tie-mem-" + System.nanoTime() + ".db").toString())) {
            sqlite.upsertMemory(a);
            sqlite.upsertMemory(b);
            List<String> before = sqlite.listMemories("u", null, true, null,
                    List.of("active"), 100, false).stream()
                    .map(Models.MemoryRecord::memoryId).toList();
            // 重写 m-aaa（REPLACE 翻 rowid）：返回序必须不变（memory_id tiebreak）。
            sqlite.upsertMemory(a);
            List<String> after = sqlite.listMemories("u", null, true, null,
                    List.of("active"), 100, false).stream()
                    .map(Models.MemoryRecord::memoryId).toList();
            assertEquals(before, after, "同刻并列组的返回序在重写后不得翻转");
            assertEquals(List.of("m-bbb", "m-aaa"), after, "tiebreak=memory_id DESC");
        }
    }

    @Test
    void inMemoryTtlFilterRunsBeforeLimitLikeSql() {
        // review B-P2：InMemory 的 8 参 listMemories 此前先按 limit 截断再
        // 做 episodic TTL 过滤——过期行占满名额后被滤掉，同调用 SQL 后端
        // 满额活候选、InMemory 短额（实现与自身 javadoc/#168 登记相反）。
        // 修复=TTL 判定移入扫描内、先于截断（对齐 SQL 下推）。真判据：
        // 2 条最新但过期的 episodic + 3 条活行、limit=3 → 满额活候选
        //（修复前 InMemory 只返回 1 条，本测试必红）。
        InMemoryMemoryStore inMemory = new InMemoryMemoryStore();
        try (SqliteMemoryStore sqlite = new SqliteMemoryStore(
                tempDir.resolve("ttl-limit-" + System.nanoTime() + ".db").toString())) {
            for (MemoryStore store : List.of(inMemory, sqlite)) {
                store.upsertMemory(new Models.MemoryRecord("stale-1", "u", "chat", "episodic",
                        "过期一", List.of(), null, null, 0.5, "active", "", List.of(),
                        4900.0, 4900.0, null, 0));
                store.upsertMemory(new Models.MemoryRecord("stale-2", "u", "chat", "episodic",
                        "过期二", List.of(), null, null, 0.5, "active", "", List.of(),
                        4800.0, 4800.0, null, 0));
                store.upsertMemory(new Models.MemoryRecord("live-1", "u", "chat", "semantic",
                        "活一", List.of(), null, null, 0.5, "active", "", List.of(),
                        3000.0, 3000.0, null, 0));
                store.upsertMemory(new Models.MemoryRecord("live-2", "u", "chat", "semantic",
                        "活二", List.of(), null, null, 0.5, "active", "", List.of(),
                        2000.0, 2000.0, null, 0));
                store.upsertMemory(new Models.MemoryRecord("live-3", "u", "chat", "semantic",
                        "活三", List.of(), null, null, 0.5, "active", "", List.of(),
                        1000.0, 1000.0, null, 0));
            }
            for (MemoryStore store : List.of(inMemory, sqlite)) {
                List<Models.MemoryRecord> view = store.listMemories("u", null, true, null,
                        List.of("active"), 3, false, 5000.0);
                assertEquals(List.of("live-1", "live-2", "live-3"),
                        view.stream().map(Models.MemoryRecord::memoryId).toList(),
                        store.getClass().getSimpleName() + "：过期行不占名额（limit=3 满额活候选）");
            }
        }
    }

    @Test
    void listOpsCrossUserThreadFilterMatchesInMemory() {
        // review B-P5：listOps(userId=null, threadId≠null) 此前生成
        // `FROM memory_ops AND thread_id = ?` 的非法 SQL（WHERE 只由 userId
        // 发起）——当前接线不可达（调用方 userId 来自认证非空），纯潜伏；
        // InMemory 版对同组合语义正常。修复=AND 改按「是否已有 WHERE」拼接。
        // 真判据：SQL 后端同组合不炸且与 InMemory 视图一致（修复前必红：
        // SQLException）。
        InMemoryMemoryStore inMemory = new InMemoryMemoryStore();
        try (SqliteMemoryStore sqlite = new SqliteMemoryStore(
                tempDir.resolve("ops-null-user-" + System.nanoTime() + ".db").toString())) {
            for (MemoryStore store : List.of(inMemory, sqlite)) {
                store.recordOp(new Models.MemoryOp("o1", "extract", "u1", "chat", "chat:t1",
                        Map.of(), "ok", "", 1, 5000.0));
                store.recordOp(new Models.MemoryOp("o2", "extract", "u2", "chat", "chat:t1",
                        Map.of(), "ok", "", 1, 5000.0));
                store.recordOp(new Models.MemoryOp("o3", "extract", "u1", "chat", "chat:t2",
                        Map.of(), "ok", "", 1, 5000.0));
            }
            List<String> expected = inMemory.listOps(null, "chat:t1", null, null, 10).stream()
                    .map(Models.MemoryOp::opId).toList();
            assertEquals(List.of("o2", "o1"), expected,
                    "InMemory 基线：跨用户按 thread 过滤（同刻 tiebreak rowid/插入序 DESC）");
            assertEquals(expected,
                    sqlite.listOps(null, "chat:t1", null, null, 10).stream()
                            .map(Models.MemoryOp::opId).toList(),
                    "SQL 后端同组合不得炸非法 SQL、语义与 InMemory 一致");
        }
    }

    // ── 跨后端语义矩阵：固定种子 → 相同观察视图 ──────────────────────────

    @Test
    void inMemoryAndSqliteProduceIdenticalViewsForSameSeed() {
        try (SqliteMemoryStore sqlite = new SqliteMemoryStore(
                tempDir.resolve("sem-" + System.nanoTime() + ".db").toString())) {
            InMemoryMemoryStore inMemory = new InMemoryMemoryStore();
            seed(inMemory);
            seed(sqlite);

            // memories：updated_at 降序的 id 序一致。
            assertEquals(memoryIds(inMemory), memoryIds(sqlite));
            // blocks：label 升序；字段逐项一致（无 byte[] 组件，record 直接比）。
            assertEquals(inMemory.listBlocks("u", "chat"), sqlite.listBlocks("u", "chat"));
            // summaries：逐项一致。
            assertEquals(inMemory.getSummary("u", "chat:t1"), sqlite.getSummary("u", "chat:t1"));
            // chunks：file/ordinal 排序的 id 序一致；标量字段一致。
            assertEquals(chunkIds(inMemory), chunkIds(sqlite));
            // ops：created_at 降序（含同秒 tiebreak）逐项一致。
            assertEquals(inMemory.listOps(null, 100), sqlite.listOps(null, 100));
            // versions：新→旧。
            assertEquals(inMemory.listVersions("m1", 50).stream().map(Models.MemoryVersion::versionId).toList(),
                    sqlite.listVersions("m1", 50).stream().map(Models.MemoryVersion::versionId).toList());
        }
    }

    @Test
    void ttlPushdownAndOpDetailSnapshotMatchAcrossBackends() {
        // #168 矩阵种子 2：InMemory 覆写 8 参 TTL 下推后，与 SQL 的
        // 「过期 episodic 不占候选名额（恰等边界行保留）」语义一致；
        // recordOp 的 detail 是快照（调用方事后 mutate 不回溯审计行）。
        try (SqliteMemoryStore sqlite = new SqliteMemoryStore(
                tempDir.resolve("ttl-" + System.nanoTime() + ".db").toString())) {
            InMemoryMemoryStore inMemory = new InMemoryMemoryStore();
            for (MemoryStore store : List.of(inMemory, sqlite)) {
                store.upsertMemory(new Models.MemoryRecord("fresh", "u", "chat", "episodic",
                        "新事件", List.of(), null, null, 0.5, "active", "", List.of(),
                        5000.0, 5000.0, null, 0));
                // updatedAt == 截止线（恰等边界，保留）。
                store.upsertMemory(new Models.MemoryRecord("boundary", "u", "chat", "episodic",
                        "边界事件", List.of(), null, null, 0.5, "active", "", List.of(),
                        4000.0, 4000.0, null, 0));
                store.upsertMemory(new Models.MemoryRecord("stale", "u", "chat", "episodic",
                        "过期事件", List.of(), null, null, 0.5, "active", "", List.of(),
                        3000.0, 3999.0, null, 0));
                store.upsertMemory(new Models.MemoryRecord("old-semantic", "u", "chat", "semantic",
                        "旧但非 episodic（不受 TTL 影响）", List.of(), null, null,
                        0.5, "active", "", List.of(), 2000.0, 2000.0, null, 0));
            }
            for (MemoryStore store : List.of(inMemory, sqlite)) {
                List<Models.MemoryRecord> view = store.listMemories("u", null, true, null,
                        List.of("active"), 100, false, 4000.0);
                assertEquals(List.of("fresh", "boundary", "old-semantic"),
                        view.stream().map(Models.MemoryRecord::memoryId).toList(),
                        store.getClass().getSimpleName() + "：过期行不占名额、边界行保留");
            }

            // detail 快照：recordOp 后 mutate 原 Map，listOps 读回不变（两后端）。
            for (MemoryStore store : List.of(inMemory, sqlite)) {
                java.util.Map<String, Object> detail = new java.util.LinkedHashMap<>();
                detail.put("k", "v");
                store.recordOp(new Models.MemoryOp("mx", "extract", "mu", "chat", "",
                        detail, "ok", "", 0, 6000.0));
                detail.put("k", "mutated-after-record");
                List<Models.MemoryOp> ops = store.listOps("mu", 10);
                assertEquals("v", ops.get(0).detail().get("k"),
                        store.getClass().getSimpleName() + "：detail 应是写入时快照");
            }
        }
    }

    /** 固定时间戳的种子数据（µs 可表示值，避免 sqlite 微秒精度漂移）。 */
    private static void seed(MemoryStore store) {
        store.upsertMemory(new Models.MemoryRecord("m1", "u", "chat", "semantic", "内容一",
                List.of("t"), null, null, 0.5, "active", "chat:t1", List.of(), 1000.0, 1002.0, null, 0));
        store.upsertMemory(new Models.MemoryRecord("m2", "u", "*", "episodic", "内容二",
                List.of(), null, null, 0.5, "active", "", List.of(), 1001.0, 1003.0, null, 0));
        store.upsertBlock(new Models.MemoryBlock("u", "chat", "persona", "人设", 2000, 1, 2000.0));
        store.upsertBlock(new Models.MemoryBlock("u", "*", "human", "画像", 100, 1, 2001.0));
        store.upsertSummary(new Models.SessionSummary("u", "chat:t1", "摘要", "chat", 3, 3000.0));
        store.putChunks(List.of(
                new Models.DocChunk("c2", "f1", "chat:t1", "u", "chat", 1, "段二",
                        null, null, null, 4000.0),
                new Models.DocChunk("c1", "f1", "chat:t1", "u", "chat", 0, "段一",
                        List.of(new long[]{0, 2}), null, null, 4000.0)));
        store.recordOp(new Models.MemoryOp("o1", "extract", "u", "chat", "chat:t1",
                Map.of("n", 1), "ok", "", 1, 5000.0));
        store.recordOp(new Models.MemoryOp("o2", "consolidate", "u", "chat", "",
                Map.of(), "ok", "", 1, 5000.0));
        store.recordVersion(new Models.MemoryVersion("v1", "m1", "u", "create", "内容一",
                "", "active", 6000.0));
    }

    private static List<String> memoryIds(MemoryStore store) {
        return store.listMemories("u", null, true, null, List.of("active"), 100, false)
                .stream().map(Models.MemoryRecord::memoryId).toList();
    }

    private static List<String> chunkIds(MemoryStore store) {
        return store.listChunks("u", null, null, 100)
                .stream().map(Models.DocChunk::chunkId).toList();
    }
}
