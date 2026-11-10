package io.github.paopaonb666.agentbase.memory.retrieval;

import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.memory.embeddings.EmbeddingClient;
import io.github.paopaonb666.agentbase.memory.store.InMemoryMemoryStore;
import io.github.paopaonb666.agentbase.memory.store.MemoryStore;
import io.github.paopaonb666.agentbase.memory.store.Models;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 关键词下推的召回语义（#326/327，MEMORY_BM25_PUSHDOWN）：InMemory
 * brute-force 默认实现下推与全量打分排序逐条一致（#325 同款语义基线，
 * 生产 minScore 证据门槛口径）；后端未实现（null）回退全量；与向量下推
 * 并集组合；minScore=0 下的候选收窄语义锁定（无豁免类——登记的容量
 * 取舍）。SQL 分数不进混合分，一致口径是排序与集合。
 */
class Bm25PushdownTest {

    private static final double NOW = 1_800_000_000.0;

    private static EmbeddingClient neverEmbedder() {
        return new EmbeddingClient() {
            @Override public Integer dims() { return null; }
            @Override public List<List<Float>> embed(List<String> texts) { return null; }
            @Override public boolean probe() { return false; }
        };
    }

    private static Models.MemoryRecord record(String id, String content, int ageHours) {
        double t = NOW - ageHours * 3600.0;
        return new Models.MemoryRecord(id, "u1", "chat", "semantic", content,
                List.of(), null, null, 0.5, "active", "", List.of(), t, t, null, 0);
    }

    private static Settings settings(boolean bm25Pushdown) {
        return Settings.load(io.github.paopaonb666.agentbase.memory.retrieval.Bm25PushdownTest
                .envOf(bm25Pushdown), null);
    }

    private static java.util.Map<String, String> envOf(boolean bm25Pushdown) {
        java.util.Map<String, String> env = new java.util.HashMap<>();
        env.put("ENV", "development");
        env.put("MEMORY_BM25_PUSHDOWN", Boolean.toString(bm25Pushdown));
        return java.util.Collections.unmodifiableMap(env);
    }

    private static List<String> rank(List<Retrieval.ScoredMemory> scored) {
        return scored.stream().map(item -> item.record().memoryId()).toList();
    }

    @Test
    void inMemoryBruteForcePushdownPreservesBaselineRanking() {
        // 生产口径（minScore 证据门槛）：brute-force 关键词下推（N≥命中数）
        // 与全量打分排序逐条一致。
        Settings baselineSettings = settings(false);
        Settings pushdownSettings = settings(true);
        InMemoryMemoryStore store = new InMemoryMemoryStore();
        store.upsertMemory(record("k1", "生产数据库部署在 Ubuntu 服务器", 1));
        store.upsertMemory(record("k2", "数据库扩容先加只读副本", 2));
        store.upsertMemory(record("k3", "数据库密码只放密钥管理服务", 3));
        store.upsertMemory(record("n1", "团队周会固定在周一上午", 4));

        List<Retrieval.ScoredMemory> baseline = Retrieval.recallMemories(store, neverEmbedder(),
                "u1", null, "数据库", 10, 0.05, baselineSettings);
        List<Retrieval.ScoredMemory> pushdown = Retrieval.recallMemories(store, neverEmbedder(),
                "u1", null, "数据库", 10, 0.05, pushdownSettings);

        assertEquals(rank(baseline), rank(pushdown), "brute-force 下推不得改变排序");
        assertEquals(Set.of("k1", "k2", "k3"), Set.copyOf(rank(baseline)));
    }

    @Test
    void storeWithoutKeywordIndexFallsBackToFullScan() {
        // SPI 默认实现返回 null（未实现）：开关打开时回退全量——检索结果
        // 与关闭开关完全一致。
        Settings pushdownSettings = settings(true);
        InMemoryMemoryStore raw = new InMemoryMemoryStore();
        raw.upsertMemory(record("k1", "生产数据库部署在 Ubuntu 服务器", 1));
        raw.upsertMemory(record("n1", "团队周会固定在周一上午", 2));
        // SPI 默认 searchMemoryIds 返回 null 的后端：动态代理剥掉 raw 的
        // 关键词下推能力（其余方法全部委托）。
        MemoryStore noKeywordIndex = (MemoryStore) java.lang.reflect.Proxy.newProxyInstance(
                MemoryStore.class.getClassLoader(), new Class<?>[]{MemoryStore.class},
                (proxy, method, args) -> {
                    if (method.getName().equals("searchMemoryIds")) {
                        return null;
                    }
                    try {
                        return method.invoke(raw, args);
                    } catch (java.lang.reflect.InvocationTargetException e) {
                        throw e.getCause();
                    }
                });

        List<Retrieval.ScoredMemory> pushdown = Retrieval.recallMemories(noKeywordIndex,
                neverEmbedder(), "u1", null, "数据库", 10, 0.05, pushdownSettings);
        assertEquals(List.of("k1"), rank(pushdown), "null 回退应等同全量打分");
    }

    @Test
    void strictNarrowingDropsNonKeywordCandidatesAtZeroGate() {
        // 语义锁定（登记的容量取舍）：minScore=0 时基线会返回纯 recency
        // 候选；关键词下推无豁免类——非命中候选被挤出候选池。
        InMemoryMemoryStore store = new InMemoryMemoryStore();
        store.upsertMemory(record("k1", "数据库备份在凌晨三点", 5));
        store.upsertMemory(record("n1", "无关的会议记录", 1)); // 更新 → recency 最前

        List<Retrieval.ScoredMemory> baseline = Retrieval.recallMemories(store, neverEmbedder(),
                "u1", null, "数据库", 10, 0.0, settings(false));
        List<Retrieval.ScoredMemory> pushdown = Retrieval.recallMemories(store, neverEmbedder(),
                "u1", null, "数据库", 10, 0.0, settings(true));

        assertTrue(rank(baseline).contains("n1"), "基线（minScore=0）含纯 recency 候选");
        assertFalse(rank(pushdown).contains("n1"), "下推收窄挤出非命中候选");
        assertEquals(List.of("k1"), rank(pushdown));
    }

    @Test
    void unionWithVectorPushdownKeepsKeywordRelevantFarVectors() {
        // 双开：候选池 = 关键词 top-N ∪ KNN top-100——关键词命中但向量远离
        // 的候选仍在池（排序由混合打分决定，不丢）。
        EmbeddingClient stub = new EmbeddingClient() {
            @Override public Integer dims() { return 2; }
            @Override public List<List<Float>> embed(List<String> texts) {
                return texts.stream().map(t -> List.of(1.0f, 0.0f)).toList();
            }
            @Override public boolean probe() { return true; }
        };
        InMemoryMemoryStore store = new InMemoryMemoryStore();
        // k1：关键词命中、向量与查询正交（远离）。
        byte[] farVector = Models.encodeEmbedding(List.of(0.0f, 1.0f));
        double t = NOW - 3600.0;
        store.upsertMemory(new Models.MemoryRecord("k1", "u1", "chat", "semantic",
                "数据库备份在凌晨三点", List.of(), farVector, 2, 0.5, "active", "",
                List.of(), t, t, null, 0));
        // v1：向量贴近查询、无关键词交集。
        byte[] nearVector = Models.encodeEmbedding(List.of(1.0f, 0.0f));
        store.upsertMemory(new Models.MemoryRecord("v1", "u1", "chat", "semantic",
                "完全不同的表述甲乙丙丁", List.of(), nearVector, 2, 0.5, "active", "",
                List.of(), t, t, null, 0));

        VectorIndexPort index = VectorIndexPort.bruteForce();
        index.upsert("k1", Models.decodeEmbedding(farVector));
        index.upsert("v1", Models.decodeEmbedding(nearVector));

        Settings both = Settings.load(java.util.Map.of(
                "ENV", "development",
                "MEMORY_BM25_PUSHDOWN", "true",
                "MEMORY_VECTOR_PUSHDOWN", "true"), null);
        List<Retrieval.ScoredMemory> pushed = Retrieval.recallMemories(store, stub,
                "u1", null, "数据库", 10, 0.05, both, index);
        assertTrue(rank(pushed).contains("k1"),
                "关键词命中但向量远离的候选应在并集池中：rank=" + rank(pushed));
    }

    @Test
    void mysqlFulltextDdlDeclaresNgramParser() {
        // MySQL 侧静态形态锁定（#327）：ngram parser 声明在守卫 ALTER 里
        // （本地无 MySQL 服务器，SQL 正确性由部署集成验证）。
        String ddl = io.github.paopaonb666.agentbase.memory.store.MemoryDdl.MYSQL_FULLTEXT_MEM_CONTENT;
        assertTrue(ddl.contains("ADD FULLTEXT INDEX"), ddl);
        assertTrue(ddl.contains("WITH PARSER ngram"), ddl);
        assertTrue(ddl.contains("idx_mem_content_ft"), ddl);
    }
}
