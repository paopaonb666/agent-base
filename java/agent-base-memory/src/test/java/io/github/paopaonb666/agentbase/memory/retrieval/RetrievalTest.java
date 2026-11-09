package io.github.paopaonb666.agentbase.memory.retrieval;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.memory.embeddings.EmbeddingClient;
import io.github.paopaonb666.agentbase.memory.retrieval.Retrieval.ScoredChunk;
import io.github.paopaonb666.agentbase.memory.retrieval.Retrieval.ScoredMemory;
import io.github.paopaonb666.agentbase.memory.store.MemoryStore;
import io.github.paopaonb666.agentbase.memory.store.Models;
import io.github.paopaonb666.agentbase.memory.store.Models.DocChunk;
import io.github.paopaonb666.agentbase.memory.store.Models.MemoryRecord;

class RetrievalTest {

    private static final double DAY = 86400.0;

    /** 测试替身：只实现召回路径用到的两个方法。 */
    static final class FakeStore implements MemoryStore {
        final List<MemoryRecord> records;
        final List<String> touched = new ArrayList<>();
        int listCalls = 0;

        FakeStore(List<MemoryRecord> records) {
            this.records = records;
        }

        @Override
        public List<MemoryRecord> listMemories(String userId, String agentId, boolean includeGlobal,
                                               List<String> kinds, List<String> statuses,
                                               int limit, boolean excludeProfile) {
            listCalls++;
            assertTrue(limit == Retrieval.CANDIDATE_LIMIT, "候选上限应为 " + Retrieval.CANDIDATE_LIMIT);
            assertEquals(List.of("active"), statuses);
            return records;
        }

        @Override
        public void touchMemories(List<String> memoryIds) {
            touched.addAll(memoryIds);
        }

        private static UnsupportedOperationException unsupported() {
            return new UnsupportedOperationException("测试替身未实现");
        }

        @Override public void upsertMemory(MemoryRecord record) { throw unsupported(); }
        @Override public MemoryRecord getMemory(String memoryId) { throw unsupported(); }
        @Override public boolean deleteMemory(String memoryId) { throw unsupported(); }
        @Override public List<MemoryRecord> listMemoriesNeedingEmbedding(int dims, int limit) { throw unsupported(); }
        @Override public void upsertBlock(Models.MemoryBlock block) { throw unsupported(); }
        @Override public Models.MemoryBlock getBlock(String userId, String agentId, String label) { throw unsupported(); }
        @Override public List<Models.MemoryBlock> listBlocks(String userId, String agentId) { throw unsupported(); }
        @Override public boolean deleteBlock(String userId, String agentId, String label) { throw unsupported(); }
        @Override public Models.SessionSummary getSummary(String userId, String threadId) { throw unsupported(); }
        @Override public void upsertSummary(Models.SessionSummary summary) { throw unsupported(); }
        @Override public void putChunks(List<DocChunk> chunks) { throw unsupported(); }
        @Override public List<DocChunk> listChunks(String userId, String agentId, String fileId, int limit) { throw unsupported(); }
        @Override public int countChunksForFile(String fileId) { throw unsupported(); }
        @Override public Map<String, Integer> chunkCountsByFile(String userId) { throw unsupported(); }
        @Override public int deleteChunksForFile(String fileId) { throw unsupported(); }
        @Override public List<DocChunk> listChunksNeedingEmbedding(int dims, int limit) { throw unsupported(); }
        @Override public void recordOp(Models.MemoryOp op) { throw unsupported(); }
        @Override public List<Models.MemoryOp> listOps(String userId, int limit) { throw unsupported(); }
        @Override public void recordVersion(Models.MemoryVersion version) { throw unsupported(); }
        @Override public List<Models.MemoryVersion> listVersions(String memoryId, int limit) { throw unsupported(); }
        @Override public Models.MemoryVersion getVersion(String versionId) { throw unsupported(); }
        @Override public void deleteForThread(String threadId) { throw unsupported(); }
    }

    /** 固定向量的测试替身 embedder。 */
    record FixedEmbedding(int dimension, List<List<Float>> vectors) implements EmbeddingClient {
        @Override
        public Integer dims() {
            return dimension;
        }

        @Override
        public List<List<Float>> embed(List<String> texts) {
            return vectors;
        }

        @Override
        public boolean probe() {
            return true;
        }
    }

    private static MemoryRecord rec(String id, String kind, String content, byte[] embedding,
                                    double salience, double updatedAt) {
        return new MemoryRecord(id, "u1", "*", kind, content, List.of(), embedding,
                embedding == null || embedding.length == 0 ? null : embedding.length / 4,
                salience, "active", "", List.of(), updatedAt, updatedAt, null, 0);
    }

    // ── 分词 ─────────────────────────────────────────────────────────

    @Test
    void tokenizeMixedAsciiAndCjk() {
        // ASCII 词在前（小写），CJK run 的二元组在后。
        assertEquals(List.of("hello", "world", "abc_123", "用户", "户偏", "偏好", "世界"),
                Retrieval.tokenize("Hello World 用户偏好 ABC_123 世界"));
        assertEquals(List.of("好"), Retrieval.tokenize("好"));
        assertEquals(List.of("用户", "户偏", "偏好"), Retrieval.tokenize("用户偏好"));
        assertEquals(List.of(), Retrieval.tokenize(" .,! "));
    }

    // ── BM25 手算对照 ────────────────────────────────────────────────

    @Test
    void bm25ScoresMatchHandComputation() {
        // docs = [[cat], [cat, dog]]；query = [cat]
        // n=2, avgdl=1.5, df(cat)=2
        // idf = ln(1 + (2-2+0.5)/(2+0.5)) = ln(1.2)
        // doc0: denom = 1 + 1.5*(1-0.75+0.75*(1/1.5)) = 2.125 → ln(1.2)*1*2.5/2.125
        // doc1: denom = 1 + 1.5*(1-0.75+0.75*(2/1.5)) = 2.875 → ln(1.2)*1*2.5/2.875
        Retrieval.Bm25 bm25 = new Retrieval.Bm25(List.of(List.of("cat"), List.of("cat", "dog")));
        double[] scores = bm25.scores(List.of("cat"));
        assertEquals(2, scores.length);
        assertEquals(Math.log(1.2) * 1.0 * (Retrieval.BM25_K1 + 1.0) / 2.125, scores[0], 1e-12);
        assertEquals(Math.log(1.2) * 1.0 * (Retrieval.BM25_K1 + 1.0) / 2.875, scores[1], 1e-12);
        // 重复查询词按 Python 语义重复计分。
        assertEquals(scores[0] * 2, bm25.scores(List.of("cat", "cat"))[0], 1e-12);
        // 空候选集返回空分数表。
        assertEquals(0, new Retrieval.Bm25(List.of()).scores(List.of("cat")).length);
    }

    @Test
    void bm25CacheReusesAndEvolves() {
        Retrieval.Bm25Cache cache = new Retrieval.Bm25Cache(2);
        List<String> texts = List.of("cat", "cat dog");
        double[] first = cache.scores(texts, List.of("cat"));
        assertEquals(1, cache.size());
        double[] second = cache.scores(texts, List.of("cat"));
        assertEquals(1, cache.size(), "相同候选集（内容即身份）应命中同一索引");
        org.junit.jupiter.api.Assertions.assertArrayEquals(first, second);
        // 内容变更 → 不同键 → 新索引条目。
        cache.scores(List.of("cat", "cat dog CHANGED"), List.of("cat"));
        assertEquals(2, cache.size());
        // 容量淘汰：超过 maxsize 后最久未用的旧键被逐出。
        cache.scores(List.of("unrelated"), List.of("x"));
        assertEquals(2, cache.size());
        // 逐出后再次访问仍能得到一致结果（重建）。
        double[] rebuilt = cache.scores(texts, List.of("cat"));
        org.junit.jupiter.api.Assertions.assertArrayEquals(first, rebuilt);
        // maxsize=0 显式禁用：不驻留任何条目。
        Retrieval.Bm25Cache disabled = new Retrieval.Bm25Cache(0);
        org.junit.jupiter.api.Assertions.assertArrayEquals(first, disabled.scores(texts, List.of("cat")));
        assertEquals(0, disabled.size());
    }

    // ── 基础因子 ─────────────────────────────────────────────────────

    @Test
    void cosineSimilarityContract() {
        assertEquals(1.0, Retrieval.cosineSimilarity(List.of(1.0f, 0.0f), List.of(1.0f, 0.0f)),
                1e-12);
        assertEquals(-1.0, Retrieval.cosineSimilarity(List.of(1.0f, 0.0f), List.of(-2.0f, 0.0f)),
                1e-12);
        assertNull(Retrieval.cosineSimilarity(List.of(1.0f, 0.0f), List.of(1.0f, 0.0f, 0.0f)),
                "维度不一致不可比");
        assertNull(Retrieval.cosineSimilarity(List.of(), List.of()));
        assertNull(Retrieval.cosineSimilarity(List.of(0.0f, 0.0f), List.of(0.0f, 0.0f)),
                "零向量不可比");
    }

    @Test
    void recencyFactorHalfLife() {
        double now = 1_700_000_000.0;
        assertEquals(1.0, Retrieval.recencyFactor(now, now, 14.0), 1e-12);
        assertEquals(0.5, Retrieval.recencyFactor(now - 14 * DAY, now, 14.0), 1e-12);
        assertEquals(0.25, Retrieval.recencyFactor(now - 28 * DAY, now, 14.0), 1e-12);
        assertTrue(Retrieval.recencyFactor(now - 3650 * DAY, now, 14.0) > 0.0, "永不归零");
        assertEquals(1.0, Retrieval.recencyFactor(now + 5, now, 14.0), 1e-12, "未来时间戳按 0 年龄");
    }

    @Test
    void episodicTtlOnlyFiltersEpisodic() {
        double now = 1_700_000_000.0;
        MemoryRecord oldEpisodic = rec("e1", "episodic", "旧情节", null, 0.5, now - 31 * DAY);
        MemoryRecord freshEpisodic = rec("e2", "episodic", "新情节", null, 0.5, now - 29 * DAY);
        MemoryRecord oldSemantic = rec("s1", "semantic", "旧知识", null, 0.5, now - 400 * DAY);
        List<MemoryRecord> kept = Retrieval.filterExpired(
                List.of(oldEpisodic, freshEpisodic, oldSemantic), 30, now);
        assertEquals(List.of("e2", "s1"), kept.stream().map(MemoryRecord::memoryId).toList());
        // 0 = 不过期。
        List<MemoryRecord> all = Retrieval.filterExpired(
                List.of(oldEpisodic, freshEpisodic, oldSemantic), 0, now);
        assertEquals(3, all.size());
    }

    // ── 四分量加权 ───────────────────────────────────────────────────

    @Test
    void hybridScoreCombinesFourComponents() {
        double now = 1_700_000_000.0;
        MemoryRecord record = rec("m1", "semantic", "用户偏好设置",
                Models.encodeEmbedding(List.of(1.0f, 0.0f)), 0.9, now - 7 * DAY);
        List<ScoredMemory> out = Retrieval.scoreMemories(List.of(record), "偏好设置",
                List.of(1.0f, 0.0f), now, 14.0, Retrieval.Weights.DEFAULT);
        assertEquals(1, out.size());
        ScoredMemory scored = out.get(0);
        Map<String, Double> components = scored.components();

        // 分量装配顺序与 Python dict 一致：keyword → recency → salience → vector。
        assertEquals(List.of("keyword", "recency", "salience", "vector"),
                List.copyOf(components.keySet()));

        // 单候选命中全部 3 个二元组 → 归一化关键词分量 = 1。
        assertEquals(1.0, scored.keywordScore(), 1e-12);
        assertEquals(Math.pow(0.5, 0.5), scored.recencyScore(), 1e-12);
        assertEquals(0.9, scored.salienceScore(), 1e-12);
        assertEquals(1.0, scored.vectorScore(), 1e-12);

        double expected = 0.3 * 1.0 + 0.15 * Math.pow(0.5, 0.5) + 0.15 * 0.9 + 0.4 * 1.0;
        assertEquals(expected, scored.score(), 1e-12);
    }

    @Test
    void missingVectorCountsZeroAndCeilingIsRecencyPlusSalience() {
        double now = 1_700_000_000.0;
        // 无 embedding 且与查询无关键词交集：score ≤ 0.3（无证据下限语义）。
        MemoryRecord noise = rec("m1", "semantic", "完全无关的内容", null, 1.0, now);
        List<ScoredMemory> out = Retrieval.scoreMemories(List.of(noise), "偏好设置",
                List.of(1.0f, 0.0f), now, 14.0, Retrieval.Weights.DEFAULT);
        ScoredMemory scored = out.get(0);
        assertFalse(scored.components().containsKey("vector"));
        // keyword/recency 恒在（缺失按 0 计），只有 salience/vector 可缺席。
        assertEquals(0.0, scored.keywordScore(), 1e-12);
        assertEquals(0.15 + 0.15, scored.score(), 1e-12);
    }

    @Test
    void dimensionMismatchAndNegativeCosineHandled() {
        double now = 1_700_000_000.0;
        MemoryRecord threeDim = rec("m1", "semantic", "任意",
                Models.encodeEmbedding(List.of(1.0f, 0.0f, 0.0f)), 0.5, now);
        List<ScoredMemory> out = Retrieval.scoreMemories(List.of(threeDim), "任意",
                List.of(1.0f, 0.0f), now, 14.0, Retrieval.Weights.DEFAULT);
        assertFalse(out.get(0).components().containsKey("vector"), "维度不一致不计向量分量");

        MemoryRecord opposite = rec("m2", "semantic", "任意",
                Models.encodeEmbedding(List.of(-3.0f, 0.0f)), 0.5, now);
        List<ScoredMemory> out2 = Retrieval.scoreMemories(List.of(opposite), "任意",
                List.of(1.0f, 0.0f), now, 14.0, Retrieval.Weights.DEFAULT);
        assertEquals(0.0, out2.get(0).vectorScore(), 1e-12, "负余弦截到 0");
    }

    @Test
    void recencyUsesUpdatedAtNotLastAccessedAt() {
        double now = 1_700_000_000.0;
        MemoryRecord record = new MemoryRecord("m1", "u1", "*", "semantic", "任意", List.of(),
                null, null, 0.5, "active", "", List.of(), now - 28 * DAY, now - 28 * DAY,
                now, 99);
        List<ScoredMemory> out = Retrieval.scoreMemories(List.of(record), "不命中",
                null, now, 14.0, Retrieval.Weights.DEFAULT);
        assertEquals(0.25, out.get(0).recencyScore(), 1e-9,
                "时间基点必须是 updated_at（若用 last_accessed_at 会得到 ~1.0）");
    }

    @Test
    void scoreChunksUseCreatedAtAndNoSalience() {
        double now = 1_700_000_000.0;
        DocChunk chunk = new DocChunk("c1", "f1", "t1", "u1", "*", 0, "用户偏好设置",
                null, Models.encodeEmbedding(List.of(1.0f, 0.0f)), 2, now - 7 * DAY);
        List<ScoredChunk> out = Retrieval.scoreChunks(List.of(chunk), "偏好设置",
                List.of(1.0f, 0.0f), now, 14.0, Retrieval.Weights.DEFAULT);
        ScoredChunk scored = out.get(0);
        assertEquals(1.0, scored.keywordScore(), 1e-12);
        assertEquals(1.0, scored.vectorScore(), 1e-12);
        // ScoredChunk 没有 recencyScore() 便捷访问器（分块语义上无显著度）；
        // 时间分量经 components 读取。
        assertEquals(Math.pow(0.5, 0.5), scored.components().get("recency"), 1e-12);
        assertFalse(scored.components().containsKey("salience"), "分块没有显著度分量");
        assertEquals(0.3 + 0.15 * Math.pow(0.5, 0.5) + 0.4, scored.score(), 1e-12);
    }

    @Test
    void scoreMemoriesReturnsScoreDescendingOrder() {
        double now = 1_700_000_000.0;
        MemoryRecord strong = rec("strong", "semantic", "用户偏好设置", null, 0.9, now);
        MemoryRecord weak = rec("weak", "semantic", "完全无关", null, 0.1, now - 100 * DAY);
        List<ScoredMemory> out = Retrieval.scoreMemories(List.of(weak, strong), "偏好设置",
                null, now, 14.0, Retrieval.Weights.DEFAULT);
        assertEquals(List.of("strong", "weak"), out.stream().map(s -> s.record().memoryId()).toList());
    }

    // ── 一次完整召回 ─────────────────────────────────────────────────

    @Test
    void recallMemoriesEndToEndWithDoubleGate() {
        double now = Models.now();
        MemoryRecord profile = rec("profile:u1", "semantic", "画像内容", null, 1.0, now);
        MemoryRecord hit = rec("m1", "episodic", "Java 咖啡 爪哇", null, 0.5, now);
        MemoryRecord noise = rec("m2", "semantic", "python 蛇", null, 0.5, now - 400 * DAY);
        MemoryRecord expired = rec("m3", "episodic", "Java 咖啡 过期情节", null, 0.5,
                now - 31 * DAY);
        FakeStore store = new FakeStore(List.of(profile, hit, noise, expired));
        Settings config = new Settings();
        config.memory.episodicTtlDays = 30;

        List<ScoredMemory> result = Retrieval.recallMemories(store,
                new FixedEmbedding(2, List.of(List.of(1.0f, 0.0f))),
                "u1", "*", "Java 咖啡", 1, 0.35, config);

        // 画像被排除、过期 episodic 被过滤、噪音记忆被双门槛挡住。
        assertEquals(1, result.size());
        assertEquals("m1", result.get(0).record().memoryId());
        assertEquals(1.0, result.get(0).keywordScore(), 1e-12);
        // 访问记账只针对被召回者。
        assertEquals(List.of("m1"), store.touched);
        assertEquals(1, store.listCalls);

        // min_score=0：门槛跳过，噪音记忆（分数低）仍可按 top_k 返回。
        FakeStore store2 = new FakeStore(new ArrayList<>(List.of(profile, hit, noise, expired)));
        List<ScoredMemory> result2 = Retrieval.recallMemories(store2, null,
                "u1", "*", "Java 咖啡", 2, 0.0, config);
        assertEquals(List.of("m1", "m2"),
                result2.stream().map(s -> s.record().memoryId()).toList());
        // embedder 为 null → 无向量分量，纯关键词路径照常工作。
        assertNull(result2.get(0).components().get("vector"));
    }

    @Test
    void recallMemoriesTopKAndCandidateLimit() {
        double now = Models.now();
        Settings config = new Settings();
        config.memory.episodicTtlDays = 0;
        List<MemoryRecord> records = new ArrayList<>();
        for (int i = 0; i < Retrieval.CANDIDATE_LIMIT; i++) {
            records.add(rec("m" + i, "semantic", "噪音内容", null, 0.5, now));
        }
        FakeStore store = new FakeStore(records);
        List<ScoredMemory> result = Retrieval.recallMemories(store, null, "u1", "*",
                "毫无交集的查询", 3, 0.0, config);
        // 达到候选上限时仍正常返回 top-k（不因容量告警而失败）。
        assertEquals(3, result.size());
        assertEquals(1, store.listCalls);
    }

    @Test
    void settingsBasedConvenienceOverloads() {
        Settings config = new Settings();
        MemoryRecord record = rec("m1", "semantic", "用户偏好设置", null, 0.9, Models.now());
        List<ScoredMemory> out = Retrieval.scoreMemories(List.of(record), "偏好设置", null, config);
        assertEquals(1, out.size());
        assertTrue(out.get(0).score() > 0.29);
        // 自定义权重生效（Settings.memory.weightXxx 可覆盖默认）。
        config.memory.weightVector = 0.10;
        config.memory.weightKeyword = 0.70;
        config.memory.weightRecency = 0.10;
        config.memory.weightSalience = 0.10;
        List<ScoredMemory> reweighted = Retrieval.scoreMemories(List.of(record), "偏好设置",
                null, config);
        ScoredMemory s = reweighted.get(0);
        double expected = 0.7 * s.keywordScore() + 0.1 * s.recencyScore() + 0.1 * s.salienceScore();
        assertEquals(expected, s.score(), 1e-12);
    }
}
