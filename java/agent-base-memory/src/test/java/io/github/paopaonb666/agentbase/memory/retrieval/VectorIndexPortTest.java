package io.github.paopaonb666.agentbase.memory.retrieval;

import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.memory.embeddings.EmbeddingClient;
import io.github.paopaonb666.agentbase.memory.store.MemoryStore;
import io.github.paopaonb666.agentbase.memory.store.Models;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * VectorIndexPort 测试（roadmap #324/旧 #35）：brute-force 默认实现的
 * 余弦 top-K 语义、维度失配排除、upsert/delete 跟随；以及 #325 的下推
 * 语义基线——brute-force 下推与全量打分逐条一致（golden 语料回归）。
 */
class VectorIndexPortTest {

    private static List<Float> vec(float... values) {
        List<Float> out = new java.util.ArrayList<>(values.length);
        for (float v : values) {
            out.add(v);
        }
        return out;
    }

    @Test
    void bruteForceReturnsCosineTopK() {
        VectorIndexPort index = VectorIndexPort.bruteForce();
        index.upsert("a", vec(1.0f, 0.0f));
        index.upsert("b", vec(0.0f, 1.0f));
        index.upsert("c", vec(0.9f, 0.1f));

        List<VectorIndexPort.VectorHit> hits = index.search(vec(1.0f, 0.0f), 2);
        assertEquals(2, hits.size());
        assertEquals("a", hits.get(0).id());
        assertEquals(1.0, hits.get(0).similarity(), 1e-9);
        assertEquals("c", hits.get(1).id());
        // 删除后不再命中。
        index.delete("a");
        hits = index.search(vec(1.0f, 0.0f), 2);
        assertEquals("c", hits.get(0).id());
    }

    @Test
    void dimensionMismatchIsExcludedAndNullQueryIsEmpty() {
        VectorIndexPort index = VectorIndexPort.bruteForce();
        index.upsert("good", vec(1.0f, 0.0f));
        index.upsert("odd", vec(1.0f, 0.0f, 0.0f));
        List<VectorIndexPort.VectorHit> hits = index.search(vec(1.0f, 0.0f), 5);
        assertEquals(1, hits.size());
        assertEquals("good", hits.get(0).id());
        // null 查询 / topK<=0 → 空表（下推在无查询向量时自然不收窄）。
        assertTrue(index.search(null, 5).isEmpty());
        assertTrue(index.search(vec(1.0f), 0).isEmpty());
        // 空向量不入索引（无异常维度条目混入）。
        index.upsert("empty", List.of());
        assertEquals(1, index.search(vec(1.0f, 0.0f), 10).size());
    }

    @Test
    void pushdownWithBruteForcePreservesBaselineRanking() {
        // #325 语义基线：带向量的语料下，brute-force 下推（KNN top-100
        // 圈定候选池）与全量打分的排序逐条一致。
        double now = 1_800_000_000.0;
        List<Models.MemoryRecord> corpus = new java.util.ArrayList<>();
        for (int i = 0; i < 40; i++) {
            List<Float> vector = vec((i % 5) * 0.2f + 0.1f, ((i * 7) % 5) * 0.2f + 0.1f);
            corpus.add(new Models.MemoryRecord("m" + i, "u", "chat", "semantic",
                    "记忆内容 " + i + " 主题" + (i % 5), List.of(),
                    Models.encodeEmbedding(vector), 2, 0.5, "active", "",
                    List.of(), now - i, now - i, null, 0));
        }
        Settings settings = Settings.load(java.util.Map.of("ENV", "development"), null);
        // 假 embedder：查询向量固定，dims 非空（判定 hasEmbedding 用）。
        EmbeddingClient stub = new EmbeddingClient() {
            @Override public Integer dims() { return 2; }
            @Override public List<List<Float>> embed(List<String> texts) {
                return texts.stream().map(t -> vec(0.3f, 0.5f)).toList();
            }
            @Override public boolean probe() { return true; }
        };
        MemoryStore store = new RetrievalTest.FakeStore(corpus);
        // 索引跟随写路径填充（生产由 MemoryService 完成；测试直接喂入）。
        VectorIndexPort index = VectorIndexPort.bruteForce();
        for (Models.MemoryRecord record : corpus) {
            index.upsert(record.memoryId(), Models.decodeEmbedding(record.embedding()));
        }
        List<Retrieval.ScoredMemory> baseline = Retrieval.recallMemories(store, stub,
                "u", null, "主题3", 10, 0.0, settings);
        List<Retrieval.ScoredMemory> pushdown = Retrieval.recallMemories(store, stub,
                "u", null, "主题3", 10, 0.0, settings, index);
        assertEquals(baseline.size(), pushdown.size());
        for (int i = 0; i < baseline.size(); i++) {
            assertEquals(baseline.get(i).record().memoryId(), pushdown.get(i).record().memoryId(),
                    "下推（brute-force）不应改变排序：rank " + i);
        }
    }
}
