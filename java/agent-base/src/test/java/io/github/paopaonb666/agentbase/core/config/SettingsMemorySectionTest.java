package io.github.paopaonb666.agentbase.core.config;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

/**
 * memory 节配置契约的装载面（#93 覆盖率补强批）：applyMemory 的
 * containsKey 分支此前只有零散键被既有测试触达——本类用一组完整合法键
 * 走通全部解析分支（值断言），再用少量非法值锁定错误分支的文案。
 * 键名与 .env.example 逐项同源（MEMORY_EMBEDDING_* / MEMORY_WEIGHT_* /
 * MEMORY_DOC_CHUNK_* 等）。
 */
class SettingsMemorySectionTest {

    @Test
    void memorySectionKeysAreHonoredEndToEnd() {
        Map<String, String> env = new HashMap<>();
        env.put("ENV", "development");
        env.put("MEMORY_EMBEDDING_ENABLED", "true");
        env.put("MEMORY_EMBEDDING_BASE_URL", "http://emb.example.com/v1");
        env.put("MEMORY_EMBEDDING_API_KEY", "sk-test");
        env.put("MEMORY_EMBEDDING_MODEL", "text-embedding-3-small");
        env.put("MEMORY_EMBEDDING_DIMS", "1536");
        env.put("MEMORY_EMBEDDING_BATCH_SIZE", "16");
        env.put("MEMORY_EMBEDDING_TIMEOUT_SECONDS", "5.5");
        env.put("MEMORY_RECALL_TOP_K", "8");
        env.put("MEMORY_RECALL_MIN_SCORE", "0.35");
        env.put("MEMORY_DEDUP_PREFILTER", "false");
        env.put("MEMORY_DEDUP_SKIP_COSINE", "0.9");
        env.put("MEMORY_SALIENCE_EXEMPT_THRESHOLD", "0.95");
        // 四权重和必须为 1.0（±0.01）。
        env.put("MEMORY_WEIGHT_VECTOR", "0.40");
        env.put("MEMORY_WEIGHT_KEYWORD", "0.30");
        env.put("MEMORY_WEIGHT_RECENCY", "0.20");
        env.put("MEMORY_WEIGHT_SALIENCE", "0.10");
        env.put("MEMORY_CONTEXT_MAX_CHARS", "9000");
        env.put("MEMORY_CONTEXT_MAX_TOKENS", "6000");
        env.put("MEMORY_CONTEXT_SNAPSHOT_MAX", "64");
        env.put("MEMORY_CAPTURE_ENABLED", "false");
        env.put("MEMORY_CAPTURE_EVERY_TURNS", "3");
        env.put("MEMORY_PIPELINE_PROFILE", "all_main");
        env.put("MEMORY_SUMMARY_ENABLED", "false");
        env.put("MEMORY_SUMMARY_TRIGGER_MESSAGES", "20");
        env.put("MEMORY_PROFILE_ENABLED", "false");
        env.put("MEMORY_PROFILE_MAX_CHARS", "1500");
        env.put("MEMORY_EPISODIC_TTL_DAYS", "7");
        env.put("MEMORY_TIME_DECAY_HALF_LIFE_DAYS", "30");
        env.put("MEMORY_VECTOR_PUSHDOWN", "true");
        env.put("MEMORY_BM25_PUSHDOWN", "true");
        env.put("MEMORY_EXTRACTION_MAX_INPUT_CHARS", "9000");
        env.put("MEMORY_EXTRACTION_MAX_MESSAGES", "24");
        env.put("MEMORY_DOC_CHUNK_CHARS", "800");
        env.put("MEMORY_DOC_CHUNK_OVERLAP", "80");
        env.put("MEMORY_MYSQL_POOL_SIZE", "4");
        env.put("MEMORY_AUTH_SECRET", "coverage-secret");
        env.put("MEMORY_AUTH_REPLAY_WINDOW_SECONDS", "120");
        env.put("MEMORY_AUTH_REPLAY_CACHE_LIMIT", "5000");

        Settings s = Settings.load(env, null);
        var t = s.memory;
        assertTrue(t.embeddingEnabled);
        assertEquals("http://emb.example.com/v1", t.embeddingBaseUrl);
        assertEquals("sk-test", t.embeddingApiKey.value());
        assertEquals("text-embedding-3-small", t.embeddingModel);
        assertEquals(1536, t.embeddingDims);
        assertEquals(16, t.embeddingBatchSize);
        assertEquals(5.5, t.embeddingTimeoutSeconds, 1e-9);
        assertEquals(8, t.recallTopK);
        assertEquals(0.35, t.recallMinScore, 1e-9);
        assertFalse(t.dedupPrefilter);
        assertEquals(0.9, t.dedupSkipCosine, 1e-9);
        assertEquals(0.95, t.salienceExemptThreshold, 1e-9);
        assertEquals(0.40, t.weightVector, 1e-9);
        assertEquals(0.30, t.weightKeyword, 1e-9);
        assertEquals(0.20, t.weightRecency, 1e-9);
        assertEquals(0.10, t.weightSalience, 1e-9);
        assertEquals(9000, t.contextMaxChars);
        assertEquals(6000, t.contextMaxTokens);
        assertEquals(64, t.contextSnapshotMax);
        assertFalse(t.captureEnabled);
        assertEquals(3, t.captureEveryTurns);
        assertEquals("all_main", t.pipelineProfile);
        assertFalse(t.summaryEnabled);
        assertEquals(20, t.summaryTriggerMessages);
        assertFalse(t.profileEnabled);
        assertEquals(1500, t.profileMaxChars);
        assertEquals(7, t.episodicTtlDays);
        assertEquals(30.0, t.timeDecayHalfLifeDays, 1e-9);
        assertTrue(t.vectorPushdown);
        assertTrue(t.bm25Pushdown);
        assertEquals(9000, t.extractionMaxInputChars);
        assertEquals(24, t.extractionMaxMessages);
        assertEquals(800, t.docChunkChars);
        assertEquals(80, t.docChunkOverlap);
        assertEquals(4, t.mysqlPoolSize);
        assertEquals("coverage-secret", t.authSecret.value());
        assertEquals(120, t.authReplayWindowSeconds);
        assertEquals(5000, t.authReplayCacheLimit);
    }

    @Test
    void csvOrJsonListAcceptsJsonArrayForm() {
        // parseCsvOrJsonList/splitJsonStrings 的 JSON 数组分支：
        // 带引号元素与前后空白都按 JSON 语义解析（非逐字符 CSV 切）。
        Settings s = Settings.load(Map.of(
                "ENV", "development",
                "CORS_ORIGINS", " [ \"https://a.example.com\" , \"https://b.example.com\" ] "), null);
        assertEquals(List.of("https://a.example.com", "https://b.example.com"), s.corsOrigins);
    }

    @Test
    void memorySectionValueErrorsCarryKeyNames() {
        assertThrows(Settings.SettingsError.class,
                () -> Settings.load(Map.of(
                        "ENV", "development", "MEMORY_EMBEDDING_DIMS", "0"), null),
                "embedding_dims 必须 > 0");
        assertThrows(Settings.SettingsError.class,
                () -> Settings.load(Map.of(
                        "ENV", "development", "MEMORY_RECALL_MIN_SCORE", "1.5"), null),
                "recall_min_score 必须在 [0,1)");
        Settings.SettingsError weightSum = assertThrows(Settings.SettingsError.class,
                () -> Settings.load(Map.of(
                        "ENV", "development", "MEMORY_WEIGHT_VECTOR", "0.9"), null),
                "权重和偏离 1.0 即拒");
        assertTrue(weightSum.getMessage().contains("must sum to 1.0"), weightSum.getMessage());
        Settings.SettingsError overlap = assertThrows(Settings.SettingsError.class,
                () -> Settings.load(Map.of(
                        "ENV", "development",
                        "MEMORY_DOC_CHUNK_CHARS", "500",
                        "MEMORY_DOC_CHUNK_OVERLAP", "500"), null),
                "overlap >= chunk_chars 是死循环配置");
        assertTrue(overlap.getMessage().contains("doc_chunk_overlap"), overlap.getMessage());
        assertThrows(Settings.SettingsError.class,
                () -> Settings.load(Map.of(
                        "ENV", "development", "MEMORY_PIPELINE_PROFILE", "nope"), null),
                "未知 pipeline profile 拒绝");
    }
}
