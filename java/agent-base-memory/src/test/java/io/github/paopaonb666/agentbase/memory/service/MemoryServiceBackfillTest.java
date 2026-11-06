package io.github.paopaonb666.agentbase.memory.service;

import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.memory.embeddings.EmbeddingClient;
import io.github.paopaonb666.agentbase.memory.store.MemoryStore;
import io.github.paopaonb666.agentbase.memory.store.Models;
import io.github.paopaonb666.agentbase.memory.store.Models.MemoryRecord;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 向量回填任务测试（roadmap #164）：换 embedding 模型后的迁移入口——
 * 单飞（重复启动拒绝）、进度可见、完成状态落账；嵌入未配置快速失败。
 */
class MemoryServiceBackfillTest {

    /** 只实现回填路径的存储替身。 */
    static final class BackfillStore implements MemoryStore {
        final List<MemoryRecord> rows;
        final List<MemoryRecord> upserted = new CopyOnWriteArrayList<>();

        BackfillStore(List<MemoryRecord> pending) {
            this.rows = new java.util.concurrent.CopyOnWriteArrayList<>(pending);
        }

        @Override
        public List<MemoryRecord> listMemoriesNeedingEmbedding(int dims, int limit) {
            // 口径对齐 SQL（embedding IS NULL OR embedding_dim IS NULL OR
            // embedding_dim != ?）；注意 MemoryRecord 紧凑构造把 null 归一为
            // 空 byte[]——"未向量化"的判据是长度 0。
            return rows.stream()
                    .filter(r -> r.embedding().length == 0 || r.embeddingDim() == null
                            || r.embeddingDim() != dims)
                    .limit(limit)
                    .toList();
        }

        @Override
        public void upsertMemory(MemoryRecord record) {
            upserted.add(record);
            // 与真实库同语义：upsert 覆盖行（重查不再命中待回填条件）。
            rows.removeIf(r -> r.memoryId().equals(record.memoryId()));
            rows.add(record);
        }

        @Override
        public List<Models.DocChunk> listChunksNeedingEmbedding(int dims, int limit) {
            return List.of();
        }

        @Override
        public int deleteChunksForFile(String fileId) {
            return 0;
        }

        private static UnsupportedOperationException unsupported() {
            return new UnsupportedOperationException("回填测试未实现");
        }

        @Override public void upsertBlock(Models.MemoryBlock block) { throw unsupported(); }
        @Override public Models.MemoryBlock getBlock(String userId, String agentId, String label) { throw unsupported(); }
        @Override public List<Models.MemoryBlock> listBlocks(String userId, String agentId) { throw unsupported(); }
        @Override public boolean deleteBlock(String userId, String agentId, String label) { throw unsupported(); }
        @Override public void upsertSummary(Models.SessionSummary summary) { throw unsupported(); }
        @Override public Models.SessionSummary getSummary(String userId, String threadId) { throw unsupported(); }
        @Override public void putChunks(List<Models.DocChunk> chunks) { throw unsupported(); }
        @Override public List<Models.DocChunk> listChunks(String userId, String agentId, String fileId, int limit) { throw unsupported(); }
        @Override public Map<String, Integer> chunkCountsByFile(String userId) { throw unsupported(); }
        @Override public void recordOp(Models.MemoryOp op) { throw unsupported(); }
        @Override public List<Models.MemoryOp> listOps(String userId, int limit) { throw unsupported(); }
        @Override public void recordVersion(Models.MemoryVersion version) { throw unsupported(); }
        @Override public List<Models.MemoryVersion> listVersions(String memoryId, int limit) { throw unsupported(); }
        @Override public Models.MemoryVersion getVersion(String versionId) { throw unsupported(); }
        @Override public int countChunksForFile(String fileId) { throw unsupported(); }
        @Override public void deleteForThread(String threadId) { throw unsupported(); }
        @Override public MemoryRecord getMemory(String memoryId) { throw unsupported(); }
        @Override public boolean deleteMemory(String memoryId) { throw unsupported(); }
        @Override public List<MemoryRecord> listMemories(String userId, String agentId, boolean includeGlobal,
                                                        List<String> kinds, List<String> statuses,
                                                        int limit, boolean excludeProfile) { throw unsupported(); }
        @Override public void touchMemories(List<String> memoryIds) { throw unsupported(); }
    }

    private static MemoryRecord record(String id) {
        return new MemoryRecord(id, "u1", "chat", "semantic", "内容 " + id, List.of(),
                null, null, 0.5, "active", "", List.of(), 1000.0, 1000.0, null, 0);
    }

    private static EmbeddingClient stubEmbedder(CountDownLatch slow) {
        return new EmbeddingClient() {
            @Override public Integer dims() { return 2; }
            @Override public List<List<Float>> embed(List<String> texts) {
                if (slow != null) {
                    try {
                        slow.await(5, TimeUnit.SECONDS);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                }
                return texts.stream().map(t -> List.of(0.5f, 0.5f)).toList();
            }
            @Override public boolean probe() { return true; }
        };
    }

    @Test
    void backfillRunsToDoneWithProgressAndSingleFlight() throws Exception {
        CountDownLatch slow = new CountDownLatch(1);
        BackfillStore store = new BackfillStore(List.of(record("m1"), record("m2"), record("m3")));
        MemoryService service = new MemoryService(store, stubEmbedder(slow),
                Settings.load(Map.of("ENV", "development")), null, null);

        assertTrue(service.startBackfill(2), "首次启动应成功");
        assertFalse(service.startBackfill(2), "运行中重复启动应被拒绝（端点转 409）");
        assertEquals("running", service.backfillStatus().get("state"));
        slow.countDown(); // 放行 embed

        long deadline = System.currentTimeMillis() + 5_000;
        while (!"done".equals(service.backfillStatus().get("state"))
                && System.currentTimeMillis() < deadline) {
            Thread.sleep(20);
        }
        assertEquals("done", service.backfillStatus().get("state"));
        assertEquals(3, service.backfillStatus().get("processed"));
        assertEquals(3, store.upserted.size());
        // 回填保持原 updated_at（维护操作不刷时间衰减分）。
        assertEquals(1000.0, store.upserted.get(0).updatedAt());
        // 2 维向量已写入。
        assertEquals(2, Models.decodeEmbedding(store.upserted.get(0).embedding()).size());
    }

    @Test
    void backfillFailsFastWithoutEmbedding() {
        BackfillStore store = new BackfillStore(List.of(record("m1")));
        EmbeddingClient nullEmbedder = new EmbeddingClient() {
            @Override public Integer dims() { return null; }
            @Override public List<List<Float>> embed(List<String> texts) { return null; }
            @Override public boolean probe() { return true; }
        };
        MemoryService service = new MemoryService(store, nullEmbedder,
                Settings.load(Map.of("ENV", "development")), null, null);
        assertThrows(Models.MemoryStoreError.class, () -> service.startBackfill(64));
        assertEquals("idle", service.backfillStatus().get("state"));
    }
}
