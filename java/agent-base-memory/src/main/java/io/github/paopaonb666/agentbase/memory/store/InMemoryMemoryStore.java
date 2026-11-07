package io.github.paopaonb666.agentbase.memory.store;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 进程内字典实现的记忆后端：memory 后端（重启即丢）与测试用。
 *
 * <p>对应 Python 侧 {@code agent_base_memory/store/memory_backend.py} 的
 * {@code MemoryMemoryStore}：按表分 map、插入序保序（Java 用 LinkedHashMap
 * 复刻 Python dict 的插入序），审计/版本的 {@code (op, 插入序号)} 二元组
 * 保证同秒并列时按插入顺序稳定排序。Python 依赖 asyncio 单线程语义不加锁；
 * Java 侧接口契约要求并发安全，这里统一在实例锁上同步（行为等价）。
 */
public final class InMemoryMemoryStore implements MemoryStore {

    private final Object lock = new Object();
    private final Map<String, Models.MemoryRecord> memories = new LinkedHashMap<>();
    private final Map<BlockKey, Models.MemoryBlock> blocks = new LinkedHashMap<>();
    private final Map<SummaryKey, Models.SessionSummary> summaries = new LinkedHashMap<>();
    private final Map<String, Models.DocChunk> chunks = new LinkedHashMap<>();
    // (op, 插入序号) 二元组：同秒并列时按插入顺序稳定排序。
    private final List<OpEntry> ops = new ArrayList<>();
    private final List<VersionEntry> versions = new ArrayList<>();
    private int opCounter = 0;

    /** memory_blocks 的三元组主键（对应 Python 的 (user_id, agent_id, label) 元组键）。 */
    private record BlockKey(String userId, String agentId, String label) {}

    /** session_summaries 的二元组主键（对应 Python 的 (user_id, thread_id) 元组键）。 */
    private record SummaryKey(String userId, String threadId) {}

    private record OpEntry(Models.MemoryOp op, int seq) {}

    private record VersionEntry(Models.MemoryVersion version, int seq) {}

    @Override
    public void upsertMemory(Models.MemoryRecord record) {
        if (!Models.KNOWN_MEMORY_KINDS.contains(record.kind())) {
            throw new Models.MemoryStoreError(
                    "unknown memory kind '%s'; expected one of %s"
                            .formatted(record.kind(), Models.KNOWN_MEMORY_KINDS));
        }
        if (!Models.KNOWN_MEMORY_STATUSES.contains(record.status())) {
            throw new Models.MemoryStoreError(
                    "unknown memory status '%s'; expected one of %s"
                            .formatted(record.status(), Models.KNOWN_MEMORY_STATUSES));
        }
        synchronized (lock) {
            memories.put(record.memoryId(), record);
        }
    }

    @Override
    public Models.MemoryRecord getMemory(String memoryId) {
        synchronized (lock) {
            return memories.get(memoryId);
        }
    }

    @Override
    public boolean deleteMemory(String memoryId) {
        synchronized (lock) {
            return memories.remove(memoryId) != null;
        }
    }

    @Override
    public List<Models.MemoryRecord> listMemories(String userId, String agentId,
                                                  boolean includeGlobal, List<String> kinds,
                                                  List<String> statuses, int limit,
                                                  boolean excludeProfile) {
        List<String> kindList = kinds == null ? null : List.copyOf(kinds);
        List<String> statusList = statuses == null ? List.of() : List.copyOf(statuses);
        synchronized (lock) {
            List<Models.MemoryRecord> matched = new ArrayList<>();
            for (Models.MemoryRecord record : memories.values()) {
                if (!record.userId().equals(userId)) {
                    continue;
                }
                if (!statusList.isEmpty() && !statusList.contains(record.status())) {
                    continue;
                }
                // 与 Python 一致：kinds 为 null 不过滤；kinds 为空表则什么都匹配不上
                // （SQL 侧对空表是不加 IN 子句——两实现的既有差异，保持原样移植）。
                if (kindList != null && !kindList.contains(record.kind())) {
                    continue;
                }
                if (excludeProfile && record.memoryId().startsWith(Models.PROFILE_ID_PREFIX)) {
                    continue;
                }
                if (agentId != null && !record.agentId().equals(agentId)
                        && !(includeGlobal && record.agentId().equals("*"))) {
                    continue;
                }
                matched.add(record);
            }
            matched.sort(Comparator.comparingDouble(Models.MemoryRecord::updatedAt).reversed());
            return new ArrayList<>(matched.subList(0, Math.min(matched.size(), Math.max(1, limit))));
        }
    }

    @Override
    public void touchMemories(List<String> memoryIds) {
        if (memoryIds == null || memoryIds.isEmpty()) {
            return;
        }
        double now = Models.now();
        synchronized (lock) {
            for (String mid : memoryIds) {
                Models.MemoryRecord record = memories.get(mid);
                if (record == null) {
                    continue;
                }
                memories.put(mid, new Models.MemoryRecord(
                        record.memoryId(), record.userId(), record.agentId(), record.kind(),
                        record.content(), record.tags(), record.embedding(), record.embeddingDim(),
                        record.salience(), record.status(), record.sourceThreadId(),
                        record.sourceRefs(), record.createdAt(), record.updatedAt(),
                        now, record.accessCount() + 1));
            }
        }
    }

    @Override
    public List<Models.MemoryRecord> listMemoriesNeedingEmbedding(int dims, int limit) {
        synchronized (lock) {
            List<Models.MemoryRecord> matched = new ArrayList<>();
            for (Models.MemoryRecord record : memories.values()) {
                if (!record.status().equals("active")
                        || record.memoryId().startsWith(Models.PROFILE_ID_PREFIX)
                        || (record.embedding().length > 0 && record.embeddingDim() != null
                                && record.embeddingDim() == dims)) {
                    continue;
                }
                matched.add(record);
            }
            matched.sort(Comparator.comparingDouble(Models.MemoryRecord::updatedAt).reversed());
            return new ArrayList<>(matched.subList(0, Math.min(matched.size(), Math.max(1, limit))));
        }
    }

    @Override
    public void upsertBlock(Models.MemoryBlock block) {
        synchronized (lock) {
            blocks.put(new BlockKey(block.userId(), block.agentId(), block.label()), block);
        }
    }

    @Override
    public Models.MemoryBlock getBlock(String userId, String agentId, String label) {
        synchronized (lock) {
            return blocks.get(new BlockKey(userId, agentId, label));
        }
    }

    @Override
    public List<Models.MemoryBlock> listBlocks(String userId, String agentId) {
        synchronized (lock) {
            List<Models.MemoryBlock> out = new ArrayList<>();
            for (Map.Entry<BlockKey, Models.MemoryBlock> e : blocks.entrySet()) {
                if (e.getKey().userId().equals(userId) && e.getKey().agentId().equals(agentId)) {
                    out.add(e.getValue());
                }
            }
            out.sort(Comparator.comparing(Models.MemoryBlock::label));
            return out;
        }
    }

    @Override
    public boolean deleteBlock(String userId, String agentId, String label) {
        synchronized (lock) {
            return blocks.remove(new BlockKey(userId, agentId, label)) != null;
        }
    }

    @Override
    public Models.SessionSummary getSummary(String userId, String threadId) {
        synchronized (lock) {
            return summaries.get(new SummaryKey(userId, threadId));
        }
    }

    @Override
    public void upsertSummary(Models.SessionSummary summary) {
        synchronized (lock) {
            summaries.put(new SummaryKey(summary.userId(), summary.threadId()), summary);
        }
    }

    @Override
    public void putChunks(List<Models.DocChunk> put) {
        synchronized (lock) {
            for (Models.DocChunk chunk : put) {
                chunks.put(chunk.chunkId(), chunk);
            }
        }
    }

    @Override
    public List<Models.DocChunk> listChunks(String userId, String agentId, String fileId,
                                            int limit) {
        synchronized (lock) {
            List<Models.DocChunk> matched = new ArrayList<>();
            for (Models.DocChunk chunk : chunks.values()) {
                if (!chunk.userId().equals(userId)) {
                    continue;
                }
                if (agentId != null && !chunk.agentId().equals(agentId)
                        && !chunk.agentId().equals("*")) {
                    continue;
                }
                if (fileId != null && !chunk.fileId().equals(fileId)) {
                    continue;
                }
                matched.add(chunk);
            }
            matched.sort(Comparator.comparing(Models.DocChunk::fileId)
                    .thenComparingInt(Models.DocChunk::ordinal));
            return new ArrayList<>(matched.subList(0, Math.min(matched.size(), Math.max(1, limit))));
        }
    }

    @Override
    public int countChunksForFile(String fileId) {
        synchronized (lock) {
            int count = 0;
            for (Models.DocChunk chunk : chunks.values()) {
                if (chunk.fileId().equals(fileId)) {
                    count++;
                }
            }
            return count;
        }
    }

    @Override
    public Map<String, Integer> chunkCountsByFile(String userId) {
        synchronized (lock) {
            Map<String, Integer> counts = new LinkedHashMap<>();
            for (Models.DocChunk chunk : chunks.values()) {
                if (chunk.userId().equals(userId)) {
                    counts.merge(chunk.fileId(), 1, Integer::sum);
                }
            }
            return counts;
        }
    }

    @Override
    public int deleteChunksForFile(String fileId) {
        synchronized (lock) {
            List<String> stale = new ArrayList<>();
            for (Map.Entry<String, Models.DocChunk> e : chunks.entrySet()) {
                if (e.getValue().fileId().equals(fileId)) {
                    stale.add(e.getKey());
                }
            }
            for (String cid : stale) {
                chunks.remove(cid);
            }
            return stale.size();
        }
    }

    @Override
    public void deleteForThread(String threadId) {
        synchronized (lock) {
            // memories 刻意不删：跨会话记忆独立于线程存活。
            chunks.values().removeIf(chunk -> chunk.threadId().equals(threadId));
            summaries.values().removeIf(summary -> summary.threadId().equals(threadId));
        }
    }

    @Override
    public void recordOp(Models.MemoryOp op) {
        synchronized (lock) {
            ops.add(new OpEntry(op, ++opCounter));
        }
    }

    @Override
    public List<Models.MemoryOp> listOps(String userId, int limit) {
        return listOps(userId, null, null, null, limit);
    }

    /**
     * #160/#176：过滤 + 游标与 SQL 后端同语义——游标严格早于边界行，
     * 同秒并列按插入序号前移（边界行的序号由其 op_id 反查；未带 op_id
     * 时退化为纯时间戳截断）。
     */
    @Override
    public List<Models.MemoryOp> listOps(String userId, String threadId,
                                         Double sinceEpochSeconds, MemoryStore.OpsCursor before,
                                         int limit) {
        synchronized (lock) {
            // 边界行的插入序号：op_id 只在写入时登记，反查需要全量扫描一次。
            Integer cursorSeq = null;
            if (before != null && before.beforeOpId() != null) {
                for (OpEntry entry : ops) {
                    if (entry.op().opId().equals(before.beforeOpId())) {
                        cursorSeq = entry.seq();
                        break;
                    }
                }
            }
            List<OpEntry> pairs = new ArrayList<>();
            for (OpEntry entry : ops) {
                Models.MemoryOp op = entry.op();
                if (userId != null && !op.userId().equals(userId)) {
                    continue;
                }
                if (threadId != null && !op.threadId().equals(threadId)) {
                    continue;
                }
                if (sinceEpochSeconds != null && op.createdAt() < sinceEpochSeconds) {
                    continue;
                }
                if (before != null && !strictlyAfterCursor(op, entry.seq(), before, cursorSeq)) {
                    continue;
                }
                pairs.add(entry);
            }
            pairs.sort(OP_ORDER);
            List<Models.MemoryOp> out = new ArrayList<>();
            int bound = Math.min(pairs.size(), Math.max(1, limit));
            for (int i = 0; i < bound; i++) {
                out.add(pairs.get(i).op());
            }
            return out;
        }
    }

    /** 游标判定：严格晚于边界行（排序序）——同刻并列时按插入序号前移。 */
    private static boolean strictlyAfterCursor(Models.MemoryOp op, int seq,
                                               MemoryStore.OpsCursor before, Integer cursorSeq) {
        if (op.createdAt() < before.beforeEpochSeconds()) {
            return true;
        }
        return op.createdAt() == before.beforeEpochSeconds()
                && cursorSeq != null && seq < cursorSeq;
    }

    @Override
    public void recordVersion(Models.MemoryVersion version) {
        synchronized (lock) {
            versions.add(new VersionEntry(version, ++opCounter));
        }
    }

    @Override
    public List<Models.MemoryVersion> listVersions(String memoryId, int limit) {
        synchronized (lock) {
            List<VersionEntry> pairs = new ArrayList<>();
            for (VersionEntry entry : versions) {
                if (entry.version().memoryId().equals(memoryId)) {
                    pairs.add(entry);
                }
            }
            pairs.sort(VERSION_ORDER);
            List<Models.MemoryVersion> out = new ArrayList<>();
            int bound = Math.min(pairs.size(), Math.max(1, limit));
            for (int i = 0; i < bound; i++) {
                out.add(pairs.get(i).version());
            }
            return out;
        }
    }

    @Override
    public Models.MemoryVersion getVersion(String versionId) {
        synchronized (lock) {
            for (VersionEntry entry : versions) {
                if (entry.version().versionId().equals(versionId)) {
                    return entry.version();
                }
            }
            return null;
        }
    }

    @Override
    public List<Models.DocChunk> listChunksNeedingEmbedding(int dims, int limit) {
        synchronized (lock) {
            List<Models.DocChunk> matched = new ArrayList<>();
            for (Models.DocChunk chunk : chunks.values()) {
                if (chunk.embedding().length > 0 && chunk.embeddingDim() != null
                        && chunk.embeddingDim() == dims) {
                    continue;
                }
                matched.add(chunk);
            }
            matched.sort(Comparator.comparingDouble(Models.DocChunk::createdAt).reversed());
            return new ArrayList<>(matched.subList(0, Math.min(matched.size(), Math.max(1, limit))));
        }
    }

    /** 审计的排序键：created_at 降序 + 插入序号降序（同秒并列最新在前）。 */
    private static final Comparator<OpEntry> OP_ORDER =
            Comparator.comparingDouble((OpEntry e) -> e.op().createdAt()).reversed()
                    .thenComparing(Comparator.comparingInt(OpEntry::seq).reversed());

    /** 版本史的排序键（同 {@link #OP_ORDER}）。 */
    private static final Comparator<VersionEntry> VERSION_ORDER =
            Comparator.comparingDouble((VersionEntry e) -> e.version().createdAt()).reversed()
                    .thenComparing(Comparator.comparingInt(VersionEntry::seq).reversed());
}
