package io.github.paopaonb666.agentbase.memory.store;

import java.util.List;
import java.util.Map;

/**
 * 完整记忆存储接口：按聚合根拆分的小接口的组合（方法集与拆分前一致），
 * {@code MemoryService} 与入口层依赖的仍是本接口。实现必须保证并发安全。
 *
 * <p>对应 Python 侧 {@code agent_base_memory/store/protocol.py} 的
 * MemoryStore 组合协议（六个 Repo + deleteForThread）。
 */
public interface MemoryStore extends AutoCloseable {

    // ── MemoryRepo：跨会话长期记忆（memories 表） ────────────────────

    void upsertMemory(Models.MemoryRecord record);

    Models.MemoryRecord getMemory(String memoryId);

    boolean deleteMemory(String memoryId);

    /**
     * 列出记忆；{@code agentId} 为 null 时跨模块；{@code includeGlobal}
     * 控制 {@code (agent_id=? OR agent_id='*')} 的全局分支；
     * {@code excludeProfile} 排除画像记录（id 前缀 profile:）。
     */
    List<Models.MemoryRecord> listMemories(String userId, String agentId, boolean includeGlobal,
                                           List<String> kinds, List<String> statuses,
                                           int limit, boolean excludeProfile);

    /**
     * 带 episodic TTL 下推的变体（#180）：{@code episodicCutoffBefore}
     * 为 unix 秒，过期 episodic 行在 SQL 层被排除（不再挤占候选名额）；
     * null = 不过滤。默认实现忽略 cutoff 委托 7 参版本——TTL 仍由调用方
     * 内存过滤兜底；SQL 后端应覆写以获得下推收益。
     */
    default List<Models.MemoryRecord> listMemories(String userId, String agentId, boolean includeGlobal,
                                                   List<String> kinds, List<String> statuses,
                                                   int limit, boolean excludeProfile,
                                                   Double episodicCutoffBefore) {
        return listMemories(userId, agentId, includeGlobal, kinds, statuses, limit, excludeProfile);
    }

    /** 访问记账：批量更新 last_accessed_at + access_count。 */
    void touchMemories(List<String> memoryIds);

    /** 换模型后的向量回填查询（embedding 缺失或维度不符）。 */
    List<Models.MemoryRecord> listMemoriesNeedingEmbedding(int dims, int limit);

    // ── BlockRepo：常驻上下文的记忆块（memory_blocks 表） ─────────────

    void upsertBlock(Models.MemoryBlock block);

    Models.MemoryBlock getBlock(String userId, String agentId, String label);

    List<Models.MemoryBlock> listBlocks(String userId, String agentId);

    boolean deleteBlock(String userId, String agentId, String label);

    // ── SummaryRepo：线程滚动摘要（session_summaries 表） ─────────────

    Models.SessionSummary getSummary(String userId, String threadId);

    void upsertSummary(Models.SessionSummary summary);

    // ── KnowledgeRepo：文档知识库分块（doc_chunks 表） ────────────────

    void putChunks(List<Models.DocChunk> chunks);

    List<Models.DocChunk> listChunks(String userId, String agentId, String fileId, int limit);

    int countChunksForFile(String fileId);

    Map<String, Integer> chunkCountsByFile(String userId);

    int deleteChunksForFile(String fileId);

    List<Models.DocChunk> listChunksNeedingEmbedding(int dims, int limit);

    // ── AuditRepo：记忆系统操作审计（memory_ops 表） ──────────────────

    void recordOp(Models.MemoryOp op);

    /**
     * 审计列表游标（#160）：定位上一页的边界行。{@code beforeEpochSeconds}
     * 是边界行的 created_at（epoch 秒，含小数）；{@code beforeOpId} 是边界
     * 行的 op_id——同秒并列时 SQL 侧用它反查 rowid/seq 做 tiebreak 前移
     * 比较（调用方无需感知该内部序号）。null 的 beforeOpId = 游标仅按
     * 时间戳严格早于截断（同刻并列行的余量不再返回，见 listOps 说明）。
     */
    record OpsCursor(double beforeEpochSeconds, String beforeOpId) {}

    /** 审计列表：{@code userId} 为 null 时跨用户；按 created_at 降序取最近 N 条。 */
    List<Models.MemoryOp> listOps(String userId, int limit);

    /**
     * #160/#176：审计列表的过滤 + 游标下推变体，参数均可选（null = 不过滤）：
     * {@code threadId} 非空按 thread_id 精确匹配；{@code sinceEpochSeconds}
     * 为 epoch 秒、含边界（created_at &gt;= since，口径与列写入一致——
     * toSqlTs）；{@code before} 为上一页边界，语义是严格早于该边界行
     * （同秒并列按 opsTiebreak 前移，sqlite=rowid / mysql=seq）。排序与
     * limit 语义同旧签名。默认实现忽略过滤委托旧签名（过滤由调用方内存
     * 兜底）；SQL 后端与 InMemory 后端覆写以获得下推收益。
     */
    default List<Models.MemoryOp> listOps(String userId, String threadId,
                                          Double sinceEpochSeconds, OpsCursor before,
                                          int limit) {
        return listOps(userId, limit);
    }

    // ── VersionRepo：记忆内容版本史（memory_versions 表） ─────────────

    void recordVersion(Models.MemoryVersion version);

    List<Models.MemoryVersion> listVersions(String memoryId, int limit);

    Models.MemoryVersion getVersion(String versionId);

    // ── 线程级联清理 ──────────────────────────────────────────────────

    /** 线程级联清理（摘要/分块；跨会话记忆刻意保留——memories 不删）。 */
    void deleteForThread(String threadId);

    @Override
    default void close() {}
}
