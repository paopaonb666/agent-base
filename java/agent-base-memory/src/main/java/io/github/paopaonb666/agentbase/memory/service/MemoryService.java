package io.github.paopaonb666.agentbase.memory.service;

import dev.langchain4j.data.message.ChatMessage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.ReentrantLock;

import io.github.paopaonb666.agentbase.core.AgentModule;
import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.core.llm.ModelRuntime;
import io.github.paopaonb666.agentbase.extensions.Metrics;
import io.github.paopaonb666.agentbase.memory.context.MemoryContext;
import io.github.paopaonb666.agentbase.memory.embeddings.EmbeddingClient;
import io.github.paopaonb666.agentbase.memory.embeddings.Embeddings;
import io.github.paopaonb666.agentbase.memory.pipeline.MemoryPipeline;
import io.github.paopaonb666.agentbase.memory.retrieval.Retrieval;
import io.github.paopaonb666.agentbase.memory.retrieval.VectorIndexPort;
import io.github.paopaonb666.agentbase.memory.store.MemoryStore;
import io.github.paopaonb666.agentbase.memory.store.MemoryStores;
import io.github.paopaonb666.agentbase.memory.store.Models;
import io.github.paopaonb666.agentbase.memory.store.Models.DocChunk;
import io.github.paopaonb666.agentbase.memory.store.Models.MemoryBlock;
import io.github.paopaonb666.agentbase.memory.store.Models.MemoryOp;
import io.github.paopaonb666.agentbase.memory.store.Models.MemoryRecord;
import io.github.paopaonb666.agentbase.memory.store.Models.MemoryStoreError;
import io.github.paopaonb666.agentbase.memory.store.Models.MemoryVersion;
import io.github.paopaonb666.agentbase.memory.store.Models.SessionSummary;
import io.github.paopaonb666.agentbase.memory.textkit.TextKit;

/**
 * 记忆系统门面：一个实例服务整个运行时（三种后端一致）。
 *
 * <p>持有四个子域并委托——记忆 CRUD + 混合检索、知识库（切块摄取/
 * 检索/撤销）、画像（结构化 JSON 读写）、形成管线编排（capture 锁 +
 * 冻结注入快照缓存）。对应 Python 侧 service.py + orchestrator.py +
 * profile.py + knowledge.py + audit.py 的合并门面（Java 侧按模块内聚
 * 收敛，行为一一对应）。
 */
public final class MemoryService implements AgentModule.MemoryServicePort {

    private static final Logger LOG = LoggerFactory.getLogger(MemoryService.class);

    /** #182 capture 重试队列容量（有界；满则丢弃可观测——对齐 #212 有界池风格）。 */
    private static final int CAPTURE_RETRY_QUEUE_CAPACITY = 32;
    /** #182 重试上限：累计失败次数（含首次实时失败）达到即丢弃。 */
    private static final int CAPTURE_RETRY_MAX_ATTEMPTS = 3;
    /** #182 tryLock 让路后的退避毫秒：实时轮持锁期间不空转。 */
    private static final long CAPTURE_RETRY_BACKOFF_MS = 100;

    private final MemoryStore store;
    private final EmbeddingClient embedder;
    private final Settings settings;
    private final MemoryPipeline pipeline;
    private final ModelRuntime llm;
    /** 向量索引（#324）：vectorPushdown 开启时装配（默认 brute-force 基线）。 */
    private final VectorIndexPort vectorIndex;

    /** 按 user_id 的捕获串行化锁（锐评 #10：并发轮次重复 ADD 防护）。 */
    private final Map<String, ReentrantLock> locks = new ConcurrentHashMap<>();

    /** #182 capture 失败重试队列：有界、满则丢弃可观测；不做持久化死信——
     * 每轮 captureTurn 持全量快照（非增量），下一轮成功的 capture 天然
     * 覆盖丢轮，内存队列足够。 */
    private final ArrayBlockingQueue<CaptureTask> captureRetries =
            new ArrayBlockingQueue<>(CAPTURE_RETRY_QUEUE_CAPACITY);
    /** 重试线程懒启动单飞锁（首次入队时创建，单守护线程）。 */
    private final Object captureRetryStartLock = new Object();
    /** 重试单守护线程；volatile 供 close() 中断与判空。 */
    private volatile Thread captureRetryWorker;
    /** 关闭标志：置位后不再入队/重放（关闭期不再打 LLM），剩余任务由 close 排空计数。 */
    private volatile boolean stopped = false;

    /** 线程级冻结注入段的 LRU 缓存（成本治理 T2.1），键 (user,agent,thread)。
     * accessOrder=true：get 即 touch（对应 Python orchestrator 的
     * move_to_end + popitem(last=False)——驱逐最近未用，而不是最早写入）。 */
    private final LinkedHashMap<List<String>, String> snapshots = new LinkedHashMap<>(16, 0.75f, true);
    private final Object snapshotLock = new Object();

    public MemoryService(MemoryStore store, EmbeddingClient embedder, Settings settings,
                         ModelRuntime llm, AgentModule.ChatInvokerPort fastLlm) {
        this(store, embedder, settings, llm, fastLlm,
                settings.memory.vectorPushdown ? VectorIndexPort.bruteForce() : null);
    }

    public MemoryService(MemoryStore store, EmbeddingClient embedder, Settings settings,
                         ModelRuntime llm, AgentModule.ChatInvokerPort fastLlm,
                         VectorIndexPort vectorIndex) {
        this.store = store;
        this.embedder = embedder;
        this.settings = settings;
        this.llm = llm;
        this.vectorIndex = vectorIndex;
        this.pipeline = llm == null ? null : new MemoryPipeline(llm, fastLlm, settings, new Ops());
    }

    /**
     * 包私有（测试注入点）：直接给定已装配的管线——{@link MemoryPipeline}
     * 为 final 类且由服务内部构造，手写桩测试经其公开构造器注入脚本化模型
     * 与 MemoryOps，才能驱动服务级 catch 兜底（#182 重试队列的入队口）。
     * 静态工厂形态：不与公开构造器在 null 字面量下产生重载二义性。
     */
    static MemoryService withPipeline(MemoryStore store, EmbeddingClient embedder,
                                      Settings settings, MemoryPipeline pipeline) {
        return new MemoryService(store, embedder, settings, pipeline, null);
    }

    private MemoryService(MemoryStore store, EmbeddingClient embedder, Settings settings,
                          MemoryPipeline pipeline, VectorIndexPort vectorIndex) {
        this.store = store;
        this.embedder = embedder;
        this.settings = settings;
        this.llm = null;
        this.vectorIndex = vectorIndex;
        this.pipeline = pipeline;
    }

    /** 按配置装配；未启用或后端不可用返回 null（对应 build_memory_service）。 */
    public static MemoryService build(Settings settings, ModelRuntime llm,
                                      AgentModule.ChatInvokerPort fastLlm) {
        if (!settings.memory.enabled) {
            LOG.info("memory: 记忆系统已禁用（MEMORY_ENABLED=false）");
            return null;
        }
        MemoryStore store = MemoryStores.build(settings);
        if (store == null) {
            return null;
        }
        return new MemoryService(store, Embeddings.build(settings), settings, llm, fastLlm);
    }

    public MemoryStore store() {
        return store;
    }

    // ── 审计与版本（失败安全旁路写入） ──────────────────────────────────

    /** 写一条操作审计（成功与失败都记录）；审计失败只记日志。 */
    public void recordOp(String op, String userId, String agentId, String threadId,
                         Map<String, Object> detail, String status, String errorText, long durationMs) {
        try {
            store.recordOp(new MemoryOp(io.github.paopaonb666.agentbase.memory.store.Models.newMemoryId(),
                    op, userId == null ? "" : userId, agentId == null ? "" : agentId,
                    threadId == null ? "" : threadId, detail, status, errorText, (int) durationMs,
                    Models.now()));
        } catch (Exception e) {
            LOG.warn("memory: 审计写入失败（不影响主路径）：{}", e.toString());
        }
    }

    private void recordVersionSafe(MemoryVersion version) {
        try {
            store.recordVersion(version);
        } catch (Exception e) {
            LOG.warn("memory: 版本史写入失败（不影响主写入路径）：{}", e.toString());
        }
    }

    // ── 写路径 ──────────────────────────────────────────────────────────

    private List<Float> embedOne(String text) {
        List<List<Float>> vectors = embedder.embed(List.of(text));
        if (vectors == null || vectors.isEmpty()) {
            return null;
        }
        return vectors.get(0);
    }

    /**
     * 敏感内容红线（#48 的 service 层兜底，review R3-P1）：抽取路径由管线
     * 拦截、memory_save 由工具层拦截，但 REST 写路径（POST/PATCH /v1/memory、
     * PUT blocks）与 memory_update_block 之前不过滤——模型改走旁路即可把
     * 密钥写进每轮注入的常驻块。所有写路径在 service 层统一拦截；开关
     * （MEMORY_SENSITIVE_CONTENT_FILTER）与工具层共用。
     */
    private void assertContentClean(String content) {
        if (!settings.memory.sensitiveContentFilter || content == null || content.isEmpty()) {
            return;
        }
        List<String> rules =
                io.github.paopaonb666.agentbase.memory.textkit.SensitiveContentPolicy
                        .INSTANCE.violations(content);
        if (!rules.isEmpty()) {
            throw new MemoryStoreError("内容命中敏感信息规则（" + String.join("、", rules)
                    + "），已拒绝入库；请移除敏感信息后重试");
        }
    }

    /** 写入一条记忆；向量尽力而为（不可用就存纯文本）。 */
    public MemoryRecord addMemory(String userId, String agentId, String content, String kind,
                                  List<String> tags, double salience, String sourceThreadId) {
        if (!Models.KNOWN_MEMORY_KINDS.contains(kind)) {
            throw new MemoryStoreError("unknown memory kind '" + kind + "'; expected one of "
                    + Models.KNOWN_MEMORY_KINDS);
        }
        assertContentClean(content);
        long started = System.nanoTime();
        double now = Models.now();
        List<Float> embedding = embedOne(content);
        MemoryRecord record = new MemoryRecord(Models.newMemoryId(), userId, agentId, kind,
                content, tags == null ? List.of() : tags,
                embedding == null ? null : Models.encodeEmbedding(embedding),
                embedding == null ? null : embedder.dims(),
                Math.min(Math.max(salience, 0.0), 1.0), "active",
                sourceThreadId == null ? "" : sourceThreadId, List.of(), now, now, null, 0);
        store.upsertMemory(record);
        indexUpsert(record.memoryId(), embedding);
        recordVersionSafe(new MemoryVersion(Models.newMemoryId(), record.memoryId(), userId,
                "create", content, "", record.status()));
        Metrics.memory().observe("add", "ok", (System.nanoTime() - started) / 1_000_000_000.0);
        return record;
    }

    /** 更新一条记忆；内容变化时重新向量化。{@code versionOp}: update/restore。 */
    public MemoryRecord updateMemory(String memoryId, String content, List<String> tags,
                                     Double salience, String status, String versionOp) {
        MemoryRecord existing = store.getMemory(memoryId);
        if (existing == null) {
            return null;
        }
        if (status != null && !Models.KNOWN_MEMORY_STATUSES.contains(status)) {
            throw new MemoryStoreError("unknown memory status '" + status + "'; expected one of "
                    + Models.KNOWN_MEMORY_STATUSES);
        }
        if (content != null) {
            assertContentClean(content);
        }
        long started = System.nanoTime();
        String newContent = content != null ? content : existing.content();
        boolean reembed = content != null && !content.equals(existing.content());
        List<Float> embedding = reembed ? embedOne(newContent) : null;
        MemoryRecord updated = new MemoryRecord(existing.memoryId(), existing.userId(),
                existing.agentId(), existing.kind(), newContent,
                tags != null ? tags : existing.tags(),
                embedding != null ? Models.encodeEmbedding(embedding) : existing.embedding(),
                embedding != null ? embedder.dims() : existing.embeddingDim(),
                salience != null ? Math.min(Math.max(salience, 0.0), 1.0) : existing.salience(),
                status != null ? status : existing.status(),
                existing.sourceThreadId(), existing.sourceRefs(), existing.createdAt(),
                Models.now(), existing.lastAccessedAt(), existing.accessCount());
        store.upsertMemory(updated);
        // 向量索引跟随写路径（#324）：重嵌入后以新向量覆盖；降级为
        // 无向量（cooldown）时移除旧条目，防止陈旧向量参与 KNN。
        if (embedding != null) {
            indexUpsert(updated.memoryId(), embedding);
        } else if (reembed && vectorIndex != null) {
            vectorIndex.delete(updated.memoryId());
        }
        boolean changed = !updated.content().equals(existing.content())
                || !updated.tags().equals(existing.tags())
                || updated.salience() != existing.salience()
                || !updated.status().equals(existing.status());
        if (changed) {
            recordVersionSafe(new MemoryVersion(Models.newMemoryId(), memoryId, existing.userId(),
                    versionOp, updated.content(), existing.content(), updated.status()));
        }
        Metrics.memory().observe("update", "ok", (System.nanoTime() - started) / 1_000_000_000.0);
        return updated;
    }

    public boolean deleteMemory(String memoryId) {
        long started = System.nanoTime();
        MemoryRecord existing = store.getMemory(memoryId);
        boolean deleted = store.deleteMemory(memoryId);
        if (deleted) {
            if (vectorIndex != null) {
                vectorIndex.delete(memoryId);
            }
            if (existing != null) {
                // 墓碑版本：内容差分的终点（previous_content 保留最后内容）。
                recordVersionSafe(new MemoryVersion(Models.newMemoryId(), memoryId, existing.userId(),
                        "delete", "", existing.content(), existing.status()));
            }
        }
        Metrics.memory().observe("delete", "ok", (System.nanoTime() - started) / 1_000_000_000.0);
        return deleted;
    }

    /** 向量索引跟随写路径（#324）：索引未装配或向量缺失为无操作。 */
    private void indexUpsert(String memoryId, List<Float> embedding) {
        if (vectorIndex != null && embedding != null) {
            vectorIndex.upsert(memoryId, embedding);
        }
    }

    // ── 版本史 ──────────────────────────────────────────────────────────

    public List<MemoryVersion> listVersions(String memoryId, int limit) {
        return store.listVersions(memoryId, limit);
    }

    /**
     * 把记忆内容恢复到指定版本（产生一条 op=restore 的新版本）。仅对
     * 仍然存在的记忆有效——恢复已删除的记忆需要重建完整记录，返回 null。
     */
    public MemoryRecord restoreVersion(String memoryId, String versionId) {
        MemoryRecord existing = store.getMemory(memoryId);
        if (existing == null) {
            return null;
        }
        MemoryVersion version = store.getVersion(versionId);
        if (version == null || !version.memoryId().equals(memoryId) || version.content().isEmpty()) {
            return null;
        }
        return updateMemory(memoryId, version.content(), null, null, null, "restore");
    }

    // ── 读路径 ──────────────────────────────────────────────────────────

    public MemoryRecord getMemory(String memoryId) {
        return store.getMemory(memoryId);
    }

    /** 浏览列表（管理端点用）：默认排除画像记录。 */
    public List<MemoryRecord> listMemories(String userId, String agentId, boolean includeGlobal,
                                           List<String> kinds, List<String> statuses, int limit,
                                           boolean excludeProfile) {
        return store.listMemories(userId, agentId, includeGlobal, kinds,
                statuses == null ? List.of() : statuses, limit, excludeProfile);
    }

    /** 混合检索（注入路径与搜索端点共用）——内部打分视图。 */
    public List<Retrieval.ScoredMemory> searchScored(String userId, String agentId, String query,
                                                     Integer topK) {
        long started = System.nanoTime();
        boolean hasEmbedding = embedder.dims() != null;
        double minScore = settings.memory.recallMinScore;
        if (!hasEmbedding && minScore > 0) {
            // 降级关键词路径：混合分缺向量主力分量（权重 0.4），硬门槛会误杀
            // 老而准的关键词命中——阈值放宽一半。
            minScore = minScore / 2;
        }
        List<Retrieval.ScoredMemory> results = Retrieval.recallMemories(store, embedder, userId,
                agentId, query, topK != null ? topK : settings.memory.recallTopK, minScore, settings,
                vectorIndex);
        // 降级观测（#195）：embedding 缺席的检索打 degraded（纯 BM25 路径
        // 与正常混合召回在监控上可区分），而不是一律 ok。
        Metrics.memory().observe("search", hasEmbedding ? "ok" : "degraded",
                (System.nanoTime() - started) / 1_000_000_000.0);
        return results;
    }

    @Override
    public List<Map<String, Object>> search(String userId, String agentId, String query,
                                            Integer topK) {
        // 端口契约：跨边界载荷是 Map（score + meta + expired 标志）。
        List<Retrieval.ScoredMemory> scored = searchScored(userId, agentId, query, topK);
        List<Map<String, Object>> out = new ArrayList<>(scored.size());
        for (Retrieval.ScoredMemory item : scored) {
            Map<String, Object> view = new LinkedHashMap<>(item.record().meta());
            view.put("score", Math.round(item.score() * 10000.0) / 10000.0);
            view.put("expired", isExpired(item.record()));
            out.add(view);
        }
        return out;
    }

    /** episodic 记忆的 TTL 过期标志（0 天 = 不过期；其余类别恒 false）。 */
    public boolean isExpired(MemoryRecord record) {
        int ttl = settings.memory.episodicTtlDays;
        return "episodic".equals(record.kind()) && ttl > 0
                && (Models.now() - record.updatedAt()) > ttl * 86400.0;
    }

    // ── 知识库 ──────────────────────────────────────────────────────────

    @Override
    public int ingestDocument(String fileId, String userId, String agentId, String text,
                              String threadId) {
        if (text == null || text.strip().isEmpty()) {
            return 0;
        }
        long started = System.nanoTime();
        try {
            List<io.github.paopaonb666.agentbase.memory.textkit.TextKit.ChunkSpan> parts =
                    TextKit.chunkText(text, settings.memory.docChunkChars, settings.memory.docChunkOverlap);
            // 幂等：同一 file_id 重复摄取先清旧分块。
            store.deleteChunksForFile(fileId);
            List<List<Float>> vectors = embedder.embed(parts.stream()
                    .map(TextKit.ChunkSpan::text).toList());
            List<DocChunk> chunks = new ArrayList<>(parts.size());
            for (int i = 0; i < parts.size(); i++) {
                var span = parts.get(i);
                chunks.add(new DocChunk(Models.newMemoryId(), fileId,
                        threadId == null ? "" : threadId, userId, agentId, i, span.text(),
                        span.segments(), vectors != null ? Models.encodeEmbedding(vectors.get(i)) : null,
                        vectors != null ? embedder.dims() : null, Models.now()));
            }
            store.putChunks(chunks);
            long durationMs = (System.nanoTime() - started) / 1_000_000;
            Metrics.memory().observe("ingest", "ok", durationMs / 1000.0);
            recordOp("ingest", userId, agentId, threadId,
                    Map.of("file_id", fileId, "chunks", chunks.size()), "ok", "", durationMs);
            return chunks.size();
        } catch (Exception exc) {
            LOG.warn("memory: 文档摄取失败（不影响上传）：{}", exc.toString());
            Metrics.memory().observe("ingest", "error",
                    (System.nanoTime() - started) / 1_000_000_000.0);
            recordOp("ingest", userId, agentId, threadId, Map.of("file_id", fileId), "error",
                    (exc.getClass().getSimpleName() + ": " + exc.getMessage()).length() > 300
                            ? (exc.getClass().getSimpleName() + ": " + exc.getMessage()).substring(0, 300)
                            : exc.getClass().getSimpleName() + ": " + exc.getMessage(),
                    (System.nanoTime() - started) / 1_000_000);
            return 0;
        }
    }

    @Override
    public List<Map<String, Object>> searchKnowledge(String userId, String agentId, String query,
                                                     Integer topK) {
        return searchKnowledge(userId, agentId, query, topK, null);
    }

    /** #184：file_id 下推——指定单文件时候选集在存储层收窄。 */
    public List<Map<String, Object>> searchKnowledge(String userId, String agentId, String query,
                                                     Integer topK, String fileId) {
        long started = System.nanoTime();
        List<DocChunk> chunks = store.listChunks(userId, agentId, fileId, 2000);
        if (chunks.size() >= 2000) {
            LOG.warn("memory: 知识库候选分块达到上限 2000（user='{}', file='{}'）——更早的文件不参与本轮检索",
                    userId, fileId);
        }
        if (chunks.isEmpty()) {
            return List.of();
        }
        List<List<Float>> vectors = embedder.embed(List.of(query));
        List<Float> queryEmbedding = vectors == null || vectors.isEmpty() ? null : vectors.get(0);
        List<Retrieval.ScoredChunk> scored = Retrieval.scoreChunks(chunks, query, queryEmbedding,
                Models.now(), settings.memory.timeDecayHalfLifeDays,
                Retrieval.Weights.fromSettings(settings));
        double minScore = settings.memory.recallMinScore;
        if (embedder.dims() == null && minScore > 0) {
            // 降级关键词路径与 search 同规则：放宽一半。
            minScore = minScore / 2;
        }
        List<Retrieval.ScoredChunk> filtered = new ArrayList<>();
        for (Retrieval.ScoredChunk item : scored) {
            if (item.score() >= minScore
                    && (item.components().getOrDefault("keyword", 0.0) > 0
                            || item.components().getOrDefault("vector", 0.0) > 0)) {
                filtered.add(item);
            }
        }
        int limit = Math.max(1, topK != null ? topK : settings.memory.recallTopK);
        // 降级观测（#195）：知识库检索与主检索同口径——embedding 缺席打
        // degraded（此前该路径完全不发指标，降级不可见）。
        Metrics.memory().observe("search_knowledge", embedder.dims() != null ? "ok" : "degraded",
                (System.nanoTime() - started) / 1_000_000_000.0);
        List<Map<String, Object>> out = new ArrayList<>();
        List<Retrieval.ScoredChunk> selected = new ArrayList<>(
                filtered.subList(0, Math.min(filtered.size(), limit)));
        List<DocChunk> selectedChunks = new ArrayList<>();
        for (Retrieval.ScoredChunk item : selected) {
            selectedChunks.add(item.chunk());
        }
        // 父块推导（#50 small-to-big）：检索命中子块、返回父块上下文——
        // 同 file 内 ordinal 连续的命中块合并为一个父区间（段落级），零
        // schema 变更（offsets 天然支持推导）。独立块 parent=自身；旧数据
        // offsets 缺失 → parent_span=null（消费端降级为逐块口径）。
        Map<String, ParentView> parents = deriveParents(selectedChunks);
        for (Retrieval.ScoredChunk item : selected) {
            DocChunk chunk = item.chunk();
            Map<String, Object> view = new LinkedHashMap<>();
            view.put("score", Math.round(item.score() * 10000.0) / 10000.0);
            view.put("file_id", chunk.fileId());
            view.put("chunk_id", chunk.chunkId());
            view.put("ordinal", chunk.ordinal());
            view.put("text", chunk.text());
            view.put("offsets", chunk.offsets().isEmpty() ? null
                    : chunk.offsets().stream().map(seg -> List.of(seg[0], seg[1])).toList());
            view.put("has_embedding", chunk.embedding().length > 0);
            ParentView parent = parents.get(chunk.chunkId());
            view.put("parent_text", parent == null ? null : parent.text());
            view.put("parent_span", parent == null || parent.span() == null ? null
                    : List.of(parent.span()[0], parent.span()[1]));
            out.add(view);
        }
        return out;
    }

    /** 父块推导的产物：合并文本 + 原文区间（null = offsets 缺失不可推导）。 */
    private record ParentView(String text, long[] span) {
    }

    /**
     * 把本轮命中的分块按「同 file_id + ordinal 连续」聚成父块：跨 file 不
     * 合并（父块是文档内概念）；offsets 拼不齐（旧数据 null/空）时 span
     * 为 null 但文本仍合并（拼接近似，可视化降级）。合并文本以
     * {@code "\n\n"} 连接——chunkText 的段落打包即以该分隔符组段。
     */
    private static Map<String, ParentView> deriveParents(List<DocChunk> selected) {
        Map<String, ParentView> parents = new LinkedHashMap<>();
        int i = 0;
        while (i < selected.size()) {
            int j = i + 1;
            while (j < selected.size()
                    && selected.get(j).fileId().equals(selected.get(j - 1).fileId())
                    && selected.get(j).ordinal() == selected.get(j - 1).ordinal() + 1) {
                j++;
            }
            List<DocChunk> run = selected.subList(i, j);
            StringBuilder text = new StringBuilder();
            long minStart = -1;
            long maxEnd = -1;
            boolean spanKnown = true;
            for (DocChunk chunk : run) {
                if (text.length() > 0) {
                    text.append("\n\n");
                }
                text.append(chunk.text());
                if (chunk.offsets().isEmpty()) {
                    spanKnown = false;
                    continue;
                }
                for (long[] seg : chunk.offsets()) {
                    if (minStart < 0 || seg[0] < minStart) {
                        minStart = seg[0];
                    }
                    if (maxEnd < 0 || seg[1] > maxEnd) {
                        maxEnd = seg[1];
                    }
                }
            }
            ParentView parent = new ParentView(text.toString(),
                    spanKnown && minStart >= 0 ? new long[]{minStart, maxEnd} : null);
            for (DocChunk chunk : run) {
                parents.put(chunk.chunkId(), parent);
            }
            i = j;
        }
        return parents;
    }

    @Override
    public int revokeDocument(String fileId, String userId) {
        long started = System.nanoTime();
        int deleted = store.deleteChunksForFile(fileId);
        Metrics.memory().observe("revoke", "ok", (System.nanoTime() - started) / 1_000_000_000.0);
        recordOp("revoke", userId, "", "", Map.of("file_id", fileId, "chunks", deleted), "ok", "", 0);
        return deleted;
    }

    @Override
    public List<Map<String, Object>> listFileChunks(String userId, String fileId, int limit) {
        List<DocChunk> chunks = store.listChunks(userId, null, fileId, Math.max(1, Math.min(limit, 2000)));
        List<Map<String, Object>> out = new ArrayList<>(chunks.size());
        for (DocChunk chunk : chunks) {
            Map<String, Object> view = new LinkedHashMap<>();
            view.put("chunk_id", chunk.chunkId());
            view.put("ordinal", chunk.ordinal());
            view.put("text", chunk.text());
            view.put("offsets", chunk.offsets().isEmpty() ? null
                    : chunk.offsets().stream()
                            .map(seg -> List.of(seg[0], seg[1])).toList());
            view.put("char_len", chunk.text().length());
            view.put("has_embedding", chunk.embedding().length > 0);
            view.put("embedding_dim", chunk.embeddingDim());
            out.add(view);
        }
        return out;
    }

    @Override
    public Map<String, Integer> chunkCountsByFile(String userId) {
        return store.chunkCountsByFile(userId);
    }

    // ── 生命周期 ────────────────────────────────────────────────────────

    @Override
    public boolean captureReady() {
        return pipeline != null;
    }

    @Override
    public String healthProbe() {
        try {
            store.listOps(null, 1);
        } catch (Exception e) {
            LOG.warn("memory: health probe failed", e);
            return "error";
        }
        // embedding 端点不可达 = degraded（检索自动降级 BM25，服务仍在）。
        if (!embedder.probe()) {
            return "degraded";
        }
        return "ok";
    }

    /**
     * 为向量缺失/维度不符的 active 记忆回填 embedding（换模型后运行）。
     * 回填保持原 updated_at——维护操作不刷时间衰减分。
     */
    public int backfillEmbeddings(int batch) {
        return backfillEmbeddings(batch, null);
    }

    /** 回填（带进度回调，每批回调一次已处理条数；#164 端点进度查询用）。 */
    public int backfillEmbeddings(int batch, java.util.function.IntConsumer progress) {
        if (embedder.dims() == null) {
            LOG.info("memory: 未配置 embedding，回填无事可做");
            return 0;
        }
        int total = 0;
        while (true) {
            List<MemoryRecord> records = store.listMemoriesNeedingEmbedding(embedder.dims(), batch);
            if (records.isEmpty()) {
                break;
            }
            List<List<Float>> vectors = embedder.embed(records.stream()
                    .map(MemoryRecord::content).toList());
            if (vectors == null) {
                LOG.warn("memory: embedding 服务不可用，回填中断（已回填 {} 条）", total);
                break;
            }
            for (int i = 0; i < records.size(); i++) {
                MemoryRecord record = records.get(i);
                store.upsertMemory(new MemoryRecord(record.memoryId(), record.userId(),
                        record.agentId(), record.kind(), record.content(), record.tags(),
                        Models.encodeEmbedding(vectors.get(i)), embedder.dims(), record.salience(),
                        record.status(), record.sourceThreadId(), record.sourceRefs(),
                        record.createdAt(), record.updatedAt(), record.lastAccessedAt(),
                        record.accessCount()));
                total++;
                if (progress != null) {
                    progress.accept(total);
                }
            }
            if (records.size() < batch) {
                break;
            }
        }
        while (true) {
            List<DocChunk> chunks = store.listChunksNeedingEmbedding(embedder.dims(), batch);
            if (chunks.isEmpty()) {
                break;
            }
            List<List<Float>> vectors = embedder.embed(chunks.stream().map(DocChunk::text).toList());
            if (vectors == null) {
                LOG.warn("memory: embedding 服务不可用，分块回填中断（已回填 {} 条）", total);
                break;
            }
            for (int i = 0; i < chunks.size(); i++) {
                DocChunk chunk = chunks.get(i);
                store.putChunks(List.of(new DocChunk(chunk.chunkId(), chunk.fileId(),
                        chunk.threadId(), chunk.userId(), chunk.agentId(), chunk.ordinal(),
                        chunk.text(), chunk.offsets(), Models.encodeEmbedding(vectors.get(i)),
                        embedder.dims(), chunk.createdAt())));
                total++;
            }
            if (chunks.size() < batch) {
                break;
            }
        }
        LOG.info("memory: 向量回填完成，共 {} 条（含知识库分块）", total);
        return total;
    }

    // ── 向量回填任务（#164：管理端点入口） ──────────────────────────────

    /** 单飞保证 + 进度状态（进程内口径；重启即清，对维护任务足够）。 */
    private final java.util.concurrent.atomic.AtomicBoolean backfillRunning =
            new java.util.concurrent.atomic.AtomicBoolean();
    private final java.util.concurrent.atomic.AtomicInteger backfillProcessed =
            new java.util.concurrent.atomic.AtomicInteger();
    private volatile String backfillState = "idle"; // idle | running | done | error

    /**
     * 启动异步回填任务（换 embedding 模型后的迁移入口）。进程内同时只允许
     * 一个任务：已有任务在跑返回 false（端点转 409）。嵌入未配置抛
     * {@link Models.MemoryStoreError}（端点转 400）。
     */
    public boolean startBackfill(int batch) {
        if (embedder.dims() == null) {
            throw new Models.MemoryStoreError("未配置 embedding（MEMORY_EMBEDDING_*），回填不可用");
        }
        if (!backfillRunning.compareAndSet(false, true)) {
            return false;
        }
        backfillState = "running";
        backfillProcessed.set(0);
        int boundedBatch = Math.min(Math.max(batch, 1), 512);
        Thread worker = new Thread(() -> {
            try {
                int total = backfillEmbeddings(boundedBatch, backfillProcessed::set);
                backfillState = "done";
                backfillProcessed.set(total);
                LOG.info("memory: 异步回填任务结束，共 {} 条", total);
            } catch (Exception e) {
                backfillState = "error";
                LOG.warn("memory: 异步回填任务失败：{}", e.toString());
            } finally {
                backfillRunning.set(false);
            }
        }, "memory-backfill");
        worker.setDaemon(true);
        worker.start();
        return true;
    }

    /** 回填任务状态（#164 端点查询面）。 */
    public Map<String, Object> backfillStatus() {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("state", backfillState);
        out.put("processed", backfillProcessed.get());
        return out;
    }

    // ── 用户画像 ────────────────────────────────────────────────────────

    /**
     * 读取结构化用户画像（JSON dict）；不存在返回 null。内容畸形（解析
     * 失败或顶层不是 JSON object）同样返回 null（#177：composeFrozen 与
     * GET /v1/memory/profile 的契约不变），但不再静默——WARN + 一条
     * {@code profile_corrupt} 审计让「失忆」可诊断。
     */
    public Map<String, Object> getProfile(String userId) {
        String memoryId = Models.profileMemoryId(userId);
        MemoryRecord record = store.getMemory(memoryId);
        if (record == null) {
            return null;
        }
        Map<String, Object> profile;
        try {
            Object parsed = new com.fasterxml.jackson.databind.ObjectMapper()
                    .readValue(record.content(), Object.class);
            profile = parsed instanceof Map<?, ?> map ? asStringMap(map) : null;
        } catch (Exception e) {
            profileCorrupt(userId, memoryId,
                    e.getClass().getSimpleName() + ": " + e.getMessage());
            return null;
        }
        if (profile == null) {
            // 解析成功但顶层不是 dict（如 JSON 数组/字符串）：同样按损坏处理。
            profileCorrupt(userId, memoryId, "画像内容不是 JSON object");
            return null;
        }
        return profile;
    }

    /** #177：画像损坏的可诊断旁路（WARN + 审计），不改变返回形态（仍 null）。 */
    private void profileCorrupt(String userId, String memoryId, String reason) {
        String summary = reason.length() > 300 ? reason.substring(0, 300) : reason;
        LOG.warn("memory: 用户画像损坏（user='{}', memory_id='{}'）：{}——按无画像处理",
                userId, memoryId, summary);
        recordOp("profile_corrupt", userId, "", "",
                Map.of("memory_id", memoryId, "reason", summary), "error", summary, 0);
    }

    /**
     * 整包保存用户画像：不需要向量（整体注入不走召回），保存前做结构级
     * 硬截断，每次演化写版本快照。
     */
    public void saveProfile(String userId, Map<String, Object> profile) {
        String memoryId = Models.profileMemoryId(userId);
        MemoryRecord existing = store.getMemory(memoryId);
        double now = Models.now();
        Map<String, Object> trimmed = TextKit.trimProfileToChars(profile, settings.memory.profileMaxChars);
        String content;
        try {
            // json.dumps(profile, ensure_ascii=False) 形态（默认分隔符），
            // 与 Python profile.save_profile 的存储字节逐位一致。
            content = io.github.paopaonb666.agentbase.memory.textkit.PyJson.dumps(trimmed);
        } catch (Exception e) {
            content = "{}";
        }
        store.upsertMemory(new MemoryRecord(memoryId, userId, "*", "semantic", content,
                List.of("profile"), null, null, 1.0, "active", "", List.of("profile"),
                existing != null ? existing.createdAt() : now, now,
                existing != null ? existing.lastAccessedAt() : null,
                existing != null ? existing.accessCount() : 0));
        recordVersionSafe(new MemoryVersion(Models.newMemoryId(), memoryId, userId,
                existing != null ? "update" : "create", content,
                existing != null ? existing.content() : ""));
    }

    // ── 记忆块 ──────────────────────────────────────────────────────────

    public List<MemoryBlock> listBlocks(String userId, String agentId) {
        return store.listBlocks(userId, agentId);
    }

    public MemoryBlock getBlock(String userId, String agentId, String label) {
        return store.getBlock(userId, agentId, label);
    }

    /** 手工写块（覆盖式，版本递增）。 */
    public MemoryBlock putBlock(String userId, String agentId, String label, String content,
                                int charLimit) {
        // 对应 Python api.py put_memory_block：content 原样入库，不做截断
        // （char_limit 只是块的声明容量，注入组装时才按预算裁剪——在这里
        // 截断会永久丢字，版本号还照常递增）。
        MemoryBlock existing = store.getBlock(userId, agentId, label);
        assertContentClean(content);
        int version = existing != null ? existing.version() + 1 : 1;
        MemoryBlock block = new MemoryBlock(userId, agentId, label,
                content, charLimit, version, Models.now());
        store.upsertBlock(block);
        return block;
    }

    public boolean deleteBlock(String userId, String agentId, String label) {
        return store.deleteBlock(userId, agentId, label);
    }

    // ── 审计查询 ────────────────────────────────────────────────────────

    public List<MemoryOp> listOps(String userId, int limit) {
        return store.listOps(userId, limit);
    }

    /** #160/#176：审计列表过滤 + 游标变体（参数语义见 MemoryStore#listOps）。 */
    public List<MemoryOp> listOps(String userId, String threadId, Double sinceEpochSeconds,
                                  MemoryStore.OpsCursor before, int limit) {
        return store.listOps(userId, threadId, sinceEpochSeconds, before, limit);
    }

    // ── 形成管线编排（M6c） ─────────────────────────────────────────────

    @Override
    public Map<String, Object> captureTurn(String userId, String agentId, String threadId,
                                           List<ChatMessage> messages) {
        return captureTurn(userId, agentId, threadId, messages, false);
    }

    /**
     * 一轮对话结束后的记忆形成（后台调用；绝不抛异常）。消息是该线程的
     * 全部持久化消息（checkpointer 快照）；force 旁路门控（夜间批脚本）。
     */
    public Map<String, Object> captureTurn(String userId, String agentId, String threadId,
                                           List<ChatMessage> messages, boolean force) {
        if (pipeline == null) {
            return null;
        }
        int humanCount = 0;
        for (ChatMessage message : messages) {
            if (message instanceof dev.langchain4j.data.message.UserMessage) {
                humanCount++;
            }
        }
        List<ChatMessage> window = messages.subList(
                Math.max(0, messages.size() - settings.memory.extractionMaxMessages), messages.size());
        String transcript = MemoryPipeline.renderTranscript(window, settings.memory.extractionMaxInputChars);
        if (transcript.strip().isEmpty()) {
            return null;
        }
        ReentrantLock lock = locks.computeIfAbsent(userId, k -> new ReentrantLock());
        lock.lock();
        Map<String, Object> detail;
        boolean captureFailed = false;
        try {
            detail = pipeline.captureTurn(userId, agentId, threadId, transcript, humanCount, force);
        } catch (Exception e) {
            // 管线内部已逐步失败安全；这里兜底防任何漏网异常干扰调用方。
            LOG.error("memory: capture_turn 意外失败", e);
            detail = null;
            captureFailed = true;
        } finally {
            lock.unlock();
        }
        if (captureFailed) {
            // #182：解锁后再入队——重试线程的 tryLock 不与本轮持锁窗口相撞。
            // 队列有界、入队失败只 WARN（可观测不阻塞）；本轮丢失由下一轮
            // 全量快照 capture 天然覆盖，无需持久化死信。
            enqueueCaptureRetry(userId, agentId, threadId, transcript, humanCount, force, 1);
        }
        return detail;
    }

    // ── capture 失败重试队列（#182） ─────────────────────────────────────

    /** 重试任务载荷：已渲染转写快照（不重渲染）+ 定位元组 + 失败计数
     * （attempt = 已失败次数，含首次实时失败；初值 1 = 实时轮失败后入队）。 */
    private record CaptureTask(String userId, String agentId, String threadId,
                               String transcript, int humanCount, boolean force, int attempt) {

        CaptureTask withAttempt(int nextAttempt) {
            return new CaptureTask(userId, agentId, threadId, transcript,
                    humanCount, force, nextAttempt);
        }
    }

    /**
     * 失败转写入队（包私有：catch 分支与测试共用入口）。队列满或已关闭时
     * 丢弃（WARN 可观测、不阻塞调用方）；首次入队懒启动单守护重试线程
     * （daemon、名字 {@code memory-capture-retry}，风格对齐 backfill worker
     * 先例——M 包不反向依赖基座 Background 池，自建线程）。
     */
    void enqueueCaptureRetry(String userId, String agentId, String threadId,
                             String transcript, int humanCount, boolean force, int attempt) {
        if (stopped) {
            return; // 已关闭：不再接收（close 之后的 captureTurn 不属于正常流程）
        }
        if (!captureRetries.offer(new CaptureTask(userId, agentId, threadId,
                transcript, humanCount, force, attempt))) {
            LOG.warn("memory: capture 重试队列已满（{}），丢弃本轮失败转写（user='{}'）",
                    CAPTURE_RETRY_QUEUE_CAPACITY, userId);
            return;
        }
        synchronized (captureRetryStartLock) {
            if (captureRetryWorker == null) {
                Thread worker = new Thread(this::drainCaptureRetries, "memory-capture-retry");
                worker.setDaemon(true);
                worker.start();
                captureRetryWorker = worker;
            }
        }
    }

    /** 待重试任务数（包私有：测试观测面）。 */
    int pendingCaptureRetries() {
        return captureRetries.size();
    }

    /**
     * 重试主循环：单守护线程串行消费。取任务用带超时的 poll（响应 close
     * 的中断，中断即退）；执行前抢同一 per-user 锁——tryLock 失败说明实时
     * 轮持锁，任务回队尾让路（锐评 #10：重试绝不拖慢实时路径），短退避
     * 防持锁期间空转。
     */
    private void drainCaptureRetries() {
        while (!stopped) {
            CaptureTask task;
            try {
                task = captureRetries.poll(500, TimeUnit.MILLISECONDS);
            } catch (InterruptedException e) {
                break; // close() 中断：退出（剩余任务由 close 排空计数 WARN）
            }
            if (task == null) {
                continue;
            }
            if (stopped) {
                // close 竞态：任务已出队但服务在关闭——不执行（关闭期不打 LLM）。
                LOG.warn("memory: 关闭时丢弃 1 条已出队待重试 capture 任务（user='{}'）",
                        task.userId());
                break;
            }
            ReentrantLock lock = locks.computeIfAbsent(task.userId(), k -> new ReentrantLock());
            if (!lock.tryLock()) {
                requeueCapture(task);
                sleepQuietly(CAPTURE_RETRY_BACKOFF_MS);
                continue;
            }
            try {
                replayCapture(task);
            } finally {
                lock.unlock();
            }
        }
    }

    /** 单次重放（持 per-user 锁调用）：成功或达上限丢弃都写 capture_retry
     * 审计 + 指标；中间失败仅 attempt+1 回队（不写终态审计）。 */
    private void replayCapture(CaptureTask task) {
        long started = System.nanoTime();
        try {
            pipeline.captureTurn(task.userId(), task.agentId(), task.threadId(),
                    task.transcript(), task.humanCount(), task.force());
        } catch (Exception e) {
            long durationMs = (System.nanoTime() - started) / 1_000_000;
            String errorText = e.getClass().getSimpleName() + ": " + e.getMessage();
            if (errorText.length() > 300) {
                errorText = errorText.substring(0, 300);
            }
            int attempts = task.attempt() + 1;
            if (attempts >= CAPTURE_RETRY_MAX_ATTEMPTS) {
                LOG.warn("memory: capture 重试 {} 次仍失败，丢弃（user='{}', thread='{}'）：{}",
                        attempts, task.userId(), task.threadId(), e.toString());
                recordOp("capture_retry", task.userId(), task.agentId(), task.threadId(),
                        Map.of("attempts", attempts), "error", errorText, durationMs);
                Metrics.memory().observe("capture_retry", "error", durationMs / 1000.0);
            } else {
                requeueCapture(task.withAttempt(attempts));
            }
            return;
        }
        long durationMs = (System.nanoTime() - started) / 1_000_000;
        recordOp("capture_retry", task.userId(), task.agentId(), task.threadId(),
                Map.of("attempts", task.attempt()), "ok", "", durationMs);
        Metrics.memory().observe("capture_retry", "ok", durationMs / 1000.0);
    }

    /** 回队尾；服务已关闭或队列满时丢弃（WARN 可观测不阻塞，对齐 #212 风格）。 */
    private void requeueCapture(CaptureTask task) {
        if (stopped) {
            return; // 关闭竞态：close 的排空计数已覆盖剩余任务
        }
        if (!captureRetries.offer(task)) {
            LOG.warn("memory: capture 重试任务回队失败（队列已满 {}），丢弃（user='{}'）",
                    CAPTURE_RETRY_QUEUE_CAPACITY, task.userId());
        }
    }

    private static void sleepQuietly(long millis) {
        try {
            Thread.sleep(millis);
        } catch (InterruptedException e) {
            // close 中断：恢复标志位即可，外层循环按 stopped 退出。
            Thread.currentThread().interrupt();
        }
    }

    // ── 上下文注入（M6d + T2.1） ────────────────────────────────────────

    @Override
    public String composeContext(String userId, String agentId, String threadId, String query) {
        // 线程级冻结注入段（画像+记忆块+摘要快照）：同线程返回逐字节相同的
        // 缓存内容（供应商前缀缓存的地基）；逐轮变化的召回走 recallContext。
        return composeFrozen(userId, agentId, threadId);
    }

    /** 线程级冻结注入段：同线程返回逐字节相同的缓存内容。 */
    public String composeFrozen(String userId, String agentId, String threadId) {
        List<String> key = List.of(userId, agentId, threadId);
        synchronized (snapshotLock) {
            String cached = snapshots.get(key); // accessOrder=true：get 即 LRU touch
            if (cached != null) {
                return cached;
            }
        }
        String frozen;
        try {
            List<MemoryBlock> globalBlocks = store.listBlocks(userId, "*");
            List<MemoryBlock> agentBlocks = store.listBlocks(userId, agentId);
            Map<String, Object> profile = settings.memory.profileEnabled ? getProfile(userId) : null;
            SessionSummary summaryRecord = store.getSummary(userId, threadId);
            List<MemoryBlock> blocks = new ArrayList<>(globalBlocks);
            blocks.addAll(agentBlocks);
            frozen = MemoryContext.composeFrozenContext(profile, blocks,
                    summaryRecord != null ? summaryRecord.summary() : null,
                    settings.memory.contextMaxChars);
        } catch (Exception e) {
            LOG.warn("memory: 冻结注入段组装失败（本轮不注入）", e);
            return null;
        }
        if (frozen != null) {
            synchronized (snapshotLock) {
                snapshots.put(key, frozen);
                while (snapshots.size() > settings.memory.contextSnapshotMax) {
                    IteratorRemoveFirst();
                }
            }
        }
        return frozen;
    }

    private void IteratorRemoveFirst() {
        List<String> eldest = snapshots.keySet().iterator().next();
        snapshots.remove(eldest);
    }

    @Override
    public String recallContext(String userId, String query) {
        try {
            List<Retrieval.ScoredMemory> recalled = searchScored(userId, null, query, null);
            return MemoryContext.composeRecallContext(recalled, settings.memory.contextMaxChars);
        } catch (Exception e) {
            LOG.warn("memory: 召回注入段组装失败（本轮不注入召回）", e);
            return null;
        }
    }

    /**
     * 线程删除的级联清理（#172 隐私语义）：默认 <b>保留跨会话记忆</b>——
     * 随线程消亡的只有滚动摘要、知识库分块与注入快照；memories 表的行
     * 独立于线程存活，出处字段 {@code source_thread_id} 原样保留，删除
     * 线程不会抹掉用户攒下的长期记忆。与 Python 侧
     * {@code service.py purge_thread} 同语义同保留（那边只删两表）。
     */
    @Override
    public void purgeThread(String threadId) {
        purgeThread(threadId, false);
    }

    /**
     * #172 可选参数变体：{@code purgeMemories=true} 时连跨会话记忆一起删
     * （按 {@code source_thread_id} 清理，出处字段随之消失）——供用户显式
     * 要求彻底清除（如 GDPR 式删除）时使用。默认（false）行为与旧签名
     * 完全一致，非 breaking；Python 端点暂未跟进该参数（行为分歧登记 #315）。
     */
    @Override
    public void purgeThread(String threadId, boolean purgeMemories) {
        store.deleteForThread(threadId);
        if (purgeMemories) {
            store.deleteMemoriesForThread(threadId);
        }
        synchronized (snapshotLock) {
            snapshots.keySet().removeIf(key -> key.size() == 3 && key.get(2).equals(threadId));
        }
    }

    private List<MemoryBlock> listBlocksFor(String agentId, String userId) {
        List<MemoryBlock> blocks = new ArrayList<>(store.listBlocks(userId, "*"));
        blocks.addAll(store.listBlocks(userId, agentId));
        return blocks;
    }

    private String getSummaryText(String userId, String threadId) {
        SessionSummary summary = store.getSummary(userId, threadId);
        return summary == null ? null : summary.summary();
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> asStringMap(Map<?, ?> map) {
        Map<String, Object> out = new LinkedHashMap<>();
        for (Map.Entry<?, ?> e : map.entrySet()) {
            out.put(String.valueOf(e.getKey()), e.getValue());
        }
        return out;
    }

    @Override
    public void close() {
        // #182：先停 capture 重试队列——置 stopped + 中断重试线程（有界 join
        // 等它退出，保证 close 返回后不再有新重放起跑），剩余任务只计数 WARN
        // 不执行（关闭期不再打 LLM）；丢轮由下一轮全量快照 capture 覆盖。
        stopped = true;
        Thread worker = captureRetryWorker;
        if (worker != null) {
            worker.interrupt();
            try {
                worker.join(1000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
        int dropped = captureRetries.size();
        captureRetries.clear();
        if (dropped > 0) {
            LOG.warn("memory: 关闭时丢弃 {} 条待重试 capture 任务（不再执行）", dropped);
        }
        try {
            store.close();
        } catch (Exception ignored) {
            // 存储关闭失败不影响进程退出。
        }
    }

    /** 管线操作适配器（MemoryPipeline.MemoryOps → 门面方法）。 */
    private class Ops implements MemoryPipeline.MemoryOps {
        @Override
        public List<Retrieval.ScoredMemory> search(String userId, String agentId, String query,
                                                   Integer topK) {
            return MemoryService.this.searchScored(userId, agentId, query, topK);
        }

        @Override
        public void addMemory(String userId, String agentId, String content, String kind,
                              List<String> tags, double salience, String sourceThreadId) {
            MemoryService.this.addMemory(userId, agentId, content, kind, tags, salience, sourceThreadId);
        }

        @Override
        public boolean updateMemory(String memoryId, String content) {
            return MemoryService.this.updateMemory(memoryId, content, null, null, null, "update") != null;
        }

        @Override
        public boolean deleteMemory(String memoryId) {
            return MemoryService.this.deleteMemory(memoryId);
        }

        @Override
        public Map<String, Object> getProfile(String userId) {
            return MemoryService.this.getProfile(userId);
        }

        @Override
        public void saveProfile(String userId, Map<String, Object> profile) {
            MemoryService.this.saveProfile(userId, profile);
        }

        @Override
        public SessionSummary getSummary(String userId, String threadId) {
            return store.getSummary(userId, threadId);
        }

        @Override
        public void upsertSummary(SessionSummary summary) {
            store.upsertSummary(summary);
        }

        @Override
        public void recordOp(String op, String userId, String agentId, String threadId,
                             Map<String, Object> detail, String status, String errorText,
                             long durationMs) {
            MemoryService.this.recordOp(op, userId, agentId, threadId, detail, status, errorText, durationMs);
        }

        @Override
        public List<Float> embedCandidate(String content) {
            return MemoryService.this.embedOne(content);
        }
    }
}
