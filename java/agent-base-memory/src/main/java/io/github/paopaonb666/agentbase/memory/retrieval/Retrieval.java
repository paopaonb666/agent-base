package io.github.paopaonb666.agentbase.memory.retrieval;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.core.text.QueryTerms;
import io.github.paopaonb666.agentbase.memory.embeddings.EmbeddingClient;
import io.github.paopaonb666.agentbase.memory.store.MemoryStore;
import io.github.paopaonb666.agentbase.memory.store.Models;

/**
 * 混合检索（M6b）：向量 + BM25 + 时间衰减 + 显著度。
 *
 * <p>四个信号（缺失分量按 0 计）：
 *
 * <ul>
 * <li><b>向量余弦</b>——语义召回（同义改写也能命中）；只在查询与候选双方
 *     都有<b>同维</b>向量时参与；</li>
 * <li><b>BM25</b>——关键词精确召回（型号、人名、代码标识符等语义模型容易
 *     漏的东西）；中文用二元组（bigram）、英文用词元；</li>
 * <li><b>时间衰减</b>——指数半衰期（默认 14 天），久未访问的记忆权重走低
 *     （时间基点是 {@code updated_at}，{@code last_accessed_at} 不参与）；</li>
 * <li><b>显著度</b>——写入时由管线/人工标定的 0..1 权重。</li>
 * </ul>
 *
 * <p>遗忘策略：episodic 记忆按 {@code MEMORY_EPISODIC_TTL_DAYS} 过滤（0 表示
 * 不过期）；semantic/procedural 不过期——它们的价值恰恰在长期。
 *
 * <p>对应 Python 侧 {@code agent_base_memory/retrieval.py}，逐算法移植
 * （打分公式、分词与分量装配顺序逐字节对齐）。
 */
public final class Retrieval {

    private static final Logger LOG = LoggerFactory.getLogger(Retrieval.class);

    /** 召回候选集上限（H2 容量契约）：与 store.listMemories 的默认上限一致。 */
    public static final int CANDIDATE_LIMIT = 2000;

    /** 向量下推的 KNN 候选池大小（#323/#324 方案口径：KNN top-100 再混合打分）。 */
    public static final int KNN_CANDIDATES = 100;

    /**
     * 关键词下推的候选池大小（#326/327）：宽松取值——BM25>0 但排在
     * SQL top-N 之外的候选会被挤出（关键词下推无天然豁免类），200 给
     * 生产 minScore 证据门槛下的真实命中留足余量。
     */
    public static final int BM25_CANDIDATES = 200;

    // 混合权重默认值（和为 1）。向量是语义召回的主力，BM25 兜关键词，
    // 时间与显著度做轻量排序先验。可经 MEMORY_WEIGHT_* 配置覆盖
    // （Settings 校验和与范围）；调权是检索质量工程，改动应有评测依据。
    public static final double W_VECTOR = 0.40;
    public static final double W_KEYWORD = 0.30;
    public static final double W_RECENCY = 0.15;
    public static final double W_SALIENCE = 0.15;

    /** BM25 经典参数。 */
    public static final double BM25_K1 = 1.5;
    public static final double BM25_B = 0.75;

    /** 一天的秒数（时间衰减/TTL 换算）。 */
    private static final double DAY_SECONDS = 86400.0;

    private Retrieval() {}

    /** 混合检索权重（对应 Python 的 weights mapping，四个固定键）。 */
    public record Weights(double vector, double keyword, double recency, double salience) {

        /** 默认权重（和为 1，与 M6b 一致）。 */
        public static final Weights DEFAULT =
                new Weights(W_VECTOR, W_KEYWORD, W_RECENCY, W_SALIENCE);

        /** 从 Settings 的 memory 节读混合检索权重（锐评 #4：可配置化）。 */
        public static Weights fromSettings(Settings settings) {
            return new Weights(settings.memory.weightVector, settings.memory.weightKeyword,
                    settings.memory.weightRecency, settings.memory.weightSalience);
        }

        double forName(String name) {
            return switch (name) {
                case "vector" -> vector;
                case "keyword" -> keyword;
                case "recency" -> recency;
                case "salience" -> salience;
                default -> throw new IllegalArgumentException("未知分量：" + name);
            };
        }
    }

    /** 一条带混合分的记忆；{@code components} 保留各分量的原始值供调试。 */
    public record ScoredMemory(Models.MemoryRecord record, double score,
                               Map<String, Double> components) {

        public ScoredMemory {
            components = components == null ? Map.of()
                    : java.util.Collections.unmodifiableMap(new LinkedHashMap<>(components));
        }

        /** 向量分量（缺失按 0）。 */
        public double vectorScore() {
            return components.getOrDefault("vector", 0.0);
        }

        /** 关键词分量（缺失按 0）。 */
        public double keywordScore() {
            return components.getOrDefault("keyword", 0.0);
        }

        /** 时间衰减分量（缺失按 0）。 */
        public double recencyScore() {
            return components.getOrDefault("recency", 0.0);
        }

        /** 显著度分量（缺失按 0）。 */
        public double salienceScore() {
            return components.getOrDefault("salience", 0.0);
        }
    }

    /** 一个带混合分的知识库分块（分量语义同 ScoredMemory，无显著度）。 */
    public record ScoredChunk(Models.DocChunk chunk, double score,
                              Map<String, Double> components) {

        public ScoredChunk {
            components = components == null ? Map.of()
                    : java.util.Collections.unmodifiableMap(new LinkedHashMap<>(components));
        }

        /** 向量分量（缺失按 0）。 */
        public double vectorScore() {
            return components.getOrDefault("vector", 0.0);
        }

        /** 关键词分量（缺失按 0）。 */
        public double keywordScore() {
            return components.getOrDefault("keyword", 0.0);
        }
    }

    // ── 分词 ─────────────────────────────────────────────────────────

    /**
     * 检索分词：ASCII 词（小写）+ 中文二元组——#244 起委托基座公共件
     * {@link QueryTerms}（与搜索相关性评分同口径），行为逐值不变。
     */
    public static List<String> tokenize(String text) {
        return QueryTerms.tokenize(text);
    }

    // ── 基础因子 ─────────────────────────────────────────────────────

    /**
     * 余弦相似度；维度不一致或零向量返回 null（不可比即不计分）。
     */
    public static Double cosineSimilarity(List<Float> a, List<Float> b) {
        if (a.size() != b.size() || a.isEmpty()) {
            return null;
        }
        double dot = 0.0;
        double normA = 0.0;
        double normB = 0.0;
        for (int i = 0; i < a.size(); i++) {
            double x = a.get(i);
            double y = b.get(i);
            dot += x * y;
            normA += x * x;
            normB += y * y;
        }
        if (normA <= 0.0 || normB <= 0.0) {
            return null;
        }
        return dot / Math.sqrt(normA * normB);
    }

    /** 时间衰减因子：{@code 0.5 ** (age_days / half_life)}，永不归零。 */
    public static double recencyFactor(double updatedAt, double now, double halfLifeDays) {
        double ageDays = Math.max(0.0, now - updatedAt) / DAY_SECONDS;
        return Math.pow(0.5, ageDays / Math.max(halfLifeDays, 1e-9));
    }

    /** 按类别应用 TTL：episodic 过期剔除，其余类别不过期。 */
    public static List<Models.MemoryRecord> filterExpired(List<Models.MemoryRecord> records,
                                                          int episodicTtlDays, double now) {
        if (episodicTtlDays <= 0) {
            return List.copyOf(records);
        }
        double cutoff = now - episodicTtlDays * DAY_SECONDS;
        List<Models.MemoryRecord> kept = new ArrayList<>();
        for (Models.MemoryRecord record : records) {
            if (!"episodic".equals(record.kind()) || record.updatedAt() >= cutoff) {
                kept.add(record);
            }
        }
        return List.copyOf(kept);
    }

    // ── BM25 ─────────────────────────────────────────────────────────

    /**
     * 极简 BM25（纯 Java）：候选集量级是单用户的记忆条数（千级），
     * 全量内存打分完全够用，不引入索引基础设施。
     */
    static final class Bm25 {

        private final List<List<String>> docs;
        private final double avgdl;
        private final Map<String, Integer> df = new HashMap<>();

        Bm25(List<List<String>> docsTokens) {
            this.docs = List.copyOf(docsTokens);
            long total = 0;
            for (List<String> doc : docsTokens) {
                total += doc.size();
            }
            this.avgdl = docsTokens.isEmpty() ? 0.0 : (double) total / docsTokens.size();
            for (List<String> doc : docsTokens) {
                for (String term : new LinkedHashSet<>(doc)) {
                    df.merge(term, 1, Integer::sum);
                }
            }
        }

        double[] scores(List<String> queryTokens) {
            int n = docs.size();
            if (n == 0) {
                return new double[0];
            }
            double[] raw = new double[n];
            for (int i = 0; i < n; i++) {
                List<String> doc = docs.get(i);
                Map<String, Integer> tf = new HashMap<>();
                for (String term : doc) {
                    tf.merge(term, 1, Integer::sum);
                }
                double score = 0.0;
                for (String term : queryTokens) {
                    Integer count = tf.get(term);
                    if (count == null || count == 0) {
                        continue;
                    }
                    int docDf = df.getOrDefault(term, 0);
                    double idf = Math.log(1.0 + (n - docDf + 0.5) / (docDf + 0.5));
                    double lengthRatio = doc.size() / (avgdl != 0.0 ? avgdl : 1.0);
                    double denom = count + BM25_K1 * (1.0 - BM25_B + BM25_B * lengthRatio);
                    score += idf * count * (BM25_K1 + 1.0) / denom;
                }
                raw[i] = score;
            }
            return raw;
        }
    }

    /**
     * BM25 索引的进程内 LRU（锐评 #2 的修正实现）。
     *
     * <p>与评审建议的"按用户缓存 + 写入失效"不同，这里键是<b>候选文本列表
     * 本身</b>（内容即身份）：候选集中任何一条内容增删改都会产生不同的
     * 键，旧键随之自然淘汰——零失效钩子，不存在忘记失效导致的脏索引。
     *
     * <p>命中缓存跳过的是两段 O(N) 循环：逐文档分词 + df 统计。容量按
     * 条目数限制（一条 ≈ 一个作用域的候选集文本 + 分词结果），默认 64
     * 个作用域；{@code maxsize=0} 显式禁用（退回每次构建）。
     */
    public static final class Bm25Cache {

        private final int maxsize;
        private final LinkedHashMap<List<String>, Bm25> entries;

        public Bm25Cache() {
            this(64);
        }

        public Bm25Cache(int maxsize) {
            this.maxsize = Math.max(0, maxsize);
            this.entries = this.maxsize == 0 ? null
                    : new LinkedHashMap<>(16, 0.75f, true) {
                        @Override
                        protected boolean removeEldestEntry(Map.Entry<List<String>, Bm25> eldest) {
                            return size() > maxsize;
                        }
                    };
        }

        /** 当前缓存条目数（观测用）。 */
        public synchronized int size() {
            return maxsize == 0 ? 0 : entries.size();
        }

        /** 在 texts 候选集上对 query 打 BM25 分（命中则复用已建索引）。 */
        public synchronized double[] scores(List<String> texts, List<String> queryTokens) {
            if (maxsize == 0) { // 显式禁用：退回每次构建
                return new Bm25(tokenizeTexts(texts)).scores(queryTokens);
            }
            List<String> key = List.copyOf(texts);
            Bm25 index = entries.get(key);
            if (index == null) {
                index = new Bm25(tokenizeTexts(texts));
                entries.put(key, index); // accessOrder=true：淘汰最久未用的旧键
            }
            return index.scores(queryTokens);
        }

        private static List<List<String>> tokenizeTexts(List<String> texts) {
            List<List<String>> tokenized = new ArrayList<>(texts.size());
            for (String text : texts) {
                tokenized.add(tokenize(text));
            }
            return tokenized;
        }
    }

    /** 进程级共享缓存（Python 侧 {@code _BM25_CACHE}）。 */
    private static final Bm25Cache BM25_CACHE = new Bm25Cache(64);

    // ── 混合打分核心 ─────────────────────────────────────────────────

    /** 打分结果对：(score, components)，内部传递用。 */
    private record Hybrid(double score, Map<String, Double> components) {}

    /**
     * 混合打分的公共核心（memories 与 doc_chunks 共用）。
     *
     * <p>缺失分量按 0 计、总分恒以全部权重为分母。<b>分数不是绝对相关度</b>：
     * keyword 分量按当前候选集内 BM25 最大值归一化，是候选集相对值——同一
     * 条记忆的分数会随候选集增删而浮动，固定阈值跨查询不严格稳定。稳定的
     * 是下限性质：既无关键词命中也无语义证据的候选，天花板只有
     * recency+salience（≤0.3），检索侧的最低分阈值据此把纯噪音挡在门外。
     *
     * <p>{@code saliences} 的 null 位表示该候选无显著度分量；
     * {@code embeddings} 的 null/空位表示该候选未向量化。
     */
    private static List<Hybrid> hybridScores(List<String> texts, List<byte[]> embeddings,
                                             double[] timestamps, Double[] saliences,
                                             String query, List<Float> queryEmbedding,
                                             double now, double halfLifeDays, Weights weights) {
        Weights w = weights == null ? Weights.DEFAULT : weights;
        double[] bm25Raw = BM25_CACHE.scores(texts, tokenize(query));
        double bm25Max = 0.0;
        for (double value : bm25Raw) {
            if (value > bm25Max) {
                bm25Max = value;
            }
        }
        List<Hybrid> results = new ArrayList<>(texts.size());
        for (int i = 0; i < texts.size(); i++) {
            // LinkedHashMap 保持 Python dict 的插入序（keyword→recency→
            // salience→vector），加权求和的浮点累加顺序与 Python 一致。
            Map<String, Double> components = new LinkedHashMap<>();
            components.put("keyword", bm25Max > 0 ? bm25Raw[i] / bm25Max : 0.0);
            components.put("recency", recencyFactor(timestamps[i], now, halfLifeDays));
            Double salience = saliences[i];
            if (salience != null) {
                components.put("salience", Math.min(Math.max(salience, 0.0), 1.0));
            }
            // 向量分量：查询与候选必须同维可比。
            byte[] embedding = embeddings.get(i);
            if (queryEmbedding != null && embedding != null && embedding.length > 0) {
                Double sim = cosineSimilarity(queryEmbedding, Models.decodeEmbedding(embedding));
                if (sim != null) {
                    // 余弦可为负（反向语义）：截到 0..1 只保留正向证据。
                    components.put("vector", Math.min(Math.max(sim, 0.0), 1.0));
                }
            }
            double score = 0.0;
            for (Map.Entry<String, Double> entry : components.entrySet()) {
                score += w.forName(entry.getKey()) * entry.getValue();
            }
            results.add(new Hybrid(score, components));
        }
        return results;
    }

    /** 按分数降序稳定排序（并列保持输入序，与 Python sort(reverse=True) 一致）。 */
    private static <T> void sortByScoreDescending(List<T> scored,
                                                  java.util.function.ToDoubleFunction<T> score) {
        scored.sort((a, b) -> Double.compare(score.applyAsDouble(b), score.applyAsDouble(a)));
    }

    // ── 面向记忆的打分 ───────────────────────────────────────────────

    /**
     * 对记忆候选集统一打分（时间基点 {@code updated_at}；返回按分数降序，
     * 并列保持候选序）。
     */
    public static List<ScoredMemory> scoreMemories(List<Models.MemoryRecord> records,
                                                   String query, List<Float> queryEmbedding,
                                                   double now, double halfLifeDays,
                                                   Weights weights) {
        if (records.isEmpty()) {
            return List.of();
        }
        List<String> texts = new ArrayList<>(records.size());
        List<byte[]> embeddings = new ArrayList<>(records.size());
        double[] timestamps = new double[records.size()];
        Double[] saliences = new Double[records.size()];
        for (int i = 0; i < records.size(); i++) {
            Models.MemoryRecord record = records.get(i);
            texts.add(record.content());
            embeddings.add(record.embedding());
            timestamps[i] = record.updatedAt();
            saliences[i] = record.salience();
        }
        List<Hybrid> scored = hybridScores(texts, embeddings, timestamps, saliences, query,
                queryEmbedding, now, halfLifeDays, weights);
        List<ScoredMemory> out = new ArrayList<>(records.size());
        for (int i = 0; i < records.size(); i++) {
            out.add(new ScoredMemory(records.get(i), scored.get(i).score(), scored.get(i).components()));
        }
        sortByScoreDescending(out, ScoredMemory::score);
        return out;
    }

    /** 便捷版：now 取当前时间、半衰期与权重读 Settings.memory。 */
    public static List<ScoredMemory> scoreMemories(List<Models.MemoryRecord> records,
                                                   String query, List<Float> queryEmbedding,
                                                   Settings config) {
        return scoreMemories(records, query, queryEmbedding, Models.now(),
                config.memory.timeDecayHalfLifeDays, Weights.fromSettings(config));
    }

    // ── 面向知识库分块的打分 ─────────────────────────────────────────

    /**
     * 对知识库分块统一打分（分块没有显著度，时间戳用 created_at，其余
     * 分量一致；返回按分数降序，并列保持候选序）。
     */
    public static List<ScoredChunk> scoreChunks(List<Models.DocChunk> chunks, String query,
                                                List<Float> queryEmbedding, double now,
                                                double halfLifeDays, Weights weights) {
        if (chunks.isEmpty()) {
            return List.of();
        }
        List<String> texts = new ArrayList<>(chunks.size());
        List<byte[]> embeddings = new ArrayList<>(chunks.size());
        double[] timestamps = new double[chunks.size()];
        Double[] saliences = new Double[chunks.size()];
        for (int i = 0; i < chunks.size(); i++) {
            Models.DocChunk chunk = chunks.get(i);
            texts.add(chunk.text());
            embeddings.add(chunk.embedding());
            timestamps[i] = chunk.createdAt();
            saliences[i] = null;
        }
        List<Hybrid> scored = hybridScores(texts, embeddings, timestamps, saliences, query,
                queryEmbedding, now, halfLifeDays, weights);
        List<ScoredChunk> out = new ArrayList<>(chunks.size());
        for (int i = 0; i < chunks.size(); i++) {
            out.add(new ScoredChunk(chunks.get(i), scored.get(i).score(), scored.get(i).components()));
        }
        sortByScoreDescending(out, ScoredChunk::score);
        return out;
    }

    /** 便捷版：now 取当前时间、半衰期与权重读 Settings.memory。 */
    public static List<ScoredChunk> scoreChunks(List<Models.DocChunk> chunks, String query,
                                                List<Float> queryEmbedding, Settings config) {
        return scoreChunks(chunks, query, queryEmbedding, Models.now(),
                config.memory.timeDecayHalfLifeDays, Weights.fromSettings(config));
    }

    // ── 一次完整召回 ─────────────────────────────────────────────────

    /**
     * 一次完整召回：取候选 → TTL 过滤 → 混合打分 → top-k + 访问记账。
     *
     * <p>{@code agentId} 为 null 时跨模块；{@code topK} 为 null 时取
     * {@code settings.memory.recallTopK}。
     */
    public static List<ScoredMemory> recallMemories(MemoryStore store, EmbeddingClient embedder,
                                                    String userId, String agentId, String query,
                                                    Integer topK, double minScore,
                                                    Settings config) {
        return recallMemories(store, embedder, userId, agentId, query, topK, minScore, config, null);
    }

    /**
     * 向量下推变体（#324）：vectorIndex 非空且查询向量可用、候选集存在
     * 向量时，混合打分的候选池 = 向量 KNN top-N ∪ 无向量候选（关键词
     * 路径不受下推挤压）。否则与基线完全一致——brute-force 默认实现下
     * 排序逐条相同（#325 的语义基线）。
     */
    public static List<ScoredMemory> recallMemories(MemoryStore store, EmbeddingClient embedder,
                                                    String userId, String agentId, String query,
                                                    Integer topK, double minScore,
                                                    Settings config, VectorIndexPort vectorIndex) {
        // 容量契约（H2）：候选集上限与 store 的列表上限一致。达到上限说明
        // 用户记忆量已超出当前"全量拉取 + 内存打分"实现的设计容量——更早
        // 的记忆本轮永远召不回，必须显式告警而不是静默截断。
        double timestamp = Models.now();
        // episodic TTL 下推（#180）：过期行在 SQL 层被排除，不再挤占候选
        // 名额；内存侧 filterExpired 保留兜底（防未覆写下推的后端）。
        Double episodicCutoff = config.memory.episodicTtlDays > 0
                ? timestamp - config.memory.episodicTtlDays * DAY_SECONDS
                : null;
        List<Models.MemoryRecord> records = store.listMemories(userId, agentId, true, null,
                List.of("active"), CANDIDATE_LIMIT, false, episodicCutoff);
        if (records.size() >= CANDIDATE_LIMIT) {
            LOG.warn(
                    "memory: 召回候选达到上限 {}（user='{}', agent='{}'）——更早的记忆不参与本轮检索；"
                            + "请考虑分片或引入候选下推索引（见 README 容量契约）",
                    CANDIDATE_LIMIT, userId, agentId);
        }
        records = filterExpired(records, config.memory.episodicTtlDays, timestamp); // 兜底（#180）
        // 用户画像不是可召回记忆：它由上下文组装单独、整体注入（M6d）。
        List<Models.MemoryRecord> visible = new ArrayList<>(records.size());
        for (Models.MemoryRecord record : records) {
            if (!record.memoryId().startsWith(Models.PROFILE_ID_PREFIX)) {
                visible.add(record);
            }
        }
        records = visible;
        if (records.isEmpty()) {
            return List.of();
        }
        List<Float> queryEmbedding = null;
        if (embedder != null) {
            List<List<Float>> vectors = embedder.embed(List.of(query));
            if (vectors != null && !vectors.isEmpty()) {
                queryEmbedding = vectors.get(0);
            }
        }
        // 关键词下推（#326/327，MEMORY_BM25_PUSHDOWN）：后端关键词索引的
        // top-N id 集收窄候选池。后端未实现（null）→ 回退全量。SQL 侧分数
        // 不进混合打分——池内候选回进程内 hybridScores 重打分（分数归一
        // 基准=池内 max，与基线的差异由 golden 门守护不降）。
        java.util.Set<String> keywordPool = null;
        if (config.memory.bm25Pushdown) {
            List<String> keywordIds = store.searchMemoryIds(userId, agentId, query, BM25_CANDIDATES);
            if (keywordIds != null) {
                keywordPool = new java.util.HashSet<>(keywordIds);
            }
        }
        // 向量下推（#324）：仅在索引、查询向量与库内向量三者齐备时收窄
        // 混合打分候选池——无向量的候选（关键词路径）永远保留。与关键词
        // 下推都开时候选池 = 关键词 top-N ∪ KNN top-100（并集）。
        boolean vectorNarrowed = false;
        if (vectorIndex != null && queryEmbedding != null) {
            boolean anyEmbedded = false;
            for (Models.MemoryRecord record : records) {
                if (record.embedding() != null) {
                    anyEmbedded = true;
                    break;
                }
            }
            if (anyEmbedded) {
                vectorNarrowed = true;
                List<VectorIndexPort.VectorHit> hits =
                        vectorIndex.search(queryEmbedding, KNN_CANDIDATES);
                java.util.Set<String> keepIds = new java.util.HashSet<>(hits.size());
                for (VectorIndexPort.VectorHit hit : hits) {
                    keepIds.add(hit.id());
                }
                if (keywordPool != null) {
                    keepIds.addAll(keywordPool);
                }
                List<Models.MemoryRecord> pooled = new ArrayList<>(records.size());
                for (Models.MemoryRecord record : records) {
                    if (record.embedding() == null || keepIds.contains(record.memoryId())) {
                        pooled.add(record);
                    }
                }
                records = pooled;
            }
        }
        if (keywordPool != null && !vectorNarrowed) {
            // 向量分支未运行（索引缺席/无查询向量/库内无向量）：关键词
            // 下推独立收窄——严格按 top-N 集合（无豁免类，见容量声明）。
            List<Models.MemoryRecord> pooled = new ArrayList<>(records.size());
            for (Models.MemoryRecord record : records) {
                if (keywordPool.contains(record.memoryId())) {
                    pooled.add(record);
                }
            }
            records = pooled;
        }
        List<ScoredMemory> scored = new ArrayList<>(scoreMemories(records, query, queryEmbedding,
                timestamp, config.memory.timeDecayHalfLifeDays, Weights.fromSettings(config)));
        scored.sort((a, b) -> Double.compare(b.score(), a.score()));
        // 最低分门槛：双条件——总分达标，且至少有一个**相关性证据**
        // （关键词命中或语义相似）。纯 recency+salience 凑数（天花板 0.3）
        // 的噪音记忆，宁可空手也不召回（UI 实测发现 F3）。
        if (minScore > 0) {
            List<ScoredMemory> gated = new ArrayList<>(scored.size());
            for (ScoredMemory item : scored) {
                if (item.score() >= minScore
                        && (item.keywordScore() > 0 || item.vectorScore() > 0)) {
                    gated.add(item);
                }
            }
            scored = gated;
        }
        int k = topK != null ? topK : config.memory.recallTopK;
        List<ScoredMemory> top = new ArrayList<>(
                scored.subList(0, Math.min(Math.max(1, k), scored.size())));
        // 访问记账（被召回本身是记忆"还活着"的证据）：失败只记日志。
        try {
            store.touchMemories(top.stream().map(item -> item.record().memoryId()).toList());
        } catch (Exception e) {
            LOG.warn("memory: 召回记账失败", e);
        }
        return top;
    }
}
