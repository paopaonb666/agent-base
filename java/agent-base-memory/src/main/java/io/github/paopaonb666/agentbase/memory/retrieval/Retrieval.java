package io.github.paopaonb666.agentbase.memory.retrieval;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.memory.embeddings.EmbeddingClient;
import io.github.paopaonb666.agentbase.memory.store.MemoryStore;
import io.github.paopaonb666.agentbase.memory.store.Models;

/**
 * 混合检索（M6b）：向量 + BM25 + 时间衰减 + 显著度。
 *
 * <p>四个信号（缺失分量按 0 计，分数为绝对相关度）：
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

    private static final Pattern ASCII_WORD_RE = Pattern.compile("[a-zA-Z0-9_]+");
    private static final Pattern CJK_RUN_RE = Pattern.compile("[\\x{4E00}-\\x{9FFF}]+");

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
     * 检索分词：ASCII 词（小写）+ 中文二元组。
     *
     * <p>中文无空格分词，二元组是无词典检索的实用折中——"用户偏好"产出
     * {@code 用户}/{@code 户偏}/{@code 偏好}，足以命中改写查询。
     */
    public static List<String> tokenize(String text) {
        List<String> tokens = new ArrayList<>();
        Matcher ascii = ASCII_WORD_RE.matcher(text);
        while (ascii.find()) {
            tokens.add(ascii.group().toLowerCase(Locale.ROOT));
        }
        Matcher cjk = CJK_RUN_RE.matcher(text);
        while (cjk.find()) {
            int[] run = cjk.group().codePoints().toArray();
            if (run.length == 1) {
                tokens.add(new String(run, 0, 1));
                continue;
            }
            for (int i = 0; i + 2 <= run.length; i++) {
                tokens.add(new String(run, i, 2));
            }
        }
        return List.copyOf(tokens);
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
     * <p>缺失分量按 0 计、总分恒以全部权重为分母——分数是<b>绝对相关度</b>：
     * 既无关键词命中也无语义证据的候选，天花板只有 recency+salience
     * （≤0.3），检索侧的最低分阈值才能可靠地把纯噪音挡在门外。
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
        // 容量契约（H2）：候选集上限与 store 的列表上限一致。达到上限说明
        // 用户记忆量已超出当前"全量拉取 + 内存打分"实现的设计容量——更早
        // 的记忆本轮永远召不回，必须显式告警而不是静默截断。
        List<Models.MemoryRecord> records = store.listMemories(userId, agentId, true, null,
                List.of("active"), CANDIDATE_LIMIT, false);
        if (records.size() >= CANDIDATE_LIMIT) {
            LOG.warn(
                    "memory: 召回候选达到上限 {}（user='{}', agent='{}'）——更早的记忆不参与本轮检索；"
                            + "请考虑分片或引入候选下推索引（见 README 容量契约）",
                    CANDIDATE_LIMIT, userId, agentId);
        }
        double timestamp = Models.now();
        records = filterExpired(records, config.memory.episodicTtlDays, timestamp);
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
