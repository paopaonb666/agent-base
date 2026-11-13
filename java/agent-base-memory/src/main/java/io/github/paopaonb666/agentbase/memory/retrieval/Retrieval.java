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
 * <p>对应 Python 侧 {@code agent_base_memory/retrieval.py}：分词、分量装配
 * 与时间衰减逐算法移植；<b>证据融合有意分歧</b>——Java 用加权 RRF（#53，
 * keyword/vector 两证据通道按通道内排名贡献融合，对量纲失配鲁棒），
 * Python 仍线性加权；行为分歧由 #44 golden 门（recall@5/MRR 不降）守护。
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

    /**
     * 访问频次先验的饱和常数（A-46）：log1p 缩放的分母基准——count≥100
     * 因子饱和到 1.0（高频常驻记忆不再增益；低频新记忆仍可得小先验）。
     */
    public static final double ACCESS_PRIOR_SATURATION = 100.0;

    /**
     * 访问频次 → log 缩放先验因子（A-46）：{@code min(1, ln(1+count)/ln(1+100))}
     * ——log 缩放抑制幂律尾巴（count 10000 与 100 同值），count≤0（从未被
     * 召回记账/新记忆）= 0 不增益。
     */
    public static double accessFactor(int accessCount) {
        if (accessCount <= 0) {
            return 0.0;
        }
        return Math.min(1.0, Math.log1p(accessCount) / Math.log1p(ACCESS_PRIOR_SATURATION));
    }

    /** 一天的秒数（时间衰减/TTL 换算）。 */
    private static final double DAY_SECONDS = 86400.0;

    private Retrieval() {}

    /**
     * 混合检索权重（对应 Python 的 weights mapping，四个固定键 + A-46 的
     * 访问频次先验开关/权重——第五分量是元数据先验不是检索证据）。
     *
     * @param accessPriorEnabled 访问频次先验开关（A-46，默认关）：关时
     *                           components 不产生 access 键，打分逐字节=
     *                           四分量旧口径；开时 {@link #access} 生效。
     * @param access             access 分量权重（参与和=1 校验，默认 0.0
     *                           → 和不变）；开关关时该值不生效（inert）。
     */
    public record Weights(double vector, double keyword, double recency, double salience,
                          boolean accessPriorEnabled, double access) {

        /** 默认权重（和为 1，与 M6b 一致；访问先验关闭）。 */
        public static final Weights DEFAULT =
                new Weights(W_VECTOR, W_KEYWORD, W_RECENCY, W_SALIENCE);

        /** 四参便捷构造（访问先验关闭——既有调用方/测试的旧口径）。 */
        public Weights(double vector, double keyword, double recency, double salience) {
            this(vector, keyword, recency, salience, false, 0.0);
        }

        /** 从 Settings 的 memory 节读混合检索权重（锐评 #4：可配置化）。 */
        public static Weights fromSettings(Settings settings) {
            return new Weights(settings.memory.weightVector, settings.memory.weightKeyword,
                    settings.memory.weightRecency, settings.memory.weightSalience,
                    settings.memory.accessPriorEnabled, settings.memory.weightAccess);
        }

        double forName(String name) {
            return switch (name) {
                case "vector" -> vector;
                case "keyword" -> keyword;
                case "recency" -> recency;
                case "salience" -> salience;
                case "access" -> access;
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

        /** 访问频次先验分量（A-46；缺失按 0——开关关时恒 0）。 */
        public double accessScore() {
            return components.getOrDefault("access", 0.0);
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
     * <p>证据融合用<b>加权 RRF</b>（#53，Java 侧语义——Python 仍线性加权，
     * 差异登记）：keyword 与 vector 是两个独立的检索证据<b>通道</b>，各按
     * 通道内排名计贡献 {@link #rrfContribution}（(k+1)/(k+rank)，正证据者
     * 入榜、top-rank=1.0、未入榜=0）——排名口径对 BM25 分数量纲与余弦
     * 尺度的失配鲁棒（线性加权把两通道原始分数直接相加，一方分数量级大时
     * 会淹没另一方）；recency/salience 是元数据先验不是检索证据，仍按原值
     * 线性进总分。缺失分量按 0 计、总分恒以全部权重为分母。<b>分数不是绝对
     * 相关度</b>：RRF 贡献是通道内候选集相对值——同一条记忆的分数会随候选
     * 集增删而浮动，固定阈值跨查询不严格稳定。稳定的下限性质不变：既无
     * 关键词命中也无语义证据的候选（两通道均未入榜），keyword/vector 贡献
     * 为 0，天花板只有 recency+salience（≤0.3），检索侧的最低分阈值据此把
     * 纯噪音挡在门外；证据门（keyword&gt;0 或 vector&gt;0）语义同构——入榜
     * ⟺ 通道命中。
     *
     * <p>{@code saliences} 的 null 位表示该候选无显著度分量；
     * {@code embeddings} 的 null/空位表示该候选未向量化。
     */
    private static List<Hybrid> hybridScores(List<String> texts, List<byte[]> embeddings,
                                             double[] timestamps, Double[] saliences,
                                             String query, List<Float> queryEmbedding,
                                             double now, double halfLifeDays, Weights weights,
                                             int[] accessCounts) {
        Weights w = weights == null ? Weights.DEFAULT : weights;
        double[] bm25Raw = BM25_CACHE.scores(texts, tokenize(query));
        int[] keywordRanks = channelRanks(bm25Raw);
        // 向量通道第一遍：可 Comparable 候选的余弦（未向量化/维度失配/不可
        // 比 = NaN）；排名只认正向证据（sim>0，负向/零=有键无贡献，与旧
        // 「截到 0..1 只保留正向证据」同口径）。
        double[] sims = new double[texts.size()];
        java.util.Arrays.fill(sims, Double.NaN);
        if (queryEmbedding != null) {
            for (int i = 0; i < texts.size(); i++) {
                byte[] embedding = embeddings.get(i);
                if (embedding != null && embedding.length > 0) {
                    Double sim = cosineSimilarity(queryEmbedding, Models.decodeEmbedding(embedding));
                    if (sim != null) {
                        sims[i] = sim;
                    }
                }
            }
        }
        double[] vectorEvidence = new double[texts.size()];
        for (int i = 0; i < sims.length; i++) {
            vectorEvidence[i] = Double.isNaN(sims[i]) ? -1.0 : sims[i];
        }
        int[] vectorRanks = channelRanks(vectorEvidence);
        List<Hybrid> results = new ArrayList<>(texts.size());
        for (int i = 0; i < texts.size(); i++) {
            // LinkedHashMap 保持 Python dict 的插入序（keyword→recency→
            // salience→vector），加权求和的浮点累加顺序与 Python 一致。
            Map<String, Double> components = new LinkedHashMap<>();
            components.put("keyword", keywordRanks[i] > 0 ? rrfContribution(keywordRanks[i]) : 0.0);
            components.put("recency", recencyFactor(timestamps[i], now, halfLifeDays));
            Double salience = saliences[i];
            if (salience != null) {
                components.put("salience", Math.min(Math.max(salience, 0.0), 1.0));
            }
            // A-46 访问频次先验：元数据层第五分量（不进 RRF 证据通道）——
            // accessCounts 非空且开关开才装配（缺位=打分逐字节旧口径）。
            if (accessCounts != null && w.accessPriorEnabled) {
                components.put("access", accessFactor(accessCounts[i]));
            }
            // 向量分量：查询与候选必须同维可比（sims[i] 非 NaN 即可比）。
            if (!Double.isNaN(sims[i])) {
                components.put("vector", vectorRanks[i] > 0 ? rrfContribution(vectorRanks[i]) : 0.0);
            }
            double score = 0.0;
            for (Map.Entry<String, Double> entry : components.entrySet()) {
                score += w.forName(entry.getKey()) * entry.getValue();
            }
            results.add(new Hybrid(score, components));
        }
        return results;
    }

    /**
     * RRF 融合常量（标准 k=60）：rank 越靠前贡献越大，被 1/(k+rank) 的
     * 长尾形态压平——头部置信度集中、长尾保留区分度的经典取值。
     */
    private static final int RRF_K = 60;

    /** 通道内排名的归一贡献：(k+1)/(k+rank) ∈ (0,1]，top-rank=1.0。 */
    private static double rrfContribution(int rank) {
        return (double) (RRF_K + 1) / (RRF_K + rank);
    }

    /**
     * 正证据候选的通道排名（1 起；分数降序，并列按候选序稳定——排序确定
     * 性对 golden 门可复现性至关重要）。0 = 未入榜（无证据或不正分）。
     */
    private static int[] channelRanks(double[] evidence) {
        int[] ranks = new int[evidence.length];
        Integer[] order = new Integer[evidence.length];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        java.util.Arrays.sort(order, (a, b) -> {
            int byScore = Double.compare(evidence[b], evidence[a]);
            return byScore != 0 ? byScore : Integer.compare(a, b);
        });
        int rank = 0;
        for (int idx : order) {
            if (evidence[idx] <= 0) {
                break;
            }
            ranks[idx] = ++rank;
        }
        return ranks;
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
        int[] accessCounts = new int[records.size()];
        for (int i = 0; i < records.size(); i++) {
            Models.MemoryRecord record = records.get(i);
            texts.add(record.content());
            embeddings.add(record.embedding());
            timestamps[i] = record.updatedAt();
            saliences[i] = record.salience();
            accessCounts[i] = record.accessCount();
        }
        List<Hybrid> scored = hybridScores(texts, embeddings, timestamps, saliences, query,
                queryEmbedding, now, halfLifeDays, weights, accessCounts);
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
                queryEmbedding, now, halfLifeDays, weights, null);
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

    // ── Stage-2 精排（#37）──────────────────────────────────────────

    /**
     * 精排参与归一的四个分量（求和顺序固定，浮点确定性同
     * {@code hybridScores} 的分量装配序）。
     */
    private static final List<String> RERANK_DIMS = List.of("keyword", "recency", "salience", "vector");

    /**
     * 显著性信息门（#37）：分量天然值域 [0,1]，spread 不足全量程 5%
     * 视为<b>无信息维度</b>，归一值恒 0.5——与「全等 → 0.5」退化规则
     * 的连续化；防「近全等」噪音被 min-max 放大成主导因子（golden 语料
     * 19 小时 recency 梯度 spread≈0.038 曾把 0.15 权重拉满，实测教训）。
     */
    private static final double RERANK_SPREAD_EPS = 0.05;

    /**
     * Stage-2 精排（#37，{@code MEMORY_RERANK_ENABLED} 默认关）：插在
     * {@link #recallMemories} 的「minScore 双门槛列表 → topK 截断」之间，
     * 对过门槛候选做可选精排。
     *
     * <p>行为（k = {@code config.memory.recallTopK}；开关关或
     * {@code gated.size() <= k} 时<b>原样返回</b>，零行为变化）：
     *
     * <ol>
     * <li>按 Stage-1 顺序分离<b>有证据候选</b>（keyword&gt;0 或 vector&gt;0）
     *     与<b>豁免候选</b>（无证据但过门槛的 #181 条目）；</li>
     * <li>有证据候选取前 {@code max(rerankPool, k)} 条构成短名单（池下限
     *     = k：pool &lt; k 时全量精排，top-k 名额不因池配置丢候选）；</li>
     * <li>短名单内对 keyword/recency/salience/vector 四分量各做 min-max
     *     归一（分量天然值域 [0,1]，spread 不足全量程 5%
     *     （{@code RERANK_SPREAD_EPS}）视为<b>无信息维度</b>——单候选、
     *     全等与近全等一样归一值恒 0.5，防「近全等」噪音被 min-max 放大
     *     成主导因子；分量缺失 = 不参与该维度——归一统计与加权和都
     *     跳过），按 {@link Weights#fromSettings} 加权和重排（并列保持
     *     Stage-1 原序）；</li>
     * <li>池外有证据候选按原序拼接在精排短名单之后（独立调用本方法时的
     *     信息保真；k 截断后不可见），豁免候选按原序排尾——#181
     *     「仍保留（排尾）」语义不变；</li>
     * <li>截 k 返回。</li>
     * </ol>
     *
     * <p>与 Stage-1 的分工：{@link #hybridScores} 的 RRF 分数是通道内
     * <b>排名</b>的相对值（同一条记忆的分数随候选集增删浮动），适合做
     * 门槛；精排在短名单内把四分量拉回可比量纲（min-max 归一），让
     * recency/salience 先验与证据通道在候选间<b>相对</b>强弱上重新仲裁。
     * 质量由 #44 golden 门（recall@5/MRR 不降）守护。注意精排截断口径是
     * {@code recallTopK}：显式 topK 大于它的 {@code recallMemories} 调用
     * 在开启精排时至多得到 {@code recallTopK} 条。
     */
    public static List<ScoredMemory> rerankShortlist(List<ScoredMemory> gated, Settings config) {
        int k = Math.max(1, config.memory.recallTopK);
        if (!config.memory.rerankEnabled || gated.size() <= k) {
            return gated;
        }
        List<ScoredMemory> evidence = new ArrayList<>(gated.size());
        List<ScoredMemory> exempt = new ArrayList<>();
        for (ScoredMemory item : gated) {
            if (item.keywordScore() > 0 || item.vectorScore() > 0) {
                evidence.add(item);
            } else {
                exempt.add(item);
            }
        }
        int poolSize = Math.min(Math.max(config.memory.rerankPool, k), evidence.size());
        List<ScoredMemory> shortlist = new ArrayList<>(evidence.subList(0, poolSize));
        List<ScoredMemory> beyondPool = evidence.subList(poolSize, evidence.size());
        // 每维度 min/max：统计只含短名单内持有该分量的候选（全员缺席 →
        // 维度整体跳过）。
        Map<String, Double> lo = new HashMap<>();
        Map<String, Double> hi = new HashMap<>();
        for (String dim : RERANK_DIMS) {
            double min = Double.POSITIVE_INFINITY;
            double max = Double.NEGATIVE_INFINITY;
            for (ScoredMemory item : shortlist) {
                Double value = item.components().get(dim);
                if (value == null) {
                    continue;
                }
                min = Math.min(min, value);
                max = Math.max(max, value);
            }
            if (min != Double.POSITIVE_INFINITY) {
                lo.put(dim, min);
                hi.put(dim, max);
            }
        }
        Weights weights = Weights.fromSettings(config);
        double[] rerankScore = new double[shortlist.size()];
        for (int i = 0; i < shortlist.size(); i++) {
            ScoredMemory item = shortlist.get(i);
            double sum = 0.0;
            for (String dim : RERANK_DIMS) {
                Double value = item.components().get(dim);
                if (value == null) {
                    continue; // 分量缺失 = 不参与该维度
                }
                double spread = hi.get(dim) - lo.get(dim);
                double norm = spread >= RERANK_SPREAD_EPS
                        ? (value - lo.get(dim)) / spread
                        : 0.5; // 显著性信息门：近全等 = 无信息维度（见 RERANK_SPREAD_EPS）
                sum += weights.forName(dim) * norm;
            }
            rerankScore[i] = sum;
        }
        Integer[] order = new Integer[shortlist.size()];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        java.util.Arrays.sort(order, (a, b) -> {
            int byScore = Double.compare(rerankScore[b], rerankScore[a]);
            return byScore != 0 ? byScore : Integer.compare(a, b); // 并列保持原序
        });
        List<ScoredMemory> out = new ArrayList<>(gated.size());
        for (int idx : order) {
            out.add(shortlist.get(idx));
        }
        out.addAll(beyondPool);
        out.addAll(exempt);
        return out.size() > k ? new ArrayList<>(out.subList(0, k)) : out;
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
        // 高显著度豁免（#181）：salience ≥ 阈值的记忆两道同豁——常驻记忆
        // 与 query 无词面/向量交集时不再永不召回；仍按总分排序（排尾）。
        if (minScore > 0) {
            List<ScoredMemory> gated = new ArrayList<>(scored.size());
            for (ScoredMemory item : scored) {
                // 两道同豁（#181）：只豁证据门则总分门（0.3 天花板 < 0.5）
                // 仍拦死，豁免无效——豁免 = 整个双门槛旁路，排序照旧。
                boolean exempt = config.memory.salienceExemptThreshold > 0
                        && item.record().salience() >= config.memory.salienceExemptThreshold;
                if (exempt || (item.score() >= minScore
                        && (item.keywordScore() > 0 || item.vectorScore() > 0))) {
                    gated.add(item);
                }
            }
            scored = gated;
        }
        // Stage-2 精排（#37）：插在门槛与 topK 截断之间；开关关（默认）时
        // rerankShortlist 原样返回，零行为变化。k 计算提前——精排池与截断
        // 的口径都是 recallTopK（显式 topK 由下方截断兜底：精排开启时至多
        // 返回 recallTopK 条）。
        int k = topK != null ? topK : config.memory.recallTopK;
        scored = rerankShortlist(scored, config);
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
