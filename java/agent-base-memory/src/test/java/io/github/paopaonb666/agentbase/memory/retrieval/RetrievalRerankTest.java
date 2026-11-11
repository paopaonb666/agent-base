package io.github.paopaonb666.agentbase.memory.retrieval;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.memory.retrieval.Retrieval.ScoredMemory;
import io.github.paopaonb666.agentbase.memory.store.MemoryStore;
import io.github.paopaonb666.agentbase.memory.store.Models;
import io.github.paopaonb666.agentbase.memory.store.Models.DocChunk;
import io.github.paopaonb666.agentbase.memory.store.Models.MemoryRecord;

/**
 * Stage-2 精排（#37，{@code MEMORY_RERANK_ENABLED} 默认关）的行为面：
 * ①默认关/名单不超 k 的零变化；②短名单 min-max 归一 + 权重和重排的
 * 数学；③#181 豁免条目不参与精排、按原序排尾；④pool&lt;k 时全量精排
 * （池下限 = k）；⑤单候选/全等分量/全豁免的退化路径；⑥recallMemories
 * 的接线（门槛后、topK 截断前，记账发生在精排后的 top-k 上）。
 */
class RetrievalRerankTest {

    private static final double NOW = 1_700_000_000.0;

    // ── 构造件 ─────────────────────────────────────────────────────

    private static MemoryRecord rec(String id, String content, double salience, double updatedAt) {
        return new MemoryRecord(id, "u1", "*", "semantic", content, List.of(), null, null,
                salience, "active", "", List.of(), updatedAt, updatedAt, null, 0);
    }

    /** 手工装配 ScoredMemory（components 受控，归一数学可手算）。 */
    private static ScoredMemory scored(String id, double score, double keyword,
                                       double recency, double salience) {
        Map<String, Double> components = new LinkedHashMap<>();
        components.put("keyword", keyword);
        components.put("recency", recency);
        components.put("salience", salience);
        return new ScoredMemory(rec(id, "内容-" + id, salience, NOW), score, components);
    }

    private static List<String> ids(List<ScoredMemory> list) {
        return list.stream().map(s -> s.record().memoryId()).toList();
    }

    private static Settings settings(String... kv) {
        Map<String, String> env = new LinkedHashMap<>();
        env.put("ENV", "development");
        for (int i = 0; i < kv.length; i += 2) {
            env.put(kv[i], kv[i + 1]);
        }
        return Settings.load(env, null);
    }

    // ── ① 默认关 / 名单不超 k：零行为变化 ──────────────────────────

    @Test
    void rerankDisabledKeepsOrderByteIdentical() {
        List<ScoredMemory> gated = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            gated.add(scored("r" + i, 0.90 - i * 0.01, 0.50 + i * 0.01,
                    0.90 - i * 0.02, 0.10 + i * 0.05));
        }
        // 默认关：10 条过门槛候选原样返回——顺序与逐条内容（含 score 与
        // components）逐字节一致，且不做 k 截断。
        List<ScoredMemory> out = Retrieval.rerankShortlist(gated, new Settings());
        assertEquals(gated, out);
        assertEquals(10, out.size());
        // 开关开但名单不超 k（默认 6）：早退路径同样零变化。
        Settings enabled = settings("MEMORY_RERANK_ENABLED", "true");
        List<ScoredMemory> six = gated.subList(0, 6);
        assertEquals(six, Retrieval.rerankShortlist(six, enabled));
    }

    // ── 配置装载（liftLegacyFlatFields 的 memory_ 前缀归一）────────

    @Test
    void rerankKeysLoadThroughEnvAndValidate() {
        // MEMORY_RERANK_ENABLED / MEMORY_RERANK_POOL 经 Settings.load 装载：
        // env 大写键走 liftLegacyFlatFields 的 memory_ 前缀进 memory 节。
        Settings s = settings("MEMORY_RERANK_ENABLED", "true", "MEMORY_RERANK_POOL", "48");
        assertTrue(s.memory.rerankEnabled);
        assertEquals(48, s.memory.rerankPool);
        // 默认值：false / 24。
        Settings d = settings();
        assertFalse(d.memory.rerankEnabled);
        assertEquals(24, d.memory.rerankPool);
        // 非法 pool：>0 校验，错误文案带键名。
        Settings.SettingsError error = assertThrows(Settings.SettingsError.class,
                () -> settings("MEMORY_RERANK_POOL", "0"));
        assertTrue(error.getMessage().contains("MEMORY_RERANK_POOL"), error.getMessage());
    }

    // ── ② 短名单 min-max 归一 + 加权和重排的数学 ────────────────────

    @Test
    void rerankMathIsMinMaxNormalizedWeightedSum() {
        // k=3、5 条有证据候选（无向量分量 → 该维度整体缺席）。归一范围：
        // kw [0.1,1.0]、rec [0.0,1.0]、sal [0.0,0.9]；默认权重
        // (0.40/0.30/0.15/0.15)：
        //   c1 → 0.30*1.0                          = 0.3000
        //   c2 → 0.30*(1/3)+0.15*1.0+0.15*1.0      = 0.4000
        //   c3 → 0.30*0.0+0.15*0.3+0.15*1.0        = 0.1950
        //   c4 → 0.30*0.5+0.15*0.5+0.15*(2/9)      ≈ 0.2583
        //   c5 → 0.30*(1/6)+0.15*0.8+0.15*(4/9)    ≈ 0.2367
        // 重排 c2 > c1 > c4 > c5 > c3，截 k=3 → [c2, c1, c4]。
        Settings config = settings("MEMORY_RERANK_ENABLED", "true",
                "MEMORY_RECALL_TOP_K", "3");
        List<ScoredMemory> gated = List.of(
                scored("c1", 0.90, 1.00, 0.00, 0.00),
                scored("c2", 0.89, 0.40, 1.00, 0.90),
                scored("c3", 0.88, 0.10, 0.30, 0.90),
                scored("c4", 0.87, 0.55, 0.50, 0.20),
                scored("c5", 0.86, 0.25, 0.80, 0.40));
        assertEquals(List.of("c2", "c1", "c4"),
                ids(Retrieval.rerankShortlist(gated, config)));
    }

    // ── ③ 豁免条目不参与精排、按原序排尾 ────────────────────────────

    @Test
    void exemptEntriesStayAtTailInOriginalOrder() {
        // 4 条有证据（同 ② 的前四条数学：e2 > e1 > e4 > e3）+ 3 条无证据
        // 豁免（#181：keyword=0、无向量、高 salience）→ 7 条 > k=6 触发
        // 精排；豁免不参与精排也不进归一统计，按原序拼接排尾；截 6 →
        // [e2, e1, e4, e3, x1, x2]（x3 被 k 截断淘汰，「仍保留（排尾）」
        // 语义不变——它们排在所有精排后有证据候选之后）。
        Settings config = settings("MEMORY_RERANK_ENABLED", "true");
        List<ScoredMemory> gated = List.of(
                scored("e1", 0.90, 1.00, 0.00, 0.00),
                scored("e2", 0.89, 0.40, 1.00, 0.90),
                scored("e3", 0.88, 0.10, 0.30, 0.90),
                scored("e4", 0.87, 0.55, 0.50, 0.20),
                scored("x1", 0.60, 0.00, 0.50, 1.00),
                scored("x2", 0.59, 0.00, 0.40, 1.00),
                scored("x3", 0.58, 0.00, 0.30, 1.00));
        assertEquals(List.of("e2", "e1", "e4", "e3", "x1", "x2"),
                ids(Retrieval.rerankShortlist(gated, config)));
    }

    // ── ④ pool < k：全量精排（池下限 = k）─────────────────────────

    @Test
    void poolBelowKReranksFullTopK() {
        // MEMORY_RERANK_POOL=2 < k=6：有效池 = max(pool, k) = 6——7 条有
        // 证据候选的前 6 条进短名单（s7 池外，截断后不可见），Stage-1
        // 第 5 名 s5 精排后登顶（若池被字面截到 2，s5 不进精排、不可能
        // 登顶）。归一范围（短名单 s1..s6）：kw [0.5,1.0]、rec [0,1]、
        // sal [0,1]：
        //   s1 → 0.30*1.0              = 0.300
        //   s2 → 0.30*0.8+0.15*0.1*2   = 0.270
        //   s3 → 0.30*0.6+0.15*0.2*2   = 0.240
        //   s4 → 0.30*0.4+0.15*0.3*2   = 0.210
        //   s5 → 0.30*0.1+0.15*1.0*2   = 0.330 ← 精排榜首
        //   s6 → 0.30*0.0+0.15*0.5*2   = 0.150
        Settings config = settings("MEMORY_RERANK_ENABLED", "true",
                "MEMORY_RERANK_POOL", "2", "MEMORY_RECALL_TOP_K", "6");
        List<ScoredMemory> gated = List.of(
                scored("s1", 0.90, 1.00, 0.00, 0.00),
                scored("s2", 0.89, 0.90, 0.10, 0.10),
                scored("s3", 0.88, 0.80, 0.20, 0.20),
                scored("s4", 0.87, 0.70, 0.30, 0.30),
                scored("s5", 0.86, 0.55, 1.00, 1.00),
                scored("s6", 0.85, 0.50, 0.50, 0.50),
                scored("s7", 0.84, 0.05, 0.10, 0.10));
        assertEquals(List.of("s5", "s1", "s2", "s3", "s4", "s6"),
                ids(Retrieval.rerankShortlist(gated, config)));
    }

    // ── ⑤ 退化路径 ─────────────────────────────────────────────────

    @Test
    void degenerateShortlistDoesNotExplode() {
        Settings config = settings("MEMORY_RERANK_ENABLED", "true",
                "MEMORY_RECALL_TOP_K", "3");
        // (a) 单证据候选：短名单只有 1 条 → 四分量归一值全部 = 0.5（防
        // 退化），豁免照常按原序排尾，截 k=3。
        List<ScoredMemory> single = List.of(
                scored("e1", 0.90, 0.70, 0.42, 0.90),
                scored("x1", 0.60, 0.00, 0.50, 1.00),
                scored("x2", 0.59, 0.00, 0.40, 1.00),
                scored("x3", 0.58, 0.00, 0.30, 1.00));
        assertEquals(List.of("e1", "x1", "x2"),
                ids(Retrieval.rerankShortlist(single, config)));
        // (b) 全等分量：短名单内精排分全部并列 → 保持 Stage-1 原序。
        List<ScoredMemory> identical = List.of(
                scored("e1", 0.90, 0.50, 0.50, 0.50),
                scored("e2", 0.89, 0.50, 0.50, 0.50),
                scored("e3", 0.88, 0.50, 0.50, 0.50),
                scored("x1", 0.60, 0.00, 0.50, 0.50));
        assertEquals(List.of("e1", "e2", "e3"),
                ids(Retrieval.rerankShortlist(identical, config)));
        // (c) 全豁免（无一条证据）：无可精排对象 → 原序截 k，不炸。
        List<ScoredMemory> allExempt = List.of(
                scored("x1", 0.60, 0.00, 0.50, 1.00),
                scored("x2", 0.59, 0.00, 0.40, 1.00),
                scored("x3", 0.58, 0.00, 0.30, 1.00),
                scored("x4", 0.57, 0.00, 0.20, 1.00));
        assertEquals(List.of("x1", "x2", "x3"),
                ids(Retrieval.rerankShortlist(allExempt, config)));
    }

    // ── ⑥ 近全等 spread 的信息门（RERANK_SPREAD_EPS）──────────────

    @Test
    void nearDegenerateSpreadFoldsDimensionToHalf() {
        // salience 维度全员 spread=0.03（< 全量程 5%）→ 无信息维度，全员
        // 归一值 0.5：排序只由 kw（spread 0.90）/rec（spread 0.80）决定，
        // 归一数学（k=3，4 证据）：
        //   n1 → 0.30*1.0+0.15*0.0+0.15*0.5 = 0.3750
        //   n2 → 0.30*0.0+0.15*1.0+0.15*0.5 = 0.2250
        //   n3 → 0.30*0.5+0.15*0.5+0.15*0.5 = 0.3000
        //   n4 → 0.30*(2/9)+0.15*0.75+0.15*0.5 ≈ 0.2542
        // 两组候选仅 salience 取值不同（第二组反转大小关系，spread 仍
        // 0.03）→ 输出序必须相同（被折叠维度对排序零影响）。
        Settings config = settings("MEMORY_RERANK_ENABLED", "true",
                "MEMORY_RECALL_TOP_K", "3");
        List<ScoredMemory> a = List.of(
                scored("n1", 0.90, 1.00, 0.10, 0.33),
                scored("n2", 0.89, 0.10, 0.90, 0.30),
                scored("n3", 0.88, 0.55, 0.50, 0.31),
                scored("n4", 0.87, 0.30, 0.70, 0.32));
        assertEquals(List.of("n1", "n3", "n4"), ids(Retrieval.rerankShortlist(a, config)));
        List<ScoredMemory> b = List.of(
                scored("n1", 0.90, 1.00, 0.10, 0.30),
                scored("n2", 0.89, 0.10, 0.90, 0.33),
                scored("n3", 0.88, 0.55, 0.50, 0.32),
                scored("n4", 0.87, 0.30, 0.70, 0.31));
        assertEquals(List.of("n1", "n3", "n4"), ids(Retrieval.rerankShortlist(b, config)),
                "spread<eps 的维度反转取值不应改变排序");
    }

    // ── ⑦ recallMemories 接线：门槛后、topK 截断前 ─────────────────

    /** 测试替身：只实现召回路径用到的两个方法（其余 fail-fast）。 */
    static final class RecallStore implements MemoryStore {
        final List<MemoryRecord> records;
        final List<String> touched = new ArrayList<>();

        RecallStore(List<MemoryRecord> records) {
            this.records = records;
        }

        @Override
        public List<MemoryRecord> listMemories(String userId, String agentId, boolean includeGlobal,
                                               List<String> kinds, List<String> statuses,
                                               int limit, boolean excludeProfile) {
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

    @Test
    void recallMemoriesReranksBetweenGateAndTopK() {
        // 语料（query=「咖啡」，无 embedder → 纯关键词路径；BM25 短文档
        // 占优 → kw 通道排名 eA>eB>eC>eD>eE，ranks 1-5 的 rrf spread
        // = 4/65 ≈ 0.0615 ≥ RERANK_SPREAD_EPS → kw 维度不被信息门折叠；
        // 3 证据语料（spread≈0.032）会被折叠成全并列、翻转消失——故取
        // 5 证据）。rec 原值 {0.4,0.7,1.0,0.2,0.5}、sal {0,0.5,1,0.2,0.9}：
        //   Stage-1（RRF+原值先验）：eC 0.5905 > eE 0.4915 > eB 0.4752
        //     > eA 0.3600 > eD 0.3459（eE 的 salience=0.9 原值抬进第 2）；
        //   Stage-2（min-max 归一）：eC 0.4452 > eB 0.3901 > eA 0.3375
        //     > eE 0.1913 > eD 0.1015（kw 归一后 eB 的综合先验压过 eE）。
        // 双门槛（minScore=0.3）全员有证据通过 → 开精排 top-2 从
        // [eC, eE] 翻转为 [eC, eB]，且与「scoreMemories → rerankShortlist
        // → 截断」的组合管线逐条一致；记账发生在精排后的 top-k 上。
        double now = Models.now();
        double halfLife = 14.0;
        List<MemoryRecord> corpus = List.of(
                rec("eA", "咖啡 coffee", 0.0, now - recencyAge(0.4, halfLife)),
                rec("eB", "咖啡 coffee extra", 0.5, now - recencyAge(0.7, halfLife)),
                rec("eC", "咖啡 coffee extra more", 1.0, now),
                rec("eD", "咖啡 coffee extra more filler", 0.2, now - recencyAge(0.2, halfLife)),
                rec("eE", "咖啡 coffee extra more filler x", 0.9, now - recencyAge(0.5, halfLife)));

        Settings off = settings("MEMORY_RECALL_TOP_K", "2");
        List<ScoredMemory> baseline = Retrieval.recallMemories(
                new RecallStore(corpus), null, "u1", null, "咖啡", null, 0.3, off);
        assertEquals(List.of("eC", "eE"), ids(baseline), "关闭时 = Stage-1 序");

        Settings on = settings("MEMORY_RECALL_TOP_K", "2", "MEMORY_RERANK_ENABLED", "true");
        RecallStore store = new RecallStore(corpus);
        List<ScoredMemory> reranked = Retrieval.recallMemories(
                store, null, "u1", null, "咖啡", null, 0.3, on);
        assertEquals(List.of("eC", "eB"), ids(reranked), "开启后 top-2 翻转");

        // 组合管线等价：与「打分 → 精排 → 截 k」逐条一致（接线位正确）。
        List<ScoredMemory> scoredAll = Retrieval.scoreMemories(corpus, "咖啡", null, now,
                on.memory.timeDecayHalfLifeDays, Retrieval.Weights.DEFAULT);
        List<ScoredMemory> expected = Retrieval.rerankShortlist(scoredAll, on);
        assertEquals(ids(expected.subList(0, Math.min(2, expected.size()))), ids(reranked));
        // 记账发生在精排后的 top-k 上（不是 Stage-1 序）。
        assertEquals(List.of("eC", "eB"), store.touched);
    }

    /** 目标 recency 因子对应的年龄（秒）：age = -log2(target) × halfLife。 */
    private static double recencyAge(double target, double halfLifeDays) {
        return Math.log(1 / target) / Math.log(2) * halfLifeDays * 86400.0;
    }
}
