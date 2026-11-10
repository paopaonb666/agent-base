package io.github.paopaonb666.agentbase.memory.textkit;

import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.memory.embeddings.EmbeddingClient;
import io.github.paopaonb666.agentbase.memory.service.MemoryService;
import io.github.paopaonb666.agentbase.memory.store.InMemoryMemoryStore;
import io.github.paopaonb666.agentbase.memory.store.Models;
import io.github.paopaonb666.agentbase.memory.store.SqliteMemoryStore;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * offsets 坐标 golden 网（roadmap #191，引用溯源的地基）：锁定
 * {@code TextKit.chunkText} 切块坐标与原文的三层恒等式——
 *
 * <ol>
 *   <li><b>码点单位契约</b>：segments 的 start/end 是<b>码点</b>偏移（与
 *       Python str 索引对齐）——不是 UTF-16 char。消费端（Java 调试切片/
 *       前端 JS slice）负责码点→char 换算；emoji 语料（代理对）下两种单位
 *       数值不同，本组用码点切片助手把契约钉死（登记为已知单位约定，
 *       非 bug）。</li>
 *   <li><b>分段重组恒等式</b>：逐段按码点切片、相邻段之间插 {@code \n\n}
 *       拼接 == chunk.text。全局「拼接重建==原文」不成立——段间分隔空白与
 *       strip 掉的首尾空白不保留（设计而非缺陷：分块用于 embedding，
 *       空隙无语义；非空内容段落级全覆盖）。</li>
 *   <li><b>跨后端往返</b>：ingest → InMemory 与临时 sqlite 库的
 *       offsets_json 解码逐值相等（坐标不因持久化漂移）。</li>
 * </ol>
 *
 * 语料覆盖现有用例未覆盖的形态：emoji 代理对/ZWJ、CRLF 分隔、NBSP 行首、
 * 首尾空白 strip 补偿、超长段滑窗与前后段落混合。
 */
class TextKitOffsetsGoldenTest {

    @TempDir
    Path tempDir;

    /** 码点切片助手：把「码点区间」按单位契约切回文本（钉死坐标单位）。 */
    private static String codePointSlice(String text, long start, long end) {
        int[] cps = text.codePoints().toArray();
        StringBuilder sb = new StringBuilder();
        for (long i = start; i < end; i++) {
            sb.appendCodePoint(cps[(int) i]);
        }
        return sb.toString();
    }

    /** 恒等式 2：分段重组（码点切片 + \n\n 连接）== 块文本。 */
    private static void assertSegmentsReassemble(String original, TextKit.ChunkSpan span) {
        StringBuilder reassembled = new StringBuilder();
        for (int i = 0; i < span.segments().size(); i++) {
            long[] seg = span.segments().get(i);
            if (i > 0) {
                reassembled.append("\n\n");
            }
            reassembled.append(codePointSlice(original, seg[0], seg[1]));
        }
        assertEquals(span.text(), reassembled.toString(),
                "分段重组恒等式失败（码点切片口径）");
    }

    /** 恒等式 1+2 对全部块生效；同时断言块内每段切片非空（无退化区间）。 */
    private static void assertGolden(String original, List<TextKit.ChunkSpan> spans) {
        assertTrue(!spans.isEmpty(), "非空语料至少一块");
        for (TextKit.ChunkSpan span : spans) {
            assertTrue(!span.segments().isEmpty());
            for (long[] seg : span.segments()) {
                assertTrue(seg[0] < seg[1], "退化区间：" + java.util.Arrays.toString(seg));
                assertTrue(!codePointSlice(original, seg[0], seg[1]).isEmpty());
            }
            assertSegmentsReassemble(original, span);
        }
    }

    @Test
    void emojiCorpusLocksCodePointUnits() {
        // 代理对（每个 emoji = 2 个 UTF-16 char = 1 个码点）：码点坐标若被
        // 当 char 用会错位——本用例是单位契约的哨兵。
        String text = "👍 相关性判断的第一段。\n\n🎯 与🎯 相关的第二段。";
        // chunkChars=12（= 单段码点数）强制走段落路径——短文本快路径
        // （≤chunkChars 单块单段）不切段，测不到段间坐标。
        List<TextKit.ChunkSpan> spans = TextKit.chunkText(text, 12, 4);
        assertGolden(text, spans);
        // 单位钉死：第二段的码点 start 必须等于原文前缀的码点数（而非 char
        // 数——char 口径会多 1，emoji 代理对让两种单位数值分叉）。
        String firstParagraph = "👍 相关性判断的第一段。";
        int prefixCodePoints = firstParagraph.codePointCount(0, firstParagraph.length()) + 2;
        TextKit.ChunkSpan withSecond = spans.stream()
                .filter(s -> s.text().contains("🎯 与"))
                .findFirst().orElseThrow();
        long secondStart = withSecond.segments().stream()
                .filter(seg -> codePointSlice(text, seg[0], seg[1]).startsWith("🎯"))
                .findFirst().orElseThrow()[0];
        assertEquals(prefixCodePoints, secondStart,
                "第二段码点起点：emoji 语料下码点数 ≠ char 数");
    }

    @Test
    void crlfNbspAndStripCompensation() {
        // CRLF 段落分隔（splitParagraphs 的 \s* 覆盖 \r）、NBSP 行首（isPySpace）、
        // 首尾空白 strip 补偿（坐标相对原始串）。
        String text = "  行首 NBSP 的段落甲。\r\n\r\n段落乙跟在 CRLF 空行后。  ";
        List<TextKit.ChunkSpan> spans = TextKit.chunkText(text, 60, 20);
        assertGolden(text, spans);
        // strip 补偿：第一块起点 > 0（前导 NBSP+空格计入 leading）。
        assertTrue(spans.get(0).segments().get(0)[0] > 0,
                "前导空白应计入 leading 补偿");
        // 尾部空白被 strip：最后一块终点 < 原文 char 长度。
        long lastEnd = spans.get(spans.size() - 1).segments()
                .get(spans.get(spans.size() - 1).segments().size() - 1)[1];
        assertTrue(lastEnd < text.length(), "尾随空白不进坐标（strip 语义）");
    }

    @Test
    void slidingWindowAfterParagraphsKeepsOriginalCoordinates() {
        // 超长段触发段内滑窗 + 前后正常段落混合：窗口坐标仍是原文码点区间。
        String longParagraph = "长".repeat(120);
        String text = "开头段落。\n\n" + longParagraph + "\n\n结尾段落。";
        List<TextKit.ChunkSpan> spans = TextKit.chunkText(text, 50, 20);
        assertGolden(text, spans);
        // 超长段退化为多窗：窗口单区间且相邻窗口推进 step = chunkChars-overlap。
        long windowCount = spans.stream()
                .filter(s -> s.segments().size() == 1
                        && codePointSlice(text, s.segments().get(0)[0], s.segments().get(0)[1])
                                .startsWith("长"))
                .count();
        assertTrue(windowCount >= 2, "超长段应产出多个滑窗块");
    }

    @Test
    void codeBlockAndTableRowsStayWhole() {
        // 代码/表格行（无空行分隔）整段打包；|---| 行不拆散。
        String text = """
                表格文档说明。

                | 列一 | 列二 |
                |---|---|
                | 甲 | 乙 |

                结尾段落。""";
        List<TextKit.ChunkSpan> spans = TextKit.chunkText(text, 200, 40);
        assertGolden(text, spans);
        boolean tableIntact = spans.stream().anyMatch(s ->
                s.text().contains("|---|---|") && s.text().contains("| 甲 | 乙 |"));
        assertTrue(tableIntact, "表格行应打包在同一块");
    }

    @Test
    void offsetsSurviveIngestAndSqliteRoundtrip() throws Exception {
        // 恒等式 3：跨后端矩阵——同一文档 ingest 进 InMemory 与临时 sqlite，
        // offsets_json 解码后逐值相等（坐标不因持久化漂移）。
        String text = "持久化矩阵的第一段，含 emoji 🙂。\n\n第二段谈 数据库 备份。\r\n\r\n第三段收尾。";
        Settings settings = Settings.load(java.util.Map.of("ENV", "development"), null);

        InMemoryMemoryStore memoryStore = new InMemoryMemoryStore();
        MemoryService memoryService = new MemoryService(memoryStore, neverEmbedder(),
                settings, null, null);
        memoryService.ingestDocument("f1", "u1", "chat", text, "chat:matrix");
        List<Models.DocChunk> inMemory = memoryStore.listChunks("u1", null, "f1", 100);
        assertTrue(!inMemory.isEmpty());

        Path db = tempDir.resolve("offsets-" + System.nanoTime() + ".db");
        try (SqliteMemoryStore sqlite = new SqliteMemoryStore(db.toString())) {
            MemoryService sqliteService = new MemoryService(sqlite, neverEmbedder(),
                    settings, null, null);
            sqliteService.ingestDocument("f1", "u1", "chat", text, "chat:matrix");
            List<Models.DocChunk> persisted = sqlite.listChunks("u1", null, "f1", 100);

            assertEquals(inMemory.size(), persisted.size());
            for (int i = 0; i < inMemory.size(); i++) {
                // chunkId 是每次 ingest 独立生成的（两次摄取 id 不同）——
                // 以 ordinal 对齐比较文本与坐标。
                assertEquals(inMemory.get(i).ordinal(), persisted.get(i).ordinal());
                assertEquals(inMemory.get(i).text(), persisted.get(i).text(),
                        "块文本跨后端一致");
                assertEquals(inMemory.get(i).offsets().size(), persisted.get(i).offsets().size(),
                        "段数跨后端一致");
                for (int j = 0; j < inMemory.get(i).offsets().size(); j++) {
                    long[] a = inMemory.get(i).offsets().get(j);
                    long[] b = persisted.get(i).offsets().get(j);
                    assertEquals(a[0], b[0]);
                    assertEquals(a[1], b[1]);
                }
                // sqlite 读回的坐标仍满足重组恒等式（解码链不破坏语义）。
                assertSegmentsReassemble(text, new TextKit.ChunkSpan(
                        persisted.get(i).text(), persisted.get(i).offsets()));
            }
        }
    }

    private static EmbeddingClient neverEmbedder() {
        return new EmbeddingClient() {
            @Override public Integer dims() { return null; }
            @Override public List<List<Float>> embed(List<String> texts) { return null; }
            @Override public boolean probe() { return false; }
        };
    }
}
