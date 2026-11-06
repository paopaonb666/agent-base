package io.github.paopaonb666.agentbase.memory.textkit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

class TextKitTest {

    /** 断言 segments 的每个区间在原文上的子串与块内对应文本一致。 */
    private static void assertSegmentsMapToOriginal(String original, TextKit.ChunkSpan span) {
        String text = span.text();
        int cursor = 0;
        for (long[] segment : span.segments()) {
            String slice = original.substring((int) segment[0], (int) segment[1]);
            assertTrue(text.startsWith(slice, cursor),
                    "区间 (" + segment[0] + "," + segment[1] + ") 与块文本不对应：" + slice);
            cursor += slice.length();
            if (cursor < text.length() && span.segments().size() > 1) {
                // 段落打包块以 \n\n 相接。
                assertEquals("\n\n", text.substring(cursor, Math.min(cursor + 2, text.length())));
                cursor += 2;
            }
        }
        assertEquals(text.length(), cursor);
    }

    // ── 短文本 / 空文本 ──────────────────────────────────────────────

    @Test
    void emptyTextReturnsEmptyList() {
        assertEquals(List.of(), TextKit.chunkText("", 10, 2));
        assertEquals(List.of(), TextKit.chunkText("   \n\n  ", 10, 2));
    }

    @Test
    void shortTextIsSingleSpanWithLeadingOffset() {
        // leading = 2（两个空格），strip 后 "abc" 落在 [2,5)。
        List<TextKit.ChunkSpan> spans = TextKit.chunkText("  abc  ", 10, 2);
        assertEquals(1, spans.size());
        assertEquals("abc", spans.get(0).text());
        assertEquals(1, spans.get(0).segments().size());
        assertEquals(2L, spans.get(0).segments().get(0)[0]);
        assertEquals(5L, spans.get(0).segments().get(0)[1]);
        assertEquals(3, spans.get(0).charLen());
    }

    // ── 段落打包 ─────────────────────────────────────────────────────

    @Test
    void paragraphsPackGreedilyAndKeepWhole() {
        String text = "甲乙丙\n\n丁戊\n\n己庚辛";
        List<TextKit.ChunkSpan> spans = TextKit.chunkText(text, 10, 2);
        // 贪心：3 + (2+2) = 7 ≤ 10 → 同块；再加己庚辛 (5) → 12 > 10 → 另起一块。
        assertEquals(2, spans.size());
        assertEquals("甲乙丙\n\n丁戊", spans.get(0).text());
        assertEquals("己庚辛", spans.get(1).text());
        // 多段落块是多区间，各段坐标指向原文。
        assertEquals(2, spans.get(0).segments().size());
        assertArrayEquals(new long[]{0, 3}, spans.get(0).segments().get(0));
        assertArrayEquals(new long[]{5, 7}, spans.get(0).segments().get(1));
        assertArrayEquals(new long[]{9, 12}, spans.get(1).segments().get(0));
        for (TextKit.ChunkSpan span : spans) {
            assertSegmentsMapToOriginal(text, span);
            assertTrue(span.text().codePointCount(0, span.text().length()) <= 10);
        }
    }

    @Test
    void tableAndListNotSplitAcrossChunks() {
        String table = "| a | b |\n|---|---|\n| 1 | 2 |"; // 无空行，单段
        String after = "尾段内容";
        String text = table + "\n\n" + after;
        List<TextKit.ChunkSpan> spans = TextKit.chunkText(text, 30, 0);
        // 表格作为完整段落整体进块，不会被从中间劈开。
        assertTrue(spans.get(0).text().startsWith("| a | b |"));
        assertTrue(spans.stream().anyMatch(s -> s.text().equals(after)));
    }

    // ── 超长段落：段内固定窗口 ───────────────────────────────────────

    @Test
    void oversizeParagraphFallsBackToSlidingWindow() {
        String para = "x".repeat(10);
        String text = "短\n\n" + para;
        List<TextKit.ChunkSpan> spans = TextKit.chunkText(text, 4, 1);
        // "短" 结算为独立块；10 字符段按 step = 4-1 = 3 滑窗。
        assertEquals(5, spans.size());
        assertEquals("短", spans.get(0).text());
        assertArrayEquals(new long[]{0, 1}, spans.get(0).segments().get(0));
        long[][] expected = {{3, 7}, {6, 10}, {9, 13}, {12, 13}};
        for (int i = 0; i < 4; i++) {
            TextKit.ChunkSpan span = spans.get(i + 1);
            assertEquals(text.substring((int) expected[i][0], (int) expected[i][1]), span.text());
            assertArrayEquals(expected[i], span.segments().get(0));
        }
        for (TextKit.ChunkSpan span : spans) {
            assertTrue(span.text().codePointCount(0, span.text().length()) <= 4);
            assertSegmentsMapToOriginal(text, span);
        }
    }

    // ── 无段落分隔：退化为纯固定窗口 ─────────────────────────────────

    @Test
    void noBlankLineDegeneratesToPlainWindow() {
        String text = "abcdef";
        List<TextKit.ChunkSpan> spans = TextKit.chunkText(text, 4, 1);
        assertEquals(2, spans.size());
        assertEquals("abcd", spans.get(0).text());
        assertArrayEquals(new long[]{0, 4}, spans.get(0).segments().get(0));
        assertEquals("def", spans.get(1).text());
        assertArrayEquals(new long[]{3, 6}, spans.get(1).segments().get(0));
        assertSegmentsMapToOriginal(text, spans.get(0));
        assertSegmentsMapToOriginal(text, spans.get(1));
    }

    @Test
    void leadingBlankLinesCompensatedInWholeTextWindow() {
        String text = "\n\n  甲乙丙丁戊己";
        List<TextKit.ChunkSpan> spans = TextKit.chunkText(text, 6, 0);
        assertEquals(1, spans.size());
        assertEquals("甲乙丙丁戊己", spans.get(0).text());
        assertArrayEquals(new long[]{4, 10}, spans.get(0).segments().get(0));
        assertEquals("甲乙丙丁戊己", text.substring(4, 10));
    }

    @Test
    void coverageNonEmptyContentAllPresent() {
        String text = "一三五\n\n二四六\n\n七九\n\n八零";
        List<TextKit.ChunkSpan> spans = TextKit.chunkText(text, 5, 0);
        StringBuilder joined = new StringBuilder();
        for (TextKit.ChunkSpan span : spans) {
            joined.append(span.text()).append('\u241F');
            assertTrue(span.text().codePointCount(0, span.text().length()) <= 5);
        }
        for (String para : List.of("一三五", "二四六", "七九", "八零")) {
            assertTrue(joined.toString().contains(para), "段落 " + para + " 丢失");
        }
    }

    // ── 画像截断 ─────────────────────────────────────────────────────

    @Test
    void trimProfileDropsFromLastSectionFirst() {
        Map<String, Object> profile = new LinkedHashMap<>();
        profile.put("name", "小明");
        profile.put("facts", new ArrayList<>(List.of("喜欢猫", "住在杭州")));
        profile.put("log", new ArrayList<>(List.of("x".repeat(40))));

        Map<String, Object> trimmed = TextKit.trimProfileToChars(profile, 30);
        // 长度按 Python json.dumps 默认分隔符（", "/": "）计——30 字符预算下
        // log 整体丢弃后 facts 仍超长、逐条丢尽（Python 侧实测同一输入输出
        // {'name': '小明'}）。
        Map<String, Object> expected = new LinkedHashMap<>();
        expected.put("name", "小明");
        assertEquals(expected, trimmed);
        assertTrue(jsonLen(trimmed) <= 30);
        // 原 profile 不被改动。
        assertEquals(2, ((List<?>) profile.get("facts")).size());
        assertTrue(profile.containsKey("log"));
    }

    @Test
    void trimProfileTruncatesOversizeStringValues() {
        Map<String, Object> profile = new LinkedHashMap<>();
        profile.put("a", "x".repeat(100));
        profile.put("b", 42);
        Map<String, Object> trimmed = TextKit.trimProfileToChars(profile, 40);
        assertEquals("x".repeat(20), trimmed.get("a"));
        assertEquals(42, trimmed.get("b"));
        assertTrue(jsonLen(trimmed) <= 40);
    }

    @Test
    void trimProfileExtremeFallbackNeverThrows() {
        Map<String, Object> profile = new LinkedHashMap<>();
        // Python 侧的 30 位大整数（Java long 装不下），用 BigInteger 保持
        // "巨型值无法收缩到预算内 → 极端兜底" 的语义。
        profile.put("n", new java.math.BigInteger("123456789012345678901234567890"));
        Map<String, Object> trimmed = TextKit.trimProfileToChars(profile, 5);
        Map<String, Object> expected = Map.of("truncated", true);
        assertEquals(expected, trimmed);
    }

    @Test
    void trimProfileNestedStructureStaysUnderBudget() {
        Map<String, Object> inner = new LinkedHashMap<>();
        inner.put("内", new ArrayList<>(List.of("a".repeat(30), "b".repeat(30))));
        Map<String, Object> profile = new LinkedHashMap<>();
        profile.put("层级", inner);
        profile.put("尾", new ArrayList<>(List.of("c".repeat(10))));
        Map<String, Object> trimmed = TextKit.trimProfileToChars(profile, 40);
        assertTrue(jsonLen(trimmed) <= 40, "截断后仍超预算：" + trimmed);
        assertFalse(trimmed.isEmpty());
        assertNotEquals(profile, trimmed);
    }

    private static int jsonLen(Map<String, Object> value) {
        try {
            String json = new com.fasterxml.jackson.databind.ObjectMapper()
                    .writeValueAsString(value);
            return json.codePointCount(0, json.length());
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }

    private static void assertArrayEquals(long[] expected, long[] actual) {
        org.junit.jupiter.api.Assertions.assertArrayEquals(expected, actual);
    }
}
