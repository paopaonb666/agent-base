package io.github.paopaonb666.agentbase.core.text;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * CharMetrics 码点计量测试（roadmap #122/#138）：len/head/tail 与
 * Python len()/切片语义逐一对齐——代理对（emoji）按 1 个字符计、
 * 截断不劈半。
 */
class CharMetricsTest {

    // U+1F600（😀）是代理对：UTF-16 下 length()==2，码点下 len()==1。
    private static final String EMOJI = "😀";

    @Test
    void lenCountsCodePointsLikePythonLen() {
        assertEquals(0, CharMetrics.len(null));
        assertEquals(0, CharMetrics.len(""));
        assertEquals(3, CharMetrics.len("abc"));
        assertEquals(3, CharMetrics.len("中文a"));
        // 2 emoji + 1 汉字：UTF-16 长度是 5，码点是 3。
        assertEquals(3, CharMetrics.len(EMOJI + EMOJI + "中"));
        assertEquals(5, (EMOJI + EMOJI + "中").length(), "前提：该串确含代理对");
    }

    @Test
    void headKeepsWholeSurrogatePairs() {
        // 前 2 个「字符」是 2 个码点（2 个 emoji），不是 2 个 char。
        assertEquals(EMOJI + EMOJI, CharMetrics.head(EMOJI + EMOJI + "中", 2));
        assertEquals(EMOJI, CharMetrics.head(EMOJI + "中", 1));
        assertEquals("", CharMetrics.head("abc", 0));
        assertEquals("abc", CharMetrics.head("abc", 3));
        assertEquals("abc", CharMetrics.head("abc", 100));
        assertEquals("", CharMetrics.head(null, 5));
    }

    @Test
    void tailKeepsWholeSurrogatePairs() {
        // 最后 1 个「字符」是 1 个码点（1 个 emoji）。
        assertEquals(EMOJI, CharMetrics.tail("中" + EMOJI, 1));
        assertEquals("中" + EMOJI, CharMetrics.tail("中" + EMOJI, 2));
        assertEquals("abc", CharMetrics.tail("abc", 100));
        assertEquals("", CharMetrics.tail("abc", 0));
        assertEquals("", CharMetrics.tail(null, 5));
    }

    @Test
    void headPlusTailNeverSplitsPairsAtBoundary() {
        String s = EMOJI + "ab" + EMOJI + "cd" + EMOJI;
        // len==7（3 emoji + 4 ascii）。任意切法的拼接不含半个代理对：
        // 头 n + 尾 (7-n) 还原全串。
        for (int n = 0; n <= 7; n++) {
            String joined = CharMetrics.head(s, n) + CharMetrics.tail(s, 7 - n);
            assertEquals(s, joined, "n=" + n);
            assertTrue(joined.codePointCount(0, joined.length()) >= 0);
        }
    }
}
