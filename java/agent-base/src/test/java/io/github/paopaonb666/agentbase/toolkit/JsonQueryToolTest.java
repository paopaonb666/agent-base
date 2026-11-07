package io.github.paopaonb666.agentbase.toolkit;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * json_query 路径语法（#246）：负下标与引号键扩展 + 旧语法回归（错误
 * 文案与 Python 语义同款）。
 */
class JsonQueryToolTest {

    private final JsonQueryTool tool = new JsonQueryTool();

    private static final String DOC = """
            {"a": {"b.c": ["x", "y", "z"]}, "items": [{"n": 1}, {"n": 2}], "plain": "v"}
            """;

    private String query(String path) {
        return tool.call("{\"json_text\": " + pyString(DOC) + ", \"path\": " + pyString(path) + "}",
                null);
    }

    private static String pyString(String value) {
        return "\"" + value.replace("\\", "\\\\").replace("\"", "\\\"")
                .replace("\n", "\\n") + "\"";
    }

    // ── 新语法 ────────────────────────────────────────────────────────

    @Test
    void negativeIndexReachesTailElements() {
        assertEquals("\"z\"", query("a[\"b.c\"][-1]"));
        assertEquals("\"x\"", query("a[\"b.c\"][-3]"));
    }

    @Test
    void negativeIndexOutOfRangeNamesValidRange() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> query("a[\"b.c\"][-4]"));
        assertTrue(e.getMessage().contains("负下标有效范围 [-3..-1]"));
    }

    @Test
    void quotedKeysReachDottedNames() {
        assertEquals("\"y\"", query("a[\"b.c\"][1]"));
        // 引号键后可继续下钻
        assertEquals("\"y\"", query("a['b.c'][1]".replace("'", "\"")));
    }

    @Test
    void quotedKeyInsideBracketOnly() {
        assertEquals("\"v\"", query("[\"plain\"]"));
    }

    // ── 旧语法回归 ────────────────────────────────────────────────────

    @Test
    void plainPathsStillWork() {
        assertEquals("{\"n\": 2}", query("items[1]"));
        assertEquals("2", query("items[1].n"));
        assertEquals("\"v\"", query("plain"));
        // 尾随点容忍（旧 split 行为）
        assertEquals("\"v\"", query("plain."));
    }

    @Test
    void errorTextsKeepTheirShape() {
        assertEquals("路径 'a..b' 含空段（连续的点号？）",
                assertThrows(IllegalArgumentException.class, () -> query("a..b")).getMessage());
        // 未闭合下标
        assertTrue(assertThrows(IllegalArgumentException.class, () -> query("items[0"))
                .getMessage().contains("缺少右括号"));
        // 引号键缺右引号
        assertTrue(assertThrows(IllegalArgumentException.class, () -> query("a[\"b.c"))
                .getMessage().contains("缺少右引号"));
        // 数组下标在 dict 上
        assertTrue(assertThrows(IllegalArgumentException.class, () -> query("plain[0]"))
                .getMessage().contains("不可下标"));
    }

    @Test
    void summaryPathStillWorks() {
        String out = tool.call("{\"json_text\": " + pyString(DOC) + "}", null);
        assertTrue(out.startsWith("JSON 对象，3 个键"));
    }
}
