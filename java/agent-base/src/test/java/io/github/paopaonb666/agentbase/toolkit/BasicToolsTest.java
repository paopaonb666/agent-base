package io.github.paopaonb666.agentbase.toolkit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import io.github.paopaonb666.agentbase.core.tools.AgentTool;

/**
 * 零依赖基础工具的测试：current_time / calculator / json_query（对应
 * Python 侧 tests/test_basic_tools.py，行为断言逐条移植）。
 */
class BasicToolsTest {

    private static String call(AgentTool tool, String argsJson) {
        try {
            return tool.call(argsJson, new AgentTool.ToolContext(null, null));
        } catch (RuntimeException e) {
            throw e;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private static IllegalArgumentException expectError(Runnable action, String contains) {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, action::run);
        assertTrue(e.getMessage().contains(contains),
                "错误信息应含 '" + contains + "'，实际：" + e.getMessage());
        return e;
    }

    // ── current_time ─────────────────────────────────────────────────

    @Test
    void currentTimeDefaultIsLocalWithTzName() {
        CurrentTimeTool tool = new CurrentTimeTool();
        String rendered = call(tool, "{}");
        assertTrue(rendered.contains("T")); // ISO 8601
        assertTrue(rendered.contains("(") && rendered.contains(")")); // 时区名注记
        // 未传 tz 与显式 "local" 行为一致。
        assertTrue(call(tool, "{\"tz\": \"local\"}").contains("T"));
    }

    @Test
    void currentTimeUtc() throws Exception {
        String rendered = call(new CurrentTimeTool(), "{\"tz\": \"UTC\"}");
        assertTrue(rendered.endsWith("(UTC)"));
        // #249：真转 UTC——偏移必须是 +00:00，且挂钟贴近真实 UTC 时刻
        // （旧实现是本地挂钟误标 UTC，时区非 UTC 的机器上会偏差数小时）。
        assertTrue(rendered.contains("+00:00"), "应带零偏移：" + rendered);
        java.time.OffsetDateTime parsed = java.time.OffsetDateTime.parse(
                rendered.substring(0, rendered.lastIndexOf(" (")));
        long skewSeconds = Math.abs(java.time.Duration.between(
                parsed.toInstant(), java.time.Instant.now()).getSeconds());
        assertTrue(skewSeconds < 60, "utc 分支应返回真实 UTC 时刻，偏差 " + skewSeconds + "s");
    }

    @Test
    void currentTimeIanaZone() {
        String rendered = call(new CurrentTimeTool(), "{\"tz\": \"Asia/Shanghai\"}");
        assertTrue(rendered.contains("+08:00"));
        assertTrue(rendered.contains("Asia/Shanghai"));
        // 区名与时区名之间是全角逗号（对应 Python 的 zone.key，tzname）。
        assertTrue(rendered.contains("Asia/Shanghai，"));
    }

    @Test
    void currentTimeRejectsUnknownZone() {
        expectError(() -> call(new CurrentTimeTool(), "{\"tz\": \"Mars/Olympus_Mons\"}"), "未知时区");
    }

    // ── calculator ───────────────────────────────────────────────────

    @Test
    void calculatorEvaluates() {
        CalculatorTool tool = new CalculatorTool();
        // 与 Python tests/test_basic_tools.py 的参数化用例一一对应。
        assertEquals("1024", call(tool, "{\"expression\": \"2 ** 10\"}"));
        assertEquals("12", call(tool, "{\"expression\": \"144 / 12\"}"));
        assertEquals("9", call(tool, "{\"expression\": \"(1 + 2) * 3\"}"));
        assertEquals("3", call(tool, "{\"expression\": \"7 // 2\"}"));
        assertEquals("1", call(tool, "{\"expression\": \"7 % 3\"}"));
        assertEquals("-7", call(tool, "{\"expression\": \"-(3 + 4)\"}"));
        assertEquals("5", call(tool, "{\"expression\": \"+5\"}"));
        assertEquals("0.75", call(tool, "{\"expression\": \"0.5 + 0.25\"}"));
    }

    @Test
    void calculatorPythonNumericSemantics() {
        CalculatorTool tool = new CalculatorTool();
        // Python 语义细化：负指数幂得 float；// 与 % 向下取整/符号随除数；
        // 巨大整数保持任意精度。
        assertEquals("0.5", call(tool, "{\"expression\": \"2 ** -1\"}"));
        assertEquals("2", call(tool, "{\"expression\": \"-7 % 3\"}"));
        assertEquals("-4", call(tool, "{\"expression\": \"-7 // 2\"}"));
        assertEquals("100000000000000000000", call(tool, "{\"expression\": \"10 ** 20\"}"));
        assertEquals("1000", call(tool, "{\"expression\": \"1e3\"}")); // 整值 float 按整数打印
        assertEquals("-4", call(tool, "{\"expression\": \"-2 ** 2\"}")); // ** 优先级高于一元负号
    }

    @Test
    void calculatorRejectsDivisionByZero() {
        expectError(() -> call(new CalculatorTool(), "{\"expression\": \"1 / 0\"}"), "除数为零");
    }

    @Test
    void calculatorFloorDivAndModByZeroFollowPythonErrors() {
        // Python 对 // 与 % 的除零抛 ZeroDivisionError（未被工具包装）。
        ArithmeticException e1 = assertThrows(ArithmeticException.class,
                () -> call(new CalculatorTool(), "{\"expression\": \"7 // 0\"}"));
        assertEquals("integer division or modulo by zero", e1.getMessage());
        assertThrows(ArithmeticException.class,
                () -> call(new CalculatorTool(), "{\"expression\": \"7 % 0\"}"));
        assertThrows(ArithmeticException.class,
                () -> call(new CalculatorTool(), "{\"expression\": \"0 ** -1\"}"));
    }

    @Test
    void calculatorRejectsSyntaxError() {
        expectError(() -> call(new CalculatorTool(), "{\"expression\": \"1 +\"}"), "语法错误");
    }

    @Test
    void calculatorRejectsNamesAndCalls() {
        // AST 白名单之外的节点（名字、调用）必须被拒绝——表达式来自模型输出，
        // 绝不能走到 eval 语义。
        expectError(() -> call(new CalculatorTool(),
                "{\"expression\": \"__import__('os').system('true')\"}"), "不支持的语法");
    }

    @Test
    void calculatorRejectsNonNumericLiteral() {
        expectError(() -> call(new CalculatorTool(), "{\"expression\": \"'hello' + 'world'\"}"),
                "只支持数字字面量");
    }

    @Test
    void calculatorRejectsBoolLiteral() {
        // bool 是 int 的子类，但 True/False 不是算术，一律拒绝。
        expectError(() -> call(new CalculatorTool(), "{\"expression\": \"True + 1\"}"),
                "只支持数字字面量");
    }

    @Test
    void calculatorIntegralFloatPrintsExactBinaryTruncation() {
        // Python str(int(float)) 截断的是 double 的精确二进制值，不是
        // Double.toString 的最短十进制（回归：1e23 曾打印成 10^23）。
        assertEquals("99999999999999991611392", call(tool(), "{\"expression\": \"1e23\"}"));
        assertEquals("200000000000000000000", call(tool(), "{\"expression\": \"2.0e20\"}"));
    }

    @Test
    void calculatorIntStrHasCpython4300DigitLimit() {
        // CPython str(int) 的默认上限：超限抛错而不是把 3 万位数字塞进
        // 模型上下文（回归：曾无上限输出）。文案与 CPython ValueError 一致。
        assertEquals(4300, call(tool(), "{\"expression\": \"10 ** 4299\"}").length());
        ArithmeticException e = assertThrows(ArithmeticException.class,
                () -> call(tool(), "{\"expression\": \"2 ** 100000\"}"));
        assertEquals("Exceeds the limit (4300 digits) for integer string conversion; "
                + "use sys.set_int_max_str_digits() to increase the limit", e.getMessage());
    }

    @Test
    void calculatorFloatPowOverflowAndComplexFollowPythonErrors() {
        // 浮点 ** 溢出：Python 抛 OverflowError（errno 元组文案）；负底数
        // 非整数指数返回 complex → isinstance 断言失败（无消息）。
        ArithmeticException overflow = assertThrows(ArithmeticException.class,
                () -> call(tool(), "{\"expression\": \"2.5 ** 1000\"}"));
        assertEquals("(34, 'Result too large')", overflow.getMessage());
        ArithmeticException complex = assertThrows(ArithmeticException.class,
                () -> call(tool(), "{\"expression\": \"(-2.0) ** 0.5\"}"));
        assertEquals("", complex.getMessage());
    }

    @Test
    void calculatorHugeExponentRejectedBeforeComputing() {
        // 2 ** 2147483647 这类表达式 bitLength 守卫拦不住（恰为 31），
        // 必须在分配结果前拒绝——否则 BigInteger.pow 撑爆堆。
        ArithmeticException e = assertThrows(ArithmeticException.class,
                () -> call(tool(), "{\"expression\": \"9 ** 2147483647\"}"));
        assertEquals("幂运算结果过大，已拒绝求值", e.getMessage());
    }

    private static CalculatorTool tool() {
        return new CalculatorTool();
    }

    @Test
    void calculatorRejectsMissingArgument() {
        expectError(() -> call(new CalculatorTool(), "{}"), "expression");
    }

    // ── json_query ───────────────────────────────────────────────────

    private static final String DOCUMENT =
            "{\"data\": {\"items\": [{\"name\": \"first\"}, {\"name\": \"second\"}]}, \"count\": 2}";

    @Test
    void jsonQueryExtractsNestedPath() {
        String result = call(new JsonQueryTool(),
                "{\"json_text\": \"" + DOCUMENT.replace("\"", "\\\"") + "\", \"path\": \"data.items[1].name\"}");
        assertEquals("\"second\"", result);
    }

    @Test
    void jsonQueryHandlesMultipleIndices() {
        String result = call(new JsonQueryTool(),
                "{\"json_text\": \"[[1, 2], [3, 4]]\", \"path\": \"[1][0]\"}");
        assertEquals("3", result);
    }

    @Test
    void jsonQueryEmptyPathSummarizesObject() {
        String result = call(new JsonQueryTool(),
                "{\"json_text\": \"" + DOCUMENT.replace("\"", "\\\"") + "\"}");
        assertTrue(result.contains("JSON 对象"));
        assertTrue(result.contains("data") && result.contains("count"));
    }

    @Test
    void jsonQueryEmptyPathSummarizesListAndScalar() {
        assertTrue(call(new JsonQueryTool(), "{\"json_text\": \"[1, 2, 3]\"}").contains("JSON 数组"));
        assertTrue(call(new JsonQueryTool(), "{\"json_text\": \"42\"}").contains("JSON 标量"));
        // 标量概要的 Python repr 语义：bool/null 用 Python 名，字符串带引号。
        assertTrue(call(new JsonQueryTool(), "{\"json_text\": \"true\"}").contains("JSON 标量：True"));
        assertTrue(call(new JsonQueryTool(), "{\"json_text\": \"null\"}").contains("JSON 标量：None"));
        assertTrue(call(new JsonQueryTool(), "{\"json_text\": \"\\\"hi\\\"\"}").contains("JSON 标量：'hi'"));
    }

    @Test
    void jsonQueryInvalidJson() {
        expectError(() -> call(new JsonQueryTool(), "{\"json_text\": \"{not json\"}"), "JSON 解析失败");
    }

    @Test
    void jsonQueryMissingKeyListsAvailable() {
        expectError(() -> call(new JsonQueryTool(),
                "{\"json_text\": \"" + DOCUMENT.replace("\"", "\\\"") + "\", \"path\": \"data.nope\"}"),
                "可用键");
    }

    @Test
    void jsonQueryIndexOutOfRange() {
        expectError(() -> call(new JsonQueryTool(),
                "{\"json_text\": \"" + DOCUMENT.replace("\"", "\\\"") + "\", \"path\": \"data.items[9]\"}"),
                "越界");
    }

    @Test
    void jsonQueryIndexOnNonList() {
        expectError(() -> call(new JsonQueryTool(),
                "{\"json_text\": \"" + DOCUMENT.replace("\"", "\\\"") + "\", \"path\": \"count[0]\"}"),
                "不可下标");
    }

    @Test
    void jsonQueryMalformedPath() {
        String doc = "{\"json_text\": \"" + DOCUMENT.replace("\"", "\\\"") + "\", \"path\": \"%s\"}";
        expectError(() -> call(new JsonQueryTool(), doc.formatted("data[0")), "缺少右括号");
        // #246：非整数下标报错放宽为"整数或引号键"（引号键是合法语法）
        expectError(() -> call(new JsonQueryTool(), doc.formatted("data[x]")), "不是整数或引号键");
        expectError(() -> call(new JsonQueryTool(), doc.formatted("data..items")), "含空段");
    }

    @Test
    void jsonQueryTruncatesHugeResult() {
        String json = "{\"json_text\": \"{\\\"blob\\\": \\\"" + "x".repeat(10_000) + "\\\"}\", \"path\": \"blob\"}";
        String result = call(new JsonQueryTool(), json);
        assertTrue(result.contains("已截断"));
        assertTrue(result.length() < 10_000);
    }

    @Test
    void jsonQueryDumpsUsesPythonSeparators() {
        // json.dumps 默认分隔符是 ", " / ": "（带空格的紧凑形式）。
        String rendered = call(new JsonQueryTool(),
                "{\"json_text\": \"{\\\"outer\\\": {\\\"a\\\": 1, \\\"b\\\": [1, 2]}}\", \"path\": \"outer\"}");
        assertEquals("{\"a\": 1, \"b\": [1, 2]}", rendered);
    }
}
