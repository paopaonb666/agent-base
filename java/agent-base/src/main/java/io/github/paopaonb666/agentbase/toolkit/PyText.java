package io.github.paopaonb666.agentbase.toolkit;

import java.math.BigDecimal;

/**
 * Python 侧"观察面"的文本对齐助手（包内工具）。
 *
 * <p>工具的错误文案与结果渲染要逐字对齐 Python 版，而 Python 的错误
 * 信息里大量使用 {@code repr()}（如 {@code f"路径 {path!r} …"}）与
 * {@code repr(float)} 的最短表示；Java 缺少直接对应物，这里补齐最小
 * 子集。仅供本包内工具与测试使用。
 */
final class PyText {

    private PyText() {
    }

    /**
     * 近似 Python {@code repr(str)}：默认单引号包裹；字符串含单引号且
     * 不含双引号时换用双引号（与 CPython 的选引号策略一致）。转义覆盖
     * 反斜杠、所选引号与常见控制符；其余控制字符不展开——错误文案场景
     * 里不会出现，注释在此声明该近似边界。
     */
    static String repr(String s) {
        return s.contains("'") && !s.contains("\"") ? quote(s, '"') : quote(s, '\'');
    }

    private static String quote(String s, char q) {
        StringBuilder sb = new StringBuilder(s.length() + 2);
        sb.append(q);
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == q || c == '\\') {
                sb.append('\\').append(c);
            } else if (c == '\n') {
                sb.append("\\n");
            } else if (c == '\r') {
                sb.append("\\r");
            } else if (c == '\t') {
                sb.append("\\t");
            } else {
                sb.append(c);
            }
        }
        return sb.append(q).toString();
    }

    /**
     * 近似 Python {@code repr(float)}（= {@code str(float)}，py3 起
     * 两者相同）：最短十进制表示，绝对值指数落在 {@code [1e-4, 1e16)}
     * 之外时用 {@code e±XX} 形式（指数至少两位数），inf/nan 用
     * {@code inf/nan}。用于 json_query 的结果渲染与 calculator 的输出。
     */
    static String floatRepr(double d) {
        if (Double.isNaN(d)) {
            return "nan";
        }
        if (d == Double.POSITIVE_INFINITY) {
            return "inf";
        }
        if (d == Double.NEGATIVE_INFINITY) {
            return "-inf";
        }
        // Double.toString（JDK 19+ 为最短表示）给出的数字经 BigDecimal
        // 重新排版成 Python 的固定/指数两种形态。
        BigDecimal bd = BigDecimal.valueOf(d).stripTrailingZeros();
        int adjusted = bd.precision() - bd.scale() - 1;
        if (adjusted >= -4 && adjusted < 16) {
            String plain = bd.toPlainString();
            return plain.contains(".") ? plain : plain + ".0";
        }
        String digits = bd.unscaledValue().abs().toString();
        String mantissa = digits.length() == 1 ? digits : digits.charAt(0) + "." + digits.substring(1);
        int exp = adjusted;
        return (d < 0 ? "-" : "") + mantissa + "e"
                + (exp < 0 ? "-" : "+") + String.format("%02d", Math.abs(exp));
    }
}
