package io.github.paopaonb666.agentbase.core.text;

/**
 * Python {@code len()} 对齐的字符计量：按 Unicode 码点计数，而非 Java
 * 的 UTF-16 编码单元。
 *
 * <p>含 emoji / 生僻字（代理对）的内容在 UTF-16 口径下长度是码点口径的
 * 两倍——长度守卫与截断若混用两种口径，边界行为会与 Python 侧不一致
 * （#122/#138）。所有面向用户的长度限制、截断与「已用 N 字符」回执
 * 统一走这里；存储序列化等 UTF-16 内部口径不受影响。
 */
public final class CharMetrics {

    private CharMetrics() {}

    /** Python {@code len()} 语义：null 视为 0，其余按码点计数。 */
    public static int len(String s) {
        return s == null ? 0 : s.codePointCount(0, s.length());
    }

    /** 保头部截断：取前 {@code maxChars} 个码点（Python {@code s[:maxChars]}）。 */
    public static String head(String s, int maxChars) {
        if (s == null || maxChars <= 0) {
            return "";
        }
        int limit = Math.min(maxChars, len(s));
        return s.substring(0, s.offsetByCodePoints(0, limit));
    }

    /** 保尾部截断：取最后 {@code maxChars} 个码点（Python {@code s[-maxChars:]}）。 */
    public static String tail(String s, int maxChars) {
        if (s == null || maxChars <= 0) {
            return "";
        }
        int total = len(s);
        if (total <= maxChars) {
            return s;
        }
        return s.substring(s.offsetByCodePoints(s.length(), -maxChars));
    }
}
