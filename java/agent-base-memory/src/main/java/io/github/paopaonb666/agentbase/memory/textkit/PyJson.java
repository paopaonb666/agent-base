package io.github.paopaonb666.agentbase.memory.textkit;

import java.util.List;
import java.util.Map;

/**
 * Python {@code json.dumps(ensure_ascii=False)} 的形态对齐器（包内工具）。
 *
 * <p>画像的分节截断阈值、存储内容与合并 prompt 的字节形态都由
 * {@code json.dumps} 的默认分隔符（{@code ", "}/{@code ": "}）决定；
 * Jackson 的紧凑输出（{@code ","}/{@code ":"}）会让截断时机与字节
 * 逐位漂移。这里补齐最小子集：紧凑带空格 + indent 模式。
 */
public final class PyJson {

    private PyJson() {}

    /** 对应 {@code json.dumps(value, ensure_ascii=False)}（默认分隔符）。 */
    public static String dumps(Object value) {
        StringBuilder sb = new StringBuilder();
        write(sb, value, -1, 0);
        return sb.toString();
    }

    /** 对应 {@code json.dumps(value, ensure_ascii=False, indent=indent)}。 */
    public static String dumpsIndent(Object value, int indent) {
        StringBuilder sb = new StringBuilder();
        write(sb, value, indent, 0);
        return sb.toString();
    }

    private static void write(StringBuilder sb, Object value, int indent, int depth) {
        if (value == null) {
            sb.append("null");
        } else if (value instanceof String s) {
            writeString(sb, s);
        } else if (value instanceof Boolean b) {
            sb.append(b ? "true" : "false");
        } else if (value instanceof Double d) {
            // Python json 对 float 恒带小数点（1.0 → "1.0"；inf/nan 按名输出）。
            if (d.isNaN()) {
                sb.append("NaN");
            } else if (d == Double.POSITIVE_INFINITY) {
                sb.append("Infinity");
            } else if (d == Double.NEGATIVE_INFINITY) {
                sb.append("-Infinity");
            } else {
                sb.append(d);
            }
        } else if (value instanceof Number n) {
            sb.append(n);
        } else if (value instanceof Map<?, ?> map) {
            if (map.isEmpty()) {
                sb.append("{}");
                return;
            }
            sb.append('{');
            if (indent >= 0) {
                sb.append('\n');
            }
            int i = 0;
            for (Map.Entry<?, ?> e : map.entrySet()) {
                if (i++ > 0) {
                    sb.append(indent >= 0 ? ",\n" : ", ");
                }
                if (indent >= 0) {
                    sb.append(" ".repeat(indent * (depth + 1)));
                }
                writeString(sb, String.valueOf(e.getKey()));
                sb.append(": ");
                write(sb, e.getValue(), indent, depth + 1);
            }
            if (indent >= 0) {
                sb.append('\n').append(" ".repeat(indent * depth));
            }
            sb.append('}');
        } else if (value instanceof Iterable<?> list) {
            java.util.Iterator<?> it = list.iterator();
            if (!it.hasNext()) {
                sb.append("[]");
                return;
            }
            sb.append('[');
            if (indent >= 0) {
                sb.append('\n');
            }
            int i = 0;
            while (it.hasNext()) {
                if (i++ > 0) {
                    sb.append(indent >= 0 ? ",\n" : ", ");
                }
                if (indent >= 0) {
                    sb.append(" ".repeat(indent * (depth + 1)));
                }
                write(sb, it.next(), indent, depth + 1);
            }
            if (indent >= 0) {
                sb.append('\n').append(" ".repeat(indent * depth));
            }
            sb.append(']');
        } else {
            // 其余类型：Python json.dumps(default=str) 的近似。
            writeString(sb, String.valueOf(value));
        }
    }

    /**
     * Python json 字符串转义（ensure_ascii=False）：仅转义 {@code "}、
     * {@code \\} 与控制字符；非 ASCII 原样输出。按码点遍历——Python
     * len 按码点计。
     */
    private static void writeString(StringBuilder sb, String s) {
        sb.append('"');
        int i = 0;
        while (i < s.length()) {
            int cp = s.codePointAt(i);
            switch (cp) {
                case '"' -> sb.append("\\\"");
                case '\\' -> sb.append("\\\\");
                case '\n' -> sb.append("\\n");
                case '\r' -> sb.append("\\r");
                case '\t' -> sb.append("\\t");
                case '\b' -> sb.append("\\b");
                case '\f' -> sb.append("\\f");
                default -> {
                    if (cp < 0x20) {
                        sb.append(String.format("\\u%04x", cp));
                    } else {
                        sb.appendCodePoint(cp);
                    }
                }
            }
            i += Character.charCount(cp);
        }
        sb.append('"');
    }
}
