package io.github.paopaonb666.agentbase.core.text;

/**
 * LLM 输出中的 JSON 块提取（#258）：平衡括号扫描（跳过字符串字面量），
 * 取<b>第一个</b>完整平衡块。贪婪 DOTALL 正则（{@code \[.*\]}）会把
 * 「多 JSON 块 + 尾注」形态的输出误取到最后一段括号——拆解错乱。
 */
public final class JsonBlocks {

    private JsonBlocks() {}

    /** 取首个平衡 JSON 数组 {@code […]}；找不到/不平衡返回 null。 */
    public static String firstBalancedArray(String text) {
        return firstBalanced(text, '[', ']');
    }

    /** 取首个平衡 JSON 对象 {@code {…}}；找不到/不平衡返回 null。 */
    public static String firstBalancedObject(String text) {
        return firstBalanced(text, '{', '}');
    }

    /**
     * 首个平衡括号块：从第一个 {@code open} 起计数到与之配对的
     * {@code close}；字符串字面量内的括号与引号转义不参与计数。
     */
    public static String firstBalanced(String text, char open, char close) {
        if (text == null) {
            return null;
        }
        int depth = 0;
        int start = -1;
        boolean inString = false;
        boolean escaped = false;
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if (inString) {
                if (escaped) {
                    escaped = false;
                } else if (c == '\\') {
                    escaped = true;
                } else if (c == '"') {
                    inString = false;
                }
                continue;
            }
            if (c == '"') {
                inString = true;
            } else if (c == open) {
                if (depth == 0) {
                    start = i;
                }
                depth++;
            } else if (c == close && depth > 0) {
                depth--;
                if (depth == 0) {
                    return text.substring(start, i + 1);
                }
            }
        }
        return null;
    }
}
