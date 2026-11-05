package io.github.paopaonb666.agentbase.toolkit.parsing;

import java.util.ArrayList;
import java.util.List;
import java.util.TreeSet;

/**
 * 页码选择解析（对应 Python 侧 {@code parsing/base.py} 的
 * {@code parse_page_selection}）：把 {@code "1-3,5"} 这样的页码选择解析
 * 成去重升序的 1-based 页号列表。
 *
 * <p>页码通常来自 LLM 的工具参数，因此任何畸形或越界输入都抛出带可读
 * 信息的 {@link DocumentParseError}——报错文本经 ToolMessage 回给模型后，
 * 它能理解规则并自行修正重试。
 */
public final class PageSelection {

    private PageSelection() {
    }

    /** 把 {@code "1-3,5"} 解析成去重升序的 1-based 页号列表。 */
    public static List<Integer> parse(String spec, int total) {
        TreeSet<Integer> selected = new TreeSet<>();
        for (String part : spec.split(",", -1)) {
            String token = part.strip();
            if (token.isEmpty()) {
                throw new DocumentParseError("无效的页码选择 '" + spec + "'：存在空的页码项");
            }
            int start;
            int end;
            if (token.contains("-")) {
                String[] bounds = token.split("-", -1);
                if (bounds.length != 2) {
                    throw new DocumentParseError(
                            "无效的页码选择 '" + spec + "'：'" + token + "' 不是页码或页码区间");
                }
                start = parsePageNumber(bounds[0], spec);
                end = parsePageNumber(bounds[1], spec);
                if (end < start) {
                    throw new DocumentParseError(
                            "无效的页码选择 '" + spec + "'：区间终点 " + end + " 小于起点 " + start);
                }
            } else {
                start = end = parsePageNumber(token, spec);
            }
            if (start < 1) {
                throw new DocumentParseError(
                        "无效的页码选择 '" + spec + "'：页码从 1 开始，收到了 " + start);
            }
            if (end > total) {
                throw new DocumentParseError(
                        "页码超出范围：文档共 " + total + " 页，但页码选择 '" + spec + "' 请求到了第 "
                                + end + " 页");
            }
            for (int p = start; p <= end; p++) {
                selected.add(p);
            }
        }
        return new ArrayList<>(selected);
    }

    /** 把单个页码 token 解析成整数；畸形输入给出指向原始 spec 的报错。 */
    private static int parsePageNumber(String token, String spec) {
        try {
            // Python 的 int() 容忍首尾空白（"1 - 3" 里的两段各带空格）。
            return Integer.parseInt(token.strip());
        } catch (NumberFormatException e) {
            throw new DocumentParseError(
                    "无效的页码选择 '" + spec + "'：'" + token + "' 不是整数页码");
        }
    }
}
