package io.github.paopaonb666.agentbase.toolkit.search;


import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;


/**
 * 搜索结果管线：缓存（见 {@link SearchCache}）→ 质量过滤 → 相关性评分
 * → 渲染（对应 Python 侧 {@code search/pipeline.py}）。
 *
 * <p>低质结果模式（URL/标题黑名单）刻意留空作为扩展点：chat-agent 里的
 * "字典站/拼音站"名单是它自己的业务垃圾源，不属于基座。部署方按需覆写
 * {@link #LOW_QUALITY_URL_PATTERNS} / {@link #LOW_QUALITY_TITLE_PATTERNS}。
 */
public final class SearchPipeline {

    /** 低质 URL 模式（子串匹配，小写比较）；默认空（扩展点）。 */
    public static final List<String> LOW_QUALITY_URL_PATTERNS = List.of();
    /** 低质标题模式（子串匹配，小写比较）；默认空（扩展点）。 */
    public static final List<String> LOW_QUALITY_TITLE_PATTERNS = List.of();

    private SearchPipeline() {
    }

    /** 按 URL/标题模式过滤低质结果；默认无模式（扩展点）。 */
    public static boolean isLowQuality(SearchResult result) {
        String url = result.url().toLowerCase();
        String title = result.title().toLowerCase();
        for (String p : LOW_QUALITY_URL_PATTERNS) {
            if (url.contains(p.toLowerCase())) {
                return true;
            }
        }
        for (String p : LOW_QUALITY_TITLE_PATTERNS) {
            if (title.contains(p.toLowerCase())) {
                return true;
            }
        }
        return false;
    }

    /** query 词项在 title/snippet 中的匹配度；标题匹配权重更高。 */
    public static double scoreRelevance(SearchResult result, String query) {
        List<String> terms = new ArrayList<>();
        // 对应 Python 的 query.lower().split()：按任意空白切分并丢弃空段。
        for (String t : query.toLowerCase().trim().split("\\s+")) {
            if (t.length() > 1) {
                terms.add(t);
            }
        }
        if (terms.isEmpty()) {
            return 0.0;
        }
        String title = result.title().toLowerCase();
        String snippet = result.snippet().toLowerCase();
        int matches = 0;
        int titleMatches = 0;
        for (String t : terms) {
            if (title.contains(t) || snippet.contains(t)) {
                matches++;
            }
            if (title.contains(t)) {
                titleMatches++;
            }
        }
        return matches + titleMatches * 2.0;
    }

    private record Scored(double score, SearchResult result) {
    }

    /** 过滤低质结果、按相关性降序排序并重新编号（排序稳定，平局保持原序）。 */
    public static List<SearchResult> filterAndRank(List<SearchResult> results, String query,
                                                   int maxResults) {
        List<Scored> scored = new ArrayList<>();
        for (SearchResult r : results) {
            if (!isLowQuality(r)) {
                scored.add(new Scored(scoreRelevance(r, query), r));
            }
        }
        scored.sort(Comparator.comparingDouble(Scored::score).reversed());
        List<SearchResult> ranked = new ArrayList<>();
        for (int i = 0; i < Math.min(maxResults, scored.size()); i++) {
            ranked.add(scored.get(i).result());
        }
        List<SearchResult> out = new ArrayList<>(ranked.size());
        for (int i = 0; i < ranked.size(); i++) {
            SearchResult r = ranked.get(i);
            out.add(new SearchResult(r.title(), r.url(), r.snippet(), i + 1, r.engine()));
        }
        return out;
    }

    /** 把结果渲染为带编号的引用上下文（编号供模型在回复中标注来源）。 */
    public static String formatWebContext(List<SearchResult> results) {
        if (results.isEmpty()) {
            return "";
        }
        // 与 Python 侧逐行同构：首行带 \n、每条结果后跟一个空行、\n join。
        // 尾行引用约定为 Java 侧扩展（#254/#338）：[n] 序号与来源事件解析
        // （Citations）配套，Python 侧差异登记在案。
        List<String> lines = new ArrayList<>(results.size() * 4 + 2);
        lines.add("以下是从互联网搜索到的相关信息：\n");
        for (SearchResult r : results) {
            lines.add("[" + r.position() + "] " + r.title());
            lines.add("    URL: " + r.url());
            lines.add("    摘要: " + r.snippet() + " (来源: " + r.engine() + ")");
            lines.add("");
        }
        lines.add("（回答引用以上来源时，请用 [n] 标注对应序号）");
        return String.join("\n", lines);
    }
}
