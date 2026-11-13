package io.github.paopaonb666.agentbase.toolkit.search;


import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;


/**
 * 搜索结果管线：缓存（见 {@link SearchCache}）→ 质量过滤 → 相关性评分
 * → 渲染（对应 Python 侧 {@code search/pipeline.py}）。
 *
 * <p>低质结果模式（URL/标题黑名单）：基座默认保守集（引擎无关的流水线
 * 产物垃圾源，宁缺毋滥——审查 A-32 裁决：模式有争议就只留最高置信集）；
 * chat-agent 的"字典站/拼音站"业务名单仍不入基座。部署方经
 * {@code SEARCH_LOW_QUALITY_URL_PATTERNS} / {@code SEARCH_LOW_QUALITY_TITLE_PATTERNS}
 * 整体覆写（逗号分隔；配置含空串=显式清空，未配置=基座默认）。
 */
public final class SearchPipeline {

    /**
     * 基座默认低质 URL 模式（子串匹配，小写比较）：只收两类无争议形态——
     * Google 缓存页（服务已停，结果恒死链）与 translate.goog 翻译代理页
     * （机器翻译镜像，agent 应引用原始来源）。
     */
    public static final List<String> BASE_LOW_QUALITY_URL_PATTERNS = List.of(
            "webcache.googleusercontent.com",
            "translate.goog");
    /**
     * 基座默认低质标题模式：刻意空——标题匹配是内容判断，不存在引擎无关
     * 的高置信集（业务名单属部署方）。
     */
    public static final List<String> BASE_LOW_QUALITY_TITLE_PATTERNS = List.of();

    private SearchPipeline() {
    }

    /**
     * Settings 串（逗号分隔模式）→ 模式表：{@code null}=未配置，回落
     * 基座默认；非 null（含空串）=整体替换（空串=显式清空）。
     */
    public static List<String> parsePatterns(String configured, List<String> baseDefaults) {
        if (configured == null) {
            return baseDefaults;
        }
        List<String> parsed = new ArrayList<>();
        for (String part : configured.split(",")) {
            String p = part.trim().toLowerCase();
            if (!p.isEmpty()) {
                parsed.add(p);
            }
        }
        return List.copyOf(parsed);
    }

    /** 按 URL/标题模式过滤低质结果；模式由调用方传入（Settings 装配）。 */
    public static boolean isLowQuality(SearchResult result,
                                       List<String> urlPatterns,
                                       List<String> titlePatterns) {
        String url = result.url().toLowerCase();
        String title = result.title().toLowerCase();
        for (String p : urlPatterns) {
            if (url.contains(p.toLowerCase())) {
                return true;
            }
        }
        for (String p : titlePatterns) {
            if (title.contains(p.toLowerCase())) {
                return true;
            }
        }
        return false;
    }

    /** 便捷形态：基座默认模式（未配置 Settings 的部署同口径）。 */
    public static boolean isLowQuality(SearchResult result) {
        return isLowQuality(result, BASE_LOW_QUALITY_URL_PATTERNS, BASE_LOW_QUALITY_TITLE_PATTERNS);
    }

    /** query 词项在 title/snippet 中的匹配度；标题匹配权重更高。 */
    public static double scoreRelevance(SearchResult result, String query) {
        // #244：分词改用基座公共件（ASCII 词 + CJK 二元组）——中文查询此前
        // 被空白切分切成不出词项，排序退化为引擎原始序。
        List<String> terms = new ArrayList<>();
        for (String t : io.github.paopaonb666.agentbase.core.text.QueryTerms.tokenize(
                query.toLowerCase())) {
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
        return filterAndRank(results, query, maxResults,
                BASE_LOW_QUALITY_URL_PATTERNS, BASE_LOW_QUALITY_TITLE_PATTERNS);
    }

    /** 同上，模式由调用方传入（Settings 装配的部署覆写口径）。 */
    public static List<SearchResult> filterAndRank(List<SearchResult> results, String query,
                                                   int maxResults,
                                                   List<String> urlPatterns,
                                                   List<String> titlePatterns) {
        List<Scored> scored = new ArrayList<>();
        for (SearchResult r : results) {
            if (!isLowQuality(r, urlPatterns, titlePatterns)) {
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
