package io.github.paopaonb666.agentbase.toolkit.search;

import java.util.ArrayList;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;

import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.core.text.CharMetrics;
import io.github.paopaonb666.agentbase.core.tools.AgentTool;

/**
 * {@code web_search} 工具：把搜索管线包装成共享池里的工具（对应 Python
 * 侧 {@code search/tool.py}）。工具 description 沿用 chat-agent 验证过的
 * Query 生成规范——它就是提示词的一部分，直接决定搜索召回质量。错误
 * 分工与 Python 版一致：
 *
 * <ul>
 *   <li><b>预期的空结果</b> → 返回带重试指引的字符串（模型能据此换词重试）；</li>
 *   <li><b>意外异常</b> → 直接抛出，由共享池的包装层归一成模型可读文本
 *       （基座已有的机制，工具层不重复兜底）。</li>
 * </ul>
 *
 * <p>流式事件（#277）：对齐 Python M3.5 的 {@code emit_step}/
 * {@code emit_sources}——搜索开始/完成各发一条 step（detail 与 Python
 * 同文案），结果来源发一条 sources（title/url）。经 {@code ctx.emit}
 * 通道（图外调用为无操作），事件失败不影响搜索本身。
 */
public final class WebSearchTool implements AgentTool {

    public static final String NAME = "web_search";

    private static final Logger LOG = LoggerFactory.getLogger(WebSearchTool.class);

    private static final ObjectMapper MAPPER = new ObjectMapper();

    /** max_results 的上下界（对应 MAX_RESULTS_BOUNDS=(1, 20)）——模型传
     * 0 或负数应得到可读的错误。 */
    static final int MIN_RESULTS = 1;
    static final int MAX_RESULTS = 20;

    /** 预期的空结果 → 带重试指引的字符串（文案逐字对齐 Python）。 */
    static final String EMPTY_RESULT_GUIDANCE =
            "未找到相关搜索结果。请检查搜索关键词：专有名词必须保持完整"
                    + "（不要拆成单字），并用空格组合多个关键词后重试。";

    /** 至少一个配置的引擎可用时才可注册（终局：Java 侧只实现 tavily+key）。 */
    public static final String UNAVAILABLE_REASON =
            "没有任何可用的搜索引擎：Java 侧只实现 tavily 引擎，"
                    + "需要配置 TAVILY_API_KEY（duckduckgo 等 ddgs 引擎仅 Python 侧可用）";

    /**
     * 查询改写提示（A-52 多查询扩展）：LLM 把原查询改写成指定数量的检索
     * 变体（JSON 字符串数组）；变体与原查询各自检索后按 URL 去重合并。
     * 占位符 {@code {variants}} / {@code {query}} 调用时替换。
     */
    static final String QUERY_REWRITE_PROMPT = """
            你是检索查询改写器。把下面的查询改写成 {variants} 个不同的检索变体，用于提升召回覆盖面：
            - 保留专有名词完整（不拆词），换用同义词、上下位词、不同表述角度；
            - 语言与原查询保持一致；
            - 每个变体都是可直接使用的独立检索词。
            只输出 JSON 字符串数组（如 ["变体一","变体二"]），不要输出其他内容。

            原查询：{query}""";

    /** 给模型看的说明书（逐字复制 Python 版 docstring）。 */
    private static final String DESCRIPTION = """
            搜索互联网获取最新信息。当需要了解实时新闻、当前事件、
            最新数据或不确定的事实时调用此工具。

            Query 生成规范：
            1. 专有名词保护 —— 人名、地名、术语必须保持完整，禁止拆成单字。
               错误示例："拉的理论"（"拉康"被拆分）
               正确示例："拉康 精神分析 理论"
            2. 多关键词组合 —— 用空格分隔多个关键词，提高搜索精度。
               错误示例："拉康"（太宽泛）
               正确示例："拉康 精神分析 镜像阶段"
            3. 保留原文语言 —— 中文问题用中文 query，英文术语保留英文。

            Args:
                query: 搜索关键词或问题，尽量精简准确。
                max_results: 返回结果条数上限（1-20，默认 8）。
            """.stripTrailing();

    private final FallbackSearchManager manager;
    private final SearchCache cache;
    private final double timeoutSeconds;
    /** A-32：低质模式表（Settings 覆写口径；测试构造回落基座默认）。 */
    private final List<String> lowQualityUrlPatterns;
    private final List<String> lowQualityTitlePatterns;
    /** A-31：语义 rerank 开关/池/批量嵌入函数（null=不可用，恒降级原序）。 */
    private final boolean rerankEnabled;
    private final int rerankPool;
    private final java.util.function.Function<List<String>, List<List<Float>>> embedBatch;
    /** A-52：查询改写开关/变体数/改写函数（null=不可用，恒单查询）。 */
    private final boolean queryRewriteEnabled;
    private final int queryRewriteVariants;
    private final java.util.function.Function<String, List<String>> queryRewriter;

    /** 生产装配：按 settings 组装引擎与缓存（注册表工厂调用）。 */
    public static WebSearchTool build(Settings settings) {
        return build(settings, null, null);
    }

    /** 全参装配（A-31/A-52）：嵌入与改写函数由 AgentRuntime 注入（null=不可用）。 */
    public static WebSearchTool build(Settings settings,
                                      java.util.function.Function<List<String>, List<List<Float>>> embedBatch,
                                      java.util.function.Function<String, List<String>> queryRewriter) {
        return new WebSearchTool(
                new FallbackSearchManager(Engines.buildEngines(
                        Engines.parsePriority(settings.search.enginePriority),
                        settings.search.tavilyApiKey.value(),
                        settings.search.timeoutSeconds)),
                new SearchCache(settings.search.cacheTtlSeconds),
                settings.search.timeoutSeconds,
                SearchPipeline.parsePatterns(settings.search.lowQualityUrlPatterns,
                        SearchPipeline.BASE_LOW_QUALITY_URL_PATTERNS),
                SearchPipeline.parsePatterns(settings.search.lowQualityTitlePatterns,
                        SearchPipeline.BASE_LOW_QUALITY_TITLE_PATTERNS),
                settings.search.rerankEnabled,
                settings.search.rerankPool,
                embedBatch,
                settings.search.queryRewriteEnabled,
                settings.search.queryRewriteVariants,
                queryRewriter);
    }

    /**
     * 可用性谓词：注册表据此决定 web_search 能否启用（对应
     * _web_search_available）。判定走 {@link Engines#usableEngines} 单源
     * （#245）——与 build() 实际拿到的引擎集合同源，不会漂移。
     */
    public static boolean available(Settings settings) {
        return !Engines.usableEngines(Engines.parsePriority(settings.search.enginePriority),
                settings.search.tavilyApiKey.value()).isEmpty();
    }

    /** 测试路径：注入替身引擎（对应 Python 的 engine_factory 注入点）。 */
    public WebSearchTool(FallbackSearchManager manager, SearchCache cache, double timeoutSeconds) {
        this(manager, cache, timeoutSeconds,
                SearchPipeline.BASE_LOW_QUALITY_URL_PATTERNS,
                SearchPipeline.BASE_LOW_QUALITY_TITLE_PATTERNS,
                false, 20, null, false, 2, null);
    }

    /** 全参形态（A-32/A-31/A-52）：模式表与 rerank/改写配置由装配传入。 */
    public WebSearchTool(FallbackSearchManager manager, SearchCache cache, double timeoutSeconds,
                         List<String> lowQualityUrlPatterns, List<String> lowQualityTitlePatterns,
                         boolean rerankEnabled, int rerankPool,
                         java.util.function.Function<List<String>, List<List<Float>>> embedBatch,
                         boolean queryRewriteEnabled, int queryRewriteVariants,
                         java.util.function.Function<String, List<String>> queryRewriter) {
        this.manager = manager;
        this.cache = cache;
        this.timeoutSeconds = timeoutSeconds;
        this.lowQualityUrlPatterns = List.copyOf(lowQualityUrlPatterns);
        this.lowQualityTitlePatterns = List.copyOf(lowQualityTitlePatterns);
        this.rerankEnabled = rerankEnabled;
        this.rerankPool = rerankPool;
        this.embedBatch = embedBatch;
        this.queryRewriteEnabled = queryRewriteEnabled;
        this.queryRewriteVariants = queryRewriteVariants;
        this.queryRewriter = queryRewriter;
    }

    @Override
    public String name() {
        return NAME;
    }

    /** #336：对外发网络请求（查询词出网），显式声明非只读——批量执行保持串行。 */
    @Override
    public boolean readOnly() {
        return false;
    }

    @Override
    public String description() {
        return DESCRIPTION;
    }

    @Override
    public ToolSpecification specification() {
        return ToolSpecification.builder()
                .name(NAME)
                .description(DESCRIPTION)
                .parameters(JsonObjectSchema.builder()
                        .addStringProperty("query", "搜索关键词或问题，尽量精简准确。")
                        .addIntegerProperty("max_results", "返回结果条数上限（1-20，默认 8）。")
                        .required("query")
                        .build())
                .build();
    }

    @Override
    public String call(String argsJson, ToolContext ctx) {
        JsonNode args = parseArgs(argsJson);
        JsonNode queryNode = args.get("query");
        if (queryNode == null || queryNode.isNull()) {
            throw new IllegalArgumentException("缺少必需参数 query：搜索关键词或问题");
        }
        String query = queryNode.asText();
        JsonNode maxNode = args.get("max_results");
        int maxResults = maxNode == null || maxNode.isNull() || maxNode.isMissingNode()
                ? 8 : maxNode.asInt(8);

        if (!(MIN_RESULTS <= maxResults && maxResults <= MAX_RESULTS)) {
            // 文案对齐 Python：f"max_results 必须在 {MAX_RESULTS_BOUNDS} 内，
            // 收到了 {max_results}"（Python 元组的 str 形如 (1, 20)）。
            throw new IllegalArgumentException(
                    "max_results 必须在 (1, " + MAX_RESULTS + ") 内，收到了 " + maxResults);
        }

        emitStep(ctx, "running", "搜索：" + CharMetrics.head(query, 40));
        List<SearchResult> results = runSearch(query, maxResults);
        if (results.isEmpty()) {
            emitStep(ctx, "completed", "找到 0 条结果");
            return EMPTY_RESULT_GUIDANCE;
        }
        emitStep(ctx, "completed", "找到 " + results.size() + " 条结果");
        emitSources(ctx, results);
        return SearchPipeline.formatWebContext(results);
    }

    /** step 进度事件（文案对齐 Python emit_step 调用点；尽力而为）。 */
    private static void emitStep(AgentTool.ToolContext ctx, String status, String detail) {
        if (ctx != null) {
            ctx.emit(new io.github.paopaonb666.agentbase.extensions.events.Events.StepEvent(
                    NAME, status, detail));
        }
    }

    /** sources 来源事件（title/url 对齐 Python emit_sources；尽力而为）。 */
    private static void emitSources(AgentTool.ToolContext ctx, List<SearchResult> results) {
        if (ctx == null) {
            return;
        }
        try {
            ctx.emit(new io.github.paopaonb666.agentbase.extensions.events.Events.SourcesEvent(
                    results.stream()
                            .map(r -> new io.github.paopaonb666.agentbase.extensions.events.Events.Source(
                                    r.title(), r.url()))
                            .toList()));
        } catch (Exception e) {
            LOG.debug("web_search: emit sources failed", e);
        }
    }

    /**
     * LLM 查询改写函数（A-52 注入缝）：AgentRuntime 用快档（缺省主力）
     * 构建——输入原查询，输出 ≤variants 个改写变体（JSON 数组解析，条目
     * trim/去空/去重/剔除与原查询相同的条目）。任何失败（LLM 异常/输出
     * 不可解析）返回空表——调用方回退仅原查询（与关闭态同结果，绝不阻断
     * 搜索）。
     */
    public static java.util.function.Function<String, List<String>> llmRewriter(
            io.github.paopaonb666.agentbase.core.llm.ModelRuntime llm,
            io.github.paopaonb666.agentbase.core.AgentModule.ChatInvokerPort fast,
            int variants) {
        return query -> {
            try {
                String prompt = QUERY_REWRITE_PROMPT
                        .replace("{variants}", String.valueOf(variants))
                        .replace("{query}", query);
                String raw = fast != null
                        ? fast.invoke(List.of(dev.langchain4j.data.message.UserMessage.from(prompt)))
                        : llm.invoke(List.of(dev.langchain4j.data.message.UserMessage.from(prompt)))
                                .text();
                String arrayJson = io.github.paopaonb666.agentbase.core.text.JsonBlocks
                        .firstBalancedArray(raw);
                com.fasterxml.jackson.databind.JsonNode node = MAPPER.readTree(arrayJson);
                List<String> out = new ArrayList<>();
                if (node.isArray()) {
                    for (com.fasterxml.jackson.databind.JsonNode item : node) {
                        if (!item.isTextual()) {
                            continue;
                        }
                        String variant = item.asText().strip();
                        if (!variant.isEmpty() && !variant.equalsIgnoreCase(query.strip())
                                && !out.contains(variant)) {
                            out.add(variant);
                        }
                        if (out.size() >= variants) {
                            break;
                        }
                    }
                }
                return out;
            } catch (Exception e) {
                LOG.warn("web_search: 查询改写 LLM 失败，回退仅原查询", e);
                return List.of();
            }
        };
    }

    /** 缓存优先的搜索执行；引擎全失败时返回空列表（管理器已记日志）。 */
    private List<SearchResult> runSearch(String query, int maxResults) {
        List<SearchResult> cached = cache.get(query);
        if (cached != null) {
            return cached;
        }
        // A-52 多查询扩展（默认关）：改写 LLM 产出变体 → 原查询 + 变体各自
        // 检索，按 URL 去重合并（先见优先——原查询结果排前）后统一打分/
        // rerank；改写失败/空表回退仅原查询（queries 单元素 = 与关闭态同
        // 路径）。预算口径：#242 整条管线 deadline 不因变体数放大——各查询
        // 均摊（下限 2s）。
        List<String> queries = List.of(query);
        if (queryRewriteEnabled && queryRewriter != null) {
            List<String> variants = queryRewriter.apply(query);
            if (!variants.isEmpty()) {
                java.util.List<String> expanded = new ArrayList<>(variants.size() + 1);
                expanded.add(query);
                expanded.addAll(variants);
                queries = List.copyOf(expanded);
            }
        }
        double share = queries.size() == 1
                ? timeoutSeconds
                : Math.max(2.0, timeoutSeconds / queries.size());
        boolean multiQuery = queries.size() > 1;
        List<SearchResult> merged = new ArrayList<>();
        java.util.Set<String> seenUrls = new java.util.LinkedHashSet<>();
        for (String q : queries) {
            for (SearchResult result : manager.search(q, maxResults * 2, share)) {
                String url = result.url();
                // 去重只发生在多查询合并（URL 先见优先）；单查询路径逐条
                // 原样进打分——关闭态行为逐字节不变。
                if (multiQuery && url != null && !url.isEmpty() && !seenUrls.add(url)) {
                    continue;
                }
                merged.add(result);
            }
        }
        List<SearchResult> ranked = SearchPipeline.filterAndRank(merged, query, maxResults,
                lowQualityUrlPatterns, lowQualityTitlePatterns);
        // A-31 语义 rerank（默认关）：打分后按 embedding 余弦重排短名单；
        // 不可用（embedder 缺席/嵌入失败）整体降级原序。
        if (rerankEnabled) {
            ranked = SearchPipeline.rerankByEmbedding(ranked, query, embedBatch, rerankPool);
        }
        if (!ranked.isEmpty()) {
            cache.put(query, ranked);
        }
        return ranked;
    }

    private static JsonNode parseArgs(String argsJson) {
        try {
            return MAPPER.readTree(argsJson);
        } catch (Exception e) {
            throw new IllegalArgumentException("工具参数不是合法 JSON：" + e.getMessage());
        }
    }
}
