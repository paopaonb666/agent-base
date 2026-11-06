package io.github.paopaonb666.agentbase.toolkit.search;

import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;

import io.github.paopaonb666.agentbase.core.config.Settings;
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
 * <p>TODO(流式事件)：Python 侧 M3.5 还经 {@code emit_step}/
 * {@code emit_sources} 发 UI 进度事件；本模块 {@code ctx.emit} 通道已具备，
 * 事件契约（step/sources）随服务侧事件批次迁移，这里先不发。
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

    /** 生产装配：按 settings 组装引擎与缓存（注册表工厂调用）。 */
    public static WebSearchTool build(Settings settings) {
        return new WebSearchTool(
                new FallbackSearchManager(Engines.buildEngines(
                        Engines.parsePriority(settings.search.enginePriority),
                        settings.search.tavilyApiKey,
                        settings.search.timeoutSeconds)),
                new SearchCache(settings.search.cacheTtlSeconds),
                settings.search.timeoutSeconds);
    }

    /**
     * 可用性谓词：注册表据此决定 web_search 能否启用（对应
     * _web_search_available）。判定走 {@link Engines#usableEngines} 单源
     * （#245）——与 build() 实际拿到的引擎集合同源，不会漂移。
     */
    public static boolean available(Settings settings) {
        return !Engines.usableEngines(Engines.parsePriority(settings.search.enginePriority),
                settings.search.tavilyApiKey).isEmpty();
    }

    /** 测试路径：注入替身引擎（对应 Python 的 engine_factory 注入点）。 */
    public WebSearchTool(FallbackSearchManager manager, SearchCache cache, double timeoutSeconds) {
        this.manager = manager;
        this.cache = cache;
        this.timeoutSeconds = timeoutSeconds;
    }

    @Override
    public String name() {
        return NAME;
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

        List<SearchResult> results = runSearch(query, maxResults);
        if (results.isEmpty()) {
            return EMPTY_RESULT_GUIDANCE;
        }
        return SearchPipeline.formatWebContext(results);
    }

    /** 缓存优先的搜索执行；引擎全失败时返回空列表（管理器已记日志）。 */
    private List<SearchResult> runSearch(String query, int maxResults) {
        List<SearchResult> cached = cache.get(query);
        if (cached != null) {
            return cached;
        }
        // Python 用 asyncio.wait_for 给整条管线设总预算；第一期只有 HTTP
        // 引擎，请求超时（timeoutSeconds）等价覆盖该预算。
        List<SearchResult> raw = manager.search(query, maxResults * 2);
        List<SearchResult> ranked = SearchPipeline.filterAndRank(raw, query, maxResults);
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
