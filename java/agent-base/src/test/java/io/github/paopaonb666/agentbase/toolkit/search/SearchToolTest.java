package io.github.paopaonb666.agentbase.toolkit.search;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.URI;
import java.net.http.HttpClient;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.HttpServer;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import io.github.paopaonb666.agentbase.core.config.Secret;
import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.core.tools.AgentTool;

/**
 * web_search 与搜索管线的测试（对应 Python 侧 tests/test_search.py）：
 * Tavily 响应映射、引擎降级、缓存命中、评分与渲染、工具行为。全部走
 * 进程内 HttpServer 替身，不出真实网络。
 */
class SearchToolTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private HttpServer server;
    private URI endpoint;
    private final AtomicInteger requestCount = new AtomicInteger();
    private final AtomicReference<String> lastAuthHeader = new AtomicReference<>();
    private final AtomicReference<String> lastBody = new AtomicReference<>();

    @BeforeEach
    void startMockServer() throws IOException {
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/", exchange -> {
            requestCount.incrementAndGet();
            lastAuthHeader.set(exchange.getRequestHeaders().getFirst("Authorization"));
            lastBody.set(new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8));
            byte[] body = SEARCH_RESPONSE_JSON.getBytes(StandardCharsets.UTF_8);
            exchange.getResponseHeaders().set("Content-Type", "application/json");
            exchange.sendResponseHeaders(200, body.length);
            exchange.getResponseBody().write(body);
            exchange.close();
        });
        server.start();
        endpoint = URI.create("http://127.0.0.1:" + server.getAddress().getPort() + "/search");
    }

    @AfterEach
    void stopMockServer() {
        if (server != null) {
            server.stop(0);
        }
    }

    /** 与 Python 侧同构的 Tavily 响应体。 */
    private static final String SEARCH_RESPONSE_JSON = """
            {"results": [
                {"title": "Alpha page", "url": "https://example.com/alpha", "content": "alpha content"},
                {"title": "Beta page", "url": "https://example.com/beta", "content": "beta content"},
                {"title": "No url", "content": "missing url field"}
            ]}
            """;

    private TavilyHttpClient engine() {
        return new TavilyHttpClient("test-key", 15.0, HttpClient.newHttpClient(), endpoint);
    }

    private String callTool(AgentTool tool, String argsJson) {
        try {
            return tool.call(argsJson, new AgentTool.ToolContext(null, null));
        } catch (RuntimeException e) {
            throw e;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    // ── Tavily 响应映射 ──────────────────────────────────────────────

    @Test
    void tavilyMapsResponseFields() throws Exception {
        List<SearchResult> results = engine().search("拉康 精神分析", 5);
        assertEquals(3, results.size());
        assertEquals(new SearchResult("Alpha page", "https://example.com/alpha", "alpha content", 1, "tavily"),
                results.get(0));
        assertEquals(2, results.get(1).position());
        assertEquals("tavily", results.get(2).engine());
        // 缺失字段回空串（对应 str(r.get("title", ""))）。
        assertEquals("", results.get(2).url());
    }

    @Test
    void tavilySendsAlignedRequestContract() throws Exception {
        engine().search("拉康 精神分析", 5);
        assertEquals("Bearer test-key", lastAuthHeader.get());
        JsonNode body = MAPPER.readTree(lastBody.get());
        assertEquals("拉康 精神分析", body.get("query").asText());
        assertEquals("basic", body.get("search_depth").asText());
        assertEquals(5, body.get("max_results").asInt());
        assertEquals(false, body.get("include_answer").asBoolean());
    }

    @Test
    void tavilyHttpErrorPropagatesToManager() throws IOException {
        server.createContext("/fail", exchange -> exchange.sendResponseHeaders(500, -1));
        TavilyHttpClient failing = new TavilyHttpClient("test-key", 15.0, HttpClient.newHttpClient(),
                URI.create("http://127.0.0.1:" + server.getAddress().getPort() + "/fail"));
        // 非 2xx → IOException（对应 raise_for_status），由管理器降级。
        assertThrows(IOException.class, () -> failing.search("q", 3));
        FallbackSearchManager manager = new FallbackSearchManager(List.of(failing));
        assertTrue(manager.search("q", 3).isEmpty());
    }

    // ── 瞬时失败重试与代理解析（#243） ──────────────────────────────

    @Test
    void tavilyRetriesOnceOnTransient5xxThenSucceeds() throws Exception {
        java.util.concurrent.atomic.AtomicInteger calls = new java.util.concurrent.atomic.AtomicInteger();
        server.createContext("/flaky", exchange -> {
            int attempt = calls.incrementAndGet();
            if (attempt == 1) {
                exchange.sendResponseHeaders(503, -1);
                exchange.close();
                return;
            }
            byte[] body = SEARCH_RESPONSE_JSON.getBytes(StandardCharsets.UTF_8);
            exchange.getResponseHeaders().set("Content-Type", "application/json");
            exchange.sendResponseHeaders(200, body.length);
            exchange.getResponseBody().write(body);
            exchange.close();
        });
        TavilyHttpClient flaky = new TavilyHttpClient("test-key", 15.0, HttpClient.newHttpClient(),
                URI.create("http://127.0.0.1:" + server.getAddress().getPort() + "/flaky"));
        assertEquals(3, flaky.search("q", 5).size());
        assertEquals(2, calls.get(), "5xx 后应恰好重试一次");
    }

    @Test
    void tavilyDoesNotRetryOnAuthError() throws Exception {
        java.util.concurrent.atomic.AtomicInteger calls = new java.util.concurrent.atomic.AtomicInteger();
        server.createContext("/unauthorized", exchange -> {
            calls.incrementAndGet();
            exchange.sendResponseHeaders(401, -1);
            exchange.close();
        });
        TavilyHttpClient unauthorized = new TavilyHttpClient("bad", 15.0,
                HttpClient.newHttpClient(),
                URI.create("http://127.0.0.1:" + server.getAddress().getPort() + "/unauthorized"));
        assertThrows(IOException.class, () -> unauthorized.search("q", 3));
        assertEquals(1, calls.get(), "4xx 语义错误不重试");
    }

    @Test
    void proxyParserAcceptsUrlAndHostPortForms() {
        var withScheme = TavilyHttpClient.parseProxy("http://127.0.0.1:8080");
        assertEquals("127.0.0.1", withScheme.getAddress().getHostAddress());
        assertEquals(8080, withScheme.getPort());
        var bare = TavilyHttpClient.parseProxy("proxy.internal:9090");
        assertEquals(9090, bare.getPort());
        // 非法形态 → null（直连），绝不让代理配置炸掉引擎构造
        assertEquals(null, TavilyHttpClient.parseProxy(null));
        assertEquals(null, TavilyHttpClient.parseProxy("  "));
        assertEquals(null, TavilyHttpClient.parseProxy("no-port"));
        assertEquals(null, TavilyHttpClient.parseProxy("http://host:notaport"));
    }

    // ── 降级管理器 ───────────────────────────────────────────────────

    @Test
    void managerTriesEnginesInPriorityOrder() {
        SearchEngine failing = new SearchEngine() {
            @Override public String name() { return "failing"; }
            @Override public List<SearchResult> search(String query, int maxResults) {
                throw new RuntimeException("boom");
            }        };
        FallbackSearchManager manager = new FallbackSearchManager(List.of(failing, engine()));
        List<SearchResult> results = manager.search("拉康", 5);
        assertEquals(3, results.size());
        // 第一个引擎在 HTTP 层之前就抛错（零请求）；只有第二个引擎打了一次 mock。
        assertEquals(1, requestCount.get());
    }

    // ── 评分与渲染 ───────────────────────────────────────────────────

    @Test
    void filterAndRankScoresAndRenumbers() {
        SearchResult weak = new SearchResult("无关标题", "https://a.example.com/x", "没有任何关键词", 1, "fake");
        SearchResult strong = new SearchResult("拉康 镜像阶段", "https://b.example.com/y", "关于拉康的介绍", 2, "fake");
        List<SearchResult> ranked = SearchPipeline.filterAndRank(List.of(weak, strong), "拉康 镜像", 5);
        assertEquals("拉康 镜像阶段", ranked.get(0).title());
        assertEquals(1, ranked.get(0).position()); // 重新编号
        assertEquals(2, ranked.get(1).position());
    }

    @Test
    void filterAndRankCapsToMaxResults() {
        List<SearchResult> raw = new java.util.ArrayList<>();
        for (int i = 0; i < 6; i++) {
            raw.add(new SearchResult("拉康 " + i, "https://e/" + i, "拉康", i + 1, "fake"));
        }
        assertEquals(2, SearchPipeline.filterAndRank(raw, "拉康", 2).size());
    }

    @Test
    void chineseQueryRanksByTermOverlapInsteadOfEngineOrder() {
        // #244：中文查询切 CJK 二元组——标题含词项的结果胜出，不再退化为
        // 引擎原始序（此前 matches 恒 0）。
        SearchResult offTopic = new SearchResult("天气预报", "https://w/1", "明天有雨", 1, "fake");
        SearchResult onTopic = new SearchResult("拉康的镜像阶段", "https://w/2", "精神分析理论介绍", 2, "fake");
        List<SearchResult> ranked = SearchPipeline.filterAndRank(
                List.of(offTopic, onTopic), "镜像阶段 精神分析", 5);
        assertEquals("拉康的镜像阶段", ranked.get(0).title());
        assertEquals(1, ranked.get(0).position());
    }

    @Test
    void asciiQueryScoringUnchanged() {
        // 英文行为回归：与旧空白切分口径一致
        SearchResult a = new SearchResult("alpha prototype", "https://a/1", "nothing", 1, "fake");
        SearchResult b = new SearchResult("unrelated", "https://a/2", "alpha prototype here", 2, "fake");
        // 标题命中权重 2x：a 应排前
        List<SearchResult> ranked = SearchPipeline.filterAndRank(List.of(b, a), "alpha prototype", 5);
        assertEquals("alpha prototype", ranked.get(0).title());
    }

    @Test
    void formatWebContextRendersExactly() {
        List<SearchResult> results = List.of(
                new SearchResult("Alpha", "https://example.com/a", "alpha content", 1, "tavily"),
                new SearchResult("Beta", "https://example.com/b", "beta content", 2, "tavily"));
        // 逐字对齐 Python format_web_context 的拼接结果；尾行引用约定为
        // Java 侧扩展（#254/#338，Python 差异登记在案）。
        String expected = "以下是从互联网搜索到的相关信息：\n"
                + "\n[1] Alpha\n    URL: https://example.com/a\n    摘要: alpha content (来源: tavily)"
                + "\n\n[2] Beta\n    URL: https://example.com/b\n    摘要: beta content (来源: tavily)\n"
                + "\n（回答引用以上来源时，请用 [n] 标注对应序号）";
        assertEquals(expected, SearchPipeline.formatWebContext(results));
        assertEquals("", SearchPipeline.formatWebContext(List.of()));
    }

    // ── 管线 deadline（#242） ────────────────────────────────────────

    @Test
    void pipelineBudgetPropagatesRemainingToEachEngine() throws Exception {
        List<Double> budgetsSeen = new java.util.concurrent.CopyOnWriteArrayList<>();
        SearchEngine recorder = new SearchEngine() {
            @Override public String name() { return "recorder"; }
            @Override public List<SearchResult> search(String query, int maxResults) {
                return List.of();
            }
            @Override public List<SearchResult> search(String query, int maxResults,
                                                       double timeoutSeconds) {
                budgetsSeen.add(timeoutSeconds);
                try {
                    Thread.sleep(120); // 消耗一部分预算
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
                return List.of(); // 空结果 → 管理器继续下一个引擎
            }
        };
        new FallbackSearchManager(List.of(recorder)).search("q", 3, 0.5);
        assertEquals(1, budgetsSeen.size());
        assertTrue(budgetsSeen.get(0) <= 0.5 && budgetsSeen.get(0) > 0.3,
                "首引擎应拿到完整预算 0.5s，实得 " + budgetsSeen.get(0));
    }

    @Test
    void pipelineBudgetStopsBeforeLaterEnginesWhenExhausted() {
        List<Double> budgetsSeen = new java.util.concurrent.CopyOnWriteArrayList<>();
        SearchEngine slow = new SearchEngine() {
            @Override public String name() { return "slow"; }
            @Override public List<SearchResult> search(String query, int maxResults) {
                return List.of();
            }
            @Override public List<SearchResult> search(String query, int maxResults,
                                                       double timeoutSeconds) {
                try {
                    Thread.sleep(150); // 超过整个预算
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
                return List.of();
            }
        };
        SearchEngine next = new SearchEngine() {
            @Override public String name() { return "next"; }
            @Override public List<SearchResult> search(String query, int maxResults) {
                budgetsSeen.add(0.0);
                return List.of();
            }
            @Override public List<SearchResult> search(String query, int maxResults,
                                                       double timeoutSeconds) {
                budgetsSeen.add(timeoutSeconds);
                return List.of();
            }
        };
        new FallbackSearchManager(List.of(slow, next)).search("q", 3, 0.1);
        assertTrue(budgetsSeen.isEmpty(), "预算耗尽后不得点燃后续引擎");
    }

    @Test
    void tavilyHonorsPipelineBudgetBelowConfiguredTimeout() throws Exception {
        // 引擎请求超时压到预算（0.2s < 配置 15s）：慢端点下请求应在预算级
        // 时间尺度失败，而不是等满配置超时
        server.createContext("/slow", exchange -> {
            try {
                Thread.sleep(2000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            exchange.sendResponseHeaders(200, 2);
            exchange.getResponseBody().write("{}".getBytes(StandardCharsets.UTF_8));
            exchange.close();
        });
        TavilyHttpClient slowEngine = new TavilyHttpClient("k", 15.0, HttpClient.newHttpClient(),
                URI.create("http://127.0.0.1:" + server.getAddress().getPort() + "/slow"));
        long started = System.nanoTime();
        assertThrows(Exception.class, () -> slowEngine.search("q", 3, 0.2));
        long elapsedMs = (System.nanoTime() - started) / 1_000_000;
        assertTrue(elapsedMs < 1500, "请求应在预算级时间失败，实耗 " + elapsedMs + "ms");
    }

    // ── web_search 工具 ──────────────────────────────────────────────

    private WebSearchTool tool() {
        return new WebSearchTool(new FallbackSearchManager(List.of(engine())),
                new SearchCache(300.0), 15.0);
    }

    @Test
    void webSearchReturnsFormattedContext() {
        String out = callTool(tool(), "{\"query\": \"拉康 精神分析\"}");
        assertTrue(out.startsWith("以下是从互联网搜索到的相关信息："));
        assertTrue(out.contains("[1] "));
        assertTrue(out.contains("(来源: tavily)"));
    }

    /** 录制事件的事件通道替身（#277：step/sources 发射断言用）。 */
    private static final class RecordingNodeContext
            implements io.github.paopaonb666.agentbase.core.graph.NodeContext {
        final List<Object> events = new java.util.ArrayList<>();
        private final io.github.paopaonb666.agentbase.core.graph.RunConfig config =
                io.github.paopaonb666.agentbase.core.graph.RunConfig.of("chat:search", "u");

        @Override
        public io.github.paopaonb666.agentbase.core.graph.RunConfig config() {
            return config;
        }

        @Override
        public void emit(Object event) {
            events.add(event);
        }

        @Override
        public String callTool(String name, String argsJson) {
            return "";
        }

        @Override
        public io.github.paopaonb666.agentbase.core.tools.ToolPool tools() {
            return null;
        }

        @Override
        public boolean cancelled() {
            return false;
        }
    }

    @Test
    void webSearchEmitsStepAndSourcesEvents() {
        RecordingNodeContext nodeCtx = new RecordingNodeContext();
        String out = tool().call("{\"query\": \"拉康 精神分析\"}",
                new AgentTool.ToolContext(nodeCtx.config(), nodeCtx));
        assertTrue(out.contains("[1] "));

        // 事件序：running → completed → sources（文案对齐 Python emit_*）。
        assertEquals(3, nodeCtx.events.size());
        var stepRunning = (io.github.paopaonb666.agentbase.extensions.events.Events.StepEvent)
                nodeCtx.events.get(0);
        assertEquals("web_search", stepRunning.name());
        assertEquals("running", stepRunning.status());
        assertEquals("搜索：拉康 精神分析", stepRunning.detail());

        var stepCompleted = (io.github.paopaonb666.agentbase.extensions.events.Events.StepEvent)
                nodeCtx.events.get(1);
        assertEquals("completed", stepCompleted.status());
        assertEquals("找到 3 条结果", stepCompleted.detail());

        var sources = (io.github.paopaonb666.agentbase.extensions.events.Events.SourcesEvent)
                nodeCtx.events.get(2);
        // Python 语义：全量结果都进 sources（无 url 条目回空串），过滤交给前端
        assertEquals(3, sources.sources().size());
        assertEquals(new io.github.paopaonb666.agentbase.extensions.events.Events.Source(
                "Alpha page", "https://example.com/alpha"), sources.sources().get(0));
        assertEquals("", sources.sources().get(2).url());
    }

    @Test
    void webSearchEmitsNothingWhenOutsideGraph() {
        // ctx 的 nodeContext 为 null（图外/池内直调）：emit 无操作不抛
        String out = callTool(tool(), "{\"query\": \"拉康 精神分析\"}");
        assertTrue(out.contains("[1] "));
    }

    @Test
    void webSearchCacheHitsDoNotHitServerTwice() {
        WebSearchTool t = tool();
        callTool(t, "{\"query\": \"拉康 精神分析\"}");
        callTool(t, "{\"query\": \"拉康 精神分析\"}");
        assertEquals(1, requestCount.get(), "相同 query 第二次应命中缓存");
        // 不同 query 不命中缓存（键规范化后仍不同，#241）。
        callTool(t, "{\"query\": \"镜像阶段\"}");
        assertEquals(2, requestCount.get());
    }

    @Test
    void webSearchCacheKeyNormalizesWhitespaceAndCase() {
        WebSearchTool t = tool();
        callTool(t, "{\"query\": \"  Lacan   Mirror  \"}");
        // trim + 空白折叠 + 小写折叠后与原查询同键（#241：省真实搜索）
        callTool(t, "{\"query\": \"lacan mirror\"}");
        assertEquals(1, requestCount.get(), "规范化变体应命中同一缓存条目");
    }

    @Test
    void webSearchEmptyResultsReturnGuidance() {
        server.createContext("/empty", exchange -> {
            byte[] body = "{\"results\": []}".getBytes(StandardCharsets.UTF_8);
            exchange.sendResponseHeaders(200, body.length);
            exchange.getResponseBody().write(body);
            exchange.close();
        });
        TavilyHttpClient empty = new TavilyHttpClient("test-key", 15.0, HttpClient.newHttpClient(),
                URI.create("http://127.0.0.1:" + server.getAddress().getPort() + "/empty"));
        WebSearchTool t = new WebSearchTool(new FallbackSearchManager(List.of(empty)),
                new SearchCache(300.0), 15.0);
        String out = callTool(t, "{\"query\": \"稀有的查询\"}");
        // 预期的空结果 → 带重试指引的字符串（文案逐字对齐 Python）。
        assertEquals(WebSearchTool.EMPTY_RESULT_GUIDANCE, out);
    }

    @Test
    void webSearchRejectsOutOfBoundsMaxResults() {
        IllegalArgumentException e1 = assertThrows(IllegalArgumentException.class,
                () -> callTool(tool(), "{\"query\": \"q\", \"max_results\": 0}"));
        assertTrue(e1.getMessage().contains("max_results 必须在 (1, 20) 内，收到了 0"));
        IllegalArgumentException e2 = assertThrows(IllegalArgumentException.class,
                () -> callTool(tool(), "{\"query\": \"q\", \"max_results\": 25}"));
        assertTrue(e2.getMessage().contains("max_results 必须在 (1, 20) 内，收到了 25"));
        assertEquals(0, requestCount.get(), "参数错误不应发起任何请求");
    }

    @Test
    void webSearchDefaultsMaxResultsToEightAndDoublesForEngine() {
        callTool(tool(), "{\"query\": \"拉康\"}");
        // 工具层 max_results 默认 8；传给引擎的是两倍预算（16），
        // 过滤排序后再截回 8（对应 Python 的 manager.search(query, max_results * 2)）。
        assertEquals(16, parseJson(lastBody.get()).path("max_results").asInt());
    }

    private static JsonNode parseJson(String json) {
        try {
            return MAPPER.readTree(json);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Test
    void unknownEnginesAreSkippedNotContractErrors() {
        // #276 终局：Java 侧搜索引擎清单收窄为 tavily；duckduckgo 等由
        // Settings 启动校验拒绝，buildEngines 层对未知名字防御性跳过。
        assertTrue(Engines.buildEngines(List.of("duckduckgo"), "", 15.0).isEmpty());
        assertTrue(Engines.usableEngines(List.of("duckduckgo", "tavily"), "key")
                .equals(List.of("tavily")));
    }

    @Test
    void availabilityPredicateSharesTheEngineUsabilitySource() {
        // #245：available() 与 buildEngines 同源（usableEngines）——
        // key 有无不漂移；默认 priority（tavily）+ 无 key = 不可用。
        Settings settings = Settings.load(Map.of("ENV", "development"), null);
        assertEquals("tavily", settings.search.enginePriority);
        assertFalse(WebSearchTool.available(settings));
        assertTrue(Engines.buildEngines(
                Engines.parsePriority(settings.search.enginePriority),
                settings.search.tavilyApiKey.value(), settings.search.timeoutSeconds).isEmpty());
        settings.search.tavilyApiKey = Secret.of("tvly-test");
        assertTrue(WebSearchTool.available(settings));
        assertEquals(1, Engines.buildEngines(
                Engines.parsePriority(settings.search.enginePriority),
                settings.search.tavilyApiKey.value(), settings.search.timeoutSeconds).size());
    }

    @Test
    void enginesWithoutKeyAreSkippedWithLog() {
        // tavily 在 key 缺失时跳过（与 Python 的 build_engines 行为一致），
        // 管理器拿到空引擎列表 → 搜索返回空。
        List<SearchEngine> engines = Engines.buildEngines(List.of("tavily"), "", 15.0);
        assertTrue(engines.isEmpty());
        assertTrue(new FallbackSearchManager(engines).search("q", 3).isEmpty());
    }

    @Test
    void searchCacheExpiresByTtl() throws InterruptedException {
        SearchCache cache = new SearchCache(0.05); // 50ms
        cache.put("q", List.of(new SearchResult("t", "u", "s", 1, "fake")));
        assertEquals(1, cache.get("q").size());
        Thread.sleep(80);
        assertNull(cache.get("q"), "过期后应返回 null");
        // LRU 容量封顶。
        for (int i = 0; i < SearchCache.CACHE_MAX_ENTRIES + 10; i++) {
            cache.put("q" + i, List.of());
        }
        assertEquals(SearchCache.CACHE_MAX_ENTRIES, cacheSize(cache));
    }

    @Test
    void baseLowQualityPatternsFilterPipelineArtifacts() {
        // A-32：基座默认保守集——Google 缓存页/翻译代理页是流水线产物垃圾源
        // （前者服务已停恒死链、后者机器翻译镜像非原始来源），宁缺毋滥；
        // 业务名单（字典站/拼音站类）不入基座，属部署方覆写。
        SearchResult cache = new SearchResult("Cached page",
                "https://webcache.googleusercontent.com/search?q=cache:x", "s", 1, "fake");
        SearchResult translated = new SearchResult("Translated",
                "https://www-example-com.translate.goog/docs", "s", 2, "fake");
        SearchResult normal = new SearchResult("Normal Result", "https://example.com/docs", "s", 3, "fake");
        List<SearchResult> ranked = SearchPipeline.filterAndRank(
                List.of(cache, translated, normal), "docs", 10);
        assertEquals(List.of("Normal Result"), ranked.stream().map(SearchResult::title).toList(),
                "基座默认模式滤掉缓存页与翻译代理页");
        assertTrue(SearchPipeline.isLowQuality(cache));
        assertTrue(SearchPipeline.isLowQuality(translated));
        assertFalse(SearchPipeline.isLowQuality(normal));
    }

    @Test
    void lowQualityPatternsParseNullFallsBackAndExplicitConfigReplaces() {
        // parsePatterns：null=未配置回落基座默认；空串=显式清空；条目 trim+小写。
        assertEquals(SearchPipeline.BASE_LOW_QUALITY_URL_PATTERNS,
                SearchPipeline.parsePatterns(null, SearchPipeline.BASE_LOW_QUALITY_URL_PATTERNS));
        assertTrue(SearchPipeline.parsePatterns("", List.of("x")).isEmpty(), "空串=显式清空");
        assertEquals(List.of("pinyin.example", "dict.example"),
                SearchPipeline.parsePatterns(" Pinyin.Example , dict.example ", List.of()),
                "条目 trim+小写");
        // 覆写整体替换（不是叠加）：业务名单接管后基座默认模式不再过滤。
        List<String> business = SearchPipeline.parsePatterns("dict.example", List.of());
        SearchResult cache = new SearchResult("t",
                "https://webcache.googleusercontent.com/x", "s", 1, "fake");
        assertFalse(SearchPipeline.isLowQuality(cache, business, List.of()),
                "覆写替换默认——基座默认模式随覆写退场");
    }

    @Test
    void settingsCarriesLowQualityPatternKeys() {
        // Settings 装配面：两键装载 + 未配置 null 语义（回落基座默认）。
        var configured = Settings.load(Map.of("ENV", "development",
                "SEARCH_LOW_QUALITY_URL_PATTERNS", "dict.example, pinyin.example",
                "SEARCH_LOW_QUALITY_TITLE_PATTERNS", ""), null);
        assertEquals("dict.example, pinyin.example", configured.search.lowQualityUrlPatterns);
        assertEquals("", configured.search.lowQualityTitlePatterns);
        var defaults = Settings.load(Map.of("ENV", "development"), null);
        assertNull(defaults.search.lowQualityUrlPatterns, "未配置=null→回落基座默认");
        assertNull(defaults.search.lowQualityTitlePatterns);
    }

    // ── A-31 语义 rerank ─────────────────────────────────────────────────

    /** 桩嵌入：按文本前缀给固定向量（"q:"=查询轴，"a:"=A 轴，"b:"=B 轴）。 */
    private static final java.util.function.Function<List<String>, List<List<Float>>>
            PREFIX_EMBEDDER = texts -> {
        List<List<Float>> out = new java.util.ArrayList<>();
        for (String text : texts) {
            if (text.startsWith("q:")) {
                out.add(List.of(1.0f, 0.0f));
            } else if (text.startsWith("a:")) {
                out.add(List.of(0.9f, 0.1f)); // 与查询余弦≈0.994
            } else {
                out.add(List.of(0.0f, 1.0f)); // 与查询余弦=0
            }
        }
        return out;
    };

    @Test
    void rerankByEmbeddingReordersShortlistAndRenumbers() {
        // 打分后序 = A,B,C；查询与 A 语义近 → rerank 后 A 第一（位置重编），
        // C 池外原序接尾并顺延编号。
        SearchResult a = new SearchResult("a:标题A", "u1", "a:摘要", 1, "fake");
        SearchResult b = new SearchResult("b:标题B", "u2", "b:摘要", 2, "fake");
        SearchResult c = new SearchResult("c:标题C", "u3", "c:摘要", 3, "fake");
        List<SearchResult> out = SearchPipeline.rerankByEmbedding(
                List.of(b, a, c), "q:查询", PREFIX_EMBEDDER, 2);
        assertEquals(List.of("a:标题A", "b:标题B", "c:标题C"),
                out.stream().map(SearchResult::title).toList());
        assertEquals(List.of(1, 2, 3), out.stream().map(SearchResult::position).toList(),
                "重排后重新编号；池外接尾顺延");
    }

    @Test
    void rerankByEmbeddingDegradesToOriginalOnAnyEmbeddingFailure() {
        SearchResult a = new SearchResult("a:标题A", "u1", "a:摘要", 1, "fake");
        SearchResult b = new SearchResult("b:标题B", "u2", "b:摘要", 2, "fake");
        // embedBatch 返回 null（端点失败/未配置的 MemoryService 口径）→ 原序。
        assertEquals(List.of(a, b), SearchPipeline.rerankByEmbedding(
                List.of(a, b), "q:查询", texts -> null, 20));
        // 数量不齐 → 原序。
        assertEquals(List.of(a, b), SearchPipeline.rerankByEmbedding(
                List.of(a, b), "q:查询", texts -> List.of(List.of(1.0f, 0.0f)), 20));
        // 零向量 → 原序。
        assertEquals(List.of(a, b), SearchPipeline.rerankByEmbedding(
                List.of(a, b), "q:查询", texts -> List.of(
                        List.of(1.0f, 0.0f), List.of(0.0f, 0.0f), List.of(1.0f, 0.0f)), 20));
        // embedder 本身缺席 → 原序。
        assertEquals(List.of(a, b), SearchPipeline.rerankByEmbedding(
                List.of(a, b), "q:查询", null, 20));
        // 嵌入调用抛异常 → 原序（整体降级口径对异常同样成立）。
        assertEquals(List.of(a, b), SearchPipeline.rerankByEmbedding(
                List.of(a, b), "q:查询", texts -> {
                    throw new IllegalStateException("endpoint down");
                }, 20));
    }

    @Test
    void rerankByEmbeddingKeepsOriginalOrderForTiesAndSmallInputs() {
        SearchResult a = new SearchResult("a:标题A", "u1", "a:摘要", 1, "fake");
        SearchResult b = new SearchResult("a:标题A2", "u2", "a:摘要2", 2, "fake");
        // 同前缀=同向量=平局 → 保持原序（稳定排序）。
        List<SearchResult> out = SearchPipeline.rerankByEmbedding(
                List.of(a, b), "q:查询", PREFIX_EMBEDDER, 20);
        assertEquals(List.of("a:标题A", "a:标题A2"), out.stream().map(SearchResult::title).toList());
        // 单候选无可重排。
        assertEquals(1, SearchPipeline.rerankByEmbedding(
                List.of(a), "q:查询", PREFIX_EMBEDDER, 20).size());
    }

    @Test
    void settingsCarriesSearchRerankKeys() {
        var configured = Settings.load(Map.of("ENV", "development",
                "SEARCH_RERANK_ENABLED", "true",
                "SEARCH_RERANK_POOL", "8"), null);
        assertTrue(configured.search.rerankEnabled);
        assertEquals(8, configured.search.rerankPool);
        var defaults = Settings.load(Map.of("ENV", "development"), null);
        assertFalse(defaults.search.rerankEnabled, "默认关（关闭态零行为变化）");
        assertEquals(20, defaults.search.rerankPool);
        assertThrows(Settings.SettingsError.class,
                () -> Settings.load(Map.of("ENV", "development", "SEARCH_RERANK_POOL", "1"), null),
                "池下限 2（单候选无可重排）");
    }

    private static int cacheSize(SearchCache cache) {
        try {
            var field = SearchCache.class.getDeclaredField("entries");
            field.setAccessible(true);
            return ((java.util.Map<?, ?>) field.get(cache)).size();
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }
    }
}
