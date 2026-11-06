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
        // 非 2xx → IOException（对应 raise_for_status），由管理器降级吞掉。
        assertThrows(IOException.class, () -> failing.search("q", 3));
        FallbackSearchManager manager = new FallbackSearchManager(List.of(failing));
        assertTrue(manager.search("q", 3).isEmpty());
    }

    // ── 降级管理器 ───────────────────────────────────────────────────

    @Test
    void managerTriesEnginesInPriorityOrder() {
        SearchEngine failing = new SearchEngine() {
            @Override public String name() { return "failing"; }
            @Override public List<SearchResult> search(String query, int maxResults) {
                throw new RuntimeException("boom");
            }
        };
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
    void formatWebContextRendersExactly() {
        List<SearchResult> results = List.of(
                new SearchResult("Alpha", "https://example.com/a", "alpha content", 1, "tavily"),
                new SearchResult("Beta", "https://example.com/b", "beta content", 2, "tavily"));
        // 逐字对齐 Python format_web_context 的拼接结果。
        String expected = "以下是从互联网搜索到的相关信息：\n"
                + "\n[1] Alpha\n    URL: https://example.com/a\n    摘要: alpha content (来源: tavily)"
                + "\n\n[2] Beta\n    URL: https://example.com/b\n    摘要: beta content (来源: tavily)\n";
        assertEquals(expected, SearchPipeline.formatWebContext(results));
        assertEquals("", SearchPipeline.formatWebContext(List.of()));
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

    @Test
    void webSearchCacheHitsDoNotHitServerTwice() {
        WebSearchTool t = tool();
        callTool(t, "{\"query\": \"拉康 精神分析\"}");
        callTool(t, "{\"query\": \"拉康 精神分析\"}");
        assertEquals(1, requestCount.get(), "相同 query 第二次应命中缓存");
        // 不同 query 不命中缓存（缓存键是原始 query）。
        callTool(t, "{\"query\": \"镜像阶段\"}");
        assertEquals(2, requestCount.get());
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
                settings.search.tavilyApiKey, settings.search.timeoutSeconds).isEmpty());
        settings.search.tavilyApiKey = "tvly-test";
        assertTrue(WebSearchTool.available(settings));
        assertEquals(1, Engines.buildEngines(
                Engines.parsePriority(settings.search.enginePriority),
                settings.search.tavilyApiKey, settings.search.timeoutSeconds).size());
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
