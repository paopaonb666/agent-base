package io.github.paopaonb666.agentbase.toolkit.search;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

/**
 * 通过 Tavily REST API 执行搜索（需要 API key；对应 Python 侧
 * {@code TavilyEngine}）。
 *
 * <p>HTTP 细节逐项对齐 Python 版：POST {@code https://api.tavily.com/search}，
 * 请求头 {@code Authorization: Bearer <TAVILY_API_KEY>}，JSON 体为
 * {@code {"query", "search_depth": "basic", "max_results", "include_answer": false}}；
 * 响应的 {@code results} 数组按序映射——{@code title}/{@code url} 直取，
 * 摘要取 {@code content} 字段（缺失字段回空串，对应 Python 的
 * {@code str(r.get("title", ""))}）。非 2xx 抛 {@link IOException}，由
 * 管理器降级。
 *
 * <p>超时来自 {@code SEARCH_TIMEOUT_SECONDS}（Python 侧 AsyncClient 的
 * {@code timeout=10.0} 固定值已改为随配置，注释说明见
 * {@link SearchEngine#search}）。
 */
public final class TavilyHttpClient implements SearchEngine {

    public static final String NAME = "tavily";
    public static final String ENDPOINT = "https://api.tavily.com/search";

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final String apiKey;
    private final Duration timeout;
    private final HttpClient http;
    private final URI endpoint;

    /** 生产路径：默认端点 + 默认 HttpClient。 */
    public TavilyHttpClient(String apiKey, double timeoutSeconds) {
        this(apiKey, timeoutSeconds, HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(10)).build(), URI.create(ENDPOINT));
    }

    /** 测试路径：注入 HttpClient 与端点（对应 Python 的 MockTransport）。 */
    public TavilyHttpClient(String apiKey, double timeoutSeconds, HttpClient http, URI endpoint) {
        this.apiKey = apiKey;
        this.timeout = Duration.ofMillis((long) (timeoutSeconds * 1000));
        this.http = http;
        this.endpoint = endpoint;
    }

    @Override
    public String name() {
        return NAME;
    }

    @Override
    public List<SearchResult> search(String query, int maxResults) throws Exception {
        ObjectNode body = MAPPER.createObjectNode();
        body.put("query", query);
        body.put("search_depth", "basic");
        body.put("max_results", maxResults);
        body.put("include_answer", false);

        HttpRequest request = HttpRequest.newBuilder(endpoint)
                .timeout(timeout)
                .header("Content-Type", "application/json")
                .header("Authorization", "Bearer " + apiKey)
                .POST(HttpRequest.BodyPublishers.ofString(MAPPER.writeValueAsString(body)))
                .build();

        HttpResponse<String> resp = http.send(request, HttpResponse.BodyHandlers.ofString());
        if (resp.statusCode() < 200 || resp.statusCode() >= 300) {
            // 对应 Python 的 resp.raise_for_status()：交由管理器降级。
            throw new IOException("Tavily HTTP " + resp.statusCode() + ": "
                    + abbreviate(resp.body()));
        }
        JsonNode data = MAPPER.readTree(resp.body());
        List<SearchResult> results = new ArrayList<>();
        JsonNode items = data.get("results");
        if (items != null && items.isArray()) {
            for (int i = 0; i < items.size(); i++) {
                JsonNode r = items.get(i);
                results.add(new SearchResult(
                        textOrEmpty(r, "title"),
                        textOrEmpty(r, "url"),
                        textOrEmpty(r, "content"),
                        i + 1,
                        NAME));
            }
        }
        return results;
    }

    private static String textOrEmpty(JsonNode node, String field) {
        JsonNode v = node.get(field);
        return v == null || v.isNull() ? "" : v.asText();
    }

    private static String abbreviate(String body) {
        if (body == null) {
            return "";
        }
        return body.length() <= 200 ? body : body.substring(0, 200) + "…";
    }
}
