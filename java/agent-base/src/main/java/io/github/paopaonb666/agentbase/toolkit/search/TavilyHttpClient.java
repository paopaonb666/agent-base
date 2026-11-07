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

    /** 瞬时失败的退避重试间隔（#243：查询幂等，1 次重试安全）。 */
    static final long RETRY_BACKOFF_MS = 500;

    /** 生产路径：默认端点 + 默认 HttpClient（HTTP(S)_PROXY 生效，#243）。 */
    public TavilyHttpClient(String apiKey, double timeoutSeconds) {
        this(apiKey, timeoutSeconds, defaultHttpClient(), URI.create(ENDPOINT));
    }

    /**
     * 默认 HttpClient：连接超时 10s；企业内网经 {@code HTTPS_PROXY} /
     * {@code HTTP_PROXY}（小写同认）走代理——tavily 端点是 https，
     * HTTPS_PROXY 优先。解析失败静默忽略（直连）。
     */
    static HttpClient defaultHttpClient() {
        HttpClient.Builder builder = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(10));
        String proxy = firstNonBlank(System.getenv("HTTPS_PROXY"), System.getenv("https_proxy"),
                System.getenv("HTTP_PROXY"), System.getenv("http_proxy"));
        java.net.InetSocketAddress address = parseProxy(proxy);
        if (address != null) {
            builder.proxy(java.net.ProxySelector.of(address));
        }
        return builder.build();
    }

    static String firstNonBlank(String... values) {
        for (String value : values) {
            if (value != null && !value.isBlank()) {
                return value.strip();
            }
        }
        return null;
    }

    /** 解析 {@code http://host:port} / {@code host:port}；非法返回 null（直连）。 */
    static java.net.InetSocketAddress parseProxy(String value) {
        if (value == null || value.isBlank()) {
            return null;
        }
        String text = value.strip();
        if (text.startsWith("http://") || text.startsWith("https://")) {
            text = text.substring(text.indexOf("://") + 3);
        }
        int colon = text.lastIndexOf(':');
        if (colon <= 0 || colon == text.length() - 1) {
            return null;
        }
        try {
            // InetSocketAddress(host, port) 不做即时 DNS 解析（未知主机生成
            // unresolved 地址），离线环境下构造也不炸
            return new java.net.InetSocketAddress(text.substring(0, colon),
                    Integer.parseInt(text.substring(colon + 1)));
        } catch (Exception e) {
            return null;
        }
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
    public List<SearchResult> search(String query, int maxResults,
                                     double budgetSeconds) throws Exception {
        // #242：剩余预算小于自身配置超时时按预算收口（管线 deadline 贯穿）。
        Duration budget = Duration.ofMillis((long) (budgetSeconds * 1000));
        return searchWith(query, maxResults,
                budgetSeconds > 0 && budget.compareTo(timeout) < 0 ? budget : timeout);
    }

    @Override
    public List<SearchResult> search(String query, int maxResults) throws Exception {
        return searchWith(query, maxResults, timeout);
    }

    private List<SearchResult> searchWith(String query, int maxResults, Duration effective)
            throws Exception {
        // #243：瞬时失败（网络异常 / 5xx / 429）1 次退避重试——查询幂等；
        // 4xx（鉴权/配额语义错误）不重试。重试沿用同一有效超时，整体仍受
        // 管线 deadline（#242）与共享池挂钟预算双重约束。
        int attempt = 0;
        while (true) {
            attempt++;
            try {
                return searchOnce(query, maxResults, effective);
            } catch (java.net.http.HttpTimeoutException | java.net.ConnectException e) {
                if (attempt >= 2) {
                    throw e;
                }
                backoff();
            } catch (HttpStatusException e) {
                boolean transientStatus = e.status >= 500 || e.status == 429;
                if (attempt >= 2 || !transientStatus) {
                    throw e;
                }
                backoff();
            }
        }
    }

    private static void backoff() throws IOException {
        try {
            Thread.sleep(RETRY_BACKOFF_MS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("Tavily 重试等待被中断");
        }
    }

    /** 带 HTTP 状态码的失败（区分可重试的 5xx/429 与终态 4xx）。 */
    private static final class HttpStatusException extends IOException {
        final int status;

        HttpStatusException(int status, String bodyAbbreviated) {
            super("Tavily HTTP " + status + ": " + bodyAbbreviated);
            this.status = status;
        }
    }

    private List<SearchResult> searchOnce(String query, int maxResults, Duration effective)
            throws Exception {
        ObjectNode body = MAPPER.createObjectNode();
        body.put("query", query);
        body.put("search_depth", "basic");
        body.put("max_results", maxResults);
        body.put("include_answer", false);

        HttpRequest request = HttpRequest.newBuilder(endpoint)
                .timeout(effective)
                .header("Content-Type", "application/json")
                .header("Authorization", "Bearer " + apiKey)
                .POST(HttpRequest.BodyPublishers.ofString(MAPPER.writeValueAsString(body)))
                .build();

        HttpResponse<String> resp = http.send(request, HttpResponse.BodyHandlers.ofString());
        if (resp.statusCode() < 200 || resp.statusCode() >= 300) {
            // 对应 Python 的 resp.raise_for_status()：交由管理器降级。
            throw new HttpStatusException(resp.statusCode(), abbreviate(resp.body()));
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
