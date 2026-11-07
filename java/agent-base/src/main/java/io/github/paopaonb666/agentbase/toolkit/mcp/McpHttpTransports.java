package io.github.paopaonb666.agentbase.toolkit.mcp;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.TimeUnit;

/**
 * MCP HTTP 系传输的公共底座（#332）：JDK HttpClient + 内部队列把
 * 请求/响应语义适配到 {@link McpTransport} 的行式接口。
 *
 * <p>两种子形态：
 *
 * <ul>
 *   <li><b>streamable_http</b>（MCP 2025-03-26 规范）：每条消息 POST 到
 *       单一端点；响应可以是 application/json 单包，也可以是
 *       text/event-stream（事件里的 data 才是 JSON-RPC 消息）；
 *       {@code Mcp-Session-Id} 在 initialize 响应头出现时随后续请求回带，
 *       关闭时 DELETE 终止会话。</li>
 *   <li><b>sse</b>（2024-11-05 旧规范）：GET 建一条常驻 event-stream，
 *       server 先发 {@code endpoint} 事件指明 POST 地址；请求 POST 过去，
 *       响应都从 SSE 流回来。</li>
 * </ul>
 */
abstract class McpHttpTransportBase implements McpTransport {

    static final ObjectMapper MAPPER = new ObjectMapper();

    protected final String serverName;
    protected final double requestTimeoutSeconds;
    private final HttpClient http;
    private final LinkedBlockingDeque<String> inbound = new LinkedBlockingDeque<>();
    private volatile boolean closed;

    protected McpHttpTransportBase(String serverName, double requestTimeoutSeconds) {
        this.serverName = serverName;
        this.requestTimeoutSeconds = requestTimeoutSeconds > 0 ? requestTimeoutSeconds : 30.0;
        this.http = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(10))
                .build();
    }

    /** 把响应 JSON-RPC 消息喂进读通道（由各子形态在收到响应时调用）。 */
    protected void deliver(JsonNode message) {
        if (message == null || closed) {
            return;
        }
        try {
            inbound.put(message.toString());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    @Override
    public String readLine() throws IOException {
        while (!closed) {
            try {
                String line = inbound.poll(200, TimeUnit.MILLISECONDS);
                if (line != null) {
                    return line;
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return null;
            }
        }
        return null;
    }

    /** 共用的 POST 构造（子形态补 URL 与会话头）。 */
    protected HttpRequest.Builder post(String url, String jsonRpcLine) {
        HttpRequest.Builder builder = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .timeout(Duration.ofMillis((long) (requestTimeoutSeconds * 1000)))
                .header("Content-Type", "application/json")
                .header("Accept", "application/json, text/event-stream")
                .POST(HttpRequest.BodyPublishers.ofString(jsonRpcLine, StandardCharsets.UTF_8));
        applySessionHeaders(builder);
        return builder;
    }

    /** 子形态的会话/协议头（streamable_http：Mcp-Session-Id + 协议版本）。 */
    protected abstract void applySessionHeaders(HttpRequest.Builder builder);

    /**
     * 发送 POST 并把响应体拆成 JSON-RPC 消息投递。202/无 body = 只有通知，
     * 不投递；非 2xx 抛 IOException（由 McpException 语义包装在客户端层）。
     */
    protected void postAndDeliver(String url, String jsonRpcLine) throws IOException {
        HttpResponse<byte[]> response;
        try {
            response = http.send(post(url, jsonRpcLine).build(),
                    HttpResponse.BodyHandlers.ofByteArray());
        } catch (java.net.http.HttpTimeoutException e) {
            throw new IOException("MCP server '" + serverName + "' HTTP 超时", e);
        } catch (IOException e) {
            throw e;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("MCP server '" + serverName + "' 请求被中断", e);
        }
        if (response.statusCode() < 200 || response.statusCode() >= 300) {
            throw new IOException("MCP server '" + serverName + "' HTTP "
                    + response.statusCode() + "：" + truncateBody(response.body()));
        }
        String contentType = response.headers().firstValue("Content-Type").orElse("");
        byte[] body = response.body();
        if (body == null || body.length == 0) {
            return; // 202 Accepted：请求已受理，无响应包
        }
        String text = new String(body, StandardCharsets.UTF_8);
        if (contentType.contains("text/event-stream")) {
            for (String data : parseSseData(text)) {
                deliverJsonOrIgnore(data);
            }
        } else {
            deliverJsonOrIgnore(text);
        }
    }

    protected static String truncateBody(byte[] body) {
        String text = body == null ? "" : new String(body, StandardCharsets.UTF_8);
        return text.length() > 200 ? text.substring(0, 200) + "…" : text;
    }
    /** 解析 SSE 文本，按事件返回拼接后的 data 负载（多行 data 以 \n 相连）。 */
    static List<String> parseSseData(String sseText) {
        List<String> out = new ArrayList<>();
        StringBuilder data = new StringBuilder();
        for (String rawLine : sseText.split("\n", -1)) {
            String line = rawLine.stripTrailing();
            if (line.isEmpty()) {
                if (data.length() > 0) {
                    out.add(data.toString());
                    data.setLength(0);
                }
                continue;
            }
            if (line.startsWith("data:")) {
                String payload = line.substring(5);
                if (payload.startsWith(" ")) {
                    payload = payload.substring(1);
                }
                if (data.length() > 0) {
                    data.append('\n');
                }
                data.append(payload);
            }
            // event:/id:/注释行：JSON-RPC 消息只在 data 里，其余忽略
        }
        if (data.length() > 0) {
            out.add(data.toString());
        }
        return out;
    }

    /** data 负载是合法 JSON（对象/数组）才投递，其余静默丢弃。 */
    protected void deliverJsonOrIgnore(String payload) {
        String text = payload.strip();
        if (text.isEmpty() || (text.charAt(0) != '{' && text.charAt(0) != '[')) {
            return;
        }
        try {
            deliver(MAPPER.readTree(text));
        } catch (IOException ignored) {
            // 非 JSON 的 SSE data（心跳等）：忽略
        }
    }

    @Override
    public void close() {
        closed = true;
    }

    protected boolean isClosed() {
        return closed;
    }

    protected HttpClient http() {
        return http;
    }
}
