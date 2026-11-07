package io.github.paopaonb666.agentbase.toolkit.mcp;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

/**
 * MCP streamable_http 传输（#332，MCP 2025-03-26 规范）：单一端点，
 * 每条 JSON-RPC 消息一个 POST；响应可能是 JSON 单包或 SSE 流，
 * {@code Mcp-Session-Id} 会话随 initialize 建立并随后续请求回带。
 */
public final class McpStreamableHttpTransport extends McpHttpTransportBase {

    private static final Logger LOG = LoggerFactory.getLogger(McpStreamableHttpTransport.class);

    private final String endpoint;
    private volatile String sessionId;
    private volatile String protocolVersion;

    public McpStreamableHttpTransport(String serverName, String endpoint,
                                      double requestTimeoutSeconds) {
        super(serverName, requestTimeoutSeconds);
        this.endpoint = endpoint;
    }

    @Override
    public void start() {
        // 无独立握手动作：会话随首个 initialize POST 建立
    }

    @Override
    public void onProtocolVersion(String version) {
        this.protocolVersion = version;
    }

    @Override
    protected void applySessionHeaders(HttpRequest.Builder builder) {
        String session = sessionId;
        if (session != null) {
            builder.header("Mcp-Session-Id", session);
        }
        String version = protocolVersion;
        if (version != null) {
            builder.header("MCP-Protocol-Version", version);
        }
    }

    @Override
    public void sendLine(String line) throws IOException {
        if (isClosed()) {
            throw new IOException("MCP server '" + serverName + "' 通道已关闭");
        }
        HttpResponse<byte[]> response;
        try {
            response = http().send(post(endpoint, line).build(),
                    HttpResponse.BodyHandlers.ofByteArray());
        } catch (java.net.http.HttpTimeoutException e) {
            throw new IOException("MCP server '" + serverName + "' HTTP 超时", e);
        } catch (IOException e) {
            throw e;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("MCP server '" + serverName + "' 请求被中断", e);
        }
        // 会话头在首个（initialize）响应上出现，之后每条请求回带
        response.headers().firstValue("Mcp-Session-Id").ifPresent(id -> sessionId = id);
        if (response.statusCode() == 404 && sessionId != null) {
            sessionId = null; // 会话过期：下条请求按无会话重试语义重建
        }
        if (response.statusCode() < 200 || response.statusCode() >= 300) {
            throw new IOException("MCP server '" + serverName + "' HTTP "
                    + response.statusCode() + "：" + truncateBody(response.body()));
        }
        byte[] body = response.body();
        if (body == null || body.length == 0) {
            return; // 202 Accepted：只有通知，无响应包
        }
        String contentType = response.headers().firstValue("Content-Type").orElse("");
        String text = new String(body, java.nio.charset.StandardCharsets.UTF_8);
        if (contentType.contains("text/event-stream")) {
            for (String data : parseSseData(text)) {
                deliverJsonOrIgnore(data);
            }
        } else {
            deliverJsonOrIgnore(text);
        }
    }

    @Override
    public void close() {
        String session = sessionId;
        if (session != null && !isClosed()) {
            // DELETE 终止会话（尽力而为：失败不影响本地关闭）
            try {
                HttpRequest.Builder builder = HttpRequest.newBuilder()
                        .uri(java.net.URI.create(endpoint))
                        .timeout(java.time.Duration.ofSeconds(3))
                        .header("Mcp-Session-Id", session)
                        .DELETE();
                http().send(builder.build(), HttpResponse.BodyHandlers.discarding());
            } catch (Exception e) {
                LOG.debug("MCP server {}: 会话终止请求失败（忽略）", serverName, e);
            }
        }
        super.close();
    }
}
