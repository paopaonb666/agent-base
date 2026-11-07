package io.github.paopaonb666.agentbase.toolkit.mcp;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * MCP sse 传输（#332，2024-11-05 旧规范）：GET 建常驻 event-stream，
 * server 发 {@code endpoint} 事件指明 POST 地址；请求 POST 过去，响应
 * 从同一条 SSE 流回来（内部后台读线程喂队列）。
 */
public final class McpSseTransport extends McpHttpTransportBase {

    private static final Logger LOG = LoggerFactory.getLogger(McpSseTransport.class);

    private final String sseUrl;
    private volatile String postEndpoint;
    private volatile Thread reader;
    private volatile java.io.InputStream sseStream;
    private final AtomicBoolean started = new AtomicBoolean(false);

    public McpSseTransport(String serverName, String sseUrl, double requestTimeoutSeconds) {
        super(serverName, requestTimeoutSeconds);
        this.sseUrl = sseUrl;
    }

    @Override
    public void start() throws IOException {
        if (!started.compareAndSet(false, true)) {
            return;
        }
        // 先把 POST 地址置为空哨兵：endpoint 事件到达前发请求即失败
        postEndpoint = null;
        Thread thread = new Thread(this::readSseLoop, "mcp-sse-" + serverName);
        thread.setDaemon(true);
        reader = thread;
        thread.start();
        // 等 endpoint 事件（POST 地址就绪），最多等请求预算
        long deadline = System.currentTimeMillis() + (long) (requestTimeoutSeconds * 1000);
        while (postEndpoint == null && !isClosed() && System.currentTimeMillis() < deadline) {
            try {
                Thread.sleep(20);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IOException("MCP server '" + serverName + "' 连接被中断");
            }
        }
        if (postEndpoint == null) {
            throw new IOException("MCP server '" + serverName
                    + "'：SSE 流未在预算内给出 endpoint 事件");
        }
    }

    @Override
    protected void applySessionHeaders(HttpRequest.Builder builder) {
        // 旧规范无会话头
    }

    @Override
    public void sendLine(String line) throws IOException {
        String target = postEndpoint;
        if (isClosed() || target == null) {
            throw new IOException("MCP server '" + serverName + "' 通道未就绪或已关闭");
        }
        try {
            HttpResponse<Void> response = http().send(
                    post(target, line).build(), HttpResponse.BodyHandlers.discarding());
            if (response.statusCode() >= 300) {
                throw new IOException("MCP server '" + serverName + "' POST HTTP "
                        + response.statusCode());
            }
            // 响应（若有）从 SSE 流回来，由 readSseLoop 投递
        } catch (java.net.http.HttpTimeoutException e) {
            throw new IOException("MCP server '" + serverName + "' POST 超时", e);
        } catch (IOException e) {
            throw e;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("MCP server '" + serverName + "' POST 被中断", e);
        }
    }

    private void readSseLoop() {
        StringBuilder data = new StringBuilder();
        try {
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(sseUrl))
                    .timeout(Duration.ofSeconds(30))
                    .header("Accept", "text/event-stream")
                    .GET()
                    .build();
            HttpResponse<java.io.InputStream> response = http().send(request,
                    HttpResponse.BodyHandlers.ofInputStream());
            if (response.statusCode() != 200) {
                throw new IOException("MCP server '" + serverName + "' SSE HTTP "
                        + response.statusCode());
            }
            sseStream = response.body();
            try (BufferedReader httpReader = new BufferedReader(new InputStreamReader(
                    response.body(), StandardCharsets.UTF_8))) {
                String line;
                while (!isClosed() && (line = httpReader.readLine()) != null) {
                    if (line.isEmpty()) {
                        if (data.length() > 0) {
                            handleEvent(data.toString());
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
                }
            }
        } catch (Exception e) {
            if (!isClosed()) {
                LOG.warn("MCP server {}: SSE 流断开：{}", serverName, e.getMessage());
            }
        } finally {
            if (!isClosed()) {
                // 流断了：置空 POST 地址，后续请求快速失败
                postEndpoint = null;
            }
        }
    }

    private void handleEvent(String dataPayload) {
        String text = dataPayload.strip();
        if (text.isEmpty()) {
            return;
        }
        // endpoint 事件：data 是 POST 地址（可能相对路径），相对 base 解析
        if (postEndpoint == null && !text.startsWith("{")) {
            try {
                postEndpoint = URI.create(sseUrl).resolve(text).toString();
            } catch (IllegalArgumentException e) {
                LOG.warn("MCP server {}: endpoint 事件不可解析：{}", serverName, text);
            }
            return;
        }
        deliverJsonOrIgnore(text);
    }

    @Override
    public void close() {
        super.close();
        // 读线程阻塞在流的 readLine 上（不响应 interrupt），直接关流解锁
        java.io.InputStream stream = sseStream;
        if (stream != null) {
            try {
                stream.close();
            } catch (IOException ignored) {
                // 关流失败不影响本地关闭语义
            }
        }
        Thread r = reader;
        if (r != null) {
            r.interrupt();
        }
    }
}
