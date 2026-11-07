package io.github.paopaonb666.agentbase.toolkit.mcp;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicReference;

/**
 * 测试夹具：最小 HTTP 系 MCP server（JDK HttpServer，无外部依赖）。
 *
 * <p>{@code streamable} 模式：POST /mcp 收 JSON-RPC，回 JSON 单包
 * （initialize 附 {@code Mcp-Session-Id} 头）。{@code sse} 模式：GET /sse
 * 建常驻流先发 {@code endpoint} 事件，POST /messages 的响应从流回推。
 * 仅服务 #332 的测试。
 */
public final class FakeHttpMcpServer {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    public static void main(String[] args) throws Exception {
        String mode = args.length > 0 ? args[0] : "streamable";
        int port = Integer.parseInt(args[1]);
        String toolName = args.length > 2 ? args[2] : "echo";
        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", port), 0);
        if (mode.equals("streamable")) {
            server.createContext("/mcp", exchange -> handleStreamable(exchange, toolName));
        } else {
            handleSse(server, port, toolName);
        }
        // 不设 executor 的话所有 exchange 在单线程上串行处理——SSE handler
        // 一阻塞，POST /messages 就永远排队（实测超时）
        server.setExecutor(java.util.concurrent.Executors.newCachedThreadPool());
        server.start();
        new CountDownLatch(1).await(); // 测试结束后 JVM 退出带走
    }

    private static void handleStreamable(HttpExchange exchange, String toolName)
            throws java.io.IOException {
        byte[] body = exchange.getRequestBody().readAllBytes();
        JsonNode message = MAPPER.readTree(new String(body, StandardCharsets.UTF_8));
        if (!message.has("id")) {
            exchange.sendResponseHeaders(202, -1);
            return;
        }
        String method = message.path("method").asText();
        ObjectNode response = MAPPER.createObjectNode();
        response.put("jsonrpc", "2.0");
        response.set("id", message.get("id"));
        if (method.equals("initialize")) {
            ObjectNode result = MAPPER.createObjectNode();
            result.put("protocolVersion", "2024-11-05");
            result.putObject("capabilities");
            result.putObject("serverInfo").put("name", "fake-http").put("version", "0");
            response.set("result", result);
            byte[] payload = response.toString().getBytes(StandardCharsets.UTF_8);
            exchange.getResponseHeaders().set("Content-Type", "application/json");
            exchange.getResponseHeaders().set("Mcp-Session-Id", "sess-123");
            exchange.sendResponseHeaders(200, payload.length);
            try (OutputStream out = exchange.getResponseBody()) {
                out.write(payload);
            }
            return;
        }
        response.set("result", toolResult(method, message, toolName));
        byte[] payload = response.toString().getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().set("Content-Type", "application/json");
        exchange.sendResponseHeaders(200, payload.length);
        try (OutputStream out = exchange.getResponseBody()) {
            out.write(payload);
        }
    }

    /** SSE 模式：endpoint 事件 + POST 回推。 */
    private static void handleSse(HttpServer server, int port, String toolName)
            throws java.io.IOException {
        AtomicReference<OutputStream> sseOut = new AtomicReference<>();
        server.createContext("/sse", exchange -> {
            exchange.getResponseHeaders().set("Content-Type", "text/event-stream");
            exchange.sendResponseHeaders(200, 0);
            OutputStream out = exchange.getResponseBody();
            sseOut.set(out);
            writeSse(out, "event: endpoint\ndata: /messages?session=s1\n\n");
            try {
                new CountDownLatch(1).await(); // 常驻，直到进程退出
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        });
        server.createContext("/messages", exchange -> {
            byte[] body = exchange.getRequestBody().readAllBytes();
            JsonNode message = MAPPER.readTree(new String(body, StandardCharsets.UTF_8));
            if (message.has("id")) {
                ObjectNode response = MAPPER.createObjectNode();
                response.put("jsonrpc", "2.0");
                response.set("id", message.get("id"));
                response.set("result", toolResult(message.path("method").asText(),
                        message, toolName));
                OutputStream out = sseOut.get();
                if (out != null) {
                    writeSse(out, "data: " + response + "\n\n");
                }
            }
            exchange.sendResponseHeaders(202, -1);
        });
    }

    private static void writeSse(OutputStream out, String chunk) {
        synchronized (out) {
            try {
                out.write(chunk.getBytes(StandardCharsets.UTF_8));
                out.flush();
            } catch (Exception ignored) {
                // 客户端断开后的写入失败忽略
            }
        }
    }

    private static ObjectNode toolResult(String method, JsonNode message, String toolName)
            throws com.fasterxml.jackson.core.JsonProcessingException {
        ObjectNode result = MAPPER.createObjectNode();
        if (method.equals("initialize")) {
            ObjectNode init = MAPPER.createObjectNode();
            init.put("protocolVersion", "2024-11-05");
            init.putObject("capabilities");
            init.putObject("serverInfo").put("name", "fake-http").put("version", "0");
            return init;
        }
        if (method.equals("tools/list")) {
            ObjectNode tool = MAPPER.createObjectNode();
            tool.put("name", toolName);
            tool.put("description", "Echo the query back（HTTP 夹具）");
            tool.set("inputSchema", MAPPER.readTree(
                    "{\"type\":\"object\",\"properties\":{\"query\":"
                            + "{\"type\":\"string\"}},\"required\":[\"query\"]}"));
            result.putArray("tools").add(tool);
        } else if (method.equals("tools/call")) {
            String query = message.path("params").path("arguments").path("query").asText("");
            result.put("isError", false);
            ObjectNode text = MAPPER.createObjectNode();
            text.put("type", "text");
            text.put("text", "echo:" + query);
            result.putArray("content").add(text);
        } else {
            result.putObject("capabilities");
        }
        return result;
    }
}
