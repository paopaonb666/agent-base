package io.github.paopaonb666.agentbase.toolkit.mcp;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * MCP 客户端：在一个 {@link McpTransport} 上做 JSON-RPC 2.0 的请求/响应
 * 关联（#331）。支持 initialize 握手、tools/list 与 tools/call——本期只
 * 需要这三个方法（裁决范围）。
 *
 * <p>响应由后台读线程分发给 pending 表（按 id 关联）；来自 server 的通知
 * 与对客户端的请求被忽略（本客户端不提供 sampling/roots 能力，工具调用
 * 期间 server 也不应反向请求）。线程安全：写线程串行化，读线程独立。
 */
public final class McpClient implements AutoCloseable {

    private static final Logger LOG = LoggerFactory.getLogger(McpClient.class);

    /** 客户端声明的 MCP 协议版本（server 会回它支持的版本，不强校验）。 */
    public static final String PROTOCOL_VERSION = "2024-11-05";

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final McpTransport transport;
    private final String serverName;
    private final AtomicInteger nextId = new AtomicInteger();
    private final Map<Integer, CompletableFuture<ObjectNode>> pending = new ConcurrentHashMap<>();
    private final Object writeLock = new Object();
    private volatile Thread readerThread;
    private volatile boolean closed;

    public McpClient(String serverName, McpTransport transport) {
        this.serverName = serverName;
        this.transport = transport;
    }

    /** 启动传输 + initialize 握手（对应 MCP 生命周期：initialize → initialized）。 */
    public void connect(String clientName, String clientVersion, double handshakeTimeoutSeconds) {
        try {
            transport.start();
        } catch (IOException e) {
            throw new McpException(
                    "MCP server '" + serverName + "' 启动失败：进程能拉起来吗？"
                            + e.getClass().getSimpleName() + ": " + e.getMessage(), e);
        }
        startReader();
        ObjectNode params = MAPPER.createObjectNode();
        params.put("protocolVersion", PROTOCOL_VERSION);
        params.putObject("capabilities");
        params.putObject("clientInfo").put("name", clientName).put("version", clientVersion);
        try {
            request("initialize", params, handshakeTimeoutSeconds);
        } catch (McpException e) {
            close();
            throw e;
        }
        sendNotification("notifications/initialized");
    }

    /** 单次 JSON-RPC 请求（带超时），返回 result 节点；error 响应抛 McpException。 */
    public ObjectNode request(String method, ObjectNode params, double timeoutSeconds) {
        int id = nextId.incrementAndGet();
        ObjectNode message = MAPPER.createObjectNode();
        message.put("jsonrpc", "2.0");
        message.put("id", id);
        message.put("method", method);
        message.set("params", params == null ? MAPPER.createObjectNode() : params);

        CompletableFuture<ObjectNode> future = new CompletableFuture<>();
        pending.put(id, future);
        try {
            String line;
            try {
                line = MAPPER.writeValueAsString(message);
            } catch (Exception e) {
                throw new McpException("MCP server '" + serverName + "'：请求序列化失败", e);
            }
            try {
                transport.sendLine(line);
            } catch (IOException e) {
                throw new McpException(
                        "MCP server '" + serverName + "' 写请求失败（进程还活着吗？）："
                                + e.getMessage(), e);
            }
            try {
                return future.get((long) (timeoutSeconds * 1000), TimeUnit.MILLISECONDS);
            } catch (TimeoutException e) {
                throw new McpException("MCP server '" + serverName + "' 响应超时（" + method
                        + "，" + timeoutSeconds + "s）");
            } catch (java.util.concurrent.ExecutionException e) {
                Throwable cause = e.getCause() == null ? e : e.getCause();
                throw new McpException("MCP server '" + serverName + "' 请求失败："
                        + cause.getMessage(), cause);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new McpException("MCP server '" + serverName + "' 等待响应被中断", e);
            }
        } finally {
            pending.remove(id);
        }
    }

    private void sendNotification(String method) {
        ObjectNode message = MAPPER.createObjectNode();
        message.put("jsonrpc", "2.0");
        message.put("method", method);
        try {
            transport.sendLine(MAPPER.writeValueAsString(message));
        } catch (Exception e) {
            throw new McpException("MCP server '" + serverName + "' 发送通知失败：" + e.getMessage(), e);
        }
    }

    /** tools/list（返回 server 暴露的工具定义）。 */
    public List<McpToolDef> listTools(double timeoutSeconds) {
        ObjectNode result = request("tools/list", null, timeoutSeconds);
        JsonNode tools = result.get("tools");
        List<McpToolDef> out = new ArrayList<>();
        if (tools == null || !tools.isArray()) {
            return out;
        }
        for (JsonNode tool : tools) {
            JsonNode nameNode = tool.get("name");
            if (nameNode == null || !nameNode.isTextual() || nameNode.asText().isEmpty()) {
                LOG.warn("MCP server {}: tools/list 含缺名条目，已跳过", serverName);
                continue;
            }
            JsonNode description = tool.get("description");
            JsonNode schema = tool.get("inputSchema");
            out.add(new McpToolDef(nameNode.asText(),
                    description == null || !description.isTextual() ? "" : description.asText(),
                    schema == null ? MAPPER.createObjectNode() : schema));
        }
        return out;
    }

    /**
     * tools/call：返回 server 的文本结果。server 标记 isError 时抛
     * IllegalStateException（由 ToolPool 归一成模型可读文本，与 Python 侧
     * ToolException 同通道）。
     */
    public String callTool(String toolName, JsonNode arguments, double timeoutSeconds) {
        ObjectNode params = MAPPER.createObjectNode();
        params.put("name", toolName);
        params.set("arguments", arguments == null ? MAPPER.createObjectNode() : arguments);
        ObjectNode result = request("tools/call", params, timeoutSeconds);
        boolean isError = result.path("isError").asBoolean(false);
        String text = renderContent(result.get("content"));
        if (isError) {
            throw new IllegalStateException(text.isEmpty() ? "MCP 工具返回错误（无详情）" : text);
        }
        return text;
    }

    private static String renderContent(JsonNode content) {
        if (content == null || !content.isArray()) {
            return "";
        }
        List<String> parts = new ArrayList<>();
        for (JsonNode item : content) {
            if ("text".equals(item.path("type").asText())) {
                parts.add(item.path("text").asText());
            } else if (item.isObject()) {
                parts.add(item.toString()); // 非 text 块降级为 JSON 视图
            }
        }
        return String.join("\n", parts);
    }

    private void startReader() {
        readerThread = new Thread(this::readLoop, "mcp-client-" + serverName);
        readerThread.setDaemon(true);
        readerThread.start();
    }

    private void readLoop() {
        try {
            while (!closed) {
                String line = transport.readLine();
                if (line == null) {
                    break;
                }
                line = line.strip();
                if (line.isEmpty()) {
                    continue;
                }
                handleLine(line);
            }
        } catch (IOException e) {
            if (!closed) {
                LOG.warn("MCP server {}: 读通道断开：{}", serverName, e.getMessage());
            }
        } finally {
            failAllPending("MCP server '" + serverName + "' 连接已关闭");
        }
    }

    private void handleLine(String line) {
        ObjectNode node;
        try {
            node = (ObjectNode) MAPPER.readTree(line);
        } catch (Exception e) {
            LOG.warn("MCP server {}: 非 JSON 行被丢弃：{}", serverName,
                    line.length() > 120 ? line.substring(0, 120) : line);
            return;
        }
        JsonNode id = node.get("id");
        if (id == null || !id.canConvertToInt()) {
            // 通知或 server → client 的请求：本客户端没有对应能力，忽略。
            return;
        }
        CompletableFuture<ObjectNode> future = pending.get(id.asInt());
        if (future == null) {
            return;
        }
        JsonNode error = node.get("error");
        if (error != null && error.isObject()) {
            future.completeExceptionally(new McpException(
                    "MCP error " + error.path("code").asInt() + ": "
                            + error.path("message").asText()));
        } else {
            JsonNode result = node.get("result");
            future.complete(result instanceof ObjectNode obj
                    ? obj : MAPPER.createObjectNode());
        }
    }

    private void failAllPending(String message) {
        for (Map.Entry<Integer, CompletableFuture<ObjectNode>> entry : pending.entrySet()) {
            entry.getValue().completeExceptionally(new McpException(message));
        }
        pending.clear();
    }

    @Override
    public void close() {
        closed = true;
        failAllPending("MCP server '" + serverName + "' 客户端已关闭");
        transport.close();
        Thread reader = readerThread;
        if (reader != null) {
            reader.interrupt();
            readerThread = null;
        }
    }
}
