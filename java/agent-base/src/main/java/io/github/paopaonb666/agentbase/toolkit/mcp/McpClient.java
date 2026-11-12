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
 * 与对客户端的请求（带 id 带 method）一律忽略——本客户端不提供
 * sampling/roots 能力，工具调用期间 server 也不应反向请求；反向请求即使
 * 撞上在途 pending id 也不会被当成响应（review C-P2）。线程安全：写线程
 * 串行化（各 transport 自持写锁），读线程独立。
 */
public final class McpClient implements AutoCloseable {

    private static final Logger LOG = LoggerFactory.getLogger(McpClient.class);

    /** 客户端声明的 MCP 协议版本（server 会回它支持的版本，不强校验）。 */
    public static final String PROTOCOL_VERSION = "2024-11-05";

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final McpTransportFactory transportFactory;
    private final String serverName;
    private final AtomicInteger nextId = new AtomicInteger();
    private final Map<Integer, CompletableFuture<ObjectNode>> pending = new ConcurrentHashMap<>();
    // transport 可替换（#29 重连）：重连时由工厂产出新通道并整体换芯。
    private volatile McpTransport transport;
    private volatile Thread readerThread;
    private volatile boolean closed;
    /** 读线程代际（#29）：重连换代后旧读线程不得清掉新代际的 pending。 */
    private final java.util.concurrent.atomic.AtomicLong readerGeneration =
            new java.util.concurrent.atomic.AtomicLong();
    private final Object reconnectLock = new Object();
    private final java.util.concurrent.atomic.AtomicBoolean reconnecting =
            new java.util.concurrent.atomic.AtomicBoolean();
    // 重连握手参数（首次 connect 时固化）。
    private volatile String clientName;
    private volatile String clientVersion;
    private volatile double handshakeTimeoutSeconds;

    public McpClient(String serverName, McpTransport transport) {
        this(serverName, transport, null);
    }

    /**
     * 工厂形态（#29 重连最小面）：每次（重）连产出全新通道——server 崩溃
     * 后下一次调用自愈，无需重启进程。单通道构造（上一构造器）保留给
     * 测试直构，无重连能力。
     */
    public McpClient(String serverName, McpTransportFactory transportFactory) {
        this(serverName, null, transportFactory);
    }

    private McpClient(String serverName, McpTransport transport, McpTransportFactory transportFactory) {
        this.serverName = serverName;
        this.transport = transport;
        this.transportFactory = transportFactory;
    }

    /** 启动传输 + initialize 握手（对应 MCP 生命周期：initialize → initialized）。 */
    public void connect(String clientName, String clientVersion, double handshakeTimeoutSeconds) {
        this.clientName = clientName;
        this.clientVersion = clientVersion;
        this.handshakeTimeoutSeconds = handshakeTimeoutSeconds;
        McpTransport current = transport;
        if (current == null) {
            if (transportFactory == null) {
                throw new McpException("MCP server '" + serverName + "'：无可用通道");
            }
            current = createTransport();
            transport = current;
        }
        try {
            current.start();
        } catch (IOException e) {
            throw new McpException(
                    "MCP server '" + serverName + "' 启动失败：进程能拉起来吗？"
                            + e.getClass().getSimpleName() + ": " + e.getMessage(), e);
        }
        startReader();
        ObjectNode params = initParams(clientName, clientVersion);
        try {
            ObjectNode result = request("initialize", params, handshakeTimeoutSeconds);
            // 协商版本回传给传输层（HTTP 系做后续请求的协议版本头）
            String negotiated = result.path("protocolVersion").asText(PROTOCOL_VERSION);
            current.onProtocolVersion(negotiated);
        } catch (McpException e) {
            close();
            throw e;
        }
        sendNotification("notifications/initialized");
    }

    private static ObjectNode initParams(String clientName, String clientVersion) {
        ObjectNode params = MAPPER.createObjectNode();
        params.put("protocolVersion", PROTOCOL_VERSION);
        params.putObject("capabilities");
        params.putObject("clientInfo").put("name", clientName).put("version", clientVersion);
        return params;
    }

    private McpTransport createTransport() {
        try {
            return transportFactory.create();
        } catch (IOException e) {
            throw new McpException("MCP server '" + serverName + "' 通道创建失败："
                    + e.getClass().getSimpleName() + ": " + e.getMessage(), e);
        }
    }

    /** 单次 JSON-RPC 请求（带超时），返回 result 节点；error 响应抛 McpException。 */
    public ObjectNode request(String method, ObjectNode params, double timeoutSeconds) {
        // 惰性重连预检（#29）：连接已断时先自愈再发——server 崩溃后下一次
        // 工具调用恢复；重连失败保持断开态（本次照常失败）。
        if (transportFactory != null && !closed && !reconnecting.get() && !connected()) {
            reconnect("连接不可用");
            if (!connected()) {
                throw new McpException("MCP server '" + serverName + "' 连接不可用（重连失败）");
            }
        }
        try {
            return doRequest(method, params, timeoutSeconds);
        } catch (McpException e) {
            // 仅「请求未送达」的写失败重连重试一次：响应超时/读侧断开可能
            // 意味着 server 已经执行过——工具调用不可安全重放，保守不重试
            // （下一次调用经预检自愈）。
            if (transportFactory != null && !closed && !reconnecting.get()
                    && e.getCause() instanceof IOException) {
                // force：写侧断开在 connected() 里不可见（读线程/进程都活着），
                // 明确的发送失败必须强制重连而不是被健康预检短路。
                reconnect("写请求失败：" + e.getMessage(), true);
                if (connected()) {
                    return doRequest(method, params, timeoutSeconds);
                }
            }
            throw e;
        }
    }

    private ObjectNode doRequest(String method, ObjectNode params, double timeoutSeconds) {
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
                McpTransport current = transport;
                if (current == null) {
                    throw new McpException("MCP server '" + serverName + "' 通道不可用");
                }
                current.sendLine(line);
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
        McpTransport bound = transport; // 读线程绑定代际通道：重连换芯后旧线程不碰新通道
        long generation = readerGeneration.incrementAndGet();
        Thread thread = new Thread(() -> readLoop(bound, generation), "mcp-client-" + serverName);
        readerThread = thread;
        thread.setDaemon(true);
        thread.start();
    }

    private void readLoop(McpTransport boundTransport, long generation) {
        try {
            while (!closed) {
                String line = boundTransport.readLine();
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
            // 只在仍是当前代际时清账：重连换代后旧读线程的退出不得吞掉
            // 新通道上在途请求的 future（由请求超时自然失败）。
            if (readerGeneration.get() == generation) {
                failAllPending("MCP server '" + serverName + "' 连接已关闭");
            }
        }
    }

    /**
     * 连接健康态（#29）：{@code closed} 标志 + 读线程存活 + 传输层存活
     * 信号（stdio=子进程存活；HTTP 系默认 true，断开由写失败暴露）。
     */
    public boolean connected() {
        if (closed) {
            return false;
        }
        Thread reader = readerThread;
        McpTransport current = transport;
        return reader != null && reader.isAlive()
                && current != null && current.alive();
    }

    /**
     * 断开后的重连（#29 最小面）：关旧通道 → 工厂产新通道 → start →
     * 重新握手。全程持锁单飞；失败保持断开态（下次调用再试），绝不把
     * 客户端置为 closed——与启动期的 fail-fast 不同，运行期断开是常态
     * 而非配置错误。
     */
    private void reconnect(String reason) {
        reconnect(reason, false);
    }

    /**
     * 断开后的重连（#29 最小面）：关旧通道 → 工厂产新通道 → start →
     * 重新握手。全程持锁单飞；失败保持断开态（下次调用再试），绝不把
     * 客户端置为 closed——与启动期的 fail-fast 不同，运行期断开是常态
     * 而非配置错误。{@code force}（写失败路径）：跳过健康预检——写侧
     * 断开在 connected() 的判据（读线程/进程存活）里不可见。
     */
    private void reconnect(String reason, boolean force) {
        reconnecting.set(true);
        try {
            synchronized (reconnectLock) {
                if (closed || (!force && connected())) {
                    return; // 并发调用者已重连成功 / 客户端已关闭 / 健康态无需要
                }
                LOG.warn("MCP server {}: 断开，尝试重连（{}）", serverName, reason);
                McpTransport old = transport;
                transport = null;
                if (old != null) {
                    try {
                        old.close();
                    } catch (Exception e) {
                        LOG.debug("MCP server {}: 旧通道关闭失败", serverName, e);
                    }
                }
                if (closed) {
                    return;
                }
                McpTransport fresh;
                try {
                    fresh = createTransport();
                } catch (Exception e) {
                    LOG.warn("MCP server {}: 重连失败（通道创建）：{}", serverName, e.getMessage());
                    return;
                }
                transport = fresh;
                if (closed) {
                    tryCloseQuietly(fresh);
                    transport = null;
                    return;
                }
                try {
                    fresh.start();
                    startReader();
                    ObjectNode result = request("initialize",
                            initParams(clientName, clientVersion), handshakeTimeoutSeconds);
                    fresh.onProtocolVersion(result.path("protocolVersion").asText(PROTOCOL_VERSION));
                    sendNotification("notifications/initialized");
                    LOG.info("MCP server {}: 重连成功", serverName);
                } catch (Exception e) {
                    LOG.warn("MCP server {}: 重连失败：{}", serverName, e.getMessage());
                    tryCloseQuietly(fresh);
                    if (transport == fresh) {
                        transport = null;
                    }
                }
            }
        } finally {
            reconnecting.set(false);
        }
    }

    private void tryCloseQuietly(McpTransport channel) {
        try {
            channel.close();
        } catch (Exception e) {
            LOG.debug("MCP server {}: 通道关闭异常", serverName, e);
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
        if (node.has("method")) {
            // server → client 的请求（带 id 带 method）或通知（无 id）：
            // 本客户端不提供 sampling/roots 能力，一律忽略——绝不落响应
            // 分派（review C-P2：撞上在途 pending id 的反向请求此前会被
            // 当成空 result 伪成功）。
            if (node.has("id")) {
                LOG.info("MCP server {}: 忽略 server→client 请求 method={}", serverName,
                        node.path("method").asText());
            }
            return;
        }
        JsonNode id = node.get("id");
        if (id == null || !id.canConvertToInt()) {
            // 无 id 且无 method 的畸形帧：忽略。
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
        McpTransport current = transport;
        if (current != null) {
            current.close();
        }
        Thread reader = readerThread;
        if (reader != null) {
            reader.interrupt();
            readerThread = null;
        }
    }
}
