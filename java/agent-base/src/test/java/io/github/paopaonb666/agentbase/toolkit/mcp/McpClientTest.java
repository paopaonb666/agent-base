package io.github.paopaonb666.agentbase.toolkit.mcp;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * McpClient 连接健康与惰性重连（#29）：connected() 判据（closed 标志 +
 * 读线程存活 + 传输层存活）、断开后的单飞重连握手、写失败重试一次的
 * 边界（响应超时不重试——工具调用不可安全重放）。
 */
class McpClientTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    /**
     * 脚本化 transport：sendLine 时按请求 id 回显脚本 result；sendBroken
     * 只断写侧（读线程仍活——精确覆盖「写失败重连重试」分支），broken
     * 同时断两侧（通道死亡语义）。
     */
    private static final class FakeTransport implements McpTransport {
        final LinkedBlockingQueue<String> incoming = new LinkedBlockingQueue<>();
        /** 下一条响应的 result 载荷（按脚本逐条出队）。 */
        final java.util.Deque<String> results = new java.util.concurrent.ConcurrentLinkedDeque<>();
        final List<String> sent = new ArrayList<>();
        volatile boolean broken = false;
        volatile boolean sendBroken = false;

        @Override
        public void start() {
        }

        @Override
        public void sendLine(String line) throws IOException {
            if (broken || sendBroken) {
                throw new IOException("pipe broken");
            }
            sent.add(line);
            JsonNode node = MAPPER.readTree(line);
            JsonNode id = node.get("id");
            if (id != null && id.canConvertToInt()) {
                ObjectNode reply = MAPPER.createObjectNode();
                reply.put("jsonrpc", "2.0");
                reply.set("id", id);
                String next = results.poll();
                reply.set("result", MAPPER.readTree(next == null ? "{}" : next));
                incoming.add(MAPPER.writeValueAsString(reply));
            }
        }

        @Override
        public String readLine() throws IOException {
            while (true) {
                if (broken) {
                    return null; // 通道关闭：读侧返回 null → 读线程退出
                }
                String line;
                try {
                    line = incoming.poll(50, TimeUnit.MILLISECONDS);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw new IOException("interrupted while waiting for reply", e);
                }
                if (line != null) {
                    return line;
                }
            }
        }

        @Override
        public void close() {
            broken = true;
        }
    }

    /** 断开 helper：broken + 一条空行唤醒读循环看到 broken（读侧退出）。 */
    private static void breakChannel(FakeTransport transport) {
        transport.broken = true;
        transport.incoming.add("");
    }

    @Test
    void connectedReflectsChannelDeathAndLazyReconnectRecovers() throws Exception {
        List<FakeTransport> transports = new ArrayList<>();
        AtomicInteger created = new AtomicInteger();
        McpClient client = new McpClient("fake", () -> {
            created.incrementAndGet();
            FakeTransport t = new FakeTransport();
            t.results.add("{\"protocolVersion\":\"" + McpClient.PROTOCOL_VERSION + "\"}");
            t.results.add("{\"tools\":[]}");
            transports.add(t);
            return t;
        });
        client.connect("agent-base-test", "1", 5.0);
        assertTrue(client.connected());
        assertEquals(1, created.get());

        // 通道死亡：读线程退出 + alive() false → connected() 翻转。
        FakeTransport first = transports.get(0);
        breakChannel(first);
        long deadline = System.currentTimeMillis() + 2000;
        while (client.connected() && System.currentTimeMillis() < deadline) {
            Thread.sleep(20);
        }
        assertFalse(client.connected());

        // 下一次请求经预检惰性重连：新通道 + 重新握手，调用自愈。
        ObjectNode result = client.request("tools/list", null, 5.0);
        assertNotNull(result);
        assertTrue(result.path("tools").isArray());
        assertTrue(client.connected());
        assertEquals(2, created.get());
        client.close();
        assertFalse(client.connected());
    }

    @Test
    void writeFailureReconnectsAndRetriesOnce() throws Exception {
        List<FakeTransport> transports = new ArrayList<>();
        McpClient client = new McpClient("fake", () -> {
            FakeTransport t = new FakeTransport();
            t.results.add("{}"); // 重连握手的 initialize 应答（tools/list 用不到）
            t.results.add("{\"ok\":true}");
            transports.add(t);
            return t;
        });
        client.connect("agent-base-test", "1", 5.0);
        // 写侧先坏（读线程还活着）：sendLine 抛 IOException → catch 分支
        // 重连重试一次（connected() 仍为 true，预检不触发）。
        transports.get(0).sendBroken = true;
        ObjectNode result = client.request("tools/list", null, 5.0);
        assertEquals(true, result.path("ok").asBoolean());
        assertEquals(2, transports.size());
        client.close();
    }

    @Test
    void reconnectFailureKeepsClientRetryableWithoutClosing() throws Exception {
        AtomicInteger created = new AtomicInteger();
        McpClient client = new McpClient("fake", () -> {
            if (created.incrementAndGet() == 1) {
                FakeTransport t = new FakeTransport();
                t.results.add("{\"protocolVersion\":\"" + McpClient.PROTOCOL_VERSION + "\"}");
                return t;
            }
            throw new IOException("server process gone for good");
        });
        client.connect("agent-base-test", "1", 5.0);
        breakChannel((FakeTransport) clientTransport(client));
        long deadline = System.currentTimeMillis() + 2000;
        while (client.connected() && System.currentTimeMillis() < deadline) {
            Thread.sleep(20);
        }
        // 重连失败：请求失败（原始异常语义），但客户端未被置 closed——
        // 下一次调用仍会再试（运行期断开是常态，不是配置错误）。
        McpException e = assertThrows(McpException.class,
                () -> client.request("tools/list", null, 5.0));
        assertFalse(e.getMessage().contains("客户端已关闭"));
        assertFalse(clientIsClosed(client));
        client.close();
    }

    @Test
    void singleTransportFormHasNoReconnect() throws Exception {
        FakeTransport transport = new FakeTransport();
        transport.results.add("{\"protocolVersion\":\"" + McpClient.PROTOCOL_VERSION + "\"}");
        McpClient client = new McpClient("fake", transport);
        client.connect("agent-base-test", "1", 5.0);
        assertTrue(client.connected());
        breakChannel(transport);
        long deadline = System.currentTimeMillis() + 2000;
        while (client.connected() && System.currentTimeMillis() < deadline) {
            Thread.sleep(20);
        }
        // 单通道形态无重连能力（工厂为 null）：断开即请求失败。
        assertThrows(McpException.class, () -> client.request("tools/list", null, 5.0));
        client.close();
    }

    // ── 反射观察面 ────────────────────────────────────────────────────────

    private static McpTransport clientTransport(McpClient client) throws Exception {
        var field = McpClient.class.getDeclaredField("transport");
        field.setAccessible(true);
        return (McpTransport) field.get(client);
    }

    private static boolean clientIsClosed(McpClient client) throws Exception {
        var field = McpClient.class.getDeclaredField("closed");
        field.setAccessible(true);
        return field.getBoolean(client);
    }
}
