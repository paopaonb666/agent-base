package io.github.paopaonb666.agentbase.toolkit.mcp;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.core.graph.RunConfig;
import io.github.paopaonb666.agentbase.core.tools.AgentTool;
import io.github.paopaonb666.agentbase.core.tools.ToolPool;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * MCP 装配端到端（#331/#333）：用 Java 子进程跑 {@link FakeMcpServer}，
 * 走真实的 ProcessBuilder + 行式 JSON-RPC + initialize 握手 + 工具入池。
 */
class McpSourcesTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private static String javaCommand() {
        return ProcessHandle.current().info().command()
                .or(() -> Optional.ofNullable(System.getenv("JAVA_HOME"))
                        .map(home -> home + File.separator + "bin" + File.separator + "java"))
                .orElse("java");
    }

    /** 命令/参数里反斜杠的 JSON 转义（Windows 路径）。 */
    private static String jsonEscape(String value) {
        return value.replace("\\", "\\\\");
    }

    /** 与测试 JVM 同 classpath 拉起 FakeMcpServer 的单 server 配置。 */
    private static String serverJson(String serverName, String toolName,
                                     String... extraFlags) {
        StringBuilder args = new StringBuilder();
        args.append("\"-cp\", \"").append(jsonEscape(System.getProperty("java.class.path")))
                .append("\", \"").append(FakeMcpServer.class.getName()).append("\"");
        if (!"__none__".equals(toolName)) {
            args.append(", \"--tool-name\", \"").append(jsonEscape(toolName)).append("\"");
        }
        for (String flag : extraFlags) {
            args.append(", \"").append(jsonEscape(flag)).append("\"");
        }
        return "{\"" + serverName + "\": {\"transport\": \"stdio\", \"command\": \""
                + jsonEscape(javaCommand()) + "\", \"args\": [" + args + "]}}";
    }

    private static Settings enabledSettings(String serversJson) {
        return Settings.load(Map.of(
                "MCP_ENABLED", "true",
                "MCP_SERVERS_JSON", serversJson), null);
    }

    @Test
    void disabledReturnsEmptyAssembly() {
        Settings settings = Settings.load(Map.of("MCP_ENABLED", "false"), null);
        McpSources.McpAssembly assembly = McpSources.load(settings);
        assertEquals(0, assembly.tools().size());
        assertEquals(0, assembly.timeouts().size());
        assembly.close(); // 空装配关闭无副作用
    }

    @Test
    void enabledWithEmptyServersFailsFast() {
        Settings settings = enabledSettings("");
        McpException e = assertThrows(McpException.class, () -> McpSources.load(settings));
        assertTrue(e.getMessage().contains("MCP_SERVERS_JSON 为空"));
    }

    @Test
    void websocketTransportFailsWithGuidance() {
        Settings settings = enabledSettings(
                "{\"remote\": {\"transport\": \"websocket\", \"url\": \"ws://localhost:9/mcp\"}}");
        McpException e = assertThrows(McpException.class, () -> McpSources.load(settings));
        assertTrue(e.getMessage().contains("websocket' 本期未实现"));
    }

    @Test
    void immediateExitServerFailsAtHandshake() {
        Settings settings = enabledSettings(serverJson("dead", "echo", "--exit-immediately"));
        McpException e = assertThrows(McpException.class, () -> McpSources.load(settings));
        assertTrue(e.getMessage().contains("dead"));
    }

    @Test
    void stdioServerToolsJoinPoolWithPrefix() throws Exception {
        Settings settings = enabledSettings(serverJson("fake", "echo"));
        McpSources.McpAssembly assembly = McpSources.load(settings);
        try {
            assertEquals(List.of("fake_echo"), assembly.tools().stream()
                    .map(AgentTool::name).toList());
            assertEquals(Map.of(), assembly.timeouts()); // 未配 per-server 超时

            ToolPool pool = ToolPool.build(List.of(), assembly.tools(), 30.0,
                    assembly.timeouts(), null);
            // 经池执行（超时包装 + 审计收口路径）：参数走 server 声明的 schema
            String result = pool.call("fake_echo",
                    MAPPER.writeValueAsString(Map.of("query", "你好 world")),
                    RunConfig.of("chat:t1", "u1"), null);
            assertEquals("echo:你好 world", result);

            // specification 面：名称前缀与描述来自 server
            assertEquals("fake_echo", pool.specifications().get(0).name());
            assertEquals("Echo the query back（测试夹具）",
                    pool.specifications().get(0).description());
        } finally {
            assembly.close();
        }
    }

    @Test
    void serverErrorCallNormalizesThroughPool() throws Exception {
        Settings settings = enabledSettings(serverJson("bad", "echo", "--error-call"));
        McpSources.McpAssembly assembly = McpSources.load(settings);
        try {
            ToolPool pool = ToolPool.build(List.of(), assembly.tools(), 30.0,
                    assembly.timeouts(), null);
            String result = pool.call("bad_echo",
                    MAPPER.writeValueAsString(Map.of("query", "x")),
                    RunConfig.of("chat:t1", "u1"), null);
            assertTrue(result.startsWith("tool execution failed:"),
                    "isError 响应应经池归一：" + result);
        } finally {
            assembly.close();
        }
    }

    @Test
    void perServerTimeoutFeedsPoolTimeoutMap() {
        String json = "{\"slow\": {\"transport\": \"stdio\", \"command\": \""
                + jsonEscape(javaCommand()) + "\", \"args\": [\"-cp\", \""
                + jsonEscape(System.getProperty("java.class.path")) + "\", \""
                + FakeMcpServer.class.getName() + "\"], \"timeout_seconds\": 7.5}}";
        Settings settings = enabledSettings(json);
        McpSources.McpAssembly assembly = McpSources.load(settings);
        try {
            assertEquals(Map.of("slow_echo", 7.5), assembly.timeouts());
        } finally {
            assembly.close();
        }
    }

    @Test
    void crossServerPrefixedNameCollisionFailsFastInPool() {
        // server "a" 的工具 "x_y" 与 server "a_x" 的工具 "y" 前缀化后同为
        // a_x_y：池的重名快速失败必须兜住（#333 与基座同规）。
        Settings settings = enabledSettings(
                serverJson("a", "x_y").substring(0, serverJson("a", "x_y").length() - 1)
                        + ", " + serverJson("a_x", "y").substring(1));
        McpSources.McpAssembly assembly = McpSources.load(settings);
        try {
            ToolPool.ToolPoolError e = assertThrows(ToolPool.ToolPoolError.class,
                    () -> ToolPool.build(List.of(), assembly.tools(), 30.0, Map.of(), null));
            assertTrue(e.getMessage().contains("duplicate tool name 'a_x_y'"));
        } finally {
            assembly.close();
        }
    }

    // ── #332：HTTP 系 transport（JDK HttpServer 夹具端到端） ─────────

    /** 拉起 FakeHttpMcpServer 子进程，返回「配置 JSON + 关闭句柄」。 */
    private static HttpFixture httpFixture(String mode) throws Exception {
        int port;
        try (java.net.ServerSocket socket = new java.net.ServerSocket(0)) {
            port = socket.getLocalPort();
        }
        String javaBin = jsonEscape(javaCommand());
        String classpath = jsonEscape(System.getProperty("java.class.path"));
        String[] command = {javaBin, "-cp", System.getProperty("java.class.path"),
                FakeHttpMcpServer.class.getName(), mode, String.valueOf(port)};
        Process process = new ProcessBuilder(command).start();
        // 等端口就绪（最多 10s）
        long deadline = System.currentTimeMillis() + 10_000;
        boolean ready = false;
        while (System.currentTimeMillis() < deadline && process.isAlive()) {
            try (java.net.Socket probe = new java.net.Socket()) {
                probe.connect(new java.net.InetSocketAddress("127.0.0.1", port), 200);
                ready = true;
                break;
            } catch (IOException probeFailure) {
                Thread.sleep(50);
            }
        }
        if (!ready) {
            process.destroyForcibly();
            throw new IllegalStateException("FakeHttpMcpServer 未能在预算内就绪");
        }
        String url = "http://127.0.0.1:" + port
                + ("streamable".equals(mode) ? "/mcp" : "/sse");
        return new HttpFixture("{\"http\": {\"transport\": \"" + ("streamable".equals(mode)
                ? "streamable_http" : "sse") + "\", \"url\": \"" + url + "\"}}", process);
    }

    private record HttpFixture(String serversJson, Process process) {
        void close() {
            process.destroyForcibly();
        }
    }

    @Test
    void streamableHttpEndToEnd() throws Exception {
        HttpFixture fixture = httpFixture("streamable");
        try {
            McpSources.McpAssembly asm = McpSources.load(enabledSettings(fixture.serversJson()));
            try {
                assertEquals(List.of("http_echo"), asm.tools().stream()
                        .map(AgentTool::name).toList());
                ToolPool pool = ToolPool.build(List.of(), asm.tools(), 30.0,
                        asm.timeouts(), null);
                String result = pool.call("http_echo",
                        MAPPER.writeValueAsString(Map.of("query", "http 你好")),
                        RunConfig.of("chat:t1", "u1"), null);
                assertEquals("echo:http 你好", result);
            } finally {
                asm.close();
            }
        } finally {
            fixture.close();
        }
    }

    @Test
    void assemblyHealthReportsServerConnectivity() throws Exception {
        // #29 健康可见：装配携带 server 健康面——存续=ok，close 后=down。
        Settings settings = enabledSettings(serverJson("fake", "echo"));
        McpSources.McpAssembly assembly = McpSources.load(settings);
        try {
            assertEquals(Map.of("fake", true), assembly.health());
            assertEquals(1, assembly.servers().size());
        } finally {
            assembly.close();
        }
        assertEquals(Map.of("fake", false), assembly.health());
    }

    @Test
    void stdioServerReconnectsAfterProcessDeath() throws Exception {
        // #29 重连最小面：server 子进程被杀后，下一次工具调用经惰性重连
        // 自愈（工厂产新子进程 + 重新握手）——无需重启进程、无需重建池。
        Settings settings = enabledSettings(serverJson("fake", "echo"));
        McpSources.McpAssembly assembly = McpSources.load(settings);
        try {
            McpClient client = assembly.servers().get("fake");
            assertTrue(client.connected());
            // 杀掉 server 子进程（stdio transport 的 process 句柄，包内可见）。
            McpStdioTransport transport = transportOf(client);
            Process process = transport.process();
            assertNotNull(process);
            process.destroy();
            process.waitFor(10, java.util.concurrent.TimeUnit.SECONDS);
            long deadline = System.currentTimeMillis() + 5000;
            while (client.connected() && System.currentTimeMillis() < deadline) {
                Thread.sleep(50);
            }
            // 进程死亡后 connected() 翻转（读线程退出或 alive() 判负）。
            boolean reconnected = false;
            deadline = System.currentTimeMillis() + 15000;
            while (System.currentTimeMillis() < deadline) {
                try {
                    client.listTools(3.0);
                    reconnected = true;
                    break;
                } catch (McpException e) {
                    Thread.sleep(100);
                }
            }
            assertTrue(reconnected, "惰性重连应恢复工具列表");
            assertTrue(client.connected(), "重连后必须恢复连接");
            assertEquals("echo:重生",
                    client.callTool("echo", MAPPER.readTree("{\"query\":\"重生\"}"), 5.0));
        } finally {
            assembly.close();
        }
    }

    private static McpStdioTransport transportOf(McpClient client) throws Exception {
        var field = McpClient.class.getDeclaredField("transport");
        field.setAccessible(true);
        return (McpStdioTransport) field.get(client);
    }

    @Test
    void sseEndToEnd() throws Exception {
        HttpFixture fixture = httpFixture("sse");
        try {
            McpSources.McpAssembly asm = McpSources.load(enabledSettings(fixture.serversJson()));
            try {
                assertEquals(List.of("http_echo"), asm.tools().stream()
                        .map(AgentTool::name).toList());
                ToolPool pool = ToolPool.build(List.of(), asm.tools(), 30.0,
                        asm.timeouts(), null);
                String result = pool.call("http_echo",
                        MAPPER.writeValueAsString(Map.of("query", "sse 你好")),
                        RunConfig.of("chat:t1", "u1"), null);
                assertEquals("echo:sse 你好", result);
            } finally {
                asm.close();
            }
        } finally {
            fixture.close();
        }
    }
}
