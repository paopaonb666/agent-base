package io.github.paopaonb666.agentbase.toolkit.mcp;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.core.graph.RunConfig;
import io.github.paopaonb666.agentbase.core.tools.AgentTool;
import io.github.paopaonb666.agentbase.core.tools.ToolPool;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
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
    void unimplementedHttpTransportFailsWithGuidance() {
        Settings settings = enabledSettings(
                "{\"remote\": {\"transport\": \"streamable_http\", \"url\": \"http://localhost:9/mcp\"}}");
        McpException e = assertThrows(McpException.class, () -> McpSources.load(settings));
        assertTrue(e.getMessage().contains("本期未实现"));
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
}
