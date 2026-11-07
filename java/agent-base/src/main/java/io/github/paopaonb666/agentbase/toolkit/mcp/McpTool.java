package io.github.paopaonb666.agentbase.toolkit.mcp;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import dev.langchain4j.agent.tool.ToolSpecification;
import io.github.paopaonb666.agentbase.core.tools.AgentTool;

/**
 * MCP server 工具 → {@link AgentTool} 适配器（#333）：工具名加 server
 * 前缀（{@code {server}_{tool}}）并入共享池——超时包装、审计收口与重名
 * 快速失败全部由池继承，不另开旁路（对齐 Python 侧
 * {@code tool_name_prefix=True} 契约）。
 *
 * <p>实现 {@link AutoCloseable}：{@code close()} 关掉所属 server 的
 * 客户端（同 server 的多个工具共享一个 {@link McpClient}，重复关闭幂等）；
 * 运行时 close 经池扫描收敛，server 子进程随运行时退出。
 */
public final class McpTool implements AgentTool, AutoCloseable {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final String prefixedName;
    private final String description;
    private final ToolSpecification spec;
    private final McpClient client;
    private final String rawName;
    private final double timeoutSeconds;

    McpTool(String serverName, McpToolDef def, McpClient client, double timeoutSeconds) {
        this.prefixedName = serverName + "_" + def.name();
        this.description = def.description();
        this.spec = ToolSpecification.builder()
                .name(prefixedName)
                .description(description)
                .parameters(McpSchemas.toObjectSchema(def.inputSchema()))
                .build();
        this.client = client;
        this.rawName = def.name();
        this.timeoutSeconds = timeoutSeconds;
    }

    /** 装配期给 AgentRuntime 复核重名用。 */
    String prefixedName() {
        return prefixedName;
    }

    @Override
    public String name() {
        return prefixedName;
    }

    @Override
    public String description() {
        return description;
    }

    @Override
    public ToolSpecification specification() {
        return spec;
    }

    @Override
    public String call(String argsJson, ToolContext ctx) throws Exception {
        JsonNode args;
        try {
            args = argsJson == null || argsJson.strip().isEmpty()
                    ? MAPPER.createObjectNode() : MAPPER.readTree(argsJson);
        } catch (Exception e) {
            throw new IllegalArgumentException(
                    "MCP 工具参数不是合法 JSON：" + e.getMessage());
        }
        if (args != null && !args.isObject()) {
            throw new IllegalArgumentException("MCP 工具参数必须是 JSON 对象");
        }
        // 内层预算比池的挂钟预算多 2s 余量：外层先到点、报统一的超时文案。
        return client.callTool(rawName, args, timeoutSeconds + 2.0);
    }

    @Override
    public void close() {
        client.close();
    }
}
