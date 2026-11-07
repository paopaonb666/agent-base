package io.github.paopaonb666.agentbase.toolkit.mcp;

import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.core.tools.AgentTool;
import io.github.paopaonb666.agentbase.core.tools.ToolPool;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * MCP 工具来源装配入口（对应 Python {@code tools/mcp.py} 的
 * {@code load_mcp_tools}，#331/#333）。
 *
 * <p>语义对齐工具库的「启用即承诺」：
 *
 * <ul>
 *   <li>{@code MCP_ENABLED=false}（默认）：零解析、零连接——返回空装配，
 *       调用方可以无条件并入；</li>
 *   <li>{@code MCP_ENABLED=true} 是承诺：{@code MCP_SERVERS_JSON} 必须是
 *       合法的非空配置、每个 server 必须能连上，否则启动即
 *       {@link McpException}（带修复指引）；绝不静默缺席。</li>
 * </ul>
 *
 * <p>工具经 {@link McpTool} 适配后并入共享池：超时包装、审计收口与重名
 * 快速失败自动继承。per-server 超时（{@code timeout_seconds}）经
 * {@link McpAssembly#timeouts()} 下发到共享池。
 */
public final class McpSources {

    private static final Logger LOG = LoggerFactory.getLogger(McpSources.class);

    /** 握手与 tools/list 的默认预算（server 冷启动可能含包下载，给足）。 */
    private static final double HANDSHAKE_TIMEOUT_SECONDS = 30.0;

    private static final String CLIENT_NAME = "agent-base";
    private static final String CLIENT_VERSION = "1.0";

    private McpSources() {
    }

    /** 一次装配的产物：工具、per-tool 超时表与关闭句柄。 */
    public record McpAssembly(List<AgentTool> tools, Map<String, Double> timeouts)
            implements AutoCloseable {

        public static final McpAssembly EMPTY =
                new McpAssembly(List.of(), Map.of());

        /** 关闭全部 server 客户端（幂等；重复调用无副作用）。 */
        @Override
        public void close() {
            for (AgentTool tool : tools) {
                if (tool instanceof McpTool mcpTool) {
                    try {
                        mcpTool.close();
                    } catch (Exception e) {
                        LOG.debug("MCP tool {} close failed", tool.name(), e);
                    }
                }
            }
        }
    }

    /** 按配置装配 MCP 工具；未启用返回空装配，失败抛 {@link McpException}。 */
    public static McpAssembly load(Settings settings) {
        if (!settings.mcp.enabled) {
            return McpAssembly.EMPTY;
        }
        Map<String, McpServerSpec> specs = McpServerSpecs.parse(settings.mcp.serversJson);
        List<AgentTool> tools = new ArrayList<>();
        Map<String, Double> timeouts = new LinkedHashMap<>();
        List<McpClient> opened = new ArrayList<>();
        try {
            for (McpServerSpec spec : specs.values()) {
                McpClient client = connect(spec);
                opened.add(client);
                double budget = spec.timeoutSeconds() == null
                        ? ToolPool.DEFAULT_TOOL_TIMEOUT_SECONDS : spec.timeoutSeconds();
                for (McpToolDef def : client.listTools(HANDSHAKE_TIMEOUT_SECONDS)) {
                    McpTool tool = new McpTool(spec.name(), def, client, budget);
                    LOG.info("MCP 工具接入：{}（server {}）", tool.prefixedName(), spec.name());
                    tools.add(tool);
                    if (spec.timeoutSeconds() != null) {
                        timeouts.put(tool.prefixedName(), spec.timeoutSeconds());
                    }
                }
            }
        } catch (Exception e) {
            for (McpClient client : opened) {
                try {
                    client.close();
                } catch (Exception closeError) {
                    LOG.debug("MCP client close failed during rollback", closeError);
                }
            }
            if (e instanceof McpException mcpError) {
                throw mcpError;
            }
            throw new McpException(
                    "MCP server 连接失败（server 进程起来了吗？）："
                            + e.getClass().getSimpleName() + ": " + e.getMessage(), e);
        }
        return new McpAssembly(List.copyOf(tools), Map.copyOf(timeouts));
    }

    private static McpClient connect(McpServerSpec spec) {
        McpTransport transport = openTransport(spec);
        McpClient client = new McpClient(spec.name(), transport);
        double handshake = spec.timeoutSeconds() == null
                ? HANDSHAKE_TIMEOUT_SECONDS : spec.timeoutSeconds();
        client.connect(CLIENT_NAME, CLIENT_VERSION, handshake);
        return client;
    }

    /** transport 分发：本期 stdio 先行（#331），HTTP 系随 #332 落地。 */
    private static McpTransport openTransport(McpServerSpec spec) {
        switch (spec.transport()) {
            case "stdio":
                return new McpStdioTransport(spec.name(), spec.command(), spec.args(),
                        spec.env());
            case "streamable_http", "sse", "websocket":
                throw new McpException(
                        "MCP server '" + spec.name() + "'：transport '" + spec.transport()
                                + "' 本期未实现（Java 版先支持 stdio）；"
                                + "先用 stdio 形态，HTTP 支持在后续版本跟进。");
            default:
                throw new McpException("MCP server '" + spec.name() + "'：未知 transport '"
                        + spec.transport() + "'");
        }
    }
}
