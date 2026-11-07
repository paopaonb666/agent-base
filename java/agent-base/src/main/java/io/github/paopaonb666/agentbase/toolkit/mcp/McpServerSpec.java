package io.github.paopaonb666.agentbase.toolkit.mcp;

import java.util.List;
import java.util.Map;

/**
 * 单个 MCP server 的连接配置（{@code MCP_SERVERS_JSON} 的一个条目）。
 *
 * @param name           server 名（会成为工具名前缀，小写标识符白名单）
 * @param transport      连接形态：stdio / streamable_http / sse / websocket
 * @param command        stdio：可执行命令
 * @param args           stdio：命令参数
 * @param env            stdio：附加环境变量（覆盖继承的父进程环境）
 * @param url            HTTP 系 transport：端点 URL
 * @param timeoutSeconds 可选 per-server 超时（秒）；null 沿用全局工具预算
 */
public record McpServerSpec(String name, String transport, String command, List<String> args,
                            Map<String, String> env, String url, Double timeoutSeconds) {
}
