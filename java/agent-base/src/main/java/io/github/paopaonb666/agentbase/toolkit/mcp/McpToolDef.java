package io.github.paopaonb666.agentbase.toolkit.mcp;

import com.fasterxml.jackson.databind.JsonNode;

/**
 * server 的 tools/list 返回的单个工具定义。
 *
 * @param name        工具原始名（{@code tools/call} 用它，不含前缀）
 * @param description 工具描述（可能为空串）
 * @param inputSchema 参数 JSON Schema（可能为空对象）
 */
public record McpToolDef(String name, String description, JsonNode inputSchema) {
}
