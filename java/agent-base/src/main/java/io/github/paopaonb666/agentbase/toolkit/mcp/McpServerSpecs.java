package io.github.paopaonb666.agentbase.toolkit.mcp;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * {@code MCP_SERVERS_JSON} 的解析与结构校验（对应 Python
 * {@code tools/mcp.py} 的 {@code _parse_servers}）。
 *
 * <p>只做结构性校验（名字白名单、transport 与 command/url 在位）；连接级
 * 参数由 transport 在连接时校验。错误一律 {@link McpException}（启动即
 * 承诺：配置非法拒绝启动），文案与 Python 侧同构。
 */
public final class McpServerSpecs {

    /** server 名会成为工具名前缀，与模块名同样严格（白名单，反注入）。 */
    public static final String NAME_PATTERN = "^[a-z][a-z0-9_]*$";

    /** 已知 transport 字面量（与 Python 契约一致；本期实现 stdio+streamable_http）。 */
    public static final List<String> KNOWN_TRANSPORTS =
            List.of("stdio", "streamable_http", "sse", "websocket");

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private McpServerSpecs() {
    }

    /** 解析并校验 servers 清单；非法即抛 {@link McpException}。 */
    public static Map<String, McpServerSpec> parse(String serversJson) {
        String text = serversJson == null ? "" : serversJson.strip();
        if (text.isEmpty()) {
            throw new McpException(
                    "MCP_ENABLED=true 但 MCP_SERVERS_JSON 为空；示例："
                            + "{\"docs\": {\"transport\": \"stdio\", \"command\": \"uvx\","
                            + " \"args\": [\"mcp-server-fetch\"]}}");
        }
        JsonNode data;
        try {
            data = MAPPER.readTree(text);
        } catch (Exception e) {
            throw new McpException("MCP_SERVERS_JSON 不是合法 JSON：" + e.getMessage(), e);
        }
        if (!data.isObject() || data.isEmpty()) {
            throw new McpException(
                    "MCP_SERVERS_JSON 必须是非空对象：{\"server-name\": {...}}");
        }
        Map<String, McpServerSpec> out = new LinkedHashMap<>();
        for (Map.Entry<String, JsonNode> field : data.properties()) {
            String name = field.getKey();
            JsonNode conn = field.getValue();
            if (!name.matches(NAME_PATTERN)) {
                throw new McpException(
                        "MCP server 名 '" + name + "' 非法：须匹配 " + NAME_PATTERN
                                + "（它会成为工具名前缀）");
            }
            if (!conn.isObject()) {
                throw new McpException("MCP server '" + name + "' 的配置必须是对象");
            }
            JsonNode transportNode = conn.get("transport");
            String transport = transportNode == null || transportNode.isNull()
                    ? null : transportNode.asText();
            if (transport == null || !KNOWN_TRANSPORTS.contains(transport)) {
                throw new McpException(
                        "MCP server '" + name + "' 的 transport " + repr(transport)
                                + " 不支持；expected one of " + KNOWN_TRANSPORTS);
            }
            if ("stdio".equals(transport)
                    && (conn.get("command") == null || !conn.get("command").isTextual()
                        || conn.get("command").asText().isEmpty())) {
                throw new McpException(
                        "MCP server '" + name + "'：transport \"stdio\" 需要 \"command\" 字段");
            }
            if (!"stdio".equals(transport)
                    && (conn.get("url") == null || !conn.get("url").isTextual()
                        || conn.get("url").asText().isEmpty())) {
                throw new McpException(
                        "MCP server '" + name + "'：transport '" + transport + "' 需要 \"url\" 字段");
            }
            Double timeout = null;
            JsonNode timeoutNode = conn.get("timeout_seconds");
            if (timeoutNode != null && !timeoutNode.isNull()) {
                if (!timeoutNode.isNumber() || timeoutNode.doubleValue() <= 0) {
                    throw new McpException(
                            "MCP server '" + name + "' 的 timeout_seconds 必须是正数");
                }
                timeout = timeoutNode.doubleValue();
            }
            out.put(name, new McpServerSpec(
                    name,
                    transport,
                    conn.path("command").asText(null),
                    stringList(conn.get("args"), name),
                    stringMap(conn.get("env")),
                    conn.path("url").asText(null),
                    timeout));
        }
        return out;
    }

    private static List<String> stringList(JsonNode node, String server) {
        if (node == null || node.isNull()) {
            return List.of();
        }
        if (!node.isArray()) {
            throw new McpException("MCP server '" + server + "' 的 args 必须是数组");
        }
        List<String> out = new ArrayList<>();
        for (JsonNode item : node) {
            if (!item.isTextual()) {
                throw new McpException("MCP server '" + server + "' 的 args 必须全是字符串");
            }
            out.add(item.asText());
        }
        return out;
    }

    private static Map<String, String> stringMap(JsonNode node) {
        if (node == null || node.isNull() || !node.isObject()) {
            return Map.of();
        }
        Map<String, String> out = new LinkedHashMap<>();
        for (Map.Entry<String, JsonNode> field : node.properties()) {
            out.put(field.getKey(), field.getValue().asText());
        }
        return out;
    }

    private static String repr(String value) {
        return value == null ? "None" : "'" + value + "'";
    }
}
