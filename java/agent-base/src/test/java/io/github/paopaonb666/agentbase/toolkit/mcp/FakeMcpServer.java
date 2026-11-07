package io.github.paopaonb666.agentbase.toolkit.mcp;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;

/**
 * 测试夹具：最小 stdio MCP server（每行一条 JSON-RPC）。
 *
 * <p>支持 {@code --tool-name <n>} 自报工具名、{@code --error-call} 让
 * tools/call 返回 isError、{@code --exit-immediately} 模拟秒退的坏
 * server。仅服务 #331/#333 的测试，非生产代码。
 */
public final class FakeMcpServer {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    public static void main(String[] args) throws Exception {
        String toolName = "echo";
        boolean errorCall = false;
        boolean exitImmediately = false;
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--tool-name" -> toolName = args[++i];
                case "--error-call" -> errorCall = true;
                case "--exit-immediately" -> exitImmediately = true;
                default -> {
                    // 未知旗标忽略
                }
            }
        }
        if (exitImmediately) {
            System.exit(7);
        }
        BufferedReader reader = new BufferedReader(
                new InputStreamReader(System.in, StandardCharsets.UTF_8));
        // System.out 在 Windows 上默认走本地码页（GBK/cp1252），显式转 UTF-8
        PrintStream out = new PrintStream(
                new java.io.FileOutputStream(java.io.FileDescriptor.out),
                true, StandardCharsets.UTF_8);
        String line;
        while ((line = reader.readLine()) != null) {
            JsonNode message;
            try {
                message = MAPPER.readTree(line);
            } catch (Exception e) {
                continue; // 非 JSON 行静默丢弃
            }
            if (!message.has("id")) {
                continue; // 通知（notifications/initialized）不回包
            }
            respond(out, message, toolName, errorCall);
        }
    }

    private static void respond(PrintStream out, JsonNode message, String toolName,
                                boolean errorCall) throws Exception {
        String method = message.path("method").asText();
        ObjectNode response = MAPPER.createObjectNode();
        response.put("jsonrpc", "2.0");
        response.set("id", message.get("id"));
        ObjectNode result = MAPPER.createObjectNode();
        switch (method) {
            case "initialize" -> {
                result.put("protocolVersion", "2024-11-05");
                result.putObject("capabilities");
                result.putObject("serverInfo").put("name", "fake").put("version", "0");
            }
            case "tools/list" -> {
                ObjectNode tool = MAPPER.createObjectNode();
                tool.put("name", toolName);
                tool.put("description", "Echo the query back（测试夹具）");
                tool.set("inputSchema", MAPPER.readTree(
                        "{\"type\":\"object\",\"properties\":{\"query\":"
                                + "{\"type\":\"string\",\"description\":\"要回显的文本\"}},"
                                + "\"required\":[\"query\"]}"));
                result.putArray("tools").add(tool);
            }
            case "tools/call" -> {
                String query = message.path("params").path("arguments")
                        .path("query").asText("");
                result.put("isError", errorCall);
                ObjectNode text = MAPPER.createObjectNode();
                text.put("type", "text");
                text.put("text", "echo:" + query);
                result.putArray("content").add(text);
            }
            default -> {
                response = MAPPER.createObjectNode();
                response.put("jsonrpc", "2.0");
                response.set("id", message.get("id"));
                ObjectNode error = MAPPER.createObjectNode();
                error.put("code", -32601);
                error.put("message", "method not found: " + method);
                response.set("error", error);
                out.println(response);
                out.flush();
                return;
            }
        }
        response.set("result", result);
        out.println(response);
        out.flush();
    }
}
