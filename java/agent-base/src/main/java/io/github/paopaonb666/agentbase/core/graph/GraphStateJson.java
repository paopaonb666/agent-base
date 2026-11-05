package io.github.paopaonb666.agentbase.core.graph;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.output.TokenUsage;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 图状态的 JSON 编解码（checkpointer 持久化的存储格式）。
 *
 * <p>消息形态与 OpenAI/LangChain 语义对齐：type ∈ {system, human, ai,
 * tool}；AiMessage 附 tool_calls（[{id,name,args}]）；ToolMessage 附
 * tool_call_id/name；UserMessage 可能带多段 content（文本 + 图片块）。
 * extra 通道（tasks/cursor/replans 等）原样 JSON 保存。
 */
public final class GraphStateJson {

    private GraphStateJson() {}

    private static final ObjectMapper MAPPER = new ObjectMapper();

    /** 序列化图状态为 JSON 字符串。 */
    public static String toJson(GraphState state) {
        ObjectNode root = MAPPER.createObjectNode();
        ArrayNode messages = root.putArray("messages");
        for (ChatMessage message : state.messages()) {
            messages.add(messageToJson(message));
        }
        ObjectNode extra = root.putObject("extra");
        for (Map.Entry<String, Object> e : state.extra().entrySet()) {
            extra.set(e.getKey(), MAPPER.valueToTree(e.getValue()));
        }
        return root.toString();
    }

    /** 反序列化；损坏的 JSON 返回 null（旧行当作缺失处理）。 */
    public static GraphState fromJson(String json) {
        if (json == null || json.isEmpty()) {
            return null;
        }
        try {
            JsonNode root = MAPPER.readTree(json);
            List<ChatMessage> messages = new ArrayList<>();
            if (root.has("messages") && root.get("messages").isArray()) {
                for (JsonNode node : root.get("messages")) {
                    ChatMessage message = messageFromJson(node);
                    if (message != null) {
                        messages.add(message);
                    }
                }
            }
            Map<String, Object> extra = new LinkedHashMap<>();
            if (root.has("extra") && root.get("extra").isObject()) {
                extra.putAll(MAPPER.convertValue(root.get("extra"),
                        new com.fasterxml.jackson.core.type.TypeReference<Map<String, Object>>() {}));
            }
            return new GraphState(messages, extra);
        } catch (JsonProcessingException e) {
            return null;
        }
    }

    /** 单条消息 → JSON 节点。 */
    public static ObjectNode messageToJson(ChatMessage message) {
        ObjectNode node = MAPPER.createObjectNode();
        if (message instanceof SystemMessage system) {
            node.put("type", "system");
            node.put("content", system.text());
        } else if (message instanceof UserMessage user) {
            node.put("type", "human");
            if (user.contents().size() == 1 && user.contents().get(0) instanceof dev.langchain4j.data.message.TextContent text) {
                node.put("content", text.text());
            } else {
                // 多段 content（多模态附件）：文本段 + 图片块（base64/mime）。
                ArrayNode parts = node.putArray("content_parts");
                for (dev.langchain4j.data.message.Content content : user.contents()) {
                    ObjectNode part = parts.addObject();
                    if (content instanceof dev.langchain4j.data.message.TextContent text) {
                        part.put("type", "text");
                        part.put("text", text.text());
                    } else if (content instanceof dev.langchain4j.data.message.ImageContent image) {
                        part.put("type", "image");
                        part.put("mime_type", image.image().mimeType());
                        part.put("data", image.image().base64Data());
                    }
                }
            }
        } else if (message instanceof AiMessage ai) {
            node.put("type", "ai");
            node.put("content", ai.text() == null ? "" : ai.text());
            if (ai.hasToolExecutionRequests()) {
                ArrayNode calls = node.putArray("tool_calls");
                for (dev.langchain4j.agent.tool.ToolExecutionRequest request : ai.toolExecutionRequests()) {
                    ObjectNode call = calls.addObject();
                    call.put("id", request.id());
                    call.put("name", request.name());
                    call.put("args", request.arguments());
                }
            }
        } else if (message instanceof ToolExecutionResultMessage tool) {
            node.put("type", "tool");
            node.put("content", tool.text());
            node.put("tool_call_id", tool.id());
            node.put("name", tool.toolName());
        } else {
            node.put("type", "ai");
            node.put("content", message.toString());
        }
        return node;
    }

    /** JSON 节点 → 消息；未知类型返回 null。 */
    public static ChatMessage messageFromJson(JsonNode node) {
        String type = node.path("type").asText("");
        switch (type) {
            case "system" -> {
                return SystemMessage.from(node.path("content").asText(""));
            }
            case "human" -> {
                if (node.has("content_parts")) {
                    List<dev.langchain4j.data.message.Content> contents = new ArrayList<>();
                    for (JsonNode part : node.get("content_parts")) {
                        if ("image".equals(part.path("type").asText())) {
                            contents.add(dev.langchain4j.data.message.ImageContent.from(
                                    part.path("data").asText(), part.path("mime_type").asText("image/png")));
                        } else {
                            contents.add(dev.langchain4j.data.message.TextContent.from(part.path("text").asText()));
                        }
                    }
                    return UserMessage.from(contents);
                }
                return UserMessage.from(node.path("content").asText(""));
            }
            case "ai" -> {
                if (node.has("tool_calls") && node.get("tool_calls").isArray()
                        && !node.get("tool_calls").isEmpty()) {
                    List<dev.langchain4j.agent.tool.ToolExecutionRequest> requests = new ArrayList<>();
                    for (JsonNode call : node.get("tool_calls")) {
                        requests.add(dev.langchain4j.agent.tool.ToolExecutionRequest.builder()
                                .id(call.path("id").asText(""))
                                .name(call.path("name").asText(""))
                                .arguments(call.path("args").asText("{}"))
                                .build());
                    }
                    return node.path("content").asText("").isEmpty()
                            ? AiMessage.from(requests)
                            : AiMessage.from(node.path("content").asText(), requests);
                }
                return AiMessage.from(node.path("content").asText(""));
            }
            case "tool" -> {
                return ToolExecutionResultMessage.from(
                        node.path("tool_call_id").asText(""),
                        node.path("name").asText(""),
                        node.path("content").asText(""));
            }
            default -> {
                return null;
            }
        }
    }

    /** usage 提取（成本计量从 ChatResponse 侧拿到 TokenUsage 后转 dict）。 */
    public static Map<String, Long> usageToMap(TokenUsage usage) {
        if (usage == null) {
            return Map.of();
        }
        Map<String, Long> out = new LinkedHashMap<>();
        out.put("input", usage.inputTokenCount() == null ? 0L : usage.inputTokenCount());
        out.put("output", usage.outputTokenCount() == null ? 0L : usage.outputTokenCount());
        out.put("total", usage.totalTokenCount() == null ? 0L : usage.totalTokenCount());
        return out;
    }
}
