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

    private static final org.slf4j.Logger LOG =
            org.slf4j.LoggerFactory.getLogger(GraphStateJson.class);

    private GraphStateJson() {}

    private static final ObjectMapper MAPPER = new ObjectMapper();

    /**
     * 存储格式版本（#200）：升级改序列化格式时递增，并在
     * {@link #fromJsonDetail} 里按旧版本写迁移分支——未知版本拒载并
     * 告警，绝不把升级中的旧快照静默解析成「无历史」。
     */
    public static final int VERSION = 1;

    /** 未知消息 type 的丢弃计数（#201）：丢失可观测。 */
    private static final java.util.concurrent.ConcurrentHashMap<String, java.util.concurrent.atomic.AtomicLong>
            UNKNOWN_TYPE_COUNTS = new java.util.concurrent.ConcurrentHashMap<>();

    /** 快照解析结果分类（#235）：checkpointer 据此隔离/告警。 */
    public record ParseResult(GraphState state, String kind, String detail) {
        public static final String OK = "ok";
        public static final String CORRUPT = "corrupt";
        public static final String VERSION_REFUSED = "version";
    }

    /** 序列化图状态为 JSON 字符串（带版本字段 v）。 */
    public static String toJson(GraphState state) {
        ObjectNode root = MAPPER.createObjectNode();
        root.put("v", VERSION);
        if (state.cancelledRound()) {
            // #199：取消轮快照带标记——「这轮是被中断的」在快照行上可审计。
            // 读侧不解析该键（行元数据不进状态通道），新 round 首次正常
            // 落盘整行覆盖后自然消失。
            root.put("cancelled", true);
        }
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

    /** 反序列化；损坏的 JSON / 被拒版本返回 null（旧行当作缺失处理）。 */
    public static GraphState fromJson(String json) {
        return fromJsonDetail(json).state();
    }

    /**
     * 分类解析（#200/#235）：ok（含空值/无 v 的存量行——v1 兼容）/
     * corrupt（JSON 损坏）/ version（未知版本，拒载保留行等迁移）。
     */
    public static ParseResult fromJsonDetail(String json) {
        if (json == null || json.isEmpty()) {
            return new ParseResult(null, ParseResult.OK, "empty");
        }
        JsonNode root;
        try {
            root = MAPPER.readTree(json);
        } catch (JsonProcessingException e) {
            return new ParseResult(null, ParseResult.CORRUPT, String.valueOf(e.getMessage()));
        }
        if (root.has("v")) {
            JsonNode v = root.get("v");
            if (!v.canConvertToInt() || v.asInt() != VERSION) {
                return new ParseResult(null, ParseResult.VERSION_REFUSED,
                        "snapshot v=" + v + " (supported: " + VERSION + ")");
            }
        }
        // 无 v 的存量行是 v1 兼容格式，照常解析。根级 cancelled 键（#199
        // 取消轮标记）是快照行元数据，读侧刻意不解析——不属于状态通道。
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
        return new ParseResult(new GraphState(messages, extra), ParseResult.OK, "");
    }

    /** 未知消息 type 的累计丢弃数（按 type 分桶；#201 可观测性）。 */
    public static Map<String, Long> unknownMessageTypeCounts() {
        Map<String, Long> out = new LinkedHashMap<>();
        UNKNOWN_TYPE_COUNTS.forEach((type, count) -> out.put(type, count.get()));
        return out;
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
            // #123：稳定 id 与 finish_reason 随快照持久化（审计/前端锚点）。
            // 兼容读：旧快照无这些键 → 重建的消息无 attributes，语义不变
            // （不 bump VERSION——读侧容忍缺键，写侧新增键对旧读侧透明）。
            String auditId = ai.attribute("id", String.class);
            if (auditId != null) {
                node.put("message_id", auditId);
            }
            String finishReason = ai.attribute("finish_reason", String.class);
            if (finishReason != null) {
                node.put("finish_reason", finishReason);
            }
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
                AiMessage rebuilt = aiFromJson(node);
                // #123：快照携带的稳定 id / finish_reason 还原进 attributes
                // （旧快照无键 → 无 attributes，兼容读）。
                java.util.Map<String, Object> attrs = new java.util.LinkedHashMap<>();
                if (!node.path("message_id").asText("").isEmpty()) {
                    attrs.put("id", node.path("message_id").asText());
                }
                if (!node.path("finish_reason").asText("").isEmpty()) {
                    attrs.put("finish_reason", node.path("finish_reason").asText());
                }
                if (attrs.isEmpty()) {
                    return rebuilt;
                }
                return AiMessage.builder()
                        .text(rebuilt.text())
                        .thinking(rebuilt.thinking())
                        .toolExecutionRequests(rebuilt.toolExecutionRequests())
                        .attributes(attrs)
                        .build();
            }
            case "tool" -> {
                return ToolExecutionResultMessage.from(
                        node.path("tool_call_id").asText(""),
                        node.path("name").asText(""),
                        node.path("content").asText(""));
            }
            default -> {
                // 未知 type（#201）：丢弃但 WARN + 计数——升级新增消息类型
                // 后旧代码读新快照的静默丢消息变成可观测事件。
                java.util.concurrent.atomic.AtomicLong count =
                        UNKNOWN_TYPE_COUNTS.computeIfAbsent(type, t -> new java.util.concurrent.atomic.AtomicLong());
                long seen = count.incrementAndGet();
                if (seen == 1) {
                    LOG.warn("graph state: 未知消息 type '{}' 被丢弃（快照兼容读取；"
                            + "该 type 累计计数可见 unknownMessageTypeCounts）", type);
                }
                return null;
            }
        }
    }

    /** ai 节点 → AiMessage 的基础重建（#123 前的形态；attributes 由调用方按需挂）。 */
    private static AiMessage aiFromJson(JsonNode node) {
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
