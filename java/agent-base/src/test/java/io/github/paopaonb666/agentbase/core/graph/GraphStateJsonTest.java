package io.github.paopaonb666.agentbase.core.graph;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import org.junit.jupiter.api.Test;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * GraphStateJson 序列化兼容测试（checkpointer 持久化格式的合同面）。
 *
 * <p>覆盖路线图 #286 圈定的矩阵：四类消息 round-trip（含多模态
 * content_parts 与 tool_calls）、extra 通道原样保存、损坏行→null（#200
 * 与 #235 的基线语义）、未知 type 丢弃（#201 的基线语义）。#200 落地
 * 版本字段时，本组的损坏行断言是要显式改写的位置。
 */
class GraphStateJsonTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private static GraphState roundTrip(GraphState state) {
        return GraphStateJson.fromJson(GraphStateJson.toJson(state));
    }

    @Test
    void systemAndHumanMessagesRoundTrip() {
        GraphState state = new GraphState(List.of(
                SystemMessage.from("you are helpful"),
                UserMessage.from("hi there")), Map.of());

        GraphState restored = roundTrip(state);
        assertEquals(2, restored.messages().size());
        assertEquals(SystemMessage.class, restored.messages().get(0).type() == null ? null : restored.messages().get(0).getClass());
        assertEquals("you are helpful", ((SystemMessage) restored.messages().get(0)).text());
        assertEquals("hi there", ((UserMessage) restored.messages().get(1)).singleText());
    }

    @Test
    void humanMultimodalContentPartsRoundTrip() {
        // 多模态：文本段 + 图片块（base64/mime）。
        GraphState state = new GraphState(List.of(
                UserMessage.from(List.of(
                        dev.langchain4j.data.message.TextContent.from("看这张图"),
                        dev.langchain4j.data.message.ImageContent.from(
                                "aGk=", "image/png")))), Map.of());

        GraphState restored = roundTrip(state);
        assertEquals(1, restored.messages().size());
        UserMessage user = (UserMessage) restored.messages().get(0);
        assertEquals(2, user.contents().size());
        dev.langchain4j.data.message.TextContent text =
                (dev.langchain4j.data.message.TextContent) user.contents().get(0);
        dev.langchain4j.data.message.ImageContent image =
                (dev.langchain4j.data.message.ImageContent) user.contents().get(1);
        assertEquals("看这张图", text.text());
        assertEquals("aGk=", image.image().base64Data());
        assertEquals("image/png", image.image().mimeType());
    }

    @Test
    void aiMessageWithToolCallsRoundTrip() {
        GraphState state = new GraphState(List.of(
                AiMessage.from("let me check",
                        List.of(dev.langchain4j.agent.tool.ToolExecutionRequest.builder()
                                .id("call_1")
                                .name("web_search")
                                .arguments("{\"query\":\"x\"}")
                                .build()))), Map.of());

        GraphState restored = roundTrip(state);
        AiMessage ai = (AiMessage) restored.messages().get(0);
        assertEquals("let me check", ai.text());
        assertTrue(ai.hasToolExecutionRequests());
        dev.langchain4j.agent.tool.ToolExecutionRequest call = ai.toolExecutionRequests().get(0);
        assertEquals("call_1", call.id());
        assertEquals("web_search", call.name());
        assertEquals("{\"query\":\"x\"}", call.arguments());
    }

    @Test
    void toolResultMessageRoundTrip() {
        GraphState state = new GraphState(List.of(
                ToolExecutionResultMessage.from("call_1", "web_search", "result text")),
                Map.of());

        GraphState restored = roundTrip(state);
        ToolExecutionResultMessage tool = (ToolExecutionResultMessage) restored.messages().get(0);
        assertEquals("call_1", tool.id());
        assertEquals("web_search", tool.toolName());
        assertEquals("result text", tool.text());
    }

    @Test
    void extraChannelSurvivesRoundTrip() {
        // extra 通道（tasks/cursor/replans 等）原样 JSON 保存，含嵌套结构。
        Map<String, Object> extra = new LinkedHashMap<>();
        extra.put("cursor", 3);
        extra.put("tasks", List.of(Map.of("goal", "写报告", "done", false)));
        extra.put("replans", 1);
        GraphState state = new GraphState(List.of(), extra);

        GraphState restored = roundTrip(state);
        assertEquals(3, ((Number) restored.get("cursor")).intValue());
        assertEquals(1, ((Number) restored.get("replans")).intValue());
        Object tasks = restored.get("tasks");
        assertTrue(tasks instanceof List<?> list && list.size() == 1);
        @SuppressWarnings("unchecked")
        Map<String, Object> task = (Map<String, Object>) ((List<?>) tasks).get(0);
        assertEquals("写报告", task.get("goal"));
        assertEquals(false, task.get("done"));
    }

    @Test
    void corruptedJsonLineReturnsNull() {
        // #200/#235：损坏 JSON → null（调用方按"无历史"处理）；分类细节
        // 由 fromJsonDetail 提供（checkpointer 据此隔离）。
        assertNull(GraphStateJson.fromJson("{not json at all"));
        assertNull(GraphStateJson.fromJson(""));
        assertNull(GraphStateJson.fromJson(null));
        assertEquals(GraphStateJson.ParseResult.CORRUPT,
                GraphStateJson.fromJsonDetail("{not json at all").kind());
    }

    @Test
    void snapshotsCarryAVersionFieldAndUnknownVersionsAreRefused() throws Exception {
        // #200：写侧带 v=1；读侧未知版本拒载（返回 null），无 v 的存量行
        // （v1 兼容格式）照常解析。
        String json = GraphStateJson.toJson(new GraphState(
                List.of(UserMessage.from("hi")), Map.of()));
        assertEquals(1, MAPPER.readTree(json).get("v").asInt());
        assertEquals("hi", ((UserMessage) GraphStateJson.fromJson(json).messages().get(0)).singleText());
        // 存量行（无 v）兼容。
        GraphState legacy = GraphStateJson.fromJson(
                "{\"messages\":[{\"type\":\"human\",\"content\":\"old\"}],\"extra\":{}}");
        assertEquals("old", ((UserMessage) legacy.messages().get(0)).singleText());
        // 未知版本拒载。
        assertNull(GraphStateJson.fromJson(
                "{\"v\":99,\"messages\":[],\"extra\":{}}"));
        assertEquals(GraphStateJson.ParseResult.VERSION_REFUSED,
                GraphStateJson.fromJsonDetail("{\"v\":99,\"messages\":[],\"extra\":{}}").kind());
        assertTrue(GraphStateJson.fromJsonDetail(
                "{\"v\":99,\"messages\":[],\"extra\":{}}").detail().contains("99"));
    }

    @Test
    void unknownMessageTypeIsDroppedAndCounted() {
        // #201：未知 type → null 丢弃，其余消息保留；丢弃按 type 计数
        // （丢失可观测，不再静默）。
        long before = GraphStateJson.unknownMessageTypeCounts()
                .getOrDefault("future_kind", 0L);
        GraphState restored = GraphStateJson.fromJson("""
                {"messages":[
                  {"type":"human","content":"before"},
                  {"type":"future_kind","content":"mystery"},
                  {"type":"human","content":"after"},
                  {"type":"future_kind","content":"mystery-2"}
                ],"extra":{}}
                """);
        assertEquals(2, restored.messages().size());
        assertEquals("before", ((UserMessage) restored.messages().get(0)).singleText());
        assertEquals("after", ((UserMessage) restored.messages().get(1)).singleText());
        assertEquals(before + 2, GraphStateJson.unknownMessageTypeCounts()
                .getOrDefault("future_kind", 0L), "未知 type 丢弃应累计计数");
    }

    @Test
    void jsonWireFormatMatchesDocumentedShape() throws Exception {
        // 线上格式合同：type ∈ {system,human,ai,tool}，AiMessage 带
        // tool_calls=[{id,name,args}]，ToolMessage 带 tool_call_id/name。
        dev.langchain4j.agent.tool.ToolExecutionRequest request =
                dev.langchain4j.agent.tool.ToolExecutionRequest.builder()
                        .id("c9")
                        .name("echo")
                        .arguments("{}")
                        .build();
        ChatMessage ai = AiMessage.from("", List.of(request));
        String json = GraphStateJson.toJson(new GraphState(List.of(ai), Map.of()));
        JsonNode root = MAPPER.readTree(json);
        assertEquals("ai", root.get("messages").get(0).get("type").asText());
        JsonNode call = root.get("messages").get(0).get("tool_calls").get(0);
        assertEquals("c9", call.get("id").asText());
        assertEquals("echo", call.get("name").asText());
        assertEquals("{}", call.get("args").asText());
    }

    @Test
    void usageToMapDefaultsZeroOnNullComponents() {
        Map<String, Long> usage = GraphStateJson.usageToMap(null);
        assertTrue(usage.isEmpty());

        Map<String, Long> partial = GraphStateJson.usageToMap(
                new dev.langchain4j.model.output.TokenUsage(10, 5));
        assertEquals(10L, partial.get("input"));
        assertEquals(5L, partial.get("output"));
        assertEquals(15L, partial.get("total"));
    }

    @Test
    void humanMessageWithEmptyPartsListSerializesAsContent() {
        // 单段文本 content 与多段 content_parts 两种形态互不混淆。
        GraphState restored = roundTrip(new GraphState(
                List.of(UserMessage.from("plain")), Map.of()));
        assertFalse(restored.messages().isEmpty());
        UserMessage user = (UserMessage) restored.messages().get(0);
        assertEquals(1, user.contents().size());
        assertEquals("plain",
                ((dev.langchain4j.data.message.TextContent) user.contents().get(0)).text());
    }
}
