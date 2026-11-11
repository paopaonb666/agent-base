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
    void cancelledRoundMarkerWritesOnlyForCancelledRoundsAndReadsAreTolerant() {
        // #199：取消轮快照带根级 cancelled 键；正常快照逐字节不变（无键）；
        // 读侧把该键当行元数据（不进状态通道），消息无损、存量行兼容。
        GraphState normal = new GraphState(List.of(UserMessage.from("hi")), Map.of());
        assertFalse(GraphStateJson.toJson(normal).contains("cancelled"));

        GraphState cancelled = new GraphState(List.of(UserMessage.from("hi")), Map.of());
        cancelled.cancelledRound(true);
        String json = GraphStateJson.toJson(cancelled);
        assertTrue(json.contains("\"cancelled\":true"), json);

        GraphState parsed = GraphStateJson.fromJson(json);
        assertEquals(1, parsed.messages().size());
        assertEquals("hi", ((UserMessage) parsed.messages().get(0)).singleText());
        assertFalse(parsed.cancelledRound(), "标记是行元数据，不随状态通道回读");
        // 存量行已带 cancelled 键（新写侧+旧读侧混跑）同样无害。
        GraphState legacy = GraphStateJson.fromJson(
                "{\"v\":1,\"cancelled\":true,\"messages\":[{\"type\":\"human\",\"content\":\"old\"}],\"extra\":{}}");
        assertEquals("old", ((UserMessage) legacy.messages().get(0)).singleText());
    }

    @Test
    void compactionSummarySystemMessageRoundTripsWithMetadataAndStaysRealHistory() {
        // #328/#330：压缩摘要 SystemMessage（新标记前缀，非注入前缀）type=
        // "system" 天然往返；extra 的 compaction 元数据原样保存；VERSION 不
        // bump（读侧不解析压缩键，只有 extra 通道）。
        Map<String, Object> extra = new LinkedHashMap<>();
        extra.put("compaction", Map.of(
                "original_message_count", 16,
                "original_token_estimate", 208,
                "at", "2026-11-11T00:00:00Z"));
        GraphState state = new GraphState(List.of(
                SystemMessage.from(io.github.paopaonb666.agentbase.core.context.ContextEngine
                        .COMPACTION_SUMMARY_PREFIX
                        + "【任务态】任务态\n【关键实体】实体\n【未决问题】问题"),
                UserMessage.from("下一轮")), extra);

        GraphState restored = roundTrip(state);
        assertEquals(2, restored.messages().size());
        assertTrue(restored.messages().get(0) instanceof SystemMessage);
        assertTrue(((SystemMessage) restored.messages().get(0)).text()
                .startsWith(io.github.paopaonb666.agentbase.core.context.ContextEngine
                        .COMPACTION_SUMMARY_PREFIX));
        assertEquals("下一轮", ((UserMessage) restored.messages().get(1)).singleText());
        @SuppressWarnings("unchecked")
        Map<String, Object> meta = (Map<String, Object>) restored.extra().get("compaction");
        assertEquals(16, ((Number) meta.get("original_message_count")).intValue());
        assertEquals("2026-11-11T00:00:00Z", meta.get("at"));
        // 新前缀不得命中三个注入前缀（否则 buildModelInput 会把压缩摘要当
        // 注入块去重/提升——互斥契约，见 ContextEngine.COMPACTION_SUMMARY_PREFIX）。
        assertNull(io.github.paopaonb666.agentbase.core.context.ContextEngine
                .injectedKind(restored.messages().get(0)));
        // VERSION 维持 1：压缩元数据走 extra 通道，不 bump 格式版本。
        assertEquals(1, GraphStateJson.VERSION);
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

    @Test
    void aiAuditMetadataRoundTripsAndOldSnapshotsStayCompatible() {
        // #123：id / finish_reason 随快照往返；旧快照（无 message_id 键）
        // 兼容读——重建的消息无 attributes，语义不变。
        AiMessage audited = AiMessage.builder()
                .text("回答")
                .attributes(new LinkedHashMap<>(Map.of(
                        "id", "msg-abc123", "finish_reason", "stop")))
                .build();
        AiMessage restored = (AiMessage) roundTrip(new GraphState(List.of(audited), Map.of()))
                .messages().get(0);
        assertEquals("msg-abc123", restored.attribute("id", String.class));
        assertEquals("stop", restored.attribute("finish_reason", String.class));
        assertEquals("回答", restored.text());

        AiMessage legacy = AiMessage.from("旧快照消息");
        AiMessage legacyRestored = (AiMessage) roundTrip(new GraphState(List.of(legacy), Map.of()))
                .messages().get(0);
        assertNull(legacyRestored.attribute("id", String.class));
        assertEquals("旧快照消息", legacyRestored.text());
    }

    @Test
    void aiThinkingRoundTripsIncludingToolCallVariant() {
        // R2-F3 收口：带 thinking 的 AiMessage 往返——推理轨迹不再静默
        // 丢失；text + thinking + tool_calls 同体的复合形态同样无损。
        AiMessage thinking = AiMessage.builder()
                .text("答")
                .thinking("推理过程")
                .build();
        AiMessage restored = (AiMessage) roundTrip(new GraphState(List.of(thinking), Map.of()))
                .messages().get(0);
        assertEquals("推理过程", restored.thinking());
        assertEquals("答", restored.text());
        assertNull(restored.attribute("id", String.class));

        AiMessage composite = AiMessage.builder()
                .text("先查再答")
                .thinking("需要先检索")
                .toolExecutionRequests(List.of(dev.langchain4j.agent.tool.ToolExecutionRequest.builder()
                        .id("call_7")
                        .name("web_search")
                        .arguments("{\"query\":\"x\"}")
                        .build()))
                .build();
        AiMessage compositeRestored = (AiMessage) roundTrip(new GraphState(List.of(composite), Map.of()))
                .messages().get(0);
        assertEquals("需要先检索", compositeRestored.thinking());
        assertEquals("先查再答", compositeRestored.text());
        assertTrue(compositeRestored.hasToolExecutionRequests());
        assertEquals("call_7", compositeRestored.toolExecutionRequests().get(0).id());
        assertEquals("web_search", compositeRestored.toolExecutionRequests().get(0).name());
    }

    @Test
    void aiSnapshotWithoutThinkingKeyStaysCompatible() {
        // R2-F3 兼容读（#123 先例）：旧快照 ai 节点无 thinking 键 →
        // thinking() 为 null，其余字段照常还原；不 bump VERSION。
        GraphState legacy = GraphStateJson.fromJson("""
                {"v":1,"messages":[
                  {"type":"ai","content":"旧快照消息"}
                ],"extra":{}}
                """);
        AiMessage old = (AiMessage) legacy.messages().get(0);
        assertNull(old.thinking());
        assertEquals("旧快照消息", old.text());
        assertNull(old.attribute("id", String.class));

        // 带 message_id/finish_reason 但无 thinking 的组合同样容忍。
        GraphState audited = GraphStateJson.fromJson("""
                {"v":1,"messages":[
                  {"type":"ai","content":"回答","message_id":"msg-old","finish_reason":"stop"}
                ],"extra":{}}
                """);
        AiMessage auditedRestored = (AiMessage) audited.messages().get(0);
        assertNull(auditedRestored.thinking());
        assertEquals("msg-old", auditedRestored.attribute("id", String.class));
        assertEquals("stop", auditedRestored.attribute("finish_reason", String.class));
        assertEquals("回答", auditedRestored.text());

        // thinking 键为 JSON null / 空串均视作无（读侧容忍，不炸不误挂）。
        GraphState nullThinking = GraphStateJson.fromJson("""
                {"v":1,"messages":[
                  {"type":"ai","content":"空推理","thinking":null}
                ],"extra":{}}
                """);
        assertNull(((AiMessage) nullThinking.messages().get(0)).thinking());
        GraphState emptyThinking = GraphStateJson.fromJson("""
                {"v":1,"messages":[
                  {"type":"ai","content":"空推理","thinking":""}
                ],"extra":{}}
                """);
        assertNull(((AiMessage) emptyThinking.messages().get(0)).thinking());
    }

    @Test
    void aiThinkingKeyWrittenOnlyWhenPresent() throws Exception {
        // wire 形态锁定：无 thinking 的 AiMessage 不写 thinking 键（旧行为
        // 逐字节不变，空串同样不写）；有 thinking 才写键且值原样。
        String bare = GraphStateJson.toJson(new GraphState(
                List.of(AiMessage.from("无推理")), Map.of()));
        assertFalse(MAPPER.readTree(bare).get("messages").get(0).has("thinking"));

        String empty = GraphStateJson.toJson(new GraphState(
                List.of(AiMessage.builder().text("空推理").thinking("").build()), Map.of()));
        assertFalse(MAPPER.readTree(empty).get("messages").get(0).has("thinking"));

        String withThinking = GraphStateJson.toJson(new GraphState(
                List.of(AiMessage.builder().text("答").thinking("推理过程").build()), Map.of()));
        JsonNode node = MAPPER.readTree(withThinking).get("messages").get(0);
        assertTrue(node.has("thinking"));
        assertEquals("推理过程", node.get("thinking").asText());
    }
}
