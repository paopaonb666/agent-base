package io.github.paopaonb666.agentbase.core.llm;

import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.output.TokenUsage;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@link LangChain4jRuntime} 的流式/回退/usage 语义测试：
 *
 * <ul>
 *   <li>流式：逐 delta 聚合出完整文本，最终 AiMessage 用累计文本显式重造
 *       （对应 Python _stream_and_accumulate 的显式 AIMessage 构造）。</li>
 *   <li>tool-safe 快档：绑工具退非流式（对应 _ToolSafeStreamChatOpenAI），
 *       纯聊天仍流式。</li>
 *   <li>usage 钩子：每次调用结束投递 (model, usage)。</li>
 * </ul>
 */
class LangChain4jRuntimeTest {

    private static final List<ChatMessage> MESSAGES = List.of(new UserMessage("hi"));

    private static ToolSpecification echoTool() {
        return ToolSpecification.builder()
                .name("echo")
                .description("原样回显输入")
                .build();
    }

    private static ToolExecutionRequest echoCall() {
        return ToolExecutionRequest.builder()
                .id("call1")
                .name("echo")
                .arguments("{\"text\":\"hi\"}")
                .build();
    }

    private static final class DeltaCapture implements Consumer<String> {
        final List<String> deltas = new ArrayList<>();

        @Override
        public void accept(String token) {
            deltas.add(token);
        }

        String joined() {
            return String.join("", deltas);
        }
    }

    @Test
    void streamCollectsDeltasAndRebuildsAiMessage() {
        ScriptedModels.ScriptedStreamingModel streaming =
                new ScriptedModels.ScriptedStreamingModel(List.of(ScriptedModels.ScriptedResponse.of("hello world")));
        LangChain4jRuntime runtime = new LangChain4jRuntime(
                new ScriptedModels.ScriptedChatModel(List.of()), streaming, "scripted-main", false, null);
        DeltaCapture capture = new DeltaCapture();
        AiMessage result = runtime.stream(MESSAGES, List.of(), capture);
        // "hello world"（11 字符）按 4 字符切片 → 3 段 delta。
        assertEquals(List.of("hell", "o wo", "rld"), capture.deltas);
        assertEquals("hello world", capture.joined());
        // 显式重造：内容一致但不是 onCompleteResponse 原样透传的那个实例。
        assertEquals("hello world", result.text());
    }

    @Test
    void rebuiltAiMessageCarriesStableAuditId() {
        // #123：每次模型调用产出的 AiMessage 带唯一稳定 id（审计/前端锚点）；
        // 两次调用 id 不同；invoke 路径同口径。
        ScriptedModels.ScriptedStreamingModel streaming =
                new ScriptedModels.ScriptedStreamingModel(List.of(
                        ScriptedModels.ScriptedResponse.of("a"),
                        ScriptedModels.ScriptedResponse.of("a")));
        LangChain4jRuntime runtime = new LangChain4jRuntime(
                new ScriptedModels.ScriptedChatModel(List.of()), streaming, "scripted-main", false, null);
        AiMessage first = runtime.stream(MESSAGES, List.of(), delta -> {});
        AiMessage second = runtime.stream(MESSAGES, List.of(), delta -> {});
        assertTrue(first.attribute("id", String.class) != null
                        && first.attribute("id", String.class).startsWith("msg-"),
                "重建消息应带 msg- 前缀的稳定 id");
        assertEquals("a", first.text());
        assertTrue(!first.attribute("id", String.class)
                .equals(second.attribute("id", String.class)), "不同调用 id 唯一");
        ScriptedModels.ScriptedChatModel chat =
                new ScriptedModels.ScriptedChatModel(List.of(ScriptedModels.ScriptedResponse.of("b")));
        LangChain4jRuntime plain = new LangChain4jRuntime(chat, null, "scripted-main", false, null);
        AiMessage invoked = plain.invoke(MESSAGES);
        assertTrue(invoked.attribute("id", String.class) != null, "invoke 路径同口径");
    }

    @Test
    void streamAggregatesToolCallsWithAccumulatedText() {
        ScriptedModels.ScriptedResponse scripted = ScriptedModels.ScriptedResponse.withToolCalls(
                " I will use the tool. ", List.of(echoCall()));
        ScriptedModels.ScriptedStreamingModel streaming =
                new ScriptedModels.ScriptedStreamingModel(List.of(scripted));
        LangChain4jRuntime runtime = new LangChain4jRuntime(
                new ScriptedModels.ScriptedChatModel(List.of()), streaming, "scripted-main", false, null);
        AiMessage result = runtime.stream(MESSAGES, List.of(echoTool()), new DeltaCapture());
        assertEquals(" I will use the tool. ", result.text());
        assertTrue(result.hasToolExecutionRequests());
        assertEquals("echo", result.toolExecutionRequests().get(0).name());
        assertEquals("{\"text\":\"hi\"}", result.toolExecutionRequests().get(0).arguments());
    }

    @Test
    void streamDeliversUsageToSink() {
        TokenUsage usage = new TokenUsage(11, 7);
        ScriptedModels.ScriptedStreamingModel streaming =
                new ScriptedModels.ScriptedStreamingModel(List.of(ScriptedModels.ScriptedResponse.of("hey", usage)));
        List<String> models = new ArrayList<>();
        List<TokenUsage> usages = new ArrayList<>();
        LangChain4jRuntime runtime = new LangChain4jRuntime(
                new ScriptedModels.ScriptedChatModel(List.of()), streaming, "scripted-main", false,
                (model, u) -> {
                    models.add(model);
                    usages.add(u);
                });
        runtime.stream(MESSAGES, List.of(), new DeltaCapture());
        assertEquals(List.of("scripted-main"), models);
        assertEquals(1, usages.size());
        assertSame(usage, usages.get(0), "原样转投 onCompleteResponse 携带的 usage");
    }

    @Test
    void invokeDeliversUsageToSink() {
        TokenUsage usage = new TokenUsage(3, 5);
        ScriptedModels.ScriptedChatModel chat =
                new ScriptedModels.ScriptedChatModel(List.of(ScriptedModels.ScriptedResponse.of("hi", usage)));
        LangChain4jRuntime runtime = new LangChain4jRuntime(chat, null, "scripted-main", false,
                (model, u) -> {
                    assertEquals("scripted-main", model);
                    assertSame(usage, u);
                });
        assertEquals("hi", runtime.invoke(MESSAGES).text());
    }

    @Test
    void toolSafeFastWithToolsFallsBackToNonStream() {
        // 快档 + 绑工具 → 走非流式（消费 chat 脚本，流式脚本不动）；
        // 整段文本作为单个 onDelta（对应 _ToolSafeStreamChatOpenAI 的整体 yield）。
        ScriptedModels.ScriptedRuntime fast = ScriptedModels.ScriptedRuntime.fast(
                List.of(ScriptedModels.ScriptedResponse.withToolCalls("", List.of(echoCall()))));
        DeltaCapture capture = new DeltaCapture();
        AiMessage result = fast.stream(MESSAGES, List.of(echoTool()), capture);
        assertEquals("", result.text());
        assertTrue(result.hasToolExecutionRequests());
        assertEquals(0, capture.deltas.size(), "空文本没有可流式内容，不回调 delta");
        assertEquals(1, fast.chatModel.received().size(), "非流式视图被消费");
        assertEquals(0, fast.streamingModel.received().size(), "流式视图未被触碰");
    }

    @Test
    void toolSafeFastWithToolsDeliversWholeTextAsSingleDelta() {
        ScriptedModels.ScriptedRuntime fast = ScriptedModels.ScriptedRuntime.fast(
                List.of(ScriptedModels.ScriptedResponse.of("whole answer")));
        DeltaCapture capture = new DeltaCapture();
        fast.stream(MESSAGES, List.of(echoTool()), capture);
        assertEquals(List.of("whole answer"), capture.deltas, "整段一次回调，不逐 token");
        assertEquals(0, fast.streamingModel.received().size());
    }

    @Test
    void toolSafeFastWithoutToolsStaysStreaming() {
        // 快档纯聊天（不绑工具）仍走流式，SSE delta 体验不变。
        ScriptedModels.ScriptedRuntime fast = ScriptedModels.ScriptedRuntime.fast(
                List.of(ScriptedModels.ScriptedResponse.of("hello world")));
        DeltaCapture capture = new DeltaCapture();
        AiMessage result = fast.stream(MESSAGES, List.of(), capture);
        assertEquals(List.of("hell", "o wo", "rld"), capture.deltas, "不绑工具时逐 token 流式");
        assertEquals("hello world", result.text());
        assertEquals(1, fast.streamingModel.received().size(), "流式视图被消费");
        assertEquals(0, fast.chatModel.received().size(), "非流式视图未被触碰");
    }

    @Test
    void mainProfileWithToolsStaysStreaming() {
        // 主档没有工具安全开关：绑工具也原生流式（行为零变化）。
        ScriptedModels.ScriptedRuntime main = ScriptedModels.ScriptedRuntime.main(
                List.of(ScriptedModels.ScriptedResponse.of("hello world")));
        DeltaCapture capture = new DeltaCapture();
        main.stream(MESSAGES, List.of(echoTool()), capture);
        assertEquals(3, capture.deltas.size());
        assertEquals(1, main.streamingModel.received().size());
    }

    @Test
    void missingStreamingModelFallsBackToNonStream() {
        // streaming() 为 null（未配置流式）：stream 全走非流式。
        ScriptedModels.ScriptedChatModel chat =
                new ScriptedModels.ScriptedChatModel(List.of(ScriptedModels.ScriptedResponse.of("fallback text")));
        LangChain4jRuntime runtime = new LangChain4jRuntime(chat, null, "scripted-main", false, null);
        DeltaCapture capture = new DeltaCapture();
        AiMessage result = runtime.stream(MESSAGES, List.of(), capture);
        assertEquals("fallback text", result.text());
        assertEquals(List.of("fallback text"), capture.deltas);
        assertNullStreaming(runtime);
    }

    private static void assertNullStreaming(LangChain4jRuntime runtime) {
        assertTrue(runtime.streaming() == null);
        assertFalse(runtime.chat() == null);
    }

    @Test
    void streamingErrorSurfacesAsRuntimeException() {
        ScriptedModels.ScriptedStreamingModel streaming = new ScriptedModels.ScriptedStreamingModel(
                List.of(ScriptedModels.ScriptedResponse.failing(new RuntimeException("boom mid-stream"))));
        LangChain4jRuntime runtime = new LangChain4jRuntime(
                new ScriptedModels.ScriptedChatModel(List.of()), streaming, "scripted-main", false, null);
        RuntimeException exc = assertThrows(RuntimeException.class,
                () -> runtime.stream(MESSAGES, List.of(), new DeltaCapture()));
        assertEquals("boom mid-stream", exc.getMessage(), "onError 的原始异常上抛");
    }

    @Test
    void scriptedModelExhaustionThrows() {
        // 脚本用尽（对应 Python RuntimeError: scripted responses exhausted）。
        ScriptedModels.ScriptedChatModel chat = new ScriptedModels.ScriptedChatModel(List.of());
        IllegalStateException exc = assertThrows(IllegalStateException.class,
                () -> chat.doChat(dev.langchain4j.model.chat.request.ChatRequest.builder()
                        .messages(MESSAGES)
                        .build()));
        assertEquals("ScriptedChatModel: no more responses", exc.getMessage());
    }
}
