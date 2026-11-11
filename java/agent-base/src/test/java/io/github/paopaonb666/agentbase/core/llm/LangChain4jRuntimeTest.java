package io.github.paopaonb666.agentbase.core.llm;

import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.output.TokenUsage;
import io.github.paopaonb666.agentbase.core.context.ContextEngine;
import io.github.paopaonb666.agentbase.core.text.EstimatingTokenCounter;
import io.github.paopaonb666.agentbase.extensions.Metrics;
import org.junit.jupiter.api.AfterEach;
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
 *   <li>#340 对账观测：invoke / 原生流式 / 非流式回退三条真实调用路径都在
 *       usage 记账时刻发射 estimated_input（与 buildModelInput 逐条计量
 *       同源，乘 tokenFactor），按 {model, profile} 分面可断言。</li>
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
    void streamNativePreservesAggregatedAttributesAndThinking() {
        // R2-F6：流式重造不再丢弃聚合产物的 attributes / thinking——invoke/
        // fallback 路径本就保留，三条路径对称；文本仍以回调累计为权威。
        AiMessage aggregated = AiMessage.builder()
                .text("聚合文本")
                .thinking("思考链")
                .attributes(java.util.Map.of("provider_key", "pv"))
                .build();
        ScriptedModels.ScriptedStreamingModel streaming =
                new ScriptedModels.ScriptedStreamingModel(List.of(
                        ScriptedModels.ScriptedResponse.of(aggregated)));
        LangChain4jRuntime runtime = new LangChain4jRuntime(
                new ScriptedModels.ScriptedChatModel(List.of()), streaming, "scripted-main", false, null);
        DeltaCapture capture = new DeltaCapture();
        AiMessage result = runtime.stream(MESSAGES, List.of(), capture);

        assertEquals("聚合文本", result.text(), "文本 = 回调累计（与聚合产物一致）");
        assertEquals("思考链", result.thinking(), "thinking 随重造保留");
        assertEquals("pv", result.attribute("provider_key", String.class),
                "provider attributes 穿透（修复前被丢弃）");
        assertTrue(result.attribute("id", String.class) != null,
                "withAuditMetadata 的稳定 id 照常追加");
    }

    @Test
    void stalledStreamFailsFastWithIdleWatchdog() {
        // #74：provider 半途停止发 delta 且不结束——idle 看门狗让它快速
        // 失败（StreamStalledException），不再等 600s HTTP 总超时。
        dev.langchain4j.model.chat.StreamingChatModel stalling =
                new dev.langchain4j.model.chat.StreamingChatModel() {
                    @Override
                    public void chat(dev.langchain4j.model.chat.request.ChatRequest request,
                                     dev.langchain4j.model.chat.response.StreamingChatResponseHandler handler) {
                        handler.onPartialResponse("半截");
                        // 既不 onComplete 也不 onError：模拟静默断流。
                    }
                };
        LangChain4jRuntime runtime = new LangChain4jRuntime(
                new ScriptedModels.ScriptedChatModel(List.of()), stalling, "scripted-main",
                false, 0, 200, null);
        long start = System.nanoTime();
        LangChain4jRuntime.StreamStalledException error = assertThrows(
                LangChain4jRuntime.StreamStalledException.class,
                () -> runtime.stream(MESSAGES, List.of(), delta -> {}));
        long elapsedMillis = (System.nanoTime() - start) / 1_000_000;
        assertTrue(elapsedMillis < 5_000, "idle 看门狗应快速失败，实测 " + elapsedMillis + "ms");
        assertTrue(error.getMessage().contains("idle"), error.getMessage());
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

    // ── #340 对账观测：estimated_input 发射 ────────────────────────────────

    @AfterEach
    void restoreDefaultCounter() {
        // estimated_input 口径经 ContextEngine 的全局计量器：用后恢复默认
        //（JTokkit cl100k），防串扰（ContextEngineTest 同款约定）。
        ContextEngine.setTokenCounter(null);
    }

    /** 预算边界测试用确定性计量器（估算口径与手算一致，ContextEngineTest 同款）。 */
    private static void useEstimator() {
        ContextEngine.setTokenCounter(EstimatingTokenCounter.INSTANCE);
    }

    /** 20 个 ASCII 字符：估算器口径 5 token，+4/条 固定开销 → 每条 9（对账公式可手算）。 */
    private static UserMessage meteredHuman() {
        return UserMessage.from("x".repeat(20));
    }

    /** llm_tokens_total{kind="estimated_input"} 的期望行形状（MetricsRenderTest 同款 contains 断言）。 */
    private static String estimatedInputLine(String model, String profile, long expected) {
        return "llm_tokens_total{model=\"" + model + "\",profile=\"" + profile
                + "\",kind=\"estimated_input\"} " + expected;
    }

    @Test
    void profileCarriesThroughConstructorWithMainFallback() {
        // 9 参 canonical：profile 原样承载（对账分面）；null/空白防御回落
        // "main"；旧 8/5 参重载委托后缺省主档。
        ScriptedModels.ScriptedChatModel chat = new ScriptedModels.ScriptedChatModel(List.of());
        assertEquals("fast", new LangChain4jRuntime(chat, null, "p-fast", false, 0, 0, 1.0, null, "fast").profile());
        assertEquals("main", new LangChain4jRuntime(chat, null, "p-null", false, 0, 0, 1.0, null, null).profile());
        assertEquals("main", new LangChain4jRuntime(chat, null, "p-blank", false, 0, 0, 1.0, null, "   ").profile());
        assertEquals("main", new LangChain4jRuntime(chat, null, "p-eight", false, 0, 0, 1.0, null).profile(),
                "旧 8 参 canonical 委托 main");
        assertEquals("main", new LangChain4jRuntime(chat, null, "p-legacy", false, null).profile(),
                "旧 5 参形态缺省主档");
    }

    @Test
    void invokeEmitsEstimatedInputToMetrics() {
        // #340 对账：invoke 在 usageSink 记账同一时刻发射 estimated_input，
        // 值 = Σ(估算 + 4) × factor（与 buildModelInput 逐条计量单源同式）。
        // 两条 20-ASCII 消息：(5+4)×1.0 = 9/条 → 18。脚本不投递 usage——
        // 估算只依赖消息列表，与 provider 是否回报 usage 无关。
        useEstimator();
        ScriptedModels.ScriptedChatModel chat =
                new ScriptedModels.ScriptedChatModel(List.of(ScriptedModels.ScriptedResponse.of("ok")));
        String model = "est-main-" + System.nanoTime(); // 独特 model 名防进程级单例串扰
        LangChain4jRuntime runtime = new LangChain4jRuntime(
                chat, null, model, false, 0, 0, 1.0, null, "main");
        runtime.invoke(List.of(meteredHuman(), meteredHuman()));
        assertTrue(Metrics.cost().renderCostMetrics()
                        .contains(estimatedInputLine(model, "main", 18)),
                "invoke 应发射 {model, profile=main, estimated_input}=18");
    }

    @Test
    void streamNativeEmitsEstimatedInputWithProfileFacet() {
        // 快档（profile="fast"）纯聊天不绑工具 → 原生流式路径同样发射；
        // 分面来自构造参数（9 参 canonical 尾参承载）。
        useEstimator();
        ScriptedModels.ScriptedStreamingModel streaming = new ScriptedModels.ScriptedStreamingModel(
                List.of(ScriptedModels.ScriptedResponse.of("hi")));
        ScriptedModels.ScriptedChatModel chat = new ScriptedModels.ScriptedChatModel(List.of());
        String model = "est-fast-" + System.nanoTime();
        LangChain4jRuntime runtime = new LangChain4jRuntime(
                chat, streaming, model, true, 0, 0, 1.0, null, "fast");
        runtime.stream(List.of(meteredHuman(), meteredHuman()), List.of(), new DeltaCapture());
        assertTrue(Metrics.cost().renderCostMetrics()
                        .contains(estimatedInputLine(model, "fast", 18)),
                "原生流式应发射 {model, profile=fast, estimated_input}=18");
    }

    @Test
    void estimatedInputFollowsTokenFactorScaling() {
        // 估算乘 tokenFactor（LLM_TOKEN_FACTOR 对齐 provider 分词的口径）：
        // factor=2.0 → (5+4)×2 = 18/条 → 36。
        useEstimator();
        ScriptedModels.ScriptedChatModel chat =
                new ScriptedModels.ScriptedChatModel(List.of(ScriptedModels.ScriptedResponse.of("ok")));
        String model = "est-factor-" + System.nanoTime();
        LangChain4jRuntime runtime = new LangChain4jRuntime(
                chat, null, model, false, 0, 0, 2.0, null, "main");
        runtime.invoke(List.of(meteredHuman(), meteredHuman()));
        assertTrue(Metrics.cost().renderCostMetrics()
                        .contains(estimatedInputLine(model, "main", 36)),
                "estimated_input 应随 tokenFactor 缩放（2×9/条）");
    }

    @Test
    void nonStreamFallbackEmitsEstimatedInputToo() {
        // 非流式回退同样是真实调用（cost_usage 行在此产出）：同一时刻发射
        // 保持 attempt 级 1:1——缺了这里，快档工具轮（回退高频形态）将
        // 永久缺失对账分母。单条 20-ASCII 消息 → 9。
        useEstimator();
        ScriptedModels.ScriptedChatModel chat = new ScriptedModels.ScriptedChatModel(
                List.of(ScriptedModels.ScriptedResponse.withToolCalls("", List.of(echoCall()))));
        ScriptedModels.ScriptedStreamingModel streaming =
                new ScriptedModels.ScriptedStreamingModel(List.of());
        String model = "est-fallback-" + System.nanoTime();
        LangChain4jRuntime runtime = new LangChain4jRuntime(
                chat, streaming, model, true, 0, 0, 1.0, null, "fast");
        runtime.stream(List.of(meteredHuman()), List.of(echoTool()), new DeltaCapture());
        assertTrue(Metrics.cost().renderCostMetrics()
                        .contains(estimatedInputLine(model, "fast", 9)),
                "非流式回退路径应发射 {model, profile=fast, estimated_input}=9");
    }
}
