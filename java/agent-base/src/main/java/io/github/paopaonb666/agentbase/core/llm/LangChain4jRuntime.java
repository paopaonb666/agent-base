package io.github.paopaonb666.agentbase.core.llm;

import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ChatRequestParameters;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import dev.langchain4j.model.output.TokenUsage;

import java.util.List;
import java.util.Objects;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.function.BiConsumer;
import java.util.function.Consumer;

/**
 * {@link ModelRuntime} 的 langchain4j 实现（对应 Python 侧 ChatOpenAI
 * 客户端的装配面，core/llm.py 的 {@code build_llm} 产物）。
 *
 * <p>所有支持的 provider（deepseek / zhipu / openai-compatible）共用同一个
 * OpenAI 兼容客户端：切换 provider 意味着在配置中修改 {@code LLM_BASE_URL}
 * 和 {@code LLM_MODEL}——永远不改代码（与 Python 侧模块 docstring 一致）。
 *
 * <p>构造签名接受 {@link ChatModel} / {@link StreamingChatModel} 接口而非
 * 具体的 {@code OpenAiChatModel} / {@code OpenAiStreamingChatModel}：工厂
 * （{@link LlmFactory}）传入的是 OpenAI 实现，测试传脚本化替身——生产装配
 * 语义不变，替身却能走完全相同的 stream/invoke 逻辑。
 *
 * <p>两条约定（与 Python 侧对齐，见 {@link ModelRuntime}）：
 * <ul>
 *   <li><b>流式优先</b>：{@link #stream} 走 {@link StreamingChatModel}；
 *       快档（{@code toolSafeStreaming=true}）绑定工具时退回非流式——对应
 *       Python {@code _ToolSafeStreamChatOpenAI}（见 {@link LlmFactory}
 *       与该类的注释：规避 SiliconFlow GLM-4-9B 流式 tool_calls 协议违规），
 *       整段文本作为单个 onDelta 回调，行为等价。</li>
 *   <li><b>usage 钩子</b>：每次调用结束回调 usageSink（对应 Python 在
 *       cost.enabled 时挂的 {@code CostMeterHandler}；stream_usage 参数的
 *       语义由 usageSink 的消费方承接）。usage 为 null（provider 未回报）
 *       也会投递，由 sink 自行决定是否记零。</li>
 * </ul>
 */
public final class LangChain4jRuntime implements ModelRuntime {

    private final ChatModel chatModel;
    /** 流式视图；null = 该档位没有流式客户端（stream 退回非流式）。 */
    private final StreamingChatModel streamingModel;
    private final String modelName;
    private final boolean toolSafeStreaming;
    private final int contextWindowTokens;
    /** #74：流式 idle 超时（毫秒）；0 = 禁用（旧行为，唯一兜底为 HTTP 总超时）。 */
    private final long streamIdleMillis;
    private final BiConsumer<String, TokenUsage> usageSink;
    /** #340：估算 token 修正系数（LLM_TOKEN_FACTOR；默认 1.0）。 */
    private final double tokenFactor;

    /**
     * @param chatModel         非流式客户端（单次调用 + 流式不可用时的回退）
     * @param streamingModel    流式客户端；可为 null（此时 stream 全走非流式）
     * @param modelName         模型名（usageSink 的第一个参数）
     * @param toolSafeStreaming true = 绑定工具时退回非流式（快档标志）
     * @param usageSink         usage 回调；null 视为 no-op
     */
    public LangChain4jRuntime(ChatModel chatModel,
                              StreamingChatModel streamingModel,
                              String modelName,
                              boolean toolSafeStreaming,
                              BiConsumer<String, TokenUsage> usageSink) {
        this(chatModel, streamingModel, modelName, toolSafeStreaming, 0, 0, usageSink);
    }

    /** 全参形态（#4）：contextWindowTokens 随运行时走（0 = 未配置）。 */
    public LangChain4jRuntime(ChatModel chatModel,
                              StreamingChatModel streamingModel,
                              String modelName,
                              boolean toolSafeStreaming,
                              int contextWindowTokens,
                              BiConsumer<String, TokenUsage> usageSink) {
        this(chatModel, streamingModel, modelName, toolSafeStreaming, contextWindowTokens, 0, usageSink);
    }

    /** 最全形态（#74）：streamIdleMillis = 流式 idle 超时（毫秒，0 = 禁用）。 */
    public LangChain4jRuntime(ChatModel chatModel,
                              StreamingChatModel streamingModel,
                              String modelName,
                              boolean toolSafeStreaming,
                              int contextWindowTokens,
                              long streamIdleMillis,
                              BiConsumer<String, TokenUsage> usageSink) {
        this(chatModel, streamingModel, modelName, toolSafeStreaming, contextWindowTokens,
                streamIdleMillis, 1.0, usageSink);
    }

    /** 全参形态（#340）：tokenFactor = 估算修正系数（随 profile 运行时走）。 */
    public LangChain4jRuntime(ChatModel chatModel,
                              StreamingChatModel streamingModel,
                              String modelName,
                              boolean toolSafeStreaming,
                              int contextWindowTokens,
                              long streamIdleMillis,
                              double tokenFactor,
                              BiConsumer<String, TokenUsage> usageSink) {
        this.chatModel = Objects.requireNonNull(chatModel, "chatModel is required");
        this.streamingModel = streamingModel;
        this.modelName = Objects.requireNonNull(modelName, "modelName is required");
        this.toolSafeStreaming = toolSafeStreaming;
        this.contextWindowTokens = Math.max(0, contextWindowTokens);
        this.streamIdleMillis = Math.max(0, streamIdleMillis);
        this.tokenFactor = tokenFactor > 0 ? tokenFactor : 1.0;
        this.usageSink = usageSink == null ? ModelRuntime.noopUsageSink() : usageSink;
    }

    /** #340：估算修正系数（透明面，Graphs/ExecuteNode 等取值点经 ctx.llm() 读取）。 */
    @Override
    public double tokenFactor() {
        return tokenFactor;
    }

    @Override
    public boolean toolSafeStreaming() {
        return toolSafeStreaming;
    }

    @Override
    public int contextWindowTokens() {
        return contextWindowTokens;
    }

    @Override
    public ChatModel chat() {
        return chatModel;
    }

    @Override
    public StreamingChatModel streaming() {
        return streamingModel;
    }

    /** 模型名（usageSink 的第一个参数；测试与成本归集用）。 */
    public String modelName() {
        return modelName;
    }

    @Override
    public AiMessage invoke(List<ChatMessage> messages) {
        ChatResponse response = chatModel.chat(messages);
        usageSink.accept(modelName, response.tokenUsage());
        AiMessage message = response.aiMessage();
        return message == null ? null : withAuditMetadata(message, response.finishReason());
    }

    @Override
    public AiMessage stream(List<ChatMessage> messages, List<ToolSpecification> tools, Consumer<String> onDelta) {
        boolean bindTools = tools != null && !tools.isEmpty();
        List<ToolSpecification> boundTools = bindTools ? List.copyOf(tools) : List.of();
        if (streamingModel == null || (toolSafeStreaming && bindTools)) {
            return nonStreamFallback(messages, boundTools, onDelta);
        }
        return streamNative(messages, boundTools, onDelta);
    }

    /**
     * 原生流式：逐段回调 deltas，最终用"累计文本 + 聚合后的 tool_calls"
     * 显式重造 AiMessage。
     *
     * <p>显式构造对应 Python 侧图执行 {@code _stream_and_accumulate} 的
     * 显式 AIMessage：不直接采信 provider 在 onCompleteResponse 里给出的
     * 聚合结果，而是用回调累计出的完整文本重造，保证 tool_calls 聚合校验
     * 通过（流式协议下文本与工具调用参数跨多块到达，聚合错误会在校验处
     * 显式失败而不是静默给出空回答）。
     */
    private AiMessage streamNative(List<ChatMessage> messages,
                                   List<ToolSpecification> tools,
                                   Consumer<String> onDelta) {
        // 工具规格经 ChatRequestParameters 下发（对应 Python bind_tools 后
        // 透传的 kwargs["tools"]）；空列表等价于不绑定工具。
        ChatRequest request = ChatRequest.builder()
                .messages(messages)
                .parameters(ChatRequestParameters.builder().toolSpecifications(tools).build())
                .build();
        CompletableFuture<ChatResponse> done = new CompletableFuture<>();
        StringBuilder deltas = new StringBuilder();
        // #74：流式 idle 看门狗——provider 半途停止发 delta 且不断连接时，
        // 无限 join 会把图节点线程挂死（唯一兜底是 600s HTTP 总超时）。
        // 记录最近活动时刻，等待侧轮询；idleMillis<=0 = 禁用（旧行为）。
        java.util.concurrent.atomic.AtomicLong lastActivity =
                new java.util.concurrent.atomic.AtomicLong(System.nanoTime());
        streamingModel.chat(request, new StreamingChatResponseHandler() {
            @Override
            public void onPartialResponse(String token) {
                lastActivity.set(System.nanoTime());
                deltas.append(token);
                onDelta.accept(token);
            }

            @Override
            public void onCompleteResponse(ChatResponse response) {
                done.complete(response);
            }

            @Override
            public void onError(Throwable error) {
                done.completeExceptionally(error);
            }
        });
        ChatResponse response = awaitResponse(done, lastActivity);
        AiMessage aggregated = response.aiMessage();
        List<ToolExecutionRequest> toolRequests = aggregated == null || aggregated.toolExecutionRequests() == null
                ? List.of()
                : aggregated.toolExecutionRequests();
        usageSink.accept(modelName, response.tokenUsage());
        AiMessage rebuilt = new AiMessage(deltas.toString(), List.copyOf(toolRequests));
        return withAuditMetadata(rebuilt, response.finishReason());
    }

    /**
     * 给 AiMessage 挂稳定 id 与 finish_reason（#123）：langchain4j 1.4 的
     * {@code attributes} 是现成承载位（调研候选时的「无承载位」结论对 1.4
     * 不成立）。id = "msg-" + 12 位随机 hex（与 ToolPool callId 同风格），
     * 每次模型调用唯一——审计/前端锚点用；既有 attributes 保留。usage 不
     * 入 attributes：计量走 usageSink 唯一收口，杜绝双计。
     */
    private static AiMessage withAuditMetadata(AiMessage message, dev.langchain4j.model.output.FinishReason finishReason) {
        java.util.Map<String, Object> attrs = new java.util.LinkedHashMap<>();
        if (message.attributes() != null) {
            attrs.putAll(message.attributes());
        }
        attrs.putIfAbsent("id", "msg-" + UUID.randomUUID().toString().replace("-", "").substring(0, 12));
        if (finishReason != null) {
            attrs.putIfAbsent("finish_reason", finishReason.name().toLowerCase(java.util.Locale.ROOT));
        }
        return AiMessage.builder()
                .text(message.text())
                .thinking(message.thinking())
                .toolExecutionRequests(message.toolExecutionRequests())
                .attributes(attrs)
                .build();
    }

    /**
     * 非流式回退：整段文本作为单个 onDelta 回调后原样返回 AiMessage。
     *
     * <p>对应 Python {@code _ToolSafeStreamChatOpenAI._astream}：绑定了
     * 工具（或根本没有流式客户端）时改走 {@code _generate}，并把非流式
     * generation 整体 yield 出去——SSE delta 体验退化为一次性整段，
     * 服务端聚合出的 tool_calls 格式保证正确。
     */
    private AiMessage nonStreamFallback(List<ChatMessage> messages,
                                        List<ToolSpecification> tools,
                                        Consumer<String> onDelta) {
        ChatResponse response = tools.isEmpty()
                ? chatModel.chat(messages)
                : chatModel.chat(ChatRequest.builder()
                        .messages(messages)
                        .parameters(ChatRequestParameters.builder().toolSpecifications(tools).build())
                        .build());
        AiMessage message = response.aiMessage();
        if (message.text() != null && !message.text().isEmpty()) {
            onDelta.accept(message.text());
        }
        usageSink.accept(modelName, response.tokenUsage());
        return withAuditMetadata(message, response.finishReason());
    }

    /** 等待流式完成；onError 投递的异常转成 RuntimeException 上抛。 */
    /** #74：流式断流（idle 超时）——ResilientModelRuntime 重试边界照常适用。 */
    public static final class StreamStalledException extends IllegalStateException {
        public StreamStalledException(String message) {
            super(message);
        }
    }

    private ChatResponse awaitResponse(CompletableFuture<ChatResponse> done,
                                       java.util.concurrent.atomic.AtomicLong lastActivity) {
        // #74：idle 轮询等待（不引入第三种线程族；sleep 可被 interrupt 打断
        // ——与 SSE 取消语义天然兼容）。超时 = completeExceptionally，异常
        // 经 join 的 CompletionException 还原为 StreamStalledException。
        while (!done.isDone()) {
            if (streamIdleMillis > 0
                    && System.nanoTime() - lastActivity.get() > streamIdleMillis * 1_000_000L) {
                done.completeExceptionally(new StreamStalledException(
                        "llm 流 idle 超时（%dms 无 delta 且未结束）".formatted(streamIdleMillis)));
            }
            try {
                Thread.sleep(100);
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException("llm 流等待被中断", interrupted);
            }
        }
        try {
            return done.join();
        } catch (CompletionException e) {
            Throwable cause = e.getCause() == null ? e : e.getCause();
            if (cause instanceof RuntimeException runtime) {
                throw runtime;
            }
            throw new IllegalStateException("llm streaming call failed", cause);
        }
    }
}
