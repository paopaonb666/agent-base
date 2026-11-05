package io.github.paopaonb666.agentbase.core.llm;

import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.model.ModelProvider;
import dev.langchain4j.model.chat.Capability;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.listener.ChatModelListener;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ChatRequestParameters;
import dev.langchain4j.model.chat.request.DefaultChatRequestParameters;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import dev.langchain4j.model.output.TokenUsage;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.Set;
import java.util.function.BiConsumer;
import java.util.function.Consumer;

/**
 * 共享的脚本化测试替身（对应 Python tests/fakes.py 的
 * {@code ScriptedChatModel} 语义）：
 *
 * <ul>
 *   <li>{@link ScriptedChatModel}：每次模型调用弹出一条脚本化回复；耗尽后
 *       抛 {@link IllegalStateException}（Python 侧是 RuntimeError，Java
 *       契约下用 IllegalStateException 表达同样的"脚本用尽"）。录下每次
 *       收到的完整输入供断言（对应 Python {@code received}）。</li>
 *   <li>{@link ScriptedStreamingModel}：把一条 AiMessage 按 token 切片逐段
 *       回调 {@code onPartialResponse}，最后回调完整 ChatResponse（含可选
 *       TokenUsage）；脚本化的错误经 {@code onError} 投递——与真实流式
 *       provider 的失败路径同构。</li>
 *   <li>{@link ScriptedRuntime}：把两者装进生产的 {@link LangChain4jRuntime}
 *       ——替身走的是与线上完全相同的 stream/invoke 逻辑（流式聚合、
 *       tool-safe 回退、usage 钩子都有），而不是另写一份简化实现。</li>
 * </ul>
 */
final class ScriptedModels {

    private ScriptedModels() {
    }

    /** 一条脚本化回复：正常消息（可带 TokenUsage）或直接抛出的错误。 */
    record ScriptedResponse(AiMessage message, TokenUsage usage, RuntimeException error) {

        static ScriptedResponse of(AiMessage message) {
            return new ScriptedResponse(message, null, null);
        }

        static ScriptedResponse of(String text) {
            return of(new AiMessage(text));
        }

        static ScriptedResponse of(String text, TokenUsage usage) {
            return new ScriptedResponse(new AiMessage(text), usage, null);
        }

        /** 工具调用轮回复：文本 + 聚合后的工具调用请求。 */
        static ScriptedResponse withToolCalls(String text, List<ToolExecutionRequest> calls) {
            return of(new AiMessage(text, List.copyOf(calls)));
        }

        static ScriptedResponse failing(RuntimeException error) {
            return new ScriptedResponse(null, null, error);
        }
    }

    /** 从队列弹出下一条脚本回复；耗尽即抛（Python: scripted responses exhausted）。 */
    private static ScriptedResponse pollNext(Deque<ScriptedResponse> queue) {
        ScriptedResponse next = queue.poll();
        if (next == null) {
            throw new IllegalStateException("ScriptedChatModel: no more responses");
        }
        return next;
    }

    /** langchain4j 接口的默认实现要求（默认 chat() 会合并参数/记录 provider）。 */
    private static ChatRequestParameters emptyParameters() {
        return DefaultChatRequestParameters.EMPTY;
    }

    /** 非流式替身：doChat 弹出一条脚本回复；录下收到的输入。 */
    static final class ScriptedChatModel implements ChatModel {
        private final Deque<ScriptedResponse> queue;
        private final List<List<ChatMessage>> received = new ArrayList<>();

        ScriptedChatModel(List<ScriptedResponse> responses) {
            this.queue = new ArrayDeque<>(responses);
        }

        /** 每次模型调用收到的完整输入（诊断/断言用，对应 Python received）。 */
        List<List<ChatMessage>> received() {
            return received;
        }

        @Override
        public ChatResponse doChat(ChatRequest request) {
            received.add(List.copyOf(request.messages()));
            ScriptedResponse next = pollNext(queue);
            if (next.error != null) {
                throw next.error;
            }
            return ChatResponse.builder()
                    .aiMessage(next.message)
                    .tokenUsage(next.usage)
                    .build();
        }

        @Override
        public ChatRequestParameters defaultRequestParameters() {
            return emptyParameters();
        }

        @Override
        public List<ChatModelListener> listeners() {
            return List.of();
        }

        @Override
        public ModelProvider provider() {
            return ModelProvider.OTHER;
        }

        @Override
        public Set<Capability> supportedCapabilities() {
            return Set.of();
        }
    }

    /** 流式替身：AiMessage 按 ~4 字符切片逐段回调，最后 onCompleteResponse。 */
    static final class ScriptedStreamingModel implements StreamingChatModel {
        private final Deque<ScriptedResponse> queue;
        private final List<List<ChatMessage>> received = new ArrayList<>();

        ScriptedStreamingModel(List<ScriptedResponse> responses) {
            this.queue = new ArrayDeque<>(responses);
        }

        List<List<ChatMessage>> received() {
            return received;
        }

        @Override
        public void doChat(ChatRequest request, StreamingChatResponseHandler handler) {
            received.add(List.copyOf(request.messages()));
            ScriptedResponse next = pollNext(queue);
            if (next.error != null) {
                handler.onError(next.error);
                return;
            }
            String text = next.message.text() == null ? "" : next.message.text();
            // 固定小切片模拟 token 增量：真实 provider 的切片粒度不可预期，
            // 多片到达足以驱动"逐 delta 聚合"路径。
            for (int i = 0; i < text.length(); i += 4) {
                handler.onPartialResponse(text.substring(i, Math.min(text.length(), i + 4)));
            }
            handler.onCompleteResponse(ChatResponse.builder()
                    .aiMessage(next.message)
                    .tokenUsage(next.usage)
                    .build());
        }

        @Override
        public ChatRequestParameters defaultRequestParameters() {
            return emptyParameters();
        }

        @Override
        public List<ChatModelListener> listeners() {
            return List.of();
        }

        @Override
        public ModelProvider provider() {
            return ModelProvider.OTHER;
        }

        @Override
        public Set<Capability> supportedCapabilities() {
            return Set.of();
        }
    }

    /**
     * 组合替身：ScriptedChatModel + ScriptedStreamingModel 装进生产
     * {@link LangChain4jRuntime}，全部 ModelRuntime 方法委托给它。
     */
    static final class ScriptedRuntime implements ModelRuntime {
        final ScriptedChatModel chatModel;
        final ScriptedStreamingModel streamingModel;
        private final LangChain4jRuntime delegate;

        /** 单脚本便捷构造：同一份脚本同时驱动非流式与流式视图。 */
        ScriptedRuntime(List<ScriptedResponse> responses,
                        String modelName,
                        boolean toolSafeStreaming,
                        BiConsumer<String, TokenUsage> usageSink) {
            this(responses, responses, modelName, toolSafeStreaming, usageSink);
        }

        /** 双脚本构造：invoke 走 chat 脚本，stream 走流式脚本。 */
        ScriptedRuntime(List<ScriptedResponse> chatResponses,
                        List<ScriptedResponse> streamResponses,
                        String modelName,
                        boolean toolSafeStreaming,
                        BiConsumer<String, TokenUsage> usageSink) {
            this.chatModel = new ScriptedChatModel(chatResponses);
            this.streamingModel = new ScriptedStreamingModel(streamResponses);
            this.delegate = new LangChain4jRuntime(chatModel, streamingModel, modelName,
                    toolSafeStreaming, usageSink);
        }

        /** 主档便捷工厂：toolSafeStreaming=false、no-op usage sink。 */
        static ScriptedRuntime main(List<ScriptedResponse> responses) {
            return new ScriptedRuntime(responses, "scripted-main", false, null);
        }

        /** 快档便捷工厂：toolSafeStreaming=true（绑工具退非流式）。 */
        static ScriptedRuntime fast(List<ScriptedResponse> responses) {
            return new ScriptedRuntime(responses, "scripted-fast", true, null);
        }

        @Override
        public ChatModel chat() {
            return delegate.chat();
        }

        @Override
        public StreamingChatModel streaming() {
            return delegate.streaming();
        }

        @Override
        public AiMessage stream(List<ChatMessage> messages, List<ToolSpecification> tools,
                                Consumer<String> onDelta) {
            return delegate.stream(messages, tools, onDelta);
        }

        @Override
        public AiMessage invoke(List<ChatMessage> messages) {
            return delegate.invoke(messages);
        }
    }
}
