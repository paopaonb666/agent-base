package io.github.paopaonb666.agentbase.core.llm;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.model.output.TokenUsage;

import java.util.List;
import java.util.function.BiConsumer;
import java.util.function.Consumer;

/**
 * 图执行用的模型运行时（对应 Python 侧 ChatOpenAI + astream 的封装面）。
 *
 * <p>两条约定（与 Python 侧对齐）：
 * <ul>
 *   <li><b>流式优先</b>：图节点走 {@link #stream}（token 增量直达 SSE）；
 *       实现方在"快档 + 绑定工具"场景可回退非流式（对应
 *       _ToolSafeStreamChatOpenAI：规避 SiliconFlow GLM-4-9B 流式
 *       tool_calls 协议违规——单 delta 直达，行为等价）。</li>
 *   <li><b>usage 钩子</b>：所有调用结束都回调 usageSink（成本计量的
 *       唯一入口，对应 CostMeterHandler.on_llm_end）。</li>
 * </ul>
 */
public interface ModelRuntime {

    /** 绑定的主模型（非流式单次调用 + 后台管线用）。 */
    ChatModel chat();

    /** 流式视图；不可用（未配置）时返回 null。 */
    StreamingChatModel streaming();

    /**
     * 流式（或实现回退非流式）调用：逐段回调 deltas，返回最终
     * AiMessage。返回值里的 usage 已经（或即将）投递给 usageSink。
     *
     * @param messages 模型输入（调用方已过上下文引擎修剪）
     * @param tools    可为 null/空 = 不绑定工具
     * @param onDelta  token 增量回调（非流式回退时整段一次）
     */
    AiMessage stream(List<ChatMessage> messages, List<ToolSpecification> tools, Consumer<String> onDelta);

    /** 非流式单次调用（后台管线用）。 */
    AiMessage invoke(List<ChatMessage> messages);

    /**
     * 工具安全标志（#128 可观测面）：true = 绑定工具时退回非流式
     * （对应 Python _ToolSafeStreamChatOpenAI 的选择逻辑）。快档恒 true
     * （含未配置回落主力配置的形态——回落路径同样要规避流式 tool_calls
     * 协议缺陷），主档 false。
     */
    /**
     * 估算 token 修正系数（#340）：provider 自有分词与 JTokkit 预算口径
     * 的系统性比值——按 profile 配置（LLM_TOKEN_FACTOR /
     * LLM_FAST_TOKEN_FACTOR，默认 1.0），乘在 ContextEngine 的消息计量
     * 出口上。透明层必须委托（与 contextWindowTokens 同契约）。
     */
    default double tokenFactor() {
        return 1.0;
    }

    default boolean toolSafeStreaming() {
        return false;
    }

    /**
     * 该运行时对应模型的上下文窗口（token，#4）：0 = 未配置（预算推导
     * 沿用 memory.contextMaxTokens 现状）。窗口随运行时走——快档/主力
     * 各自生效，配置面见 Settings.LlmSettings.maxContextTokens。
     */
    default int contextWindowTokens() {
        return 0;
    }

    /** usage 回调：(model, usage)；实现方在每次调用结束时投递一次。 */
    static BiConsumer<String, TokenUsage> noopUsageSink() {
        return (model, usage) -> {};
    }
}
