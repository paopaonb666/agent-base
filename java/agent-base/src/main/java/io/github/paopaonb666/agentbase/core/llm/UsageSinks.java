package io.github.paopaonb666.agentbase.core.llm;

import dev.langchain4j.model.openai.OpenAiTokenUsage;
import dev.langchain4j.model.output.TokenUsage;
import io.github.paopaonb666.agentbase.extensions.costmeter.CostMeters;

import java.util.function.BiConsumer;

/**
 * usage 回调（usageSink）的现成实现，对应 Python core/llm.py 在
 * {@code cost.enabled} 时挂在模型上的 {@code CostMeterHandler}：
 * 把每次调用结束的 {@code (model, TokenUsage)} 转投成本计量。
 *
 * <p>组合根在 cost.enabled 时把 {@link #costMeterSink} 传给
 * {@link LlmFactory#buildLlm}；未启用时传 {@link ModelRuntime#noopUsageSink()}
 * ——成本计量关闭时完全不挂任何回调（对应 Python 的条件挂载）。
 */
public final class UsageSinks {

    private UsageSinks() {
    }

    // ── A-L79+#76：attempt/fallback 写账上下文 ───────────────────────────

    /**
     * 一次真实调用所在的尝试位次（弹性包写账上下文）：ResilientModelRuntime
     * 在每次 {@code callWithLimit} 前后 set/clear——同线程可见（invoke 与
     * 流式回调同线程），costMeterSink 在记账时刻读取。fallback 序号仅在
     * 备用候选调用时非负（主档尝试恒 -1）。
     */
    public record AttemptContext(int attempt, int fallbackIndex) {
    }

    private static final ThreadLocal<AttemptContext> ATTEMPT = new ThreadLocal<>();

    /** 弹性包进入一次真实调用前设置（调用结束必须 clear）。 */
    public static void setAttemptContext(AttemptContext context) {
        ATTEMPT.set(context);
    }

    /** 弹性包退出真实调用后清除。 */
    public static void clearAttemptContext() {
        ATTEMPT.remove();
    }

    /** 记账时刻读取；无弹性包装（直连运行时/测试）返回 null（账本列写 NULL）。 */
    public static AttemptContext currentAttemptContext() {
        return ATTEMPT.get();
    }

    /**
     * 成本计量 sink：投递给成本计量单例
     * {@code CostMeters.cost().recordUsage(model, profile, input, output, cacheRead)}。
     *
     * <p>说明：{@code CostMeter.recordUsage → observe} 内部已经无条件调用
     * {@code Metrics.cost().observeTokens(model, profile, "input"/"output", …)}
     * （Prometheus 侧的 llm_tokens_total），所以这里不再显式调用
     * {@code observeTokens}——再调一次会把 token 计数翻倍。
     *
     * <p>cache-read（命中缓存的输入 token）从 OpenAI 兼容响应的
     * {@code prompt_tokens_details.cached_tokens} 提取（langchain4j 侧为
     * {@link OpenAiTokenUsage#inputTokensDetails()}）；对应 Python
     * {@code _extract_usage} 的 {@code cache_read}，命中部分由价目表按
     * {@code input_hit} 计价——硬编码 0 会把全部输入按 {@code input_miss}
     * 计价，系统性高计成本。非 OpenAI 用量对象或未回报缓存明细时按 0。
     *
     * <p>档位用调用方声明的 {@code profile}（"main"/"fast"），与 Python
     * {@code CostMeterHandler(model=model, profile=profile, …)} 的语义一致。
     * {@code usage == null}（provider 未回报 usage）时按 (0,0,0) 记账——
     * 对应 Python on_llm_end 的 {@code _extract_usage} 全零回退（账本里
     * 保留该次调用的一行，方便与请求日志对账）；input/output 为 null 按 0 计。
     *
     * @param profile 档位标签（"main" 或 "fast"），进账本与指标维度
     */
    public static BiConsumer<String, TokenUsage> costMeterSink(String profile) {
        String safeProfile = profile == null ? "main" : profile;
        return (model, usage) -> {
            long input = 0;
            long output = 0;
            long cacheRead = 0;
            if (usage != null) {
                input = usage.inputTokenCount() == null ? 0 : usage.inputTokenCount();
                output = usage.outputTokenCount() == null ? 0 : usage.outputTokenCount();
                if (usage instanceof OpenAiTokenUsage openAi) {
                    var details = openAi.inputTokensDetails();
                    if (details != null && details.cachedTokens() != null) {
                        cacheRead = details.cachedTokens();
                    }
                }
            }
            CostMeters.cost().recordUsage(model, safeProfile, input, output, cacheRead,
                    currentAttemptContext());
        };
    }
}
