package io.github.paopaonb666.agentbase.core.llm;

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

    /**
     * 成本计量 sink：投递给成本计量单例
     * {@code CostMeters.cost().recordUsage(model, profile, input, output, 0)}。
     *
     * <p>说明：{@code CostMeter.recordUsage → observe} 内部已经无条件调用
     * {@code Metrics.cost().observeTokens(model, profile, "input"/"output", …)}
     * （Prometheus 侧的 llm_tokens_total），所以这里不再显式调用
     * {@code observeTokens}——再调一次会把 token 计数翻倍。
     *
     * <p>档位用调用方声明的 {@code profile}（"main"/"fast"），与 Python
     * {@code CostMeterHandler(model=model, profile=profile, …)} 的语义一致。
     * {@code usage == null}（provider 未回报 usage）时不投递——与 Python
     * on_llm_end 拿不到 usage 就什么都不记的行为一致；input/output 为 null
     * 按 0 计。
     *
     * @param profile 档位标签（"main" 或 "fast"），进账本与指标维度
     */
    public static BiConsumer<String, TokenUsage> costMeterSink(String profile) {
        String safeProfile = profile == null ? "main" : profile;
        return (model, usage) -> {
            if (usage == null) {
                return;
            }
            long input = usage.inputTokenCount() == null ? 0 : usage.inputTokenCount();
            long output = usage.outputTokenCount() == null ? 0 : usage.outputTokenCount();
            CostMeters.cost().recordUsage(model, safeProfile, input, output, 0);
        };
    }
}
