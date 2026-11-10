package io.github.paopaonb666.agentbase.core.llm;

import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.model.openai.OpenAiStreamingChatModel;
import dev.langchain4j.model.output.TokenUsage;
import io.github.paopaonb666.agentbase.core.config.Settings;

import java.util.List;
import java.util.function.BiConsumer;

/**
 * 根据已校验的 settings 装配对话模型运行时（对应 Python core/llm.py 的
 * {@code build_llm}）。
 *
 * <p>所有 provider（deepseek / zhipu / openai-compatible）共用同一个 OpenAI
 * 兼容客户端（{@code OpenAiChatModel} / {@code OpenAiStreamingChatModel}），
 * 切换 provider 意味着修改 {@code LLM_BASE_URL} / {@code LLM_MODEL}——
 * 永远不改代码。provider 名已在 {@link Settings} 校验，这里只做拼写兜底。
 *
 * <p>双档位（成本治理）：{@code profile="fast"} 在 {@code LLM_FAST_*} 已配置
 * 时返回独立运行时（免费/低价模型吃批量任务）。未配置快档时回落主力配置
 * 构建（#128：对象不同、配置相同——对齐 Python {@code build_llm} 恒返回
 * 客户端的语义），且快档恒走工具安全装配（含回落形态，见下）。
 * 分流决策（要不要用快档）由调用方按 {@code LLM_FAST_*} 是否配置自行做出
 * ——{@code profileView} 未配置时仍显式 400（API 契约，#289 锁定）。
 *
 * <p>快档走工具安全装配（{@code toolSafeStreaming=true}）：SiliconFlow 的
 * GLM-4-9B 在流式模式下把工具名拆进 {@code arguments} delta（
 * {@code function.name} 为空、id 为 null），违反 OpenAI 流式协议——
 * langchain 聚合不出合法的 tool_calls，图执行静默拿到空工具调用与空回答
 * （同请求非流式则完全正常）。因此快档一旦进入图执行路径（Tier 1 /
 * planner fast 都会 bind_tools），工具轮必须走非流式让服务端聚合出正确
 * 格式；不绑工具的对话仍流式，SSE delta 体验不变。主档保持原生流式，
 * 行为零变化。
 */
public final class LlmFactory {

    /** LLM 档位单源（#220）：main=主力、fast=快档——校验、错误文案与 /v1/modules 发现面共用。 */
    public static final List<String> PROFILES = List.of("main", "fast");

    private LlmFactory() {
    }

    /**
     * 构建对话模型运行时。
     *
     * @param settings 已校验的配置（provider 拼写等已由 Settings 校验）
     * @param profile  {@code "main"} 或 {@code "fast"}；其他值抛
     *                 {@link IllegalArgumentException}（对应 Python ValueError）
     * @param usageSink usage 回调 {@code (model, usage)}（对应 Python 在
     *                  cost.enabled 时挂的 CostMeterHandler；null 视为 no-op）
     * @return ModelRuntime（恒非 null）。{@code profile="fast"} 且
     *         {@code LLM_FAST_*} 未配置（base_url 与 model 任一为空）时
     *         用主力配置构建——对齐 Python {@code build_llm(profile="fast")
     *         } 的回落语义；且快档恒走工具安全装配（#128：回落路径复用
     *         toolSafe 判定，不把流式 tool_calls 协议缺陷重新暴露给回落
     *         流量）。分流决策由调用方按 {@code LLM_FAST_*} 是否配置做出。
     */
    public static ModelRuntime buildLlm(Settings settings, String profile, BiConsumer<String, TokenUsage> usageSink) {
        if (!PROFILES.contains(profile)) {
            throw new IllegalArgumentException(
                    "unknown llm profile '%s'; expected '%s'".formatted(profile, String.join("' or '", PROFILES)));
        }

        String model;
        String baseUrl;
        String apiKey;
        boolean toolSafeStreaming;
        int contextWindowTokens;
        if ("fast".equals(profile)) {
            if (settings.llmFast.isConfigured()) {
                Settings.LlmFastSettings fast = settings.llmFast;
                model = fast.model;
                baseUrl = fast.baseUrl;
                // 快档未单独配 key 时共用主力 key（同一 provider 双档的常见形态）。
                String fastKey = fast.apiKey.value().strip();
                String mainKey = settings.llm.apiKey.value().strip();
                apiKey = !fastKey.isEmpty() ? fastKey : mainKey;
                contextWindowTokens = fast.maxContextTokens;
            } else {
                // #128：未配置回落主力配置（对象不同、配置相同——Python 同构），
                // 不再返回 null 让调用方拿到原生主档运行时。窗口同随主力配置。
                model = settings.llm.model;
                baseUrl = settings.llm.baseUrl;
                apiKey = settings.llm.apiKey.value().strip();
                contextWindowTokens = settings.llm.maxContextTokens;
            }
            toolSafeStreaming = true;
        } else {
            model = settings.llm.model;
            baseUrl = settings.llm.baseUrl;
            apiKey = settings.llm.apiKey.value().strip();
            toolSafeStreaming = false;
            contextWindowTokens = settings.llm.maxContextTokens;
        }
        if (apiKey.isEmpty()) {
            // 环境兜底：对应 langchain-openai 的标准回退（LLM_API_KEY 未设置
            // 时客户端读环境中的 OPENAI_API_KEY）。
            String ambient = System.getenv("OPENAI_API_KEY");
            if (ambient != null) {
                apiKey = ambient.strip();
            }
        }
        if (apiKey.isEmpty()) {
            // Python 侧 ChatOpenAI 在没有任何 key 时构造即抛 OpenAIError，
            // 这里显式转成同文案的 SettingsError——langchain4j 会接受空白
            // key 并把错误推迟到首次调用（401），fail-fast 语义就丢了。
            throw new Settings.SettingsError(
                    "LLM_API_KEY is not configured and no ambient OPENAI_API_KEY is set; "
                            + "copy .env.example to .env, set LLM_API_KEY, then retry");
        }

        try {
            // 读超时：主档走 LLM_MAIN_TIMEOUT（#1，默认对齐 Python openai
            // SDK 的 600s）——不设的话 langchain4j 默认 60s，长生成/长停顿
            // 的流会在 Java 侧被提前掐断。快档无独立配置，沿用 600s。
            java.time.Duration timeout = "main".equals(profile)
                    ? java.time.Duration.ofMillis((long) (settings.llm.mainTimeoutSeconds * 1000))
                    : java.time.Duration.ofSeconds(600);
            // #340：修正系数随 profile（快档独立配置），经运行时到达取值点。
            double tokenFactor = "fast".equals(profile)
                    ? settings.llmFast.tokenFactor : settings.llm.tokenFactor;
            return buildOpenAiRuntime(baseUrl, apiKey, model, timeout,
                    toolSafeStreaming, contextWindowTokens,
                    (long) (settings.llm.streamIdleSeconds * 1000), tokenFactor, usageSink);
        } catch (RuntimeException exc) {
            // 客户端构造失败（典型：完全没有任何 API key）转成可操作的
            // SettingsError，而不是泄漏 langchain4j 的原始异常——对应 Python
            // 把 openai.OpenAIError 包成 SettingsError 的语义。
            throw new Settings.SettingsError(
                    "LLM_API_KEY is not configured and no ambient OPENAI_API_KEY is set; "
                            + "copy .env.example to .env, set LLM_API_KEY, then retry");
        }
    }

    /** 单个 openai 兼容客户端装配（主档与 #71 fallback 链共用）。 */
    private static LangChain4jRuntime buildOpenAiRuntime(String baseUrl, String apiKey, String model,
                                                         java.time.Duration timeout,
                                                         boolean toolSafeStreaming, int contextWindowTokens,
                                                         long streamIdleMillis, double tokenFactor,
                                                         BiConsumer<String, TokenUsage> usageSink) {
        OpenAiChatModel chat = OpenAiChatModel.builder()
                .baseUrl(baseUrl)
                .apiKey(apiKey)
                .modelName(model)
                .timeout(timeout)
                .build();
        OpenAiStreamingChatModel streaming = OpenAiStreamingChatModel.builder()
                .baseUrl(baseUrl)
                .apiKey(apiKey)
                .modelName(model)
                .timeout(timeout)
                .build();
        return new LangChain4jRuntime(chat, streaming, model, toolSafeStreaming,
                contextWindowTokens, streamIdleMillis, tokenFactor, usageSink);
    }

    /**
     * fallback 链构建（#71，Java 单边增强）：按配置顺序为每个备用模型构建
     * 独立运行时（各自带 modelName——usage 按实际生效模型记账自动成立）。
     * api_key 留空共用主力 key（与快档同先例）；主力 key 也为空时该元素
     * 报 SettingsError（构造期 fail-fast，运行期不再探测配置）。
     */
    public static List<ModelRuntime> buildFallbacks(Settings settings, BiConsumer<String, TokenUsage> usageSink) {
        if (settings.llm.fallbacks.isEmpty()) {
            return List.of();
        }
        List<ModelRuntime> runtimes = new java.util.ArrayList<>();
        for (int i = 0; i < settings.llm.fallbacks.size(); i++) {
            Settings.LlmFallbackSpec spec = settings.llm.fallbacks.get(i);
            String apiKey = spec.apiKey().value().strip();
            if (apiKey.isEmpty()) {
                apiKey = settings.llm.apiKey.value().strip();
            }
            if (apiKey.isEmpty()) {
                throw new Settings.SettingsError(
                        "LLM_FALLBACKS[%d] has no api_key and LLM_API_KEY is empty; "
                                + "set an api_key on the fallback entry or configure LLM_API_KEY".formatted(i));
            }
            try {
                runtimes.add(buildOpenAiRuntime(spec.baseUrl(), apiKey, spec.model(),
                        java.time.Duration.ofMillis((long) (settings.llm.mainTimeoutSeconds * 1000)),
                        false, spec.maxContextTokens(),
                        (long) (settings.llm.streamIdleSeconds * 1000),
                        settings.llm.tokenFactor, usageSink));
            } catch (RuntimeException exc) {
                throw new Settings.SettingsError(
                        "LLM_FALLBACKS[%d] client build failed: %s".formatted(i, exc.getMessage()));
            }
        }
        return List.copyOf(runtimes);
    }
}
