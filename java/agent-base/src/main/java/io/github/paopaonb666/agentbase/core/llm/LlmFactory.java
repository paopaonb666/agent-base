package io.github.paopaonb666.agentbase.core.llm;

import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.model.openai.OpenAiStreamingChatModel;
import dev.langchain4j.model.output.TokenUsage;
import io.github.paopaonb666.agentbase.core.config.Settings;

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
 * 时返回独立运行时（免费/低价模型吃批量任务）。与 Python 的差异：
 * Python 侧未配置快档时 {@code build_llm(settings, "fast")} 返回与 main
 * 相同配置的客户端（对象不同、配置相同），而 Java 侧返回 {@code null}，
 * 由调用方（{@link ResilientLlm} / 组合根）显式回落主力运行时——可观察
 * 行为一致：快档流量最终都打到主力配置。
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
     *         流量）。调用方按 {@code LLM_FAST_*} 是否配置自行决定分流。
     */
    public static ModelRuntime buildLlm(Settings settings, String profile, BiConsumer<String, TokenUsage> usageSink) {
        if (!"main".equals(profile) && !"fast".equals(profile)) {
            throw new IllegalArgumentException(
                    "unknown llm profile '%s'; expected 'main' or 'fast'".formatted(profile));
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
                    contextWindowTokens, usageSink);
        } catch (RuntimeException exc) {
            // 客户端构造失败（典型：完全没有任何 API key）转成可操作的
            // SettingsError，而不是泄漏 langchain4j 的原始异常——对应 Python
            // 把 openai.OpenAIError 包成 SettingsError 的语义。
            throw new Settings.SettingsError(
                    "LLM_API_KEY is not configured and no ambient OPENAI_API_KEY is set; "
                            + "copy .env.example to .env, set LLM_API_KEY, then retry");
        }
    }
}
