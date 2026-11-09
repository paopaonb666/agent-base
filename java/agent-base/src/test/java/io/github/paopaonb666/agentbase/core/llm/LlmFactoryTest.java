package io.github.paopaonb666.agentbase.core.llm;

import io.github.paopaonb666.agentbase.core.config.Secret;
import io.github.paopaonb666.agentbase.core.config.Settings;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@link LlmFactory#buildLlm} 的装配语义测试（对应 Python 侧 build_llm
 * 的行为）：profile 校验、快档未配置回落 null、快档模型取 LLM_FAST_*。
 *
 * <p>快档 api_key 空则共用主 key 的回退无法从装配好的客户端外部断言
 * （langchain4j 客户端不暴露 key），由实现内一行赋值保证（fastKey 或
 * mainKey），这里只测可观察的档位/模型装配。
 */
class LlmFactoryTest {

    private static Settings settingsWithFast() {
        Settings settings = new Settings();
        settings.llm.apiKey = Secret.of("sk-test");
        settings.llm.model = "deepseek-chat";
        settings.llmFast.baseUrl = "https://api.siliconflow.cn/v1";
        settings.llmFast.model = "THUDM/glm-4-9b-chat";
        return settings;
    }

    @Test
    void mainProfileBuildsRuntimeWithMainModel() {
        Settings settings = settingsWithFast();
        ModelRuntime runtime = LlmFactory.buildLlm(settings, "main", null);
        assertNotNull(runtime);
        LangChain4jRuntime lcRuntime = assertInstanceOf(LangChain4jRuntime.class, runtime);
        assertEquals("deepseek-chat", lcRuntime.modelName());
        // 主档保持原生流式、非工具安全（Python：主档用原生 ChatOpenAI）。
        assertNotNull(runtime.streaming());
        assertNotNull(runtime.chat());
    }

    @Test
    void fastUnconfiguredFallsBackToToolSafeMainConfig() {
        // #128：未配置回落主力配置（对象不同、配置相同——Python 同构），
        // 且回落路径同样走工具安全装配（不把流式 tool_calls 协议缺陷
        // 重新暴露给回落流量）。修复前返回 null，调用方拿到原生主档。
        Settings settings = new Settings(); // LLM_FAST_* 全空 = 未配置
        settings.llm.apiKey = Secret.of("sk-test");
        settings.llm.model = "deepseek-chat";
        ModelRuntime runtime = LlmFactory.buildLlm(settings, "fast", null);
        LangChain4jRuntime lcRuntime = assertInstanceOf(LangChain4jRuntime.class, runtime);
        assertEquals("deepseek-chat", lcRuntime.modelName(), "回落主力配置");
        assertTrue(runtime.toolSafeStreaming(), "回落路径必须工具安全（#128）");
    }

    @Test
    void fastPartiallyConfiguredAlsoFallsBack() {
        Settings settings = new Settings(); // 只有 base_url，缺 model = 未配置
        settings.llm.apiKey = Secret.of("sk-test");
        settings.llm.model = "deepseek-chat";
        settings.llmFast.baseUrl = "https://api.siliconflow.cn/v1";
        LangChain4jRuntime lcRuntime = assertInstanceOf(LangChain4jRuntime.class,
                LlmFactory.buildLlm(settings, "fast", null));
        assertEquals("deepseek-chat", lcRuntime.modelName());
        assertTrue(lcRuntime.toolSafeStreaming());
    }

    @Test
    void fastProfileBuildsRuntimeWithFastModel() {
        Settings settings = settingsWithFast();
        ModelRuntime runtime = LlmFactory.buildLlm(settings, "fast", null);
        assertNotNull(runtime);
        LangChain4jRuntime lcRuntime = assertInstanceOf(LangChain4jRuntime.class, runtime);
        assertEquals("THUDM/glm-4-9b-chat", lcRuntime.modelName());
        // 快档走工具安全装配（见 LlmFactory docstring：SiliconFlow GLM-4-9B
        // 流式 tool_calls 协议违规）。
        assertTrue(runtime.toolSafeStreaming());
        assertNotNull(runtime.streaming());
    }

    @Test
    void contextWindowTravelsWithTheRuntime() {
        // #4：窗口随运行时走——主档带主窗口；快档配置了带快窗口；
        // 快档未配置回落主力配置时带主力窗口（#128 同构）。
        Settings settings = settingsWithFast();
        settings.llm.maxContextTokens = 32768;
        settings.llmFast.maxContextTokens = 8192;
        assertEquals(32768, LlmFactory.buildLlm(settings, "main", null).contextWindowTokens());
        assertEquals(8192, LlmFactory.buildLlm(settings, "fast", null).contextWindowTokens());
        Settings noFast = new Settings();
        noFast.llm.apiKey = Secret.of("sk-test");
        noFast.llm.maxContextTokens = 32768;
        assertEquals(32768, LlmFactory.buildLlm(noFast, "fast", null).contextWindowTokens());
        // 未配置 → 0（预算推导沿用现状）。
        assertEquals(0, LlmFactory.buildLlm(settingsWithFast(), "main", null).contextWindowTokens());
    }

    @Test
    void unknownProfileThrows() {
        Settings settings = settingsWithFast();
        IllegalArgumentException exc = assertThrows(IllegalArgumentException.class,
                () -> LlmFactory.buildLlm(settings, "turbo", null));
        assertEquals("unknown llm profile 'turbo'; expected 'main' or 'fast'", exc.getMessage());
    }
}
