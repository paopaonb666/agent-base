package io.github.paopaonb666.agentbase.core.config;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** #106：敏感配置值的 Secret 掩码类型——toString 恒 "***"，真值只经 value() 显式取出。 */
class SettingsSecretTest {

    @Test
    void toStringIsAlwaysMasked() {
        assertEquals("***", Secret.of("sk-real-key").toString());
        assertEquals("***", Secret.of("").toString());
        // 真值不进字符串拼接（意外日志/序列化的泄露面）。
        assertFalse(("leak=" + Secret.of("sk-real-key")).contains("sk-real-key"));
    }

    @Test
    void valueAndBlankSemantics() {
        assertEquals("sk-real-key", Secret.of("sk-real-key").value());
        assertTrue(Secret.of("").isBlank());
        assertTrue(Secret.of("   ").isBlank());
        assertTrue(Secret.of(null).isBlank());
        assertFalse(Secret.of("k").isBlank());
    }

    @Test
    void equalsByValue() {
        assertEquals(Secret.of("k"), Secret.of("k"));
        assertNotEquals(Secret.of("k"), Secret.of("j"));
    }

    @Test
    void loadedKeysLandInSecretFields() {
        Settings s = Settings.load(java.util.Map.of(
                "LLM_API_KEY", "sk-main",
                "TAVILY_API_KEY", "tvly-1",
                "MEMORY_AUTH_SECRET", "sec-1"), null);
        assertEquals("sk-main", s.llm.apiKey.value());
        assertEquals("tvly-1", s.search.tavilyApiKey.value());
        assertEquals("sec-1", s.memory.authSecret.value());
        // 默认字段为空 Secret（不是 null），isBlank 判空语义保持。
        assertTrue(new Settings().llmFast.apiKey.isBlank());
    }

    @Test
    void diagnosticExportMasksEverySecretFieldByType() {
        // A-#118：导出面 Secret 族逐键掩码——按字段类型判定（新增敏感键
        // 自动被掩），非敏感值原样，键名与 env 约定同 snake_case。
        Settings s = Settings.load(java.util.Map.of(
                "ENV", "development",
                "LLM_API_KEY", "sk-main",
                "TAVILY_API_KEY", "tvly-1",
                "CHECKPOINTER_MYSQL_PASSWORD", "pw-1"));
        var out = s.exportForDiagnostics();
        @SuppressWarnings("unchecked")
        var llm = (java.util.Map<String, Object>) out.get("llm");
        assertEquals("***", llm.get("api_key"));
        @SuppressWarnings("unchecked")
        var search = (java.util.Map<String, Object>) out.get("search");
        assertEquals("***", search.get("tavily_api_key"));
        assertEquals("tavily", search.get("engine_priority"));
        @SuppressWarnings("unchecked")
        var checkpointer = (java.util.Map<String, Object>) out.get("checkpointer");
        assertEquals("***", checkpointer.get("mysql_password"));
        // 全面性：导出树里除掩码外不出现任何 Secret 原值。
        String rendered = String.valueOf(out);
        assertFalse(rendered.contains("sk-main"));
        assertFalse(rendered.contains("tvly-1"));
        assertFalse(rendered.contains("pw-1"));

        // A-审查 P1 回归锁：LLM_FALLBACKS 的嵌套 Secret（record 内 apiKey）
        // 原样进导出树会让 Jackson（FAIL_ON_EMPTY_BEANS）500——消毒后按
        // 字符串渲染且密钥不落明文。
        Settings withFallback = Settings.load(java.util.Map.of(
                "ENV", "development",
                "LLM_FALLBACKS",
                "[{\"base_url\":\"https://fb.example/v1\",\"model\":\"fb-m\",\"api_key\":\"sk-fallback\"}]"), null);
        var fbOut = withFallback.exportForDiagnostics();
        String fbRendered = String.valueOf(fbOut);
        assertFalse(fbRendered.contains("sk-fallback"), "嵌套 Secret 不得落明文：" + fbRendered);
        assertTrue(fbRendered.contains("***"), "嵌套 Secret 渲染为掩码：" + fbRendered);
    }
}
