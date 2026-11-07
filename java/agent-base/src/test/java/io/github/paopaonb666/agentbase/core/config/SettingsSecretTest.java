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
}
