package io.github.paopaonb666.agentbase.core.config;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

/**
 * 根级平铺 env 键的配置契约（对应 Python config.py 的根字段）。
 *
 * <p>回归：{@code liftLegacyFlatFields} 曾把收集了根级键的 {@code out}
 * 直接丢弃（只返回 {@code lifted}），导致 {@code ENV} / {@code AGENT_MODULES} /
 * {@code AGENT_MODULE_PACKAGES} / {@code CORS_ORIGINS} / {@code
 * AGENT_RECURSION_LIMIT} 从 env/.env/overrides 静默失效，
 * {@code ensureProductionReady} 的生产守卫（{@code env=="production"} 分支）
 * 整体不可达。
 */
class SettingsRootFieldsTest {

    @Test
    void rootLevelEnvKeysAreHonored() {
        Settings s = Settings.load(Map.of(
                "ENV", "production",
                "AGENT_MODULES", "chat,writer",
                "AGENT_MODULE_PACKAGES", "agent_base.modules",
                "CORS_ORIGINS", "https://ui.example.com",
                "AGENT_RECURSION_LIMIT", "40"), null);
        assertEquals("production", s.env);
        assertEquals(List.of("chat", "writer"), s.agentModules);
        assertEquals(List.of("agent_base.modules"), s.agentModulePackages);
        assertEquals(List.of("https://ui.example.com"), s.corsOrigins);
        assertEquals(40, s.agentRecursionLimit);
    }

    @Test
    void uppercaseRootKeysNormalizeToLowercase() {
        Settings s = Settings.load(Map.of("ENV", "production"), null);
        assertEquals("production", s.env);
    }
}
