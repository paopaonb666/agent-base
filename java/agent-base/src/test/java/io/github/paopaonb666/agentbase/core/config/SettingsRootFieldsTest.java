package io.github.paopaonb666.agentbase.core.config;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

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

    @Test
    void healthProbeKeyControlsTheProbeSwitch() {
        // #101：正确键 HEALTH_PROBE_MODEL → observability.healthProbeModel。
        Settings on = Settings.load(Map.of("HEALTH_PROBE_MODEL", "true"), null);
        assertTrue(on.observability.healthProbeModel);
        Settings off = Settings.load(Map.of("HEALTH_PROBE_MODEL", "false"), null);
        assertFalse(off.observability.healthProbeModel);
    }

    @Test
    void staleLlmHealthProbeKeyIsRejectedNotSwallowed() {
        // #101：旧写法 LLM_HEALTH_PROBE_MODEL 会被 llm_ 前缀抬进 llm 节后
        // 被 applyLlm 静默丢弃（「照文档配置不生效」）——现在显式拒绝并指路。
        Settings.SettingsError error = assertThrows(Settings.SettingsError.class,
                () -> Settings.load(Map.of("LLM_HEALTH_PROBE_MODEL", "false"), null));
        assertTrue(error.getMessage().contains("HEALTH_PROBE_MODEL"),
                "拒绝文案要指路正确键：" + error.getMessage());
        // 小写同拒（lift 内部先归一小写）。
        assertThrows(Settings.SettingsError.class,
                () -> Settings.load(Map.of("llm_health_probe_model", "false"), null));
    }

    @Test
    void costPricesParseOnceAndNeverSilentlyZero() {
        // #105：数值串按 Python float() 口径接受（"2" → 2.0，不是静默 0）。
        Settings s = Settings.load(Map.of(
                "ENV", "development",
                "COST_PRICES_JSON", "{\"m\": {\"input_miss\": 2.5, \"input_hit\": \"1\", \"output\": 10}}"),
                null);
        assertEquals(Map.of("m", Map.of("input_miss", 2.5, "input_hit", 1.0, "output", 10.0)),
                s.prices());
        // 非数值 fail-fast（修复前 asDouble 静默回 0.0——白烧钱不可见）。
        Settings.SettingsError bad = assertThrows(Settings.SettingsError.class,
                () -> Settings.load(Map.of(
                        "ENV", "development",
                        "COST_PRICES_JSON", "{\"m\": {\"input_miss\": \"abc\"}}"), null));
        assertTrue(bad.getMessage().contains("input_miss"), bad.getMessage());
        // 结构错误照旧拒绝。
        assertThrows(Settings.SettingsError.class,
                () -> Settings.load(Map.of(
                        "ENV", "development",
                        "COST_PRICES_JSON", "[{\"m\": 1}]"), null));
        // 未配置 → 空表。
        assertEquals(Map.of(), Settings.load(Map.of("ENV", "development"), null).prices());
    }
}
