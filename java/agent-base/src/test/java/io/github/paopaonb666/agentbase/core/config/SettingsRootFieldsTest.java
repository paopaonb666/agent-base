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

    @Test
    void searchEnginesNarrowToTavilyWithGhostConfigIgnored() {
        // #30/#276 终局：默认 priority 指向真实存在的引擎（旧默认
        // duckduckgo 指向未实现引擎——照文档启用 web_search 必失败）；
        // duckduckgo 在启动时被显式拒绝；SEARCH_DDGS_BACKEND 是 Python
        // 专属幽灵配置，Java 侧忽略（不报错，Python 部署合法）。
        Settings defaults = Settings.load(Map.of("ENV", "development"), null);
        assertEquals("tavily", defaults.search.enginePriority);
        Settings.SettingsError rejected = assertThrows(Settings.SettingsError.class,
                () -> Settings.load(Map.of(
                        "ENV", "development",
                        "SEARCH_ENGINE_PRIORITY", "duckduckgo"), null));
        assertTrue(rejected.getMessage().contains("Python"), rejected.getMessage());
        Settings withGhost = Settings.load(Map.of(
                "ENV", "development",
                "SEARCH_DDGS_BACKEND", "bing",
                "SEARCH_ENGINE_PRIORITY", "tavily"), null);
        assertEquals("tavily", withGhost.search.enginePriority);
    }

    @Test
    void toolkitEnabledUnknownEntryFailsAtConfigTime() {
        // #103：名单校验前移到配置期（此前装配期才报，文案与 Toolkits 一致）。
        Settings.SettingsError error = assertThrows(Settings.SettingsError.class,
                () -> Settings.load(Map.of(
                        "ENV", "development",
                        "TOOLKIT_ENABLED", "calculator,nope"), null));
        assertTrue(error.getMessage().contains("unknown TOOLKIT_ENABLED entry 'nope'"),
                error.getMessage());
        // 未迁移条目 python_repl 仍在名单内（合法；装配期另有显式报错）。
        Settings s = Settings.load(Map.of(
                "ENV", "development",
                "TOOLKIT_ENABLED", "python_repl"), null);
        assertEquals(List.of("python_repl"), s.toolkit.enabled);
        // 默认 enabled 通过。
        assertEquals(List.of("current_time", "calculator", "json_query"),
                Settings.load(Map.of("ENV", "development"), null).toolkit.enabled);
    }

    @Test
    void baseUrlIsTrimmedAndTrailingSlashStripped() {
        // #110：base_url 规范化（trim+去尾斜杠）；自定义路径原样保留。
        Settings s = Settings.load(Map.of(
                "ENV", "development",
                "LLM_BASE_URL", "https://api.deepseek.com/v1/",
                "LLM_FAST_BASE_URL", "  https://gw.example.com/v1/openai//  "), null);
        assertEquals("https://api.deepseek.com/v1", s.llm.baseUrl);
        assertEquals("https://gw.example.com/v1/openai", s.llmFast.baseUrl);
        // 未配置时保持空串（fast 未配置语义不变）。
        assertEquals("", Settings.load(Map.of("ENV", "development"), null).llmFast.baseUrl);
    }

    @Test
    void authReplayCacheLimitParsesFromMemorySection() {
        // #207：nonce 重放缓存容量可配（MEMORY_AUTH_REPLAY_CACHE_LIMIT，
        // 默认 4096 与构造器默认一致）。
        assertEquals(4096,
                Settings.load(Map.of("ENV", "development"), null).memory.authReplayCacheLimit);
        Settings s = Settings.load(Map.of(
                "ENV", "development",
                "MEMORY_AUTH_REPLAY_CACHE_LIMIT", "2048"), null);
        assertEquals(2048, s.memory.authReplayCacheLimit);
        // 非正数 fail-fast：0 会让 LRU 淘汰分支恒真（刚记账就被挤出，
        // 缓存永远为空、防重放失效）。
        Settings.SettingsError error = assertThrows(Settings.SettingsError.class,
                () -> Settings.load(Map.of(
                        "ENV", "development",
                        "MEMORY_AUTH_REPLAY_CACHE_LIMIT", "0"), null));
        assertTrue(error.getMessage().contains("auth_replay_cache_limit"),
                "拒绝文案要点名出错字段：" + error.getMessage());
    }

    @Test
    void memoryCompactionKeysParseWithDefaultsAndValidation() {
        // #328 三键：默认关（vectorPushdown 先例）、ratio (0,1) 开区间、
        // keepRecentTurns >= 1。
        Settings defaults = Settings.load(Map.of("ENV", "development"), null);
        assertFalse(defaults.memory.compactionEnabled);
        assertEquals(0.7, defaults.memory.compactionRatio, 1e-9);
        assertEquals(3, defaults.memory.compactionKeepRecentTurns);

        Settings tuned = Settings.load(Map.of(
                "ENV", "development",
                "MEMORY_COMPACTION_ENABLED", "true",
                "MEMORY_COMPACTION_RATIO", "0.8",
                "MEMORY_COMPACTION_KEEP_RECENT_TURNS", "5"), null);
        assertTrue(tuned.memory.compactionEnabled);
        assertEquals(0.8, tuned.memory.compactionRatio, 1e-9);
        assertEquals(5, tuned.memory.compactionKeepRecentTurns);

        // ratio 开区间：两端点（0 = 必触发、1 = 永不触发）与越界值拒绝。
        for (String bad : new String[]{"0", "1.0", "1", "-0.5", "1.5"}) {
            Settings.SettingsError ratioError = assertThrows(Settings.SettingsError.class,
                    () -> Settings.load(Map.of(
                            "ENV", "development",
                            "MEMORY_COMPACTION_RATIO", bad), null),
                    "ratio=" + bad + " 应拒绝");
            assertTrue(ratioError.getMessage().contains("MEMORY_COMPACTION_RATIO"),
                    ratioError.getMessage());
        }
        // keepRecentTurns 下限：0 = 保留窗为空，全部历史折叠（违背「近 N 轮
        // 原文」契约）→ 拒绝。
        Settings.SettingsError turnsError = assertThrows(Settings.SettingsError.class,
                () -> Settings.load(Map.of(
                        "ENV", "development",
                        "MEMORY_COMPACTION_KEEP_RECENT_TURNS", "0"), null));
        assertTrue(turnsError.getMessage().contains("MEMORY_COMPACTION_KEEP_RECENT_TURNS"),
                turnsError.getMessage());
    }

    @Test
    void memoryContextBudgetKeysParseWithDefaultsAndNegativeRejected() {
        // 条目 15 四域预算键：默认全 0（=关/沿用既有口径，现状逐字节）；
        // >0 生效；负数一律 SettingsError 且点名出错字段。
        Settings defaults = Settings.load(Map.of("ENV", "development"), null);
        assertEquals(0, defaults.memory.contextBudgetSystemTokens);
        assertEquals(0, defaults.memory.contextBudgetHistoryTokens);
        assertEquals(0, defaults.memory.contextBudgetToolResultTokens);
        assertEquals(0, defaults.memory.contextBudgetInjectionTokens);

        Settings tuned = Settings.load(Map.of(
                "ENV", "development",
                "MEMORY_CONTEXT_BUDGET_SYSTEM_TOKENS", "512",
                "MEMORY_CONTEXT_BUDGET_HISTORY_TOKENS", "4096",
                "MEMORY_CONTEXT_BUDGET_TOOL_RESULT_TOKENS", "1024",
                "MEMORY_CONTEXT_BUDGET_INJECTION_TOKENS", "2048"), null);
        assertEquals(512, tuned.memory.contextBudgetSystemTokens);
        assertEquals(4096, tuned.memory.contextBudgetHistoryTokens);
        assertEquals(1024, tuned.memory.contextBudgetToolResultTokens);
        assertEquals(2048, tuned.memory.contextBudgetInjectionTokens);

        // 0 是合法值（=关）；负数拒绝（>=0 语义：0 不是无效配置而是开关）。
        for (String key : new String[] {
                "MEMORY_CONTEXT_BUDGET_SYSTEM_TOKENS",
                "MEMORY_CONTEXT_BUDGET_HISTORY_TOKENS",
                "MEMORY_CONTEXT_BUDGET_TOOL_RESULT_TOKENS",
                "MEMORY_CONTEXT_BUDGET_INJECTION_TOKENS"}) {
            Settings.SettingsError error = assertThrows(Settings.SettingsError.class,
                    () -> Settings.load(Map.of(
                            "ENV", "development",
                            key, "-1"), null),
                    key + " 负数应拒绝");
            assertTrue(error.getMessage().contains(
                            key.substring("MEMORY_".length()).toLowerCase(java.util.Locale.ROOT)),
                    "拒绝文案要点名出错字段：" + error.getMessage());
        }
    }

    @Test
    void perModelContextWindowsParseIndependently() {
        // #4：主/快档窗口独立解析；0 = 未配置（预算沿用现状）；负数拒绝。
        Settings s = Settings.load(Map.of(
                "ENV", "development",
                "LLM_MAX_CONTEXT_TOKENS", "32768",
                "LLM_FAST_MAX_CONTEXT_TOKENS", "8192"), null);
        assertEquals(32768, s.llm.maxContextTokens);
        assertEquals(8192, s.llmFast.maxContextTokens);
        Settings defaults = Settings.load(Map.of("ENV", "development"), null);
        assertEquals(0, defaults.llm.maxContextTokens);
        assertEquals(0, defaults.llmFast.maxContextTokens);
        assertThrows(Settings.SettingsError.class,
                () -> Settings.load(Map.of(
                        "ENV", "development",
                        "LLM_MAX_CONTEXT_TOKENS", "-1"), null));
        assertThrows(Settings.SettingsError.class,
                () -> Settings.load(Map.of(
                        "ENV", "development",
                        "LLM_FAST_MAX_CONTEXT_TOKENS", "-1"), null));
    }

    @Test
    void rateLimitPerMinuteParsesAndProductionRequiresIt() {
        // #78：0 = 不限（本机开发默认）；负数拒绝；production 必须显式配置
        // 限流（多用户硬前提——单用户无限并发是成本/资源无保护暴露面）。
        assertEquals(0, Settings.load(Map.of("ENV", "development"), null).rateLimitPerMinute);
        assertEquals(30, Settings.load(Map.of(
                "ENV", "development", "RATE_LIMIT_PER_MINUTE", "30"), null).rateLimitPerMinute);
        assertThrows(Settings.SettingsError.class,
                () -> Settings.load(Map.of(
                        "ENV", "development", "RATE_LIMIT_PER_MINUTE", "-1"), null));
        // production 守卫：缺限流拒绝启动，指路键名；显式配置后通过。
        Settings production = Settings.load(Map.of(
                "ENV", "production",
                "LLM_API_KEY", "sk-test",
                "MEMORY_AUTH_SECRET", "sec",
                "CHECKPOINTER_BACKEND", "memory"), null);
        Settings.SettingsError error = assertThrows(Settings.SettingsError.class,
                production::ensureProductionReady);
        assertTrue(error.getMessage().contains("RATE_LIMIT_PER_MINUTE"), error.getMessage());
        production.rateLimitPerMinute = 60;
        production.ensureProductionReady(); // 不抛
    }

    @Test
    void streamIdleSecondsParsesWithZeroMeaningDisabled() {
        // #74：默认 30s；0 = 禁用（旧行为，唯一兜底是 HTTP 总超时）；负数拒绝。
        assertEquals(30.0,
                Settings.load(Map.of("ENV", "development"), null).llm.streamIdleSeconds, 1e-9);
        assertEquals(60.0, Settings.load(Map.of(
                "ENV", "development", "LLM_STREAM_IDLE_SECONDS", "60"), null).llm.streamIdleSeconds, 1e-9);
        assertEquals(0.0, Settings.load(Map.of(
                "ENV", "development", "LLM_STREAM_IDLE_SECONDS", "0"), null).llm.streamIdleSeconds, 1e-9);
        assertThrows(Settings.SettingsError.class,
                () -> Settings.load(Map.of(
                        "ENV", "development", "LLM_STREAM_IDLE_SECONDS", "-1"), null));
    }

    @Test
    void llmFallbacksParseAsJsonArrayOfTriples() {
        // #71：JSON 数组三元组；api_key 掩码类型；缺 model/base_url 拒绝；
        // 空串=空链。
        Settings s = Settings.load(Map.of(
                "ENV", "development",
                "LLM_FALLBACKS", "[{\"base_url\":\"https://b1.invalid/v1\",\"model\":\"m1\"},"
                        + "{\"base_url\":\"https://b2.invalid/v1\",\"api_key\":\"sk-fb\","
                        + "\"model\":\"m2\",\"max_context_tokens\":8192}]"), null);
        assertEquals(2, s.llm.fallbacks.size());
        assertEquals("https://b1.invalid/v1", s.llm.fallbacks.get(0).baseUrl());
        assertEquals("m1", s.llm.fallbacks.get(0).model());
        assertTrue(s.llm.fallbacks.get(0).apiKey().isBlank(), "省略 api_key → 空 Secret（构建期回落主力 key）");
        assertEquals("sk-fb", s.llm.fallbacks.get(1).apiKey().value());
        assertEquals(8192, s.llm.fallbacks.get(1).maxContextTokens());
        assertEquals(0, s.llm.fallbacks.get(0).maxContextTokens());
        assertEquals(List.of(), Settings.load(Map.of("ENV", "development",
                "LLM_FALLBACKS", ""), null).llm.fallbacks);
        assertThrows(Settings.SettingsError.class,
                () -> Settings.load(Map.of(
                        "ENV", "development", "LLM_FALLBACKS", "[{\"model\":\"m1\"}]"), null));
        assertThrows(Settings.SettingsError.class,
                () -> Settings.load(Map.of(
                        "ENV", "development", "LLM_FALLBACKS", "{\"not\":\"array\"}"), null));
        assertThrows(Settings.SettingsError.class,
                () -> Settings.load(Map.of(
                        "ENV", "development", "LLM_FALLBACKS", "not-json"), null));
    }

    @Test
    void agentToolLoopThresholdParsesWithZeroMeaningDisabled() {
        // 条目 5：默认 2（连续 2 次同参重复后第 3 次拦截）；0 = 关闭（逐字节
        // 旧行为）；负数解析期 fail-fast（ensureValid 收敛校验另有同款守卫）。
        assertEquals(2, Settings.load(Map.of("ENV", "development"), null).agentToolLoopThreshold);
        assertEquals(5, Settings.load(Map.of(
                "ENV", "development", "AGENT_TOOL_LOOP_THRESHOLD", "5"), null).agentToolLoopThreshold);
        assertEquals(0, Settings.load(Map.of(
                "ENV", "development", "AGENT_TOOL_LOOP_THRESHOLD", "0"), null).agentToolLoopThreshold);
        Settings.SettingsError error = assertThrows(Settings.SettingsError.class,
                () -> Settings.load(Map.of(
                        "ENV", "development", "AGENT_TOOL_LOOP_THRESHOLD", "-1"), null));
        assertTrue(error.getMessage().contains("AGENT_TOOL_LOOP_THRESHOLD"),
                "拒绝文案要点名出错键：" + error.getMessage());
    }
}
