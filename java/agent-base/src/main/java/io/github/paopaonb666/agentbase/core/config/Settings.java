package io.github.paopaonb666.agentbase.core.config;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * agent 基座的配置，供运行时消费（按域嵌套的配置节）。
 *
 * <p>本模块是 {@code .env.example} 中声明的配置契约（CONTRACT）的唯一
 * 消费方。设计说明（继承自 Python 侧 config.py）：
 * <ul>
 *   <li>"配置即校验"：每个可能出错的字段都在启动时校验，而不是静默地
 *       使用默认值；校验随配置节就近放置。</li>
 *   <li>不用弱默认密钥；production 环境在没有 {@code LLM_API_KEY} 时
 *       拒绝启动。</li>
 *   <li><b>按域嵌套</b>：{@code LlmSettings} / {@code CheckpointerSettings} /
 *       {@code ToolkitSettings} / {@code SearchSettings} / {@code DocParseSettings} /
 *       {@code MemorySettings} / {@code ObservabilitySettings} 各一个配置节，
 *       子系统只消费自己那一节（接口隔离）。</li>
 *   <li><b>环境变量契约保持平铺不变</b>：嵌套节的 env 名 = 节前缀 + 内部
 *       字段名（{@code MEMORY_ENABLED} / {@code CHECKPOINTER_MYSQL_HOST}…），
 *       {@code .env.example} 无需任何迁移。</li>
 * </ul>
 *
 * <p>加载优先级：显式 overrides（测试/编程装配）&gt; 环境变量 &gt; {@code .env}。
 * 未知键被忽略。对应 Python 侧 {@code core/config.py}。
 */
public final class Settings {

    // ── 合法取值集合（与 Python 侧常量一一对应） ──────────────────────

    /** 基座知道如何装配的 openai 兼容 provider（仅用于防止配置拼写错误）。 */
    public static final Set<String> KNOWN_PROVIDERS = Set.of("deepseek", "zhipu", "openai-compatible");
    /** 基座识别的 checkpointer 后端。 */
    public static final Set<String> KNOWN_CHECKPOINTER_BACKENDS = Set.of("memory", "sqlite", "mysql");
    /** 工具库搜索引擎实现清单（SEARCH_ENGINE_PRIORITY 的合法取值）。 */
    public static final Set<String> KNOWN_SEARCH_ENGINES = Set.of("tavily", "duckduckgo");
    public static final Set<String> KNOWN_ENVIRONMENTS = Set.of("development", "production");
    /** 记忆形成管线的模型分派档位（MEMORY_PIPELINE_PROFILE 的合法取值）。 */
    public static final Set<String> KNOWN_PIPELINE_PROFILES = Set.of("split", "all_main", "all_fast");
    /** 预算超限行为（COST_ACTION 的合法取值）。 */
    public static final Set<String> KNOWN_COST_ACTIONS = Set.of("warn", "block");

    /** 当配置校验失败时抛出（快速失败）。 */
    public static final class SettingsError extends IllegalArgumentException {
        public SettingsError(String message) { super(message); }
    }

    // ── 根级字段 ─────────────────────────────────────────────────────

    /** 要加载的模块名；顺序即装配顺序。 */
    public List<String> agentModules = List.of("chat");
    /** 模块包搜索顺序；默认只有基座自带的样板模块包。 */
    public List<String> agentModulePackages = List.of("agent_base.modules");
    /** 单轮 invoke 的图步数上限（含工具往返）。 */
    public int agentRecursionLimit = 25;
    /** 允许跨域调用本服务的来源（浏览器客户端，如 agent-base-ui）。 */
    public List<String> corsOrigins = List.of("http://localhost:3000");
    /** 运行环境：development | production。 */
    public String env = "development";

    public final LlmSettings llm = new LlmSettings();
    public final LlmFastSettings llmFast = new LlmFastSettings();
    public final CheckpointerSettings checkpointer = new CheckpointerSettings();
    public final ToolkitSettings toolkit = new ToolkitSettings();
    public final SearchSettings search = new SearchSettings();
    public final DocParseSettings docParse = new DocParseSettings();
    public final MemorySettings memory = new MemorySettings();
    public final PlannerSettings planner = new PlannerSettings();
    public final ObservabilitySettings observability = new ObservabilitySettings();
    public final CostSettings cost = new CostSettings();
    public final McpSettings mcp = new McpSettings();

    // ── 配置节 ───────────────────────────────────────────────────────

    /** LLM（openai 兼容）配置节：env 前缀 {@code LLM_}。 */
    public static final class LlmSettings {
        public String provider = "deepseek";
        /** API key（SecretStr 语义：禁止写入日志）。 */
        public String apiKey = "";
        public String baseUrl = "https://api.deepseek.com/v1";
        public String model = "deepseek-chat";
    }

    /**
     * LLM 快速档配置节：env 前缀 {@code LLM_FAST_}（免费/低价模型吃批量
     * 任务）。它是优化不是依赖：baseUrl 与 model 两者皆非空才算已配置，
     * 未配置时使用 fast 档的调用方应回落主力模型。
     */
    public static final class LlmFastSettings {
        public String apiKey = "";
        public String baseUrl = "";
        public String model = "";
        /** 免费档普遍按并发数限速：包装层用它做信号量限流。 */
        public int concurrency = 2;

        public boolean isConfigured() {
            return !baseUrl.strip().isEmpty() && !model.strip().isEmpty();
        }
    }

    /** 对话状态（checkpointer）配置节：env 前缀 {@code CHECKPOINTER_}。 */
    public static final class CheckpointerSettings {
        public String backend = "sqlite";
        public String sqlitePath = "./agent_base_state.db";
        public String mysqlHost = "127.0.0.1";
        public int mysqlPort = 3306;
        public String mysqlUser = "root";
        public String mysqlPassword = "";
        public String mysqlDatabase = "agent_base";
    }

    /** 工具池与工具库配置节：env 前缀 {@code TOOLKIT_} / {@code TOOL_}。 */
    public static final class ToolkitSettings {
        /** 要装配进共享池的基座内置工具名。 */
        public List<String> enabled = List.of("current_time", "calculator", "json_query");
        /** 每次工具执行的挂钟时间预算，由池的包装器强制执行。 */
        public double timeoutSeconds = 30.0;
        /** 工具调用审计（M5）：每条调用落 tool_call_records 表。 */
        public boolean callLogEnabled = true;
    }

    /** 网络搜索（tools/search）配置节：env 前缀 {@code SEARCH_} / {@code TAVILY_}。 */
    public static final class SearchSettings {
        /** 搜索引擎优先级：按顺序依次尝试，任一成功即返回。 */
        public String enginePriority = "duckduckgo";
        public String tavilyApiKey = "";
        public double timeoutSeconds = 15.0;
        public double cacheTtlSeconds = 300.0;
        public String ddgsBackend = "duckduckgo";
    }

    /** 文档解析（tools/parsing）配置节：env 前缀 {@code DOC_PARSE_}。 */
    public static final class DocParseSettings {
        /** 单个文档的输入大小封顶（字节）。 */
        public int maxInputBytes = 10 * 1024 * 1024;
        /** 解析输出的文本长度封顶（字符）。 */
        public int maxOutputChars = 50_000;
    }

    /** 记忆系统（memory/）配置节：env 前缀 {@code MEMORY_}。 */
    public static final class MemorySettings {
        public boolean enabled = true;
        public boolean embeddingEnabled = true;
        public String embeddingBaseUrl = "https://api.siliconflow.cn/v1";
        public String embeddingApiKey = "";
        public String embeddingModel = "BAAI/bge-m3";
        public int embeddingDims = 1024;
        public int embeddingBatchSize = 16;
        public double embeddingTimeoutSeconds = 10.0;
        public int recallTopK = 6;
        /** 召回最低分阈值（0..1）：低于该分数的候选不召回。 */
        public double recallMinScore = 0.5;
        public double weightVector = 0.40;
        public double weightKeyword = 0.30;
        public double weightRecency = 0.15;
        public double weightSalience = 0.15;
        public int contextMaxChars = 3000;
        public int contextMaxTokens = 12_000;
        /** 注入快照缓存上限（成本治理 T2.1）：线程级冻结注入段的 LRU 容量。 */
        public int contextSnapshotMax = 256;
        public boolean captureEnabled = true;
        public int captureEveryTurns = 1;
        public String pipelineProfile = "split";
        public boolean summaryEnabled = true;
        public int summaryTriggerMessages = 12;
        public boolean profileEnabled = true;
        public int profileMaxChars = 1500;
        /** 遗忘策略：episodic 记忆的保留天数（0 表示不过期）。 */
        public int episodicTtlDays = 30;
        public double timeDecayHalfLifeDays = 14.0;
        public int extractionMaxInputChars = 8000;
        public int extractionMaxMessages = 24;
        public int docChunkChars = 800;
        public int docChunkOverlap = 100;
        public int mysqlPoolSize = 5;
        /** 记忆身份鉴权（P0）：配置后所有用户作用域请求需附带 HMAC 签名。 */
        public String authSecret = "";
        public int authReplayWindowSeconds = 300;
    }

    /** planner 模块的预算与防护（M9）：env 前缀 {@code PLANNER_}。 */
    public static final class PlannerSettings {
        /** 单次拆解的子任务上限（含重排后的总量）。 */
        public int maxSubtasks = 5;
        /** 单个子任务的工具轮数上限：耗尽即该子任务 failed。 */
        public int subtaskToolRounds = 8;
        /** 重规划预算：耗尽后走 best-effort synthesize。 */
        public int maxReplans = 2;
        /** 空输出门的最小摘要长度门槛；0 = 只拦空白输出。 */
        public int minSummaryChars = 0;
    }

    /** 可观测性配置节：env {@code LOG_JSON} / {@code HEALTH_PROBE_MODEL}。 */
    public static final class ObservabilitySettings {
        public boolean logJson = false;
        /** /health 是否真实探测 LLM 端点可达性（会打真实网络，默认关闭）。 */
        public boolean healthProbeModel = false;
    }

    /** 成本治理配置节：env 前缀 {@code COST_}（T4.1/T4.2）。 */
    public static final class CostSettings {
        public boolean enabled = false;
        public double dailyCny = 0.0;
        public double monthlyCny = 0.0;
        public String action = "block";
        /** 价目表 JSON（元/百万 tokens，手工维护）。 */
        public String pricesJson = "";
    }

    /** MCP 工具来源配置节：env 前缀 {@code MCP_}（第一期未迁移：启用即报错）。 */
    public static final class McpSettings {
        public boolean enabled = false;
        public String serversJson = "";
    }

    // ── 加载 ─────────────────────────────────────────────────────────

    /** 从 overrides + 环境变量 + 工作目录 .env 加载并校验（默认路径）。 */
    public static Settings load(Map<String, String> overrides) {
        return load(overrides, Path.of(".env"));
    }

    /**
     * 从三层来源合并加载：{@code overrides} &gt; 环境变量 &gt; dotenv 文件
     * （{@code envFile} 传 null 禁用 .env 加载，测试用）。
     */
    public static Settings load(Map<String, String> overrides, Path envFile) {
        Map<String, String> flat = new LinkedHashMap<>();
        if (envFile != null) {
            flat.putAll(readDotEnv(envFile));
        }
        // 环境变量里的 null 值不可能出现，但显式排除空覆盖语义与 Python 一致。
        for (Map.Entry<String, String> e : System.getenv().entrySet()) {
            if (e.getValue() != null) {
                flat.put(e.getKey(), e.getValue());
            }
        }
        if (overrides != null) {
            flat.putAll(overrides);
        }
        Settings s = new Settings();
        applyFlat(s, liftLegacyFlatFields(flat));
        s.ensureValid();
        return s;
    }

    /**
     * 兼容垫片（写入侧）：把平铺的 env 键按前缀归位到对应配置节。
     * 对应 Python 侧 {@code Settings._lift_legacy_flat_fields}。
     * 注意 {@code llm_fast_} 必须排在 {@code llm_} 之前。
     */
    /**
     * 照旧文档/记忆写的平铺键会被更宽的前缀吞进错误配置节后被静默丢弃
     * （#101：LLM_HEALTH_PROBE_MODEL 被 {@code llm_} 前缀抬进 llm 节，
     * applyLlm 不认识它——「照文档配置不生效」且无从排查）。对这类键
     * 显式拒绝并指路。
     */
    private static final Map<String, String> STALE_FLAT_KEYS = Map.of(
            "llm_health_probe_model",
            "HEALTH_PROBE_MODEL（observability 节：/health 的 LLM 探活开关）");

    static Map<String, Map<String, String>> liftLegacyFlatFields(Map<String, String> data) {
        // 有序：前缀匹配按声明顺序（llm_fast_ 先于 llm_）。
        List<Map.Entry<String, String>> prefixes = List.of(
                Map.entry("llm_fast_", "llm_fast"),
                Map.entry("llm_", "llm"),
                Map.entry("cost_", "cost"),
                Map.entry("mcp_", "mcp"),
                Map.entry("checkpointer_", "checkpointer"),
                Map.entry("toolkit_", "toolkit"),
                Map.entry("tool_timeout_seconds", "toolkit"),
                Map.entry("tool_call_log_enabled", "toolkit"),
                Map.entry("search_", "search"),
                Map.entry("tavily_api_key", "search"),
                Map.entry("doc_parse_", "doc_parse"),
                Map.entry("memory_", "memory"),
                Map.entry("planner_", "planner"),
                Map.entry("log_json", "observability"),
                Map.entry("health_probe_model", "observability"));
        // TOOL_*/观测整键迁移到节内的新名字。
        Map<String, String> sectionInner = Map.of(
                "tool_timeout_seconds", "timeout_seconds",
                "tool_call_log_enabled", "call_log_enabled",
                "log_json", "log_json",
                "health_probe_model", "health_probe_model",
                "tavily_api_key", "tavily_api_key");
        Map<String, Map<String, String>> lifted = new LinkedHashMap<>();
        Map<String, String> out = new LinkedHashMap<>();
        for (Map.Entry<String, String> e : data.entrySet()) {
            String key = e.getKey();
            String lower = key.toLowerCase(Locale.ROOT);
            String stale = STALE_FLAT_KEYS.get(lower);
            if (stale != null) {
                throw new SettingsError(
                        ("配置键 '%s' 不存在（会被 llm_ 前缀规则吞进 LLM 节后静默丢弃）；"
                                + "正确键为 %s").formatted(key, stale));
            }
            String matchedPrefix = "";
            String target = "";
            for (Map.Entry<String, String> p : prefixes) {
                if (lower.startsWith(p.getKey())) {
                    matchedPrefix = p.getKey();
                    target = p.getValue();
                    break;
                }
            }
            if (!matchedPrefix.isEmpty()) {
                String inner = sectionInner.containsKey(lower)
                        ? sectionInner.get(lower)
                        : lower.substring(matchedPrefix.length());
                lifted.computeIfAbsent(target, k -> new LinkedHashMap<>()).put(inner, e.getValue());
            } else {
                out.put(key, e.getValue());
            }
        }
        // 根级字段的 env/dotenv 键可能全大写（AGENT_MODULES）：归一小写。
        for (String key : List.of("agent_modules", "agent_module_packages", "cors_origins", "env",
                "agent_recursion_limit")) {
            String up = key.toUpperCase(Locale.ROOT);
            if (out.containsKey(up) && !out.containsKey(key)) {
                out.put(key, out.remove(up));
            }
        }
        if (!out.isEmpty()) {
            // 根级字段挂在 "" 伪节上：applyFlat 对每个节先做根级字段检查，
            // "" 又不匹配任何 switch 分支——与 Python 把 out 与 lifted 合并
            // 返回的语义一致（丢掉 out 会让 ENV/AGENT_MODULES 等静默失效）。
            lifted.computeIfAbsent("", k -> new LinkedHashMap<>()).putAll(out);
        }
        return lifted;
    }

    /** 把归位后的配置节写入实例。 */
    private static void applyFlat(Settings s, Map<String, Map<String, String>> lifted) {
        for (Map.Entry<String, Map<String, String>> e : lifted.entrySet()) {
            String section = e.getKey();
            Map<String, String> vals = e.getValue();
            // 根级平铺字段（AGENT_MODULES / ENV / AGENT_RECURSION_LIMIT / CORS_ORIGINS）
            if (vals.containsKey("agent_modules")) {
                s.agentModules = parseCsvOrJsonList(vals.get("agent_modules"), "agent_modules");
            }
            if (vals.containsKey("env")) {
                s.env = requireStr("env", vals.get("env"));
            }
            if (vals.containsKey("agent_recursion_limit")) {
                s.agentRecursionLimit = requireInt("agent_recursion_limit", vals.get("agent_recursion_limit"));
            }
            if (vals.containsKey("cors_origins")) {
                s.corsOrigins = parseCsvOrJsonList(vals.get("cors_origins"), "cors_origins");
            }
            switch (section) {
                case "llm" -> applyLlm(s, vals);
                case "llm_fast" -> applyLlmFast(s, vals);
                case "checkpointer" -> applyCheckpointer(s, vals);
                case "toolkit" -> applyToolkit(s, vals);
                case "search" -> applySearch(s, vals);
                case "doc_parse" -> applyDocParse(s, vals);
                case "memory" -> applyMemory(s, vals);
                case "planner" -> applyPlanner(s, vals);
                case "observability" -> applyObservability(s, vals);
                case "cost" -> applyCost(s, vals);
                case "mcp" -> applyMcp(s, vals);
                default -> { /* 未知节忽略（extra=ignore） */ }
            }
        }
    }

    private static String requireStr(String field, String value) {
        return value == null ? "" : value;
    }

    private static int requireInt(String field, String value) {
        try {
            return Integer.parseInt(value.trim());
        } catch (NumberFormatException ex) {
            throw new SettingsError("%s must be an integer, got '%s'".formatted(field, value));
        }
    }

    private static double requireFloat(String field, String value) {
        try {
            return Double.parseDouble(value.trim());
        } catch (NumberFormatException ex) {
            throw new SettingsError("%s must be a number, got '%s'".formatted(field, value));
        }
    }

    private static boolean requireBool(String field, String value) {
        String v = value.trim().toLowerCase(Locale.ROOT);
        return switch (v) {
            case "y", "yes", "t", "true", "on", "1" -> true;
            case "n", "no", "f", "false", "off", "0" -> false;
            default -> throw new SettingsError("%s must be a boolean, got '%s'".formatted(field, value));
        };
    }

    /**
     * 把列表类型的字段从其友好的字符串形式解析出来：接受逗号形式
     * （{@code "chat,writer"}）和 JSON 数组（{@code '["chat","writer"]'}）。
     * 空输入 -&gt; 空列表。对应 Python 侧 {@code _parse_csv_or_json_list}。
     */
    public static List<String> parseCsvOrJsonList(String value, String fieldName) {
        if (value == null) {
            return List.of();
        }
        String text = value.strip();
        if (text.isEmpty()) {
            return List.of();
        }
        if (text.startsWith("[")) {
            // 轻量 JSON 数组解析（元素为字符串），避免在此处引入完整 Jackson。
            List<String> names = new ArrayList<>();
            String body = text.strip();
            if (!body.endsWith("]")) {
                throw new SettingsError(
                        "%s starts with '[' so it is parsed as a JSON array, but the array is invalid: '%s'"
                                .formatted(fieldName, text));
            }
            String inner = body.substring(1, body.length() - 1);
            for (String item : splitJsonStrings(inner)) {
                String stripped = item.strip();
                if (stripped.length() >= 2 && stripped.startsWith("\"") && stripped.endsWith("\"")) {
                    String unquoted = stripped.substring(1, stripped.length() - 1)
                            .replace("\\\"", "\"").replace("\\\\", "\\");
                    if (!unquoted.strip().isEmpty()) {
                        names.add(unquoted.strip());
                    }
                } else {
                    throw new SettingsError(
                            "%s JSON array must be a list of strings: '%s'".formatted(fieldName, text));
                }
            }
            return List.copyOf(names);
        }
        List<String> parts = new ArrayList<>();
        for (String part : text.split(",")) {
            if (!part.strip().isEmpty()) {
                parts.add(part.strip());
            }
        }
        return List.copyOf(parts);
    }

    /** 按逗号拆分（容忍 JSON 内的引号与转义——仅覆盖列表字段的受控场景）。 */
    private static List<String> splitJsonStrings(String inner) {
        List<String> items = new ArrayList<>();
        StringBuilder cur = new StringBuilder();
        boolean inString = false;
        boolean escaped = false;
        for (int i = 0; i < inner.length(); i++) {
            char c = inner.charAt(i);
            if (escaped) {
                cur.append('\\').append(c);
                escaped = false;
                continue;
            }
            if (c == '\\' && inString) {
                cur.append(c);
                escaped = true;
                continue;
            }
            if (c == '"') {
                inString = !inString;
            }
            if (c == ',' && !inString) {
                items.add(cur.toString());
                cur.setLength(0);
            } else {
                cur.append(c);
            }
        }
        if (!cur.isEmpty() || !inner.isBlank()) {
            items.add(cur.toString());
        }
        return items;
    }

    // ── 各节应用（含逐字段校验，错误文案与 Python 侧一致） ──────────────

    private static void applyLlm(Settings s, Map<String, String> v) {
        LlmSettings t = s.llm;
        if (v.containsKey("provider")) {
            String value = v.get("provider");
            if (!KNOWN_PROVIDERS.contains(value)) {
                throw new SettingsError("unknown LLM_PROVIDER '%s'; expected one of %s"
                        .formatted(value, sorted(KNOWN_PROVIDERS)));
            }
            t.provider = value;
        }
        if (v.containsKey("api_key")) t.apiKey = v.get("api_key");
        if (v.containsKey("base_url")) t.baseUrl = v.get("base_url");
        if (v.containsKey("model")) t.model = v.get("model");
    }

    private static void applyLlmFast(Settings s, Map<String, String> v) {
        LlmFastSettings t = s.llmFast;
        if (v.containsKey("api_key")) t.apiKey = v.get("api_key");
        if (v.containsKey("base_url")) t.baseUrl = v.get("base_url");
        if (v.containsKey("model")) t.model = v.get("model");
        if (v.containsKey("concurrency")) {
            int value = requireInt("concurrency", v.get("concurrency"));
            if (value < 1) {
                throw new SettingsError("LLM_FAST_CONCURRENCY must be >= 1, got %d".formatted(value));
            }
            t.concurrency = value;
        }
    }

    private static void applyCheckpointer(Settings s, Map<String, String> v) {
        CheckpointerSettings t = s.checkpointer;
        if (v.containsKey("backend")) {
            String value = v.get("backend");
            if (!KNOWN_CHECKPOINTER_BACKENDS.contains(value)) {
                throw new SettingsError("unknown CHECKPOINTER_BACKEND '%s'; expected one of %s"
                        .formatted(value, sorted(KNOWN_CHECKPOINTER_BACKENDS)));
            }
            t.backend = value;
        }
        if (v.containsKey("sqlite_path")) t.sqlitePath = v.get("sqlite_path");
        if (v.containsKey("mysql_host")) t.mysqlHost = v.get("mysql_host");
        if (v.containsKey("mysql_port")) t.mysqlPort = requireInt("mysql_port", v.get("mysql_port"));
        if (v.containsKey("mysql_user")) t.mysqlUser = v.get("mysql_user");
        if (v.containsKey("mysql_password")) t.mysqlPassword = v.get("mysql_password");
        if (v.containsKey("mysql_database")) t.mysqlDatabase = v.get("mysql_database");
    }

    private static void applyToolkit(Settings s, Map<String, String> v) {
        ToolkitSettings t = s.toolkit;
        if (v.containsKey("enabled")) {
            t.enabled = parseCsvOrJsonList(v.get("enabled"), "TOOLKIT_ENABLED");
        }
        if (v.containsKey("timeout_seconds")) {
            double value = requireFloat("timeout_seconds", v.get("timeout_seconds"));
            // 非正数会让工具池立即超时；NaN 的所有比较都是 False，用 !(value > 0) 一并拒绝。
            if (!(value > 0)) {
                throw new SettingsError("TOOL_TIMEOUT_SECONDS must be > 0, got %s".formatted(value));
            }
            t.timeoutSeconds = value;
        }
        if (v.containsKey("call_log_enabled")) t.callLogEnabled = requireBool("call_log_enabled", v.get("call_log_enabled"));
    }

    private static void applySearch(Settings s, Map<String, String> v) {
        SearchSettings t = s.search;
        if (v.containsKey("engine_priority")) {
            String value = v.get("engine_priority");
            List<String> engines = new ArrayList<>();
            for (String part : value.split(",")) {
                String p = part.strip().toLowerCase(Locale.ROOT);
                if (!p.isEmpty()) engines.add(p);
            }
            if (engines.isEmpty()) {
                throw new SettingsError("SEARCH_ENGINE_PRIORITY must name at least one engine");
            }
            List<String> unknown = engines.stream().filter(e -> !KNOWN_SEARCH_ENGINES.contains(e)).toList();
            if (!unknown.isEmpty()) {
                throw new SettingsError("unknown SEARCH_ENGINE_PRIORITY entries %s; expected subset of %s"
                        .formatted(unknown, sorted(KNOWN_SEARCH_ENGINES)));
            }
            t.enginePriority = value;
        }
        if (v.containsKey("tavily_api_key")) t.tavilyApiKey = v.get("tavily_api_key");
        if (v.containsKey("timeout_seconds")) {
            double value = requireFloat("timeout_seconds", v.get("timeout_seconds"));
            if (!(value > 0)) {
                throw new SettingsError("timeout_seconds must be > 0, got %s".formatted(value));
            }
            t.timeoutSeconds = value;
        }
        if (v.containsKey("cache_ttl_seconds")) {
            double value = requireFloat("cache_ttl_seconds", v.get("cache_ttl_seconds"));
            if (!(value > 0)) {
                throw new SettingsError("cache_ttl_seconds must be > 0, got %s".formatted(value));
            }
            t.cacheTtlSeconds = value;
        }
        if (v.containsKey("ddgs_backend")) {
            String backend = v.get("ddgs_backend").strip().toLowerCase(Locale.ROOT);
            if (backend.isEmpty()) {
                throw new SettingsError(
                        "SEARCH_DDGS_BACKEND must name at least one ddgs engine (e.g. duckduckgo, bing)");
            }
            t.ddgsBackend = v.get("ddgs_backend");
        }
    }

    private static void applyDocParse(Settings s, Map<String, String> v) {
        DocParseSettings t = s.docParse;
        if (v.containsKey("max_input_bytes")) {
            int value = requireInt("max_input_bytes", v.get("max_input_bytes"));
            if (value <= 0) {
                throw new SettingsError("max_input_bytes must be > 0, got %d".formatted(value));
            }
            t.maxInputBytes = value;
        }
        if (v.containsKey("max_output_chars")) {
            int value = requireInt("max_output_chars", v.get("max_output_chars"));
            if (value <= 0) {
                throw new SettingsError("max_output_chars must be > 0, got %d".formatted(value));
            }
            t.maxOutputChars = value;
        }
    }

    private static void applyMemory(Settings s, Map<String, String> v) {
        MemorySettings t = s.memory;
        if (v.containsKey("enabled")) t.enabled = requireBool("enabled", v.get("enabled"));
        if (v.containsKey("embedding_enabled")) t.embeddingEnabled = requireBool("embedding_enabled", v.get("embedding_enabled"));
        if (v.containsKey("embedding_base_url")) t.embeddingBaseUrl = v.get("embedding_base_url");
        if (v.containsKey("embedding_api_key")) t.embeddingApiKey = v.get("embedding_api_key");
        if (v.containsKey("embedding_model")) t.embeddingModel = v.get("embedding_model");
        if (v.containsKey("embedding_dims")) {
            int value = requireInt("embedding_dims", v.get("embedding_dims"));
            if (value <= 0) throw new SettingsError("embedding_dims must be > 0, got %d".formatted(value));
            t.embeddingDims = value;
        }
        if (v.containsKey("embedding_batch_size")) {
            int value = requireInt("embedding_batch_size", v.get("embedding_batch_size"));
            if (value <= 0) throw new SettingsError("embedding_batch_size must be > 0, got %d".formatted(value));
            t.embeddingBatchSize = value;
        }
        if (v.containsKey("embedding_timeout_seconds")) {
            double value = requireFloat("embedding_timeout_seconds", v.get("embedding_timeout_seconds"));
            if (!(value > 0)) throw new SettingsError("embedding_timeout_seconds must be > 0, got %s".formatted(value));
            t.embeddingTimeoutSeconds = value;
        }
        if (v.containsKey("recall_top_k")) {
            int value = requireInt("recall_top_k", v.get("recall_top_k"));
            if (value <= 0) throw new SettingsError("recall_top_k must be > 0, got %d".formatted(value));
            t.recallTopK = value;
        }
        if (v.containsKey("recall_min_score")) {
            double value = requireFloat("recall_min_score", v.get("recall_min_score"));
            if (!(value >= 0.0 && value < 1.0)) {
                throw new SettingsError("MEMORY_RECALL_MIN_SCORE must be in [0, 1), got %s".formatted(value));
            }
            t.recallMinScore = value;
        }
        if (v.containsKey("weight_vector")) {
            double value = requireFloat("weight_vector", v.get("weight_vector"));
            if (!(value >= 0.0 && value <= 1.0)) throw new SettingsError("weight_vector must be in [0, 1], got %s".formatted(value));
            t.weightVector = value;
        }
        if (v.containsKey("weight_keyword")) {
            double value = requireFloat("weight_keyword", v.get("weight_keyword"));
            if (!(value >= 0.0 && value <= 1.0)) throw new SettingsError("weight_keyword must be in [0, 1], got %s".formatted(value));
            t.weightKeyword = value;
        }
        if (v.containsKey("weight_recency")) {
            double value = requireFloat("weight_recency", v.get("weight_recency"));
            if (!(value >= 0.0 && value <= 1.0)) throw new SettingsError("weight_recency must be in [0, 1], got %s".formatted(value));
            t.weightRecency = value;
        }
        if (v.containsKey("weight_salience")) {
            double value = requireFloat("weight_salience", v.get("weight_salience"));
            if (!(value >= 0.0 && value <= 1.0)) throw new SettingsError("weight_salience must be in [0, 1], got %s".formatted(value));
            t.weightSalience = value;
        }
        double weightSum = t.weightVector + t.weightKeyword + t.weightRecency + t.weightSalience;
        if (Math.abs(weightSum - 1.0) > 0.01) {
            throw new SettingsError(
                    "memory hybrid weights must sum to 1.0 (±0.01), got %s from (%s, %s, %s, %s)"
                            .formatted(weightSum, t.weightVector, t.weightKeyword, t.weightRecency, t.weightSalience));
        }
        if (v.containsKey("context_max_chars")) {
            int value = requireInt("context_max_chars", v.get("context_max_chars"));
            if (value <= 0) throw new SettingsError("context_max_chars must be > 0, got %d".formatted(value));
            t.contextMaxChars = value;
        }
        if (v.containsKey("context_max_tokens")) {
            int value = requireInt("context_max_tokens", v.get("context_max_tokens"));
            if (value <= 0) throw new SettingsError("context_max_tokens must be > 0, got %d".formatted(value));
            t.contextMaxTokens = value;
        }
        if (v.containsKey("context_snapshot_max")) {
            int value = requireInt("context_snapshot_max", v.get("context_snapshot_max"));
            if (value <= 0) throw new SettingsError("context_snapshot_max must be > 0, got %d".formatted(value));
            t.contextSnapshotMax = value;
        }
        if (v.containsKey("capture_enabled")) t.captureEnabled = requireBool("capture_enabled", v.get("capture_enabled"));
        if (v.containsKey("capture_every_turns")) {
            int value = requireInt("capture_every_turns", v.get("capture_every_turns"));
            if (value < 1) throw new SettingsError("capture_every_turns must be >= 1, got %d".formatted(value));
            t.captureEveryTurns = value;
        }
        if (v.containsKey("pipeline_profile")) {
            String value = v.get("pipeline_profile");
            if (!KNOWN_PIPELINE_PROFILES.contains(value)) {
                throw new SettingsError("unknown MEMORY_PIPELINE_PROFILE '%s'; expected one of %s"
                        .formatted(value, sorted(KNOWN_PIPELINE_PROFILES)));
            }
            t.pipelineProfile = value;
        }
        if (v.containsKey("summary_enabled")) t.summaryEnabled = requireBool("summary_enabled", v.get("summary_enabled"));
        if (v.containsKey("summary_trigger_messages")) {
            int value = requireInt("summary_trigger_messages", v.get("summary_trigger_messages"));
            if (value < 1) throw new SettingsError("summary_trigger_messages must be >= 1, got %d".formatted(value));
            t.summaryTriggerMessages = value;
        }
        if (v.containsKey("profile_enabled")) t.profileEnabled = requireBool("profile_enabled", v.get("profile_enabled"));
        if (v.containsKey("profile_max_chars")) {
            int value = requireInt("profile_max_chars", v.get("profile_max_chars"));
            if (value <= 0) throw new SettingsError("profile_max_chars must be > 0, got %d".formatted(value));
            t.profileMaxChars = value;
        }
        if (v.containsKey("episodic_ttl_days")) {
            int value = requireInt("episodic_ttl_days", v.get("episodic_ttl_days"));
            // 0 是合法值：表示 episodic 记忆不过期。
            if (value < 0) throw new SettingsError("memory_episodic_ttl_days must be >= 0, got %d".formatted(value));
            t.episodicTtlDays = value;
        }
        if (v.containsKey("time_decay_half_life_days")) {
            double value = requireFloat("time_decay_half_life_days", v.get("time_decay_half_life_days"));
            if (!(value > 0)) throw new SettingsError("time_decay_half_life_days must be > 0, got %s".formatted(value));
            t.timeDecayHalfLifeDays = value;
        }
        if (v.containsKey("extraction_max_input_chars")) {
            int value = requireInt("extraction_max_input_chars", v.get("extraction_max_input_chars"));
            if (value <= 0) throw new SettingsError("extraction_max_input_chars must be > 0, got %d".formatted(value));
            t.extractionMaxInputChars = value;
        }
        if (v.containsKey("extraction_max_messages")) {
            int value = requireInt("extraction_max_messages", v.get("extraction_max_messages"));
            if (value <= 0) throw new SettingsError("extraction_max_messages must be > 0, got %d".formatted(value));
            t.extractionMaxMessages = value;
        }
        if (v.containsKey("doc_chunk_chars")) {
            int value = requireInt("doc_chunk_chars", v.get("doc_chunk_chars"));
            if (value <= 0) throw new SettingsError("doc_chunk_chars must be > 0, got %d".formatted(value));
            t.docChunkChars = value;
        }
        if (v.containsKey("doc_chunk_overlap")) {
            int value = requireInt("doc_chunk_overlap", v.get("doc_chunk_overlap"));
            if (value < 0) throw new SettingsError("doc_chunk_overlap must be >= 0, got %d".formatted(value));
            // 重叠必须小于块长，否则分块会陷入死循环。
            if (value >= t.docChunkChars) {
                throw new SettingsError("doc_chunk_overlap must be < doc_chunk_chars (%d), got %d"
                        .formatted(t.docChunkChars, value));
            }
            t.docChunkOverlap = value;
        }
        if (v.containsKey("mysql_pool_size")) {
            int value = requireInt("mysql_pool_size", v.get("mysql_pool_size"));
            if (value <= 0) throw new SettingsError("mysql_pool_size must be > 0, got %d".formatted(value));
            t.mysqlPoolSize = value;
        }
        if (v.containsKey("auth_secret")) t.authSecret = v.get("auth_secret");
        if (v.containsKey("auth_replay_window_seconds")) {
            int value = requireInt("auth_replay_window_seconds", v.get("auth_replay_window_seconds"));
            if (value < 0) {
                throw new SettingsError("memory_auth_replay_window_seconds must be >= 0, got %d".formatted(value));
            }
            t.authReplayWindowSeconds = value;
        }
    }

    private static void applyPlanner(Settings s, Map<String, String> v) {
        PlannerSettings t = s.planner;
        if (v.containsKey("max_subtasks")) {
            int value = requireInt("max_subtasks", v.get("max_subtasks"));
            if (value < 1) throw new SettingsError("max_subtasks must be >= 1, got %d".formatted(value));
            t.maxSubtasks = value;
        }
        if (v.containsKey("subtask_tool_rounds")) {
            int value = requireInt("subtask_tool_rounds", v.get("subtask_tool_rounds"));
            if (value < 1) throw new SettingsError("subtask_tool_rounds must be >= 1, got %d".formatted(value));
            t.subtaskToolRounds = value;
        }
        if (v.containsKey("max_replans")) {
            int value = requireInt("max_replans", v.get("max_replans"));
            if (value < 1) throw new SettingsError("max_replans must be >= 1, got %d".formatted(value));
            t.maxReplans = value;
        }
        if (v.containsKey("min_summary_chars")) {
            int value = requireInt("min_summary_chars", v.get("min_summary_chars"));
            if (value < 0) throw new SettingsError("PLANNER_MIN_SUMMARY_CHARS must be >= 0, got %d".formatted(value));
            t.minSummaryChars = value;
        }
    }

    private static void applyObservability(Settings s, Map<String, String> v) {
        ObservabilitySettings t = s.observability;
        if (v.containsKey("log_json")) t.logJson = requireBool("log_json", v.get("log_json"));
        if (v.containsKey("health_probe_model")) t.healthProbeModel = requireBool("health_probe_model", v.get("health_probe_model"));
    }

    private static void applyCost(Settings s, Map<String, String> v) {
        CostSettings t = s.cost;
        if (v.containsKey("enabled")) t.enabled = requireBool("enabled", v.get("enabled"));
        if (v.containsKey("daily_cny")) t.dailyCny = requireFloat("daily_cny", v.get("daily_cny"));
        if (v.containsKey("monthly_cny")) t.monthlyCny = requireFloat("monthly_cny", v.get("monthly_cny"));
        if (v.containsKey("action")) {
            String value = v.get("action");
            if (!KNOWN_COST_ACTIONS.contains(value)) {
                throw new SettingsError("unknown COST_ACTION '%s'; expected one of %s"
                        .formatted(value, sorted(KNOWN_COST_ACTIONS)));
            }
            t.action = value;
        }
        if (v.containsKey("prices_json")) {
            String value = v.get("prices_json");
            if (!value.strip().isEmpty()) {
                validatePricesJson(value);
            }
            t.pricesJson = value;
        }
    }

    private static void validatePricesJson(String value) {
        // 结构校验：{"model": {"input_miss": …, "input_hit": …, "output": …}}。
        String trimmed = value.strip();
        if (!trimmed.startsWith("{")) {
            throw new SettingsError(
                    "COST_PRICES_JSON must be a JSON object of {\"model\": {\"input_miss\": …, \"input_hit\": …, \"output\": …}}");
        }
        // 委托 Jackson（已随 Spring 在类路径上）做完整合法性校验。
        try {
            com.fasterxml.jackson.databind.JsonNode node =
                    new com.fasterxml.jackson.databind.ObjectMapper().readTree(trimmed);
            if (!node.isObject()) {
                throw new SettingsError(
                        "COST_PRICES_JSON must be a JSON object of {\"model\": {\"input_miss\": …, \"input_hit\": …, \"output\": …}}");
            }
            node.fields().forEachRemaining(e -> {
                if (!e.getValue().isObject()) {
                    throw new SettingsError(
                            "COST_PRICES_JSON must be a JSON object of {\"model\": {\"input_miss\": …, \"input_hit\": …, \"output\": …}}");
                }
            });
        } catch (IOException ex) {
            throw new SettingsError("COST_PRICES_JSON is not valid JSON: %s".formatted(ex.getMessage()));
        }
    }

    private static void applyMcp(Settings s, Map<String, String> v) {
        McpSettings t = s.mcp;
        if (v.containsKey("enabled")) t.enabled = requireBool("enabled", v.get("enabled"));
        if (v.containsKey("servers_json")) t.serversJson = v.get("servers_json");
    }

    /** 解析后的价目表（pricesJson 已由校验保证合法）。 */
    public Map<String, Map<String, Double>> prices() {
        if (cost.pricesJson.strip().isEmpty()) {
            return Map.of();
        }
        try {
            com.fasterxml.jackson.databind.ObjectMapper mapper = new com.fasterxml.jackson.databind.ObjectMapper();
            com.fasterxml.jackson.databind.JsonNode root = mapper.readTree(cost.pricesJson);
            Map<String, Map<String, Double>> out = new LinkedHashMap<>();
            root.fields().forEachRemaining(model -> {
                Map<String, Double> table = new LinkedHashMap<>();
                model.getValue().fields().forEachRemaining(e ->
                        table.put(e.getKey(), e.getValue().asDouble()));
                out.put(model.getKey(), table);
            });
            return out;
        } catch (IOException ex) {
            throw new SettingsError("COST_PRICES_JSON is not valid JSON: %s".formatted(ex.getMessage()));
        }
    }

    // ── 校验后的整体收敛 + 快速失败 ──────────────────────────────────

    private void ensureValid() {
        if (agentModulePackages.isEmpty()) {
            throw new SettingsError("AGENT_MODULE_PACKAGES must list at least one package");
        }
        for (String pkg : agentModulePackages) {
            if (!pkg.matches("^[a-z][a-z0-9_]*(\\.[a-z][a-z0-9_]*)*$")) {
                throw new SettingsError("invalid AGENT_MODULE_PACKAGES entries: [%s]".formatted(pkg));
            }
        }
        if (agentRecursionLimit < 1) {
            throw new SettingsError("AGENT_RECURSION_LIMIT must be >= 1, got %d".formatted(agentRecursionLimit));
        }
        if (!KNOWN_ENVIRONMENTS.contains(env)) {
            throw new SettingsError("unknown ENV '%s'; expected one of %s".formatted(env, sorted(KNOWN_ENVIRONMENTS)));
        }
    }

    /**
     * 在配置错误时快速失败（安全项对所有环境生效，其余限 production）。
     * production 要求真实的 API key；没有它启动会让服务陷入静默损坏的
     * 状态（SEC-C2）。
     */
    public void ensureProductionReady() {
        if (corsOrigins.contains("*")) {
            throw new SettingsError(
                    "CORS_ORIGINS=\"*\" is not allowed (in any environment): a wildcard "
                            + "origin lets any site call the invoke endpoint on the user's behalf");
        }
        if (!"production".equals(env)) {
            return;
        }
        if (llm.apiKey.strip().isEmpty()) {
            throw new SettingsError(
                    "LLM_API_KEY is required in production (ENV=production); "
                            + "refusing to start with an empty key");
        }
        if (memory.enabled && memory.authSecret.strip().isEmpty()) {
            throw new SettingsError(
                    "MEMORY_AUTH_SECRET is required in production when MEMORY_ENABLED=true: "
                            + "memory isolation is enforced via signed X-User-Id "
                            + "(clients must send X-User-Sig = HMAC-SHA256("
                            + "user_id\\ntimestamp\\nnonce, secret))");
        }
        if ("mysql".equals(checkpointer.backend)) {
            List<String> missing = new ArrayList<>();
            if (checkpointer.mysqlUser.strip().isEmpty()) missing.add("CHECKPOINTER_MYSQL_USER");
            if (checkpointer.mysqlDatabase.strip().isEmpty()) missing.add("CHECKPOINTER_MYSQL_DATABASE");
            if (checkpointer.mysqlPassword.strip().isEmpty()) missing.add("CHECKPOINTER_MYSQL_PASSWORD");
            if (!missing.isEmpty()) {
                throw new SettingsError("%s required in production with CHECKPOINTER_BACKEND=mysql"
                        .formatted(String.join(", ", missing)));
            }
        }
    }

    /** Set 的稳定排序（对应 Python sorted(...) 的展示语义）。 */
    private static List<String> sorted(Set<String> set) {
        return set.stream().sorted().toList();
    }

    // ── .env 文件加载（python-dotenv 兼容的受控子集） ─────────────────

    /**
     * 读取 dotenv 文件：{@code KEY=VALUE} 行；支持 {@code export } 前缀、
     * 单/双引号包裹、行内/整行注释（#）与空行。文件不可读按缺失处理。
     */
    static Map<String, String> readDotEnv(Path path) {
        Map<String, String> out = new LinkedHashMap<>();
        if (!Files.isRegularFile(path)) {
            return out;
        }
        List<String> lines;
        try {
            lines = Files.readAllLines(path, StandardCharsets.UTF_8);
        } catch (IOException e) {
            return out; // .env 不可读时按缺失处理
        }
        for (String rawLine : lines) {
            String line = rawLine.strip();
            if (line.isEmpty() || line.startsWith("#")) {
                continue;
            }
            if (line.startsWith("export ")) {
                line = line.substring("export ".length()).strip();
            }
            int eq = line.indexOf('=');
            if (eq <= 0) {
                continue;
            }
            String key = line.substring(0, eq).strip();
            String value = unquote(line.substring(eq + 1).strip());
            // 行内注释：未加引号的值以 " #" 分隔时截断（dotenv 常见语义）。
            if (!value.startsWith("\"") && !value.startsWith("'")) {
                int hash = value.indexOf(" #");
                if (hash >= 0) {
                    value = value.substring(0, hash).strip();
                }
            }
            out.put(key, value);
        }
        return out;
    }

    private static String unquote(String value) {
        if (value.length() >= 2) {
            char first = value.charAt(0);
            char last = value.charAt(value.length() - 1);
            if ((first == '"' && last == '"') || (first == '\'' && last == '\'')) {
                String inner = value.substring(1, value.length() - 1);
                if (first == '"') {
                    // 双引号支持常见的转义对（与 dotenv 语义一致）。
                    return inner.replace("\\n", "\n").replace("\\r", "\r")
                            .replace("\\t", "\t").replace("\\\"", "\"").replace("\\\\", "\\");
                }
                return inner;
            }
        }
        return value;
    }
}
