package io.github.paopaonb666.agentbase.core.config;

import java.io.IOException;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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

    private static final Logger LOG = LoggerFactory.getLogger(Settings.class);

    // ── 合法取值集合（与 Python 侧常量一一对应） ──────────────────────

    /** 基座知道如何装配的 openai 兼容 provider（仅用于防止配置拼写错误）。 */
    public static final Set<String> KNOWN_PROVIDERS = Set.of("deepseek", "zhipu", "openai-compatible");
    /** 基座识别的 checkpointer 后端。 */
    public static final Set<String> KNOWN_CHECKPOINTER_BACKENDS = Set.of("memory", "sqlite", "mysql");
    /**
     * 注册表全部工具名（与 Python 注册表的键集一致，含未迁移条目）。
     * #103：名单单源下放到 Settings——TOOLKIT_ENABLED 在配置期即校验，
     * 装配期（Toolkits）只引用同一份名单。
     */
    public static final List<String> KNOWN_TOOLKIT_NAMES =
            List.of("calculator", "current_time", "json_query", "python_repl", "web_search");
    /**
     * 工具库搜索引擎实现清单（SEARCH_ENGINE_PRIORITY 的合法取值，#276 终局）：
     * Java 侧只实现了 tavily；duckduckgo 是 Python 侧的 ddgs 可选依赖引擎
     * （{@code pip install 'agent-base[search]'}），Java 配它会在启动时
     * 显式报错而不是构建途中炸出「第一期未迁移」。
     */
    public static final Set<String> KNOWN_SEARCH_ENGINES = Set.of("tavily");
    public static final Set<String> KNOWN_ENVIRONMENTS = Set.of("development", "production");
    /** 记忆形成管线的模型分派档位（MEMORY_PIPELINE_PROFILE 的合法取值）。 */
    public static final Set<String> KNOWN_PIPELINE_PROFILES = Set.of("split", "all_main", "all_fast");
    /** 预算超限行为（COST_ACTION 的合法取值）。 */
    public static final Set<String> KNOWN_COST_ACTIONS = Set.of("warn", "block");
    /** 常见 OpenAI 兼容 provider 的 host（#110：base_url 缺 /v1 的启动期提示用）。 */
    private static final Set<String> KNOWN_OPENAI_HOSTS =
            Set.of("api.deepseek.com", "open.bigmodel.cn", "api.openai.com");

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
    /**
     * 单轮 run 的 handoff 次数上限（#68）：超限后 supervisor 收到结构化
     * 提示强制收敛出最终回答——supervisor↔子模块乒乓循环不再烧满全局
     * recursionLimit 才被用户看到。0 = 不限。
     */
    public int agentHandoffMaxPerRun = 10;
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
        /** API key（Secret 掩码类型：真值只经 value() 取，禁止写入日志）。 */
        public Secret apiKey = Secret.of("");
        public String baseUrl = "https://api.deepseek.com/v1";
        public String model = "deepseek-chat";
        /**
         * 主路径弹性包（#1/#73）：读超时（对齐 openai SDK 默认 600s）、
         * 瞬时错误重试次数（指数退避，ResilientLlm 同默认 max_retries=2）
         * 与全局并发信号量（防 429 连锁雪崩；快档独立信号量不受影响）。
         */
        public double mainTimeoutSeconds = 600.0;
        public int mainRetries = 2;
        public int mainConcurrency = 8;
    }

    /**
     * LLM 快速档配置节：env 前缀 {@code LLM_FAST_}（免费/低价模型吃批量
     * 任务）。它是优化不是依赖：baseUrl 与 model 两者皆非空才算已配置，
     * 未配置时使用 fast 档的调用方应回落主力模型。
     */
    public static final class LlmFastSettings {
        public Secret apiKey = Secret.of("");
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
        public Secret mysqlPassword = Secret.of("");
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
        /**
         * 搜索引擎优先级：按顺序依次尝试，任一成功即返回。默认 tavily
         * （#30/#276：Java 侧唯一已实现引擎——旧默认 duckduckgo 指向不存在的
         * 引擎，照文档启用 web_search 必在启动时失败）。
         */
        public String enginePriority = "tavily";
        public Secret tavilyApiKey = Secret.of("");
        public double timeoutSeconds = 15.0;
        public double cacheTtlSeconds = 300.0;
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
        public Secret embeddingApiKey = Secret.of("");
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
        /**
         * 向量下推（#324/旧 #35）：启用后召回的混合打分候选池 = 向量 KNN
         * top-100 ∪ 无向量候选（检索容量天花板的结构性出口）。默认关闭
         * ——全量打分语义不变；sqlite-vec 真下推（#323）就绪后按需打开。
         */
        public boolean vectorPushdown = false;
        public double timeDecayHalfLifeDays = 14.0;
        public int extractionMaxInputChars = 8000;
        public int extractionMaxMessages = 24;
        public int docChunkChars = 800;
        public int docChunkOverlap = 100;
        public int mysqlPoolSize = 5;
        /** 记忆身份鉴权（P0）：配置后所有用户作用域请求需附带 HMAC 签名。 */
        public Secret authSecret = Secret.of("");
        public int authReplayWindowSeconds = 300;
        /** nonce 重放缓存的 LRU 容量（#207）：超出即淘汰最旧指纹并计数。 */
        public int authReplayCacheLimit = 4096;
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
        /**
         * 语义判分（#63）：空=关（默认，只跑字数门）；"llm" = check 节点
         * 对通过字数门的输出做一次 LLM 判分（done/failed/partial 三态；
         * partial 定向补做一轮）。判分调用走模块主档模型（ModuleContext
         * 无快档通道，登记差异）。
         */
        public String verdict = "";
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
        /** 解析后的价目表（load 期一次解析并校验，#105）。 */
        public Map<String, Map<String, Double>> parsedPrices = Map.of();
    }

    /** MCP 工具来源配置节：env 前缀 {@code MCP_}（stdio 先行，见 toolkit/mcp）。 */
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
                "agent_recursion_limit", "agent_handoff_max_per_run")) {
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
            if (vals.containsKey("agent_handoff_max_per_run")) {
                int value = requireInt("agent_handoff_max_per_run", vals.get("agent_handoff_max_per_run"));
                if (value < 0) {
                    throw new SettingsError("AGENT_HANDOFF_MAX_PER_RUN must be >= 0, got %d".formatted(value));
                }
                s.agentHandoffMaxPerRun = value;
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
        if (v.containsKey("api_key")) t.apiKey = Secret.of(v.get("api_key"));
        if (v.containsKey("base_url")) t.baseUrl = normalizeBaseUrl(v.get("base_url"), "LLM_BASE_URL");
        if (v.containsKey("model")) t.model = v.get("model");
        if (v.containsKey("main_timeout")) {
            double value = requireFloat("main_timeout", v.get("main_timeout"));
            if (!(value > 0)) {
                throw new SettingsError("LLM_MAIN_TIMEOUT must be > 0, got %s".formatted(value));
            }
            t.mainTimeoutSeconds = value;
        }
        if (v.containsKey("main_retries")) {
            int value = requireInt("main_retries", v.get("main_retries"));
            if (value < 0) {
                throw new SettingsError("LLM_MAIN_RETRIES must be >= 0, got %d".formatted(value));
            }
            t.mainRetries = value;
        }
        if (v.containsKey("main_concurrency")) {
            int value = requireInt("main_concurrency", v.get("main_concurrency"));
            if (value < 1) {
                throw new SettingsError("LLM_MAIN_CONCURRENCY must be >= 1, got %d".formatted(value));
            }
            t.mainConcurrency = value;
        }
    }

    private static void applyLlmFast(Settings s, Map<String, String> v) {
        LlmFastSettings t = s.llmFast;
        if (v.containsKey("api_key")) t.apiKey = Secret.of(v.get("api_key"));
        if (v.containsKey("base_url")) t.baseUrl = normalizeBaseUrl(v.get("base_url"), "LLM_FAST_BASE_URL");
        if (v.containsKey("model")) t.model = v.get("model");
        if (v.containsKey("concurrency")) {
            int value = requireInt("concurrency", v.get("concurrency"));
            if (value < 1) {
                throw new SettingsError("LLM_FAST_CONCURRENCY must be >= 1, got %d".formatted(value));
            }
            t.concurrency = value;
        }
    }

    /**
     * base_url 规范化（#110）：trim + 去尾斜杠；已知 provider 的 host 缺
     * {@code /v1} 时告警——langchain4j 的 OpenAI 兼容客户端按 baseUrl 原样
     * 拼路径，缺 /v1 的笔误常见表现为运行期 404，启动期提示一次排障减负。
     * 只提示不改值（自定义网关的路径形态不可枚举，不自动补 /v1）。
     */
    private static String normalizeBaseUrl(String raw, String field) {
        if (raw == null || raw.isBlank()) {
            return raw;
        }
        String v = raw.strip();
        while (v.endsWith("/")) {
            v = v.substring(0, v.length() - 1);
        }
        try {
            String host = URI.create(v).getHost();
            if (host != null && KNOWN_OPENAI_HOSTS.contains(host) && !v.endsWith("/v1")) {
                LOG.warn("{} 缺 /v1（{}）：OpenAI 兼容客户端按 baseUrl 原样拼接，"
                        + "缺 /v1 常见表现为运行期 404", field, v);
            }
        } catch (IllegalArgumentException ignored) {
            // 非法 URL 不在这里做第二套校验——由探活/首次调用报错。
        }
        return v;
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
        if (v.containsKey("mysql_password")) t.mysqlPassword = Secret.of(v.get("mysql_password"));
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
                        .formatted(unknown, sorted(KNOWN_SEARCH_ENGINES))
                        + "（duckduckgo 等 ddgs 引擎仅 Python 侧可用：pip install 'agent-base[search]'）");
            }
            t.enginePriority = value;
        }
        if (v.containsKey("tavily_api_key")) t.tavilyApiKey = Secret.of(v.get("tavily_api_key"));
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
            // 幽灵配置收口（#104←#132/#276）：Java 侧无 ddgs 引擎，该键是
            // Python 专属——不再解析成死字段，WARN 提示即可（Python 部署
            // 带着这个键合法，不能报错）。
            LOG.warn("SEARCH_DDGS_BACKEND 是 Python 侧（ddgs 引擎）专属配置，"
                    + "Java 侧只实现 tavily，该键被忽略；Java 侧搜索引擎清单见 "
                    + "SEARCH_ENGINE_PRIORITY（tavily）");
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
        if (v.containsKey("embedding_api_key")) t.embeddingApiKey = Secret.of(v.get("embedding_api_key"));
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
        if (v.containsKey("vector_pushdown")) {
            t.vectorPushdown = requireBool("vector_pushdown", v.get("vector_pushdown"));
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
        if (v.containsKey("auth_secret")) t.authSecret = Secret.of(v.get("auth_secret"));
        if (v.containsKey("auth_replay_window_seconds")) {
            int value = requireInt("auth_replay_window_seconds", v.get("auth_replay_window_seconds"));
            if (value < 0) {
                throw new SettingsError("memory_auth_replay_window_seconds must be >= 0, got %d".formatted(value));
            }
            t.authReplayWindowSeconds = value;
        }
        if (v.containsKey("auth_replay_cache_limit")) {
            int value = requireInt("auth_replay_cache_limit", v.get("auth_replay_cache_limit"));
            if (value < 1) {
                throw new SettingsError("memory_auth_replay_cache_limit must be >= 1, got %d".formatted(value));
            }
            t.authReplayCacheLimit = value;
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
        if (v.containsKey("verdict")) {
            String value = v.get("verdict").strip().toLowerCase(java.util.Locale.ROOT);
            if (!value.isEmpty() && !"off".equals(value) && !"llm".equals(value)) {
                throw new SettingsError(
                        "PLANNER_VERDICT must be one of '', 'off', 'llm', got '%s'".formatted(value));
            }
            t.verdict = "llm".equals(value) ? "llm" : "";
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
                t.parsedPrices = parsePricesJson(value);
            }
            t.pricesJson = value;
        }
    }

    /**
     * 价目表一次解析（#105）：load 期完成结构+数值校验并缓存进
     * parsedPrices——之前 load 只查结构、prices() 读取时二次解析，且
     * asDouble 对非数值叶子静默回 0.0（"2" 会被读成 0，白烧钱不可见）。
     * 数值字符串按 Python float() 口径接受，非数值 fail-fast。
     */
    private static Map<String, Map<String, Double>> parsePricesJson(String value) {
        // 结构契约：{"model": {"input_miss": …, "input_hit": …, "output": …}}。
        String structureError =
                "COST_PRICES_JSON must be a JSON object of {\"model\": {\"input_miss\": …, \"input_hit\": …, \"output\": …}}";
        String trimmed = value.strip();
        if (!trimmed.startsWith("{")) {
            throw new SettingsError(structureError);
        }
        try {
            com.fasterxml.jackson.databind.JsonNode root =
                    new com.fasterxml.jackson.databind.ObjectMapper().readTree(trimmed);
            if (!root.isObject()) {
                throw new SettingsError(structureError);
            }
            Map<String, Map<String, Double>> out = new LinkedHashMap<>();
            root.fields().forEachRemaining(model -> {
                if (!model.getValue().isObject()) {
                    throw new SettingsError(structureError);
                }
                Map<String, Double> table = new LinkedHashMap<>();
                model.getValue().fields().forEachRemaining(e -> {
                    com.fasterxml.jackson.databind.JsonNode leaf = e.getValue();
                    if (leaf.isNumber()) {
                        table.put(e.getKey(), leaf.asDouble());
                    } else if (leaf.isTextual()) {
                        try {
                            table.put(e.getKey(), Double.parseDouble(leaf.asText()));
                        } catch (NumberFormatException ex) {
                            throw new SettingsError(
                                    "COST_PRICES_JSON: price for %s.%s must be a number, got '%s'"
                                            .formatted(model.getKey(), e.getKey(), leaf.asText()));
                        }
                    } else {
                        throw new SettingsError(
                                "COST_PRICES_JSON: price for %s.%s must be a number"
                                        .formatted(model.getKey(), e.getKey()));
                    }
                });
                out.put(model.getKey(), table);
            });
            return out;
        } catch (IOException ex) {
            throw new SettingsError("COST_PRICES_JSON is not valid JSON: %s".formatted(ex.getMessage()));
        }
    }

    private static void applyMcp(Settings s, Map<String, String> v) {
        McpSettings t = s.mcp;
        if (v.containsKey("enabled")) t.enabled = requireBool("enabled", v.get("enabled"));
        if (v.containsKey("servers_json")) t.serversJson = v.get("servers_json");
    }

    /** 解析后的价目表（load 期一次解析缓存；未配置 → 空表）。 */
    public Map<String, Map<String, Double>> prices() {
        return cost.parsedPrices;
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
        // #103：TOOLKIT_ENABLED 名单校验前移到配置期（此前在装配期才报）。
        for (String name : toolkit.enabled) {
            if (!KNOWN_TOOLKIT_NAMES.contains(name)) {
                throw new SettingsError("unknown TOOLKIT_ENABLED entry '" + name + "'; known toolkit tools: "
                        + KNOWN_TOOLKIT_NAMES);
            }
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
        if (llm.apiKey.isBlank()) {
            throw new SettingsError(
                    "LLM_API_KEY is required in production (ENV=production); "
                            + "refusing to start with an empty key");
        }
        if (memory.enabled && memory.authSecret.isBlank()) {
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
            if (checkpointer.mysqlPassword.isBlank()) missing.add("CHECKPOINTER_MYSQL_PASSWORD");
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
            out.put(key, parseEnvValue(line.substring(eq + 1).strip()));
        }
        return out;
    }

    /**
     * 单个 .env 值的解析（解引号 + 行内注释，#114）：
     * 引号内的 " #" 保留（对齐 python-dotenv）；未加引号的值以 " #"
     * 分隔时截断行内注释（dotenv 常见语义）。引号判定必须在解引号
     * 之前做——先 unquote 再判会作用在已剥引号的值上，守卫失效。
     */
    static String parseEnvValue(String raw) {
        boolean quoted = raw.startsWith("\"") || raw.startsWith("'");
        String value = unquote(raw);
        if (!quoted) {
            int hash = value.indexOf(" #");
            if (hash >= 0) {
                value = value.substring(0, hash).strip();
            }
        }
        return value;
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
