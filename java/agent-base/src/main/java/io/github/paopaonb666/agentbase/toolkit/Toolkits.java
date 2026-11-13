package io.github.paopaonb666.agentbase.toolkit;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.core.tools.AgentTool;
import io.github.paopaonb666.agentbase.toolkit.search.WebSearchTool;

/**
 * 工具库注册表（对应 Python 侧 {@code tools/registry.py}）：把基座内置
 * 工具以声明式元数据的形式登记，{@link #buildToolkitTools} 在 bootstrap
 * 时按 {@code TOOLKIT_ENABLED} 装配启用的工具并追加进共享池。
 *
 * <p>治理语义（继承自 Python 侧注释）：
 *
 * <ul>
 *   <li><b>默认最小暴露</b>——{@code TOOLKIT_ENABLED} 默认只含零依赖工具
 *       （current_time / calculator / json_query）；依赖外部服务的
 *       {@code web_search} 必须显式开启且配置齐全才存在。</li>
 *   <li><b>启用即承诺</b>——显式启用的工具若依赖缺失或配置不全，是启动
 *       时的配置错误（{@link ToolkitException}），不是运行时的静默缺席；
 *       错误信息带上可执行的修复指引。</li>
 *   <li><b>名字即契约</b>——{@code TOOLKIT_ENABLED} 的合法取值就是注册表
 *       的键，拼错在启动时快速失败（与 {@code AGENT_MODULES} 一致）。</li>
 *   <li><b>per-tool 超时</b>——注册表声明的独立超时（如网络搜索显著短于
 *       全局预算）经 {@link #toolkitTimeouts} 下发到共享池。</li>
 * </ul>
 *
 * <p>MCP 工具来源不随工具库迁移——它由基座 bootstrap 阶段接入
 * （toolkit/mcp 包，#331/#333），此处不再留位。
 * TODO(repl)：{@code python_repl} 第一期未迁移（见 buildToolkitTools 的
 * 分支文案）。
 */
public final class Toolkits {

    /** 注册表全部工具名——名单单源在 Settings（#103），此处保留别名兼容既有引用。 */
    public static final List<String> KNOWN_TOOLKIT_NAMES = Settings.KNOWN_TOOLKIT_NAMES;

    private Toolkits() {
    }

    /**
     * 按 {@code settings.toolkit.enabled} 装配启用的内置工具。
     *
     * <p>启用但不可用 → {@link ToolkitException}（快速失败 + 修复指引）；
     * 未启用 → 工厂根本不会被调用（依赖缺失完全无感）。
     */
    public static List<AgentTool> buildToolkitTools(Settings settings) {
        return buildToolkitTools(settings, null);
    }

    /**
     * 同上，附批量嵌入函数（A-31）：记忆插件端口注入（null=embedding
     * 未配置/插件缺席）——web_search 的语义 rerank 复用它，检索侧零新增
     * LLM 成本。
     */
    public static List<AgentTool> buildToolkitTools(Settings settings,
                                                    java.util.function.Function<java.util.List<String>, java.util.List<java.util.List<Float>>> embedBatch) {
        List<AgentTool> tools = new ArrayList<>();
        for (String name : settings.toolkit.enabled) {
            switch (name) {
                case CurrentTimeTool.NAME -> tools.add(new CurrentTimeTool());
                case CalculatorTool.NAME -> tools.add(new CalculatorTool());
                case JsonQueryTool.NAME -> tools.add(new JsonQueryTool());
                case WebSearchTool.NAME -> {
                    if (!WebSearchTool.available(settings)) {
                        throw new ToolkitException(
                                "toolkit tool '" + name + "' is enabled but unavailable: "
                                        + WebSearchTool.UNAVAILABLE_REASON);
                    }
                    tools.add(WebSearchTool.build(settings, embedBatch));
                }
                case "python_repl" -> throw new ToolkitException(
                        "python_repl 第一期未迁移：启用即承诺子进程隔离执行的 exec 级语义"
                                + "（-I 隔离、超时强制 kill、资源限额、输出封顶），"
                                + "Java 重构第一期不提供该承诺；请从 TOOLKIT_ENABLED 移除"
                                + "该条目，或等待后续迁移。");
                default -> throw new ToolkitException(
                        "unknown TOOLKIT_ENABLED entry '" + name + "'; known toolkit tools: "
                                + KNOWN_TOOLKIT_NAMES);
            }
        }
        return tools;
    }

    /**
     * 收集已启用工具声明的独立超时（供共享池覆盖全局预算；对应 Python 的
     * {@code toolkit_timeouts}）：web_search 25.0（引擎自身有更短的搜索
     * 超时，池预算只做兜底）、python_repl 60.0（子进程启动/产物回收的
     * 开销余量）。未列出的工具沿用全局 {@code TOOL_TIMEOUT_SECONDS}。
     */
    public static Map<String, Double> toolkitTimeouts(Settings settings) {
        Map<String, Double> out = new LinkedHashMap<>();
        for (String name : settings.toolkit.enabled) {
            switch (name) {
                case WebSearchTool.NAME -> out.put(name, 25.0);
                case "python_repl" -> out.put(name, 60.0);
                default -> {
                    // 基础三工具无独立超时（沿用全局预算）。
                }
            }
        }
        return out;
    }

    /** 注册表里的全部工具名（错误信息与文档用）。 */
    public static List<String> knownToolkitNames() {
        return KNOWN_TOOLKIT_NAMES;
    }
}
