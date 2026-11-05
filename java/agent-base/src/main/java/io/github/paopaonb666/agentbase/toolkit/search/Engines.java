package io.github.paopaonb666.agentbase.toolkit.search;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import io.github.paopaonb666.agentbase.toolkit.ToolkitException;

/**
 * 按优先级配置组装引擎列表（对应 Python 侧 {@code search/engines.py} 的
 * {@code build_engines}）。
 *
 * <p><b>第一期契约（已获批准的裁剪）</b>：{@code SEARCH_ENGINE_PRIORITY}
 * 含 {@code duckduckgo} 且被选中构建时抛
 * {@link ToolkitException.SearchEngineError}（说明未迁移），而不是 Python
 * 版的"依赖缺席则跳过并记日志"——静默跳过会让"配了引擎却从未生效"
 * 变成无法察觉的配置漂移；显式报错把迁移缺口变成部署方的显式决策。
 * tavily 完整实现（需 {@code TAVILY_API_KEY}，缺 key 时跳过并记日志，
 * 与 Python 一致）。
 */
public final class Engines {

    private static final Logger LOG = LoggerFactory.getLogger(Engines.class);

    private Engines() {
    }

    /**
     * 按优先级配置组装引擎列表。与 Python 的差异：不再有 ddgs 可选依赖
     * 维度（引擎未迁移），{@code ddgs_backend} 配置保留在
     * {@code SearchSettings} 里但无消费方。
     */
    public static List<SearchEngine> buildEngines(List<String> priority, String tavilyApiKey,
                                                  double timeoutSeconds) {
        List<SearchEngine> engines = new ArrayList<>();
        for (String name : priority) {
            if (name.equals("tavily")) {
                if (tavilyApiKey != null && !tavilyApiKey.isEmpty()) {
                    engines.add(new TavilyHttpClient(tavilyApiKey, timeoutSeconds));
                } else {
                    LOG.warn("TAVILY_API_KEY is empty; tavily engine skipped");
                }
            } else if (name.equals("duckduckgo")) {
                throw new ToolkitException.SearchEngineError(
                        "搜索引擎 duckduckgo 第一期未迁移：Java 重构只实现了 tavily 引擎"
                                + "（需要 TAVILY_API_KEY）；请把 SEARCH_ENGINE_PRIORITY 改为"
                                + " 'tavily' 并配置 key，或移除 duckduckgo。");
            }
            // 未知名字到不了这里：Settings 已在启动时校验 SEARCH_ENGINE_PRIORITY。
        }
        return engines;
    }

    /** 解析 engine_priority 配置串（逗号分隔，去空白、小写）。 */
    public static List<String> parsePriority(String enginePriority) {
        List<String> out = new ArrayList<>();
        for (String part : enginePriority.split(",")) {
            String p = part.strip().toLowerCase(Locale.ROOT);
            if (!p.isEmpty()) {
                out.add(p);
            }
        }
        return out;
    }
}
