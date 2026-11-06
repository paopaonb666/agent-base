package io.github.paopaonb666.agentbase.toolkit.search;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 按优先级配置组装引擎列表（对应 Python 侧 {@code search/engines.py} 的
 * {@code build_engines}）。
 *
 * <p><b>终局契约（#30/#245/#276）</b>：Java 侧只实现 tavily 引擎
 * （需 {@code TAVILY_API_KEY}，缺 key 时跳过并记日志，与 Python 一致）。
 * duckduckgo 是 Python 侧的 ddgs 可选依赖引擎——Java 侧不在默认值、
 * 合法清单里（Settings 启动时显式拒绝），这里也不再保留「选中即抛
 * 未迁移」分支；未知名字防御性跳过（配置边界由 Settings 守住）。
 *
 * <p><b>可用性单源（#245）</b>：{@link #usableEngines} 是「哪些引擎此刻
 * 可用」的唯一判定——{@link #buildEngines}（构建实例）与
 * {@link WebSearchTool#available}（注册谓词）都从它派生，加引擎时不再
 * 出现两处判定漂移。
 */
public final class Engines {

    private static final Logger LOG = LoggerFactory.getLogger(Engines.class);

    private Engines() {
    }

    /**
     * 引擎可用性的唯一判定源：按优先级返回「此刻可构建」的引擎名单。
     * tavily 缺 key → 跳过并 WARN（Python 同语义）；未知名字 → 防御性
     * 跳过并 WARN（Settings 校验器是配置边界的守门员，正常到不了这里）。
     */
    public static List<String> usableEngines(List<String> priority, String tavilyApiKey) {
        boolean hasKey = tavilyApiKey != null && !tavilyApiKey.isEmpty();
        List<String> out = new ArrayList<>();
        for (String name : priority) {
            if (name.equals("tavily")) {
                if (hasKey) {
                    out.add(name);
                } else {
                    LOG.warn("TAVILY_API_KEY is empty; tavily engine skipped");
                }
            } else {
                LOG.warn("unknown search engine '{}' skipped (Java side implements tavily only;"
                        + " ddgs engines are Python-side)", name);
            }
        }
        return out;
    }

    /** 按优先级组装可用引擎实例（可用性判定走 {@link #usableEngines}）。 */
    public static List<SearchEngine> buildEngines(List<String> priority, String tavilyApiKey,
                                                  double timeoutSeconds) {
        List<SearchEngine> engines = new ArrayList<>();
        for (String name : usableEngines(priority, tavilyApiKey)) {
            // usableEngines 只返回有实例实现的引擎名。
            engines.add(new TavilyHttpClient(tavilyApiKey, timeoutSeconds));
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
