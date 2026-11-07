package io.github.paopaonb666.agentbase.toolkit.search;

import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 按优先级依次尝试多个搜索引擎；任一成功即返回，全失败返回空（对应
 * Python 侧 {@code FallbackSearchManager}）——单个引擎的抖动不构成对话
 * 失败。
 */
public final class FallbackSearchManager {

    private static final Logger LOG = LoggerFactory.getLogger(FallbackSearchManager.class);

    private final List<SearchEngine> engines;

    public FallbackSearchManager(List<SearchEngine> engines) {
        this.engines = List.copyOf(engines);
    }

    /** 按优先级依次尝试；全部失败返回空列表（管理器已记日志）。 */
    public List<SearchResult> search(String query, int maxResults) {
        return search(query, maxResults, 0);
    }

    /**
     * 带管线预算的依次尝试（#242）：每个引擎拿到 deadline 的剩余部分
     * （预算 − 已耗时）；预算耗尽立即停（不再点燃后续引擎请求——池的超时
     * 掐断调用方后引擎线程继续跑完是已记录的泄漏放大，这里从源头收口）。
     * {@code budgetSeconds <= 0} 表示不设预算（同旧语义）。
     */
    public List<SearchResult> search(String query, int maxResults, double budgetSeconds) {
        long startNanos = System.nanoTime();
        for (SearchEngine engine : engines) {
            double remaining = budgetSeconds <= 0 ? 0
                    : budgetSeconds - (System.nanoTime() - startNanos) / 1_000_000_000.0;
            if (budgetSeconds > 0 && remaining <= 0) {
                LOG.warn("Search pipeline budget {}s exhausted before engine {}",
                        budgetSeconds, engine.name());
                break;
            }
            try {
                List<SearchResult> results = budgetSeconds > 0
                        ? engine.search(query, maxResults, remaining)
                        : engine.search(query, maxResults);
                if (!results.isEmpty()) {
                    LOG.info("Search succeeded via {} ({} results)", engine.name(), results.size());
                    return results;
                }
                LOG.warn("Engine {} returned no results; trying next", engine.name());
            } catch (Exception e) {
                LOG.warn("Engine {} failed: {}", engine.name(), e.toString());
            }
        }
        LOG.error("All search engines failed for query: {}", query);
        return List.of();
    }
}
