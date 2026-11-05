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
        for (SearchEngine engine : engines) {
            try {
                List<SearchResult> results = engine.search(query, maxResults);
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
