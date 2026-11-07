package io.github.paopaonb666.agentbase.toolkit.search;

import java.util.List;

/**
 * 搜索引擎抽象：把各家引擎归一成统一的结果结构（对应 Python 侧
 * {@code SearchEngine} ABC）。失败直接抛异常，由
 * {@link FallbackSearchManager} 决定降级——单个引擎的抖动不构成对话失败。
 */
public interface SearchEngine {

    /** 引擎名（进入结果的 engine 字段与日志）。 */
    String name();

    /**
     * 执行搜索，返回统一格式的结果；失败抛异常由管理器降级。
     *
     * <p>超时语义：Python 侧用 {@code asyncio.wait_for} 在工具层给整个
     * 管线设预算；Java 侧（第一期只有 HTTP 引擎）等价地由 HTTP 客户端
     * 的请求超时（{@code SEARCH_TIMEOUT_SECONDS}）强制执行。
     */
    List<SearchResult> search(String query, int maxResults) throws Exception;

    /**
     * 带管线预算的变体（#242）：{@code timeoutSeconds} 是调用方传入的
     * <b>剩余预算</b>（整条管线的 deadline 减去已耗时），引擎应把请求
     * 超时压到不超过它。默认忽略预算、走各自配置的超时（自研引擎兼容）。
     */
    default List<SearchResult> search(String query, int maxResults,
                                      double timeoutSeconds) throws Exception {
        return search(query, maxResults);
    }
}
