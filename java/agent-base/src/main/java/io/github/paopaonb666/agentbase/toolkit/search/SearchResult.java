package io.github.paopaonb666.agentbase.toolkit.search;

/**
 * 归一化的一条搜索结果（对应 Python 侧 {@code search/engines.py} 的
 * {@code SearchResult} dataclass）：各家引擎的差异在引擎实现内抹平，
 * 管线与工具层只消费这一个结构。
 */
public record SearchResult(String title, String url, String snippet, int position, String engine) {
}
