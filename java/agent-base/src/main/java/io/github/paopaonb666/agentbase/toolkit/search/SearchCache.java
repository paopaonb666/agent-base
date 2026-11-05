package io.github.paopaonb666.agentbase.toolkit.search;

import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 进程内 TTL 缓存：命中返回缓存结果，过期/未命中返回 null（对应 Python
 * 侧 {@code search/pipeline.py} 的 {@code SearchCache}）。
 *
 * <p>刻意不进 MySQL——搜索缓存丢了无碍正确性，不值得加存储依赖。LRU
 * 容量封顶（{@value #CACHE_MAX_ENTRIES}）防止长驻进程在大量不同 query
 * 下内存无界增长；淘汰最旧的条目（搜索缓存被丢弃是安全的，只是多一次
 * 真实搜索）。TTL 的时钟用单调时钟（对应 Python 的
 * {@code time.monotonic()}，Java 用 {@link System#nanoTime()}）。
 */
public final class SearchCache {

    /** 缓存条目上限。 */
    public static final int CACHE_MAX_ENTRIES = 256;

    private static final class Entry {
        final long atNanos;
        final List<SearchResult> results;

        Entry(long atNanos, List<SearchResult> results) {
            this.atNanos = atNanos;
            this.results = results;
        }
    }

    private final long ttlNanos;
    // accessOrder=true → get 命中即视为最近使用（对应 Python 的 move_to_end）。
    private final Map<String, Entry> entries = new LinkedHashMap<>(64, 0.75f, true);

    public SearchCache(double ttlSeconds) {
        this.ttlNanos = (long) (ttlSeconds * 1_000_000_000L);
    }

    /** 命中返回缓存结果；过期条目被顺手清除。 */
    public synchronized List<SearchResult> get(String query) {
        Entry entry = entries.get(query);
        if (entry == null) {
            return null;
        }
        if (System.nanoTime() - entry.atNanos >= ttlNanos) {
            entries.remove(query);
            return null;
        }
        return entry.results;
    }

    /** 写入并按容量淘汰最旧（最久未访问）条目。 */
    public synchronized void put(String query, List<SearchResult> results) {
        entries.put(query, new Entry(System.nanoTime(), List.copyOf(results)));
        while (entries.size() > CACHE_MAX_ENTRIES) {
            Iterator<String> it = entries.keySet().iterator();
            it.next();
            it.remove();
        }
    }
}
