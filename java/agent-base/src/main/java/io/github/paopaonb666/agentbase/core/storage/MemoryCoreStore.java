package io.github.paopaonb666.agentbase.core.storage;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 进程内字典实现的基座存储：memory 后端（重启即丢）与测试用（对应
 * Python 侧 {@code memory_backend.py} 的 {@code CoreMemoryStore}）。
 */
public final class MemoryCoreStore implements CoreStore {

    private final Map<String, ThreadIndexRecord> threads = new ConcurrentHashMap<>();
    private final Map<String, Map<String, Object>> plans = new ConcurrentHashMap<>();

    // ── 会话索引（thread_index；S1 线程作用域） ─────────────────────

    @Override
    public void upsertThreadIndex(ThreadIndexRecord entry) {
        threads.put(entry.threadId(), entry);
    }

    @Override
    public ThreadIndexRecord getThreadIndex(String threadId) {
        return threads.get(threadId);
    }

    @Override
    public List<ThreadIndexRecord> listThreadIndexes(String userId, String module, int limit) {
        List<ThreadIndexRecord> matched = new ArrayList<>();
        for (ThreadIndexRecord entry : threads.values()) {
            if (entry.userId().equals(userId) && (module == null || entry.module().equals(module))) {
                matched.add(entry);
            }
        }
        matched.sort(Comparator.comparingDouble(ThreadIndexRecord::updatedAt).reversed());
        int bound = Math.min(matched.size(), Math.max(1, limit));
        return new ArrayList<>(matched.subList(0, bound));
    }

    @Override
    public void deleteThreadIndex(String threadId) {
        threads.remove(threadId);
    }

    // ── 计划快照（plan_snapshots；M10 跨轮延续） ────────────────────

    @Override
    public void savePlan(String threadId, String module, List<Map<String, Object>> tasks,
                         int cursor, int replans) {
        // 逐条浅拷贝任务 dict（对应 Python [dict(task) for task in tasks]）。
        List<Map<String, Object>> taskCopies = new ArrayList<>();
        for (Map<String, Object> task : tasks == null ? List.<Map<String, Object>>of() : tasks) {
            taskCopies.add(new LinkedHashMap<>(task));
        }
        Map<String, Object> snapshot = new LinkedHashMap<>();
        snapshot.put("module", module);
        snapshot.put("tasks", taskCopies);
        snapshot.put("cursor", cursor);
        snapshot.put("replans", replans);
        snapshot.put("updated_at", System.currentTimeMillis() / 1000.0);
        plans.put(threadId, snapshot);
    }

    @Override
    public Map<String, Object> getPlan(String threadId) {
        Map<String, Object> snapshot = plans.get(threadId);
        return snapshot == null ? null : new LinkedHashMap<>(snapshot);
    }

    @Override
    public void deletePlan(String threadId) {
        plans.remove(threadId);
    }

    /** 清空缓存（对应 Python aclose）。 */
    @Override
    public void close() {
        threads.clear();
        plans.clear();
    }
}
