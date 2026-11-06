package io.github.paopaonb666.agentbase.core.storage;

import io.github.paopaonb666.agentbase.core.AgentModule;
import io.github.paopaonb666.agentbase.core.config.Settings;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 基座自有存储（thread_index + plan_snapshots）的往返测试：memory /
 * sqlite 两后端的生命周期、工厂降级语义与 PlanStoreAdapter 的契约形态。
 * sqlite 用临时目录文件；不依赖网络/MySQL。
 */
class CoreStoreTest {

    @TempDir
    Path tempDir;

    // ── 工厂 ─────────────────────────────────────────────────────────

    @Test
    void factoryBuildsMemoryBackend() {
        Settings settings = new Settings();
        settings.checkpointer.backend = "memory";
        assertInstanceOf(MemoryCoreStore.class, CoreStores.build(settings));
    }

    @Test
    void factoryDegradesToNullOnSqliteFailure() {
        Settings settings = new Settings();
        settings.checkpointer.backend = "sqlite";
        settings.checkpointer.sqlitePath =
                tempDir.resolve("missing-dir").resolve("core.db").toString();
        // 建表/建库失败不抛出：只降级为 null（对应 Python 返回 None）。
        assertNull(CoreStores.build(settings));
    }

    @Test
    void factoryDegradesToNullOnUnknownBackend() {
        Settings settings = new Settings();
        settings.checkpointer.backend = "redis";
        assertNull(CoreStores.build(settings));
    }

    // ── 会话索引（thread_index） ─────────────────────────────────────

    @Test
    void memoryThreadIndexLifecycle() {
        try (MemoryCoreStore store = new MemoryCoreStore()) {
            threadIndexLifecycle(store);
        }
    }

    @Test
    void sqliteThreadIndexLifecycle() {
        try (SqliteCoreStore store = new SqliteCoreStore(tempDir.resolve("core.db").toString())) {
            threadIndexLifecycle(store);
        }
    }

    private void threadIndexLifecycle(CoreStore store) {
        assertNull(store.getThreadIndex("chat:a"));

        // 显式时间戳以控制 updated_at 的排序。
        store.upsertThreadIndex(new ThreadIndexRecord("chat:a", "u1", "chat", "会话A", 100.0, 200.0));
        store.upsertThreadIndex(new ThreadIndexRecord("chat:b", "u1", "chat", "会话B", 300.0, 400.0));
        store.upsertThreadIndex(new ThreadIndexRecord("chat:c", "u2", "chat", "会话C", 500.0, 600.0));

        ThreadIndexRecord got = store.getThreadIndex("chat:a");
        assertNotNull(got);
        assertEquals("u1", got.userId());
        assertEquals("chat", got.module());
        assertEquals("会话A", got.title());
        assertEquals(200.0, got.updatedAt());
        assertNull(store.getThreadIndex("chat:absent"));

        // 按用户列出：updated_at 降序，只含本用户的行。
        List<ThreadIndexRecord> mine = store.listThreadIndexes("u1", null, 50);
        assertEquals(List.of("chat:b", "chat:a"),
                mine.stream().map(ThreadIndexRecord::threadId).toList());

        // module 过滤。
        assertEquals(0, store.listThreadIndexes("u1", "planner", 50).size());
        assertEquals(2, store.listThreadIndexes("u1", "chat", 50).size());

        // limit 截断（下限 1：limit <= 1 都按 1 处理，对应 Python max(1, limit)）。
        assertEquals(1, store.listThreadIndexes("u1", null, 1).size());
        assertEquals(1, store.listThreadIndexes("u1", null, 0).size());

        // upsert 覆盖同一 thread_id。
        store.upsertThreadIndex(new ThreadIndexRecord("chat:a", "u1", "chat", "新标题", 100.0, 999.0));
        assertEquals("新标题", store.getThreadIndex("chat:a").title());
        assertEquals(List.of("chat:a", "chat:b"),
                store.listThreadIndexes("u1", null, 50).stream()
                        .map(ThreadIndexRecord::threadId).toList());

        store.deleteThreadIndex("chat:a");
        assertNull(store.getThreadIndex("chat:a"));
        assertEquals(1, store.listThreadIndexes("u1", null, 50).size());
    }

    // ── 计划快照（plan_snapshots） ───────────────────────────────────

    @Test
    void memoryPlanLifecycle() {
        try (MemoryCoreStore store = new MemoryCoreStore()) {
            planLifecycle(store);
        }
    }

    @Test
    void sqlitePlanLifecycle() {
        try (SqliteCoreStore store = new SqliteCoreStore(tempDir.resolve("plan.db").toString())) {
            planLifecycle(store);
        }
    }

    private void planLifecycle(CoreStore store) {
        assertNull(store.getPlan("chat:p"));

        List<Map<String, Object>> tasks = new ArrayList<>();
        tasks.add(task(1, "查天气"));
        tasks.add(task(2, "订机票"));

        store.savePlan("chat:p", "planner", tasks, 0, 0);

        Map<String, Object> plan = store.getPlan("chat:p");
        assertNotNull(plan);
        // 载荷与 Python get_plan 逐键一致：thread_id/module/tasks/cursor/replans。
        assertEquals("chat:p", plan.get("thread_id"));
        assertEquals("planner", plan.get("module"));
        assertEquals(0, ((Number) plan.get("cursor")).intValue());
        assertEquals(0, ((Number) plan.get("replans")).intValue());
        assertEquals(2, ((List<?>) plan.get("tasks")).size());
        assertEquals("查天气",
                ((Map<?, ?>) ((List<?>) plan.get("tasks")).get(0)).get("goal"));
        assertEquals(5, plan.size());

        // 推进 cursor 后再次保存（upsert 覆盖）。
        store.savePlan("chat:p", "planner", tasks, 1, 0);
        assertEquals(1, ((Number) store.getPlan("chat:p").get("cursor")).intValue());

        // 修改 tasks 内容后覆盖，tasks_json 序列化往返无损。
        List<Map<String, Object>> newTasks = new ArrayList<>();
        newTasks.add(task(1, "改签机票"));
        store.savePlan("chat:p", "planner", newTasks, 1, 1);
        Map<String, Object> replanned = store.getPlan("chat:p");
        assertEquals(1, ((Number) replanned.get("replans")).intValue());
        assertEquals(1, ((List<?>) replanned.get("tasks")).size());
        assertEquals("改签机票",
                ((Map<?, ?>) ((List<?>) replanned.get("tasks")).get(0)).get("goal"));

        store.deletePlan("chat:p");
        assertNull(store.getPlan("chat:p"));
    }

    // ── PlanStorePort 适配器契约 ─────────────────────────────────────

    @Test
    void planStoreAdapterDelegatesToCoreStore() {
        MemoryCoreStore store = new MemoryCoreStore();
        AgentModule.PlanStorePort port = new PlanStoreAdapter(store);

        port.savePlan("chat:p", "planner", List.of(task(1, "查天气")), 0, 1);
        Map<String, Object> plan = port.getPlan("chat:p");
        // 契约形态（与 Python get_plan 逐键一致）：
        // thread_id/module/tasks(List<Map>)/cursor(int)/replans(int)。
        assertEquals(5, plan.size());
        assertEquals("chat:p", plan.get("thread_id"));
        assertEquals(1, ((List<?>) plan.get("tasks")).size());
        assertEquals(0, ((Number) plan.get("cursor")).intValue());
        assertEquals(1, ((Number) plan.get("replans")).intValue());
        assertEquals("planner", plan.get("module"));

        port.deletePlan("chat:p");
        assertNull(port.getPlan("chat:p"));
    }

    private static Map<String, Object> task(int id, String goal) {
        Map<String, Object> task = new LinkedHashMap<>();
        task.put("id", id);
        task.put("goal", goal);
        return task;
    }
}
