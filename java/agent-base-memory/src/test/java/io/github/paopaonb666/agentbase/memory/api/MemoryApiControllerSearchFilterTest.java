package io.github.paopaonb666.agentbase.memory.api;

import java.lang.reflect.Constructor;
import java.util.List;
import java.util.Map;

import io.github.paopaonb666.agentbase.core.AgentModule;
import io.github.paopaonb666.agentbase.core.AgentRuntime;
import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.entrypoints.server.AuthBackends;
import io.github.paopaonb666.agentbase.memory.service.MemoryService;
import io.github.paopaonb666.agentbase.memory.store.InMemoryMemoryStore;
import io.github.paopaonb666.agentbase.memory.store.Models.MemoryRecord;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * GUI 验收二轮 D9/D10 回归锁（GET /v1/memory 检索分支）：
 * <ul>
 *   <li>D10——q 检索分支此前只过滤 kind，status 被静默忽略：带检索词时
 *       GUI 的状态筛选（启用中/已归档/已废弃）完全失效；</li>
 *   <li>D9——检索端点不享受 #181 显著度豁免：salience≥0.9 的无关记忆
 *       此前旁路双门槛，任何查询都返回全量列表，空态不可达。</li>
 * </ul>
 *
 * <p>装配与 {@link MemoryApiControllerAuditCursorTest} 同款：控制器直调 +
 * 真实 {@link MemoryService} + {@link InMemoryMemoryStore} + never-embedder
 * （无向量降级路径）+ 鉴权直通桩。
 */
class MemoryApiControllerSearchFilterTest {

    private final InMemoryMemoryStore store = new InMemoryMemoryStore();
    private final MemoryService memory =
            new MemoryService(store, neverEmbedder(), Settings.load(Map.of(), null), null, null);
    private final MemoryApiController controller =
            new MemoryApiController(runtimeWith(memory), PASS_THROUGH_AUTH);

    private static final AuthBackends.AuthBackend PASS_THROUGH_AUTH =
            (userId, sig, ts, nonce, settings) ->
                    userId == null || userId.isEmpty() ? "default" : userId;

    private static io.github.paopaonb666.agentbase.memory.embeddings.EmbeddingClient neverEmbedder() {
        return new io.github.paopaonb666.agentbase.memory.embeddings.EmbeddingClient() {
            @Override public Integer dims() { return null; }
            @Override public java.util.List<java.util.List<Float>> embed(java.util.List<String> texts) {
                return null;
            }
            @Override public boolean probe() { return false; }
        };
    }

    /** 反射装配最小运行时（构造器包私有且跨包）：只注入 settings 与 memory 端口。 */
    private static AgentRuntime runtimeWith(MemoryService memory) {
        try {
            Constructor<AgentRuntime> ctor = AgentRuntime.class.getDeclaredConstructor(
                    Settings.class,
                    io.github.paopaonb666.agentbase.core.llm.ModelRuntime.class,
                    Map.class,
                    io.github.paopaonb666.agentbase.core.tools.ToolPool.class,
                    io.github.paopaonb666.agentbase.core.graph.Checkpointer.class,
                    io.github.paopaonb666.agentbase.extensions.toollog.ToolCallRecorder.class,
                    io.github.paopaonb666.agentbase.extensions.filestore.UploadedFileStore.class,
                    AgentModule.MemoryServicePort.class,
                    io.github.paopaonb666.agentbase.core.storage.CoreStore.class,
                    AgentModule.PlanStorePort.class);
            ctor.setAccessible(true);
            return ctor.newInstance(Settings.load(Map.of(), null), null, Map.of(), null,
                    null, null, null, memory, null, null);
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException("AgentRuntime 反射装配失败", e);
        }
    }

    private MemoryRecord seed(String content, double salience) {
        return memory.addMemory("alice", "chat", content, "semantic", List.of(), salience, "");
    }

    @SuppressWarnings("unchecked")
    private List<Map<String, Object>> search(String q, String status) {
        Map<String, Object> out = controller.list(q, null, null, status, 20,
                "alice", null, null, null);
        return (List<Map<String, Object>>) out.get("memories");
    }

    // ── D10：q 检索分支的 status 过滤 ───────────────────────────────────

    @Test
    void statusFilterAppliesWithinQuerySearch() {
        MemoryRecord active = seed("数据库备份在凌晨三点执行", 0.5);
        MemoryRecord archived = seed("数据库扩容需要先加只读副本", 0.5);
        controller.patch(archived.memoryId(),
                new MemoryApiController.MemoryPatchRequest(null, null, null, "archived"),
                "alice", null, null, null);

        // 无 status：检索候选池默认只含 active（既有 chat 召回同款语义）——
        // 只有 active 条目可见。
        List<Map<String, Object>> unfiltered = search("数据库", null);
        assertEquals(List.of(active.memoryId()),
                unfiltered.stream().map(m -> m.get("memory_id")).toList());
        // 带 status：候选池随筛选切换——archived 查询返回归档条目本身。
        // 修复前 q 分支忽略 status：archived 查询返回的是 active 条目。
        List<Map<String, Object>> archivedOnly = search("数据库", "archived");
        assertEquals(List.of(archived.memoryId()),
                archivedOnly.stream().map(m -> m.get("memory_id")).toList());
        List<Map<String, Object>> activeOnly = search("数据库", "active");
        assertEquals(List.of(active.memoryId()),
                activeOnly.stream().map(m -> m.get("memory_id")).toList());
        // 过滤后无命中 → 空列表（空态可达）。
        assertTrue(search("数据库", "superseded").isEmpty());
    }

    // ── D9：检索端点不享受显著度豁免 ────────────────────────────────────

    @Test
    void querySearchDropsHighSalienceUnrelatedMemory() {
        seed("数据库备份在凌晨三点执行", 0.5);
        // salience 1.0 ≥ 豁免阈值 0.9：修复前旁路双门槛，无关词也整条入榜。
        seed("完全无关的常驻事实", 1.0);

        List<Map<String, Object>> hits = search("数据库", null);
        assertEquals(1, hits.size(), "无关高显著度记忆不得进入检索结果: " + hits);
        assertTrue(((String) hits.get(0).get("content")).contains("数据库备份"));
        // 与查询零交集 → 空结果（空态可达）。
        assertTrue(search("量子纠缠香蕉花园", null).isEmpty());
    }
}
