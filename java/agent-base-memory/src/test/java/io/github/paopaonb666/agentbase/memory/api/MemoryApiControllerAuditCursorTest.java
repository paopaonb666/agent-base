package io.github.paopaonb666.agentbase.memory.api;

import java.lang.reflect.Constructor;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import io.github.paopaonb666.agentbase.core.AgentModule;
import io.github.paopaonb666.agentbase.core.AgentRuntime;
import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.entrypoints.server.AuthBackends;
import io.github.paopaonb666.agentbase.entrypoints.server.HttpError;
import io.github.paopaonb666.agentbase.memory.service.MemoryService;
import io.github.paopaonb666.agentbase.memory.store.InMemoryMemoryStore;
import io.github.paopaonb666.agentbase.memory.store.Models;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * /v1/memory/audit 游标复合化（code review R3 收口）：MySQL created_at 是
 * 秒精度，同秒并列组的页界即 createdAt 相同——next_before 只带秒值时下一页
 * 「严格早于该时刻」会把同秒余量整组跳掉。锁定三层：
 * <ul>
 *   <li>next_before 形态「秒:op_id」（秒值 toPlainString，不出科学计数法），
 *       以它翻页全量覆盖、无重复无遗漏；</li>
 *   <li>旧客户端的纯秒游标仍接受（兼容路径），同秒余量按既有文档语义
 *       跳过——不抛错且单调前进；</li>
 *   <li>非法游标 422（与该控制器其他参数校验同形态）。</li>
 * </ul>
 *
 * <p>装配：控制器直调（{@code AgentRuntime} 为包私有构造器，经反射仅注入
 * 设置与真实记忆端口，基座侧 ApiEndpointMatrixTest 同款手法——本模块无
 * Mockito，全部手写桩）+ 真实 {@link MemoryService} + {@link InMemoryMemoryStore}
 * + never-embedder（{@link MemoryToolsTest} 同款：dims null / embed null /
 * probe false）。造数走 {@code store.recordOp}（store 层接口接受显式
 * createdAt——{@code MemoryService.recordOp} 固定取 now() 不可注入），同秒
 * 并列由固定 epoch 精确构造。鉴权用直通桩（headers 透传 user，不做 HMAC）。
 */
class MemoryApiControllerAuditCursorTest {

    /** 主秒组：所有并列行共享同一 epoch 秒（1.8E9 量级，贴近真实 now）。 */
    private static final double T = 1800000000.0;

    private final InMemoryMemoryStore store = new InMemoryMemoryStore();
    private final MemoryService memory =
            new MemoryService(store, neverEmbedder(), Settings.load(Map.of(), null), null, null);
    private final MemoryApiController controller =
            new MemoryApiController(runtimeWith(memory), PASS_THROUGH_AUTH);

    private static final AuthBackends.AuthBackend PASS_THROUGH_AUTH =
            (userId, sig, ts, nonce, settings) ->
                    userId == null || userId.isEmpty() ? "default" : userId;

    /** 恒不向量化（MemoryService 各路径对 null 嵌入的降级分支）。 */
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

    /** 审计造数：store 层写入口（createdAt 显式，可精确构造同秒并列）。 */
    private static void seedOp(InMemoryMemoryStore store, String opId, String threadId,
                               double createdAt) {
        store.recordOp(new Models.MemoryOp(opId, "manual", "alice", "chat", threadId,
                new LinkedHashMap<>(), "ok", "", 0, createdAt));
    }

    @SuppressWarnings("unchecked")
    private static List<String> opIds(Map<String, Object> page) {
        List<Map<String, Object>> ops = (List<Map<String, Object>>) page.get("ops");
        List<String> ids = new ArrayList<>();
        for (Map<String, Object> op : ops) {
            ids.add((String) op.get("op_id"));
        }
        return ids;
    }

    // ── 复合游标：同秒并列翻页无重无漏 ─────────────────────────────────

    @Test
    void compositeCursorPaginatesSameSecondGroupWithoutLossOrDuplication() {
        // 同秒并列 5 条（插入序 o1..o5，列表序最新在前）+ 早一秒 2 条
        // （a1..a2）：limit=3 必然把同秒组从中间切开。
        for (int i = 1; i <= 5; i++) {
            seedOp(store, "o" + i, "chat:t1", T);
        }
        seedOp(store, "a1", "chat:t1", T - 1);
        seedOp(store, "a2", "chat:t1", T - 1);

        // 逐页翻页：第一页取满 → next_before 含 ':' 与 op_id → 以它续页，
        // 直到 next_before 缺席（页行数 < limit，自然终止）。
        List<String> seen = new ArrayList<>();
        String before = null;
        int guard = 0;
        while (guard++ < 10) {
            Map<String, Object> page = controller.audit(3, null, null, before,
                    "alice", null, null, null);
            seen.addAll(opIds(page));
            before = (String) page.get("next_before");
            if (before == null) {
                break;
            }
            assertTrue(before.contains(":"), before);
        }
        // 全量且无重复无遗漏（列表序 = created_at 降序 + 同秒插入序降序）。
        assertEquals(List.of("o5", "o4", "o3", "o2", "o1", "a2", "a1"), seen);
        assertEquals(7, new HashSet<>(seen).size());
        assertNull(before, "末页（1 行 < limit）不得再带 next_before");
    }

    // ── next_before 形态：秒值 toPlainString + op_id ───────────────────

    @Test
    void nextBeforeIsPlainDecimalSecondsWithColonAndOpId() {
        for (int i = 1; i <= 3; i++) {
            seedOp(store, "o" + i, "chat:t1", T);
        }
        Map<String, Object> page = controller.audit(2, null, null, null,
                "alice", null, null, null);
        // 1.8E9 量级 Double.toString 会出科学计数法——编码必须恒为十进制
        // 展开（整数秒经 toPlainString 无 ".0" 尾巴），且携带末行 op_id
        // （uuid hex，不含冒号，切分无歧义）。
        assertEquals("1800000000:o2", page.get("next_before"));

        // 小数秒：同样十进制展开、round-trip 精确（翻页测试已验证回传）。
        seedOp(store, "f1", "chat:t1", T + 0.5);
        seedOp(store, "f2", "chat:t1", T + 0.5);
        Map<String, Object> fractional = controller.audit(2, null, null, null,
                "alice", null, null, null);
        // 页序 [f2, f1]（同秒插入序降序），末行是 f1。
        assertEquals("1800000000.5:f1", fractional.get("next_before"));
    }

    // ── 旧客户端兼容：纯秒游标仍可用 ───────────────────────────────────

    @Test
    void legacySecondsOnlyCursorStillWorksWithDocumentedSameSecondTruncation() {
        for (int i = 1; i <= 5; i++) {
            seedOp(store, "o" + i, "chat:t1", T);
        }
        seedOp(store, "a1", "chat:t1", T - 1);
        seedOp(store, "a2", "chat:t1", T - 1);

        // 纯秒旧形态：严格早于该时刻——早一秒的两行仍在，同秒余量按既有
        // 文档语义允许跳过；不抛错且单调前进（返回行严格早于游标秒）。
        Map<String, Object> page = controller.audit(50, null, null,
                Double.toString(T), "alice", null, null, null);
        assertEquals(List.of("a2", "a1"), opIds(page));
        for (Map<String, Object> op : (List<Map<String, Object>>) page.get("ops")) {
            assertTrue(((Double) op.get("created_at")) < T);
        }
        assertFalse(page.containsKey("next_before"));

        // 游标推进到最早一秒：空页且无 next_before（翻页自然终止）。
        Map<String, Object> tail = controller.audit(50, null, null,
                Double.toString(T - 1), "alice", null, null, null);
        assertTrue(opIds(tail).isEmpty());
        assertFalse(tail.containsKey("next_before"));
    }

    // ── 非法游标：422（与控制器其他参数校验同形态） ────────────────────

    @Test
    void malformedBeforeCursorRejectedWith422() {
        seedOp(store, "o1", "chat:t1", T);
        HttpError seconds = assertThrows(HttpError.class, () ->
                controller.audit(50, null, null, "not-a-number", "alice", null, null, null));
        assertEquals(422, seconds.status());
        HttpError composite = assertThrows(HttpError.class, () ->
                controller.audit(50, null, null, "abc:op1", "alice", null, null, null));
        assertEquals(422, composite.status());
    }
}
