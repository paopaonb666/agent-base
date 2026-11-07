package io.github.paopaonb666.agentbase.memory.service;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.memory.embeddings.EmbeddingClient;
import io.github.paopaonb666.agentbase.memory.store.InMemoryMemoryStore;
import io.github.paopaonb666.agentbase.memory.store.Models;

/**
 * #177 画像损坏可诊断性：{@code getProfile} 对畸形内容（解析失败 / 顶层
 * 非 JSON object）返回 null 的行为不变（composeFrozen 与
 * GET /v1/memory/profile 的契约），但必须写一条 {@code profile_corrupt}
 * status=error 的审计——「失忆」不再是静默事件。正常画像与不存在的画像
 * 不产生任何损坏审计。
 */
class MemoryServiceProfileTest {

    /** 恒不向量化：MemoryService 各路径对 null 嵌入的降级分支。 */
    private static EmbeddingClient neverEmbedder() {
        return new EmbeddingClient() {
            @Override public Integer dims() { return null; }
            @Override public List<List<Float>> embed(List<String> texts) { return null; }
            @Override public boolean probe() { return false; }
        };
    }

    /** 绕过 saveProfile 的序列化直写一条画像记录（制造损坏存储的入口）。 */
    private static Models.MemoryRecord rawProfile(String userId, String content) {
        return new Models.MemoryRecord(Models.profileMemoryId(userId), userId, "*", "semantic",
                content, List.of("profile"), null, null, 1.0, "active", "", List.of("profile"),
                Models.now(), Models.now(), null, 0);
    }

    private static List<Models.MemoryOp> corruptOps(InMemoryMemoryStore store, String userId) {
        return store.listOps(userId, 50).stream()
                .filter(op -> "profile_corrupt".equals(op.op()))
                .toList();
    }

    @Test
    void corruptProfileReturnsNullAndAudits() {
        try (InMemoryMemoryStore store = new InMemoryMemoryStore()) {
            MemoryService service = new MemoryService(store, neverEmbedder(),
                    Settings.load(Map.of(), null), null, null);
            store.upsertMemory(rawProfile("u1", "not json"));

            // 返回形态不变：畸形画像仍 null（GET /v1/memory/profile、composeFrozen 契约）。
            assertNull(service.getProfile("u1"));

            // 可诊断：一条 profile_corrupt / status=error 审计，detail 带画像 id 与异常摘要。
            List<Models.MemoryOp> ops = corruptOps(store, "u1");
            assertEquals(1, ops.size(), "损坏画像恰好一条审计");
            Models.MemoryOp op = ops.get(0);
            assertEquals("error", op.status());
            assertFalse(op.errorText().isEmpty(), "error_text 带异常摘要");
            assertEquals(Models.profileMemoryId("u1"), op.detail().get("memory_id"));
            assertFalse(String.valueOf(op.detail().get("reason")).isEmpty());
        }
    }

    @Test
    void nonDictProfileAuditsToo() {
        try (InMemoryMemoryStore store = new InMemoryMemoryStore()) {
            MemoryService service = new MemoryService(store, neverEmbedder(),
                    Settings.load(Map.of(), null), null, null);
            // 解析成功但顶层是 JSON 数组（非 dict）：同样按损坏处理。
            store.upsertMemory(rawProfile("u2", "[1, 2, 3]"));
            assertNull(service.getProfile("u2"));
            assertEquals(1, corruptOps(store, "u2").size());
            assertEquals("画像内容不是 JSON object",
                    corruptOps(store, "u2").get(0).detail().get("reason"));
        }
    }

    @Test
    void healthyOrMissingProfileWritesNoCorruptionAudit() {
        try (InMemoryMemoryStore store = new InMemoryMemoryStore()) {
            MemoryService service = new MemoryService(store, neverEmbedder(),
                    Settings.load(Map.of(), null), null, null);

            // 不存在的画像：null 且无审计（未损坏，只是还没有）。
            assertNull(service.getProfile("u3"));
            assertEquals(0, corruptOps(store, "u3").size());

            // 正常画像：内容原样返回，无 profile_corrupt 审计。
            service.saveProfile("u3", Map.of("name", "张三", "偏好", "简洁回复"));
            Map<String, Object> profile = service.getProfile("u3");
            assertEquals("张三", profile.get("name"));
            assertEquals(0, corruptOps(store, "u3").size(), "正常画像不产生损坏审计");
            assertTrue(store.listOps("u3", 50).isEmpty(), "健康路径完全不写审计");
        }
    }
}
