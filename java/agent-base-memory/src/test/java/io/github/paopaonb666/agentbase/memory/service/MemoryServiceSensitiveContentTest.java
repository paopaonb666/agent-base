package io.github.paopaonb666.agentbase.memory.service;

import io.github.paopaonb666.agentbase.core.MemoryScope;
import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.memory.store.InMemoryMemoryStore;
import io.github.paopaonb666.agentbase.memory.store.Models.MemoryStoreError;
import io.github.paopaonb666.agentbase.memory.store.MemoryStore;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * review R3-P1（#48 兜底）：service 层写路径统一红线——REST 直写与
 * memory_update_block 不再是旁路（此前只有抽取路径与 memory_save 有检查）。
 */
class MemoryServiceSensitiveContentTest {

    @AfterEach
    void clearScope() {
        MemoryScope.clear();
    }

    private static final String SECRET = "密钥 sk-abcdefghijklmnop123 在这里";

    private static MemoryService service(boolean filterOn) {
        Settings s = Settings.load(Map.of(
                "ENV", "development",
                "MEMORY_SENSITIVE_CONTENT_FILTER", filterOn ? "true" : "false"), null);
        return new MemoryService(new InMemoryMemoryStore(),
                io.github.paopaonb666.agentbase.memory.service.MemoryServiceBlockTest
                        .neverEmbedder(),
                s, null, null);
    }

    @Test
    void allServiceWritePathsRejectSensitiveContent() {
        MemoryService service = service(true);
        // addMemory / updateMemory / putBlock 三条写路径全部拦截。
        MemoryStoreError add = assertThrows(MemoryStoreError.class,
                () -> service.addMemory("u1", "chat", SECRET, "semantic", List.of(), 0.5, ""));
        assertTrue(add.getMessage().contains("敏感信息规则"), add.getMessage());
        MemoryStoreError put = assertThrows(MemoryStoreError.class,
                () -> service.putBlock("u1", "chat", "notes", SECRET, 2000));
        assertTrue(put.getMessage().contains("敏感信息规则"), put.getMessage());
        // update：先写干净内容，再更新成敏感内容时拦截。
        var record = service.addMemory("u1", "chat", "干净内容", "semantic", List.of(), 0.5, "");
        MemoryStoreError update = assertThrows(MemoryStoreError.class,
                () -> service.updateMemory(record.memoryId(), SECRET, null, null, null, "update"));
        assertTrue(update.getMessage().contains("敏感信息规则"), update.getMessage());
    }

    @Test
    void filterOffKeepsLegacyBehavior() {
        MemoryService service = service(false);
        assertEquals("semantic", service.addMemory("u1", "chat", SECRET, "semantic",
                List.of(), 0.5, "").kind());
        service.putBlock("u1", "chat", "notes", SECRET, 2000);
    }
}
