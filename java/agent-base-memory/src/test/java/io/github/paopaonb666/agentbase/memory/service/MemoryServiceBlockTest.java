package io.github.paopaonb666.agentbase.memory.service;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.memory.embeddings.EmbeddingClient;
import io.github.paopaonb666.agentbase.memory.store.InMemoryMemoryStore;
import io.github.paopaonb666.agentbase.memory.store.MemoryStore;

/**
 * 记忆块 API 契约（对应 Python api.py put_memory_block）。
 *
 * <p>回归：{@code putBlock} 曾在写入前把 content 截断到 char_limit——
 * Python 原样入库（char_limit 只是声明容量，注入组装时才按预算裁剪），
 * 截断会导致管理端写入的数据永久丢字、版本号照常递增。
 */
class MemoryServiceBlockTest {

    @Test
    void putBlockStoresContentVerbatim() {
        try (MemoryStore store = new InMemoryMemoryStore()) {
            EmbeddingClient never = new EmbeddingClient() {
                @Override
                public Integer dims() {
                    return null;
                }

                @Override
                public List<List<Float>> embed(List<String> texts) {
                    return null;
                }

                @Override
                public boolean probe() {
                    return false;
                }
            };
            MemoryService service = new MemoryService(store, never, Settings.load(Map.of(), null),
                    null, null);
            String longContent = "字".repeat(5000);
            var block = service.putBlock("u1", "chat", "persona", longContent, 2000);
            assertEquals(5000, block.content().length(),
                    "putBlock 必须原样存储 content（截断只发生在注入组装）");
            assertEquals(5000, store.getBlock("u1", "chat", "persona").content().length());
        }
    }
}
