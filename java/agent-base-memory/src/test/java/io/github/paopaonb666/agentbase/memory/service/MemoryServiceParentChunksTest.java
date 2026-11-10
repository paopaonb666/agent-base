package io.github.paopaonb666.agentbase.memory.service;

import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.memory.embeddings.EmbeddingClient;
import io.github.paopaonb666.agentbase.memory.service.MemoryService;
import io.github.paopaonb666.agentbase.memory.store.InMemoryMemoryStore;
import io.github.paopaonb666.agentbase.memory.store.Models;
import io.github.paopaonb666.agentbase.memory.store.MemoryStore;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * searchKnowledge 父块推导聚合（#50 small-to-big）：同 file 内 ordinal
 * 连续的命中块合并为父区间（零 schema 变更，offsets 推导）；跨 file 不
 * 合并；旧数据 offsets 缺失 → parent_span=null 降级。只改返回组装不改
 * 打分——#44 golden 门（recall@5≥0.90/MRR≥0.75）随全量套件护航。
 */
class MemoryServiceParentChunksTest {

    private MemoryStore store;

    @AfterEach
    void tearDown() {
        if (store != null) {
            store.close();
        }
    }

    private static EmbeddingClient neverEmbedder() {
        return new EmbeddingClient() {
            @Override public Integer dims() { return null; }
            @Override public List<List<Float>> embed(List<String> texts) { return null; }
            @Override public boolean probe() { return false; }
        };
    }

    private MemoryService service() {
        store = new InMemoryMemoryStore();
        return new MemoryService(store, neverEmbedder(),
                Settings.load(Map.of("ENV", "development")), null, null);
    }

    private static Models.DocChunk chunk(String fileId, int ordinal, String text,
                                         List<long[]> offsets) {
        return new Models.DocChunk("c-" + fileId + "-" + ordinal, fileId, "chat:doc",
                "u1", "chat", ordinal, text, offsets, new byte[0], null,
                Models.now());
    }

    @Test
    void adjacentHitsMergeIntoParentSpanAndText() {
        MemoryService service = service();
        // 直插三个相邻分块（offsets 指向假设原文的段落区间）。
        store.putChunks(List.of(
                chunk("f1", 0, "第一段：部署在 Ubuntu",
                        List.of(new long[]{0, 12})),
                chunk("f1", 1, "第二段：部署的数据库 PostgreSQL",
                        List.of(new long[]{13, 40})),
                chunk("f1", 2, "第三段：部署的备份策略",
                        List.of(new long[]{41, 55}))));

        List<Map<String, Object>> hits = service.searchKnowledge("u1", "chat", "部署", 5);

        assertEquals(3, hits.size());
        // 相邻 ordinal 全命中 → 三块共享同一父区间 [0,50]。
        Map<String, Object> first = hits.get(0);
        assertEquals(List.of(0L, 55L), first.get("parent_span"));
        Object parentText = first.get("parent_text");
        assertTrue(parentText.toString().contains("第一段")
                && parentText.toString().contains("第二段")
                && parentText.toString().contains("第三段"), String.valueOf(parentText));
        // 段落拼接以 \n\n 连接（与 chunkText 的组段分隔一致）。
        assertEquals("第一段：部署在 Ubuntu\n\n第二段：部署的数据库 PostgreSQL\n\n第三段：部署的备份策略",
                parentText);
        // 三个条目的父块视图一致（同一父区间）。
        assertEquals(first.get("parent_text"), hits.get(1).get("parent_text"));
        assertEquals(first.get("parent_span"), hits.get(2).get("parent_span"));
        // 子块字段保持原样（wire 向后兼容加键，不动既有键）。
        assertEquals(1, ((Number) hits.get(1).get("ordinal")).intValue());
        assertEquals("c-f1-2", hits.get(2).get("chunk_id"));
    }

    @Test
    void hitsAcrossFilesNeverMerge() {
        MemoryService service = service();
        store.putChunks(List.of(
                chunk("f1", 0, "文件一的相关段落", List.of(new long[]{0, 8})),
                chunk("f2", 1, "文件二的相关段落", List.of(new long[]{0, 8}))));

        List<Map<String, Object>> hits = service.searchKnowledge("u1", "chat", "相关段落", 5);

        assertEquals(2, hits.size());
        // 跨 file 即便 ordinal 相邻也不合并——父块是文档内概念。
        assertEquals(List.of(0L, 8L), hits.get(0).get("parent_span"));
        assertEquals("文件一的相关段落", hits.get(0).get("parent_text"));
        assertEquals("文件二的相关段落", hits.get(1).get("parent_text"));
    }

    @Test
    void isolatedHitGetsItselfAsParent() {
        MemoryService service = service();
        store.putChunks(List.of(
                chunk("f1", 0, "第一段", List.of(new long[]{0, 3})),
                chunk("f1", 1, "第二段", List.of(new long[]{4, 7})),
                chunk("f1", 2, "第三段", List.of(new long[]{8, 11})),
                chunk("f1", 3, "第四段", List.of(new long[]{12, 15}))));

        // limit=1 只取排序首位；父块推导只看「命中集合」而非全文档——
        // 即使相邻块存在也不会外扩。
        List<Map<String, Object>> hits = service.searchKnowledge("u1", "chat", "第一段", 1);
        assertEquals(1, hits.size());
        Map<String, Object> hit = hits.get(0);
        // 独立命中块 parent=自身（不外扩到未命中的相邻块）。
        assertEquals(hit.get("text"), hit.get("parent_text"));
        assertEquals(List.of(0L, 3L), hit.get("parent_span"));
    }

    @Test
    void legacyChunksWithoutOffsetsDegradeToNullSpan() {
        MemoryService service = service();
        // 旧数据：offsets null（DocChunk 契约归一为空列表）→ span 不可推导。
        store.putChunks(List.of(
                chunk("f1", 0, "旧数据的相关段落甲", null),
                chunk("f1", 1, "旧数据的相关段落乙", null)));

        List<Map<String, Object>> hits = service.searchKnowledge("u1", "chat", "相关段落", 5);
        assertEquals(2, hits.size());
        assertNull(hits.get(0).get("parent_span"), "offsets 缺失 → span 不可推导");
        // 文本仍按相邻合并（拼接近似，可视化降级为逐块）。
        assertEquals("旧数据的相关段落甲\n\n旧数据的相关段落乙", hits.get(0).get("parent_text"));
        assertNull(hits.get(0).get("offsets"));
    }
}
