package io.github.paopaonb666.agentbase.memory.service;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.UserMessage;
import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.memory.embeddings.EmbeddingClient;
import io.github.paopaonb666.agentbase.memory.store.InMemoryMemoryStore;
import io.github.paopaonb666.agentbase.memory.store.Models;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * MemoryService 主管线编排测试（roadmap #300）：服务层职责——captureTurn
 * 的服务级门控（无管线/空转写本）、composeFrozen 的同线程逐字节缓存与
 * LRU 驱逐（T2.1 供应商前缀缓存地基）、recallContext 的降级召回、
 * purgeThread 的级联清理（快照/摘要/分块，跨会话记忆保留）、文档摄取的
 * 分块/幂等/撤销闭环。LLM 判分细节在 MemoryPipelineTest（#296），不重复。
 */
class MemoryServicePipelineTest {

    private static EmbeddingClient neverEmbedder() {
        return new EmbeddingClient() {
            @Override public Integer dims() { return null; }
            @Override public List<List<Float>> embed(List<String> texts) { return null; }
            @Override public boolean probe() { return false; }
        };
    }

    private static MemoryService service(InMemoryMemoryStore store, Settings settings) {
        return new MemoryService(store, neverEmbedder(), settings, null, null);
    }

    private static Settings settings() {
        return Settings.load(Map.of(), null);
    }

    // ── captureTurn 服务级门控 ────────────────────────────────────────────

    @Test
    void captureTurnReturnsNullWithoutPipelineOrTranscript() {
        // llm=null → pipeline=null：直接返回 null（不抛）。
        InMemoryMemoryStore store = new InMemoryMemoryStore();
        MemoryService bare = service(store, settings());
        assertNull(bare.captureTurn("u", "chat", "chat:t1",
                List.of(UserMessage.from("hi"))));

        // 有管线（有 llm）但转写为空（无消息）→ null。
        MemoryService withPipeline = new MemoryService(store, neverEmbedder(), settings(),
                new MemoryServicePipelineTest.NeverMain(), null);
        assertNull(withPipeline.captureTurn("u", "chat", "chat:t1", List.of()));
        // 只有无文本的消息（工具消息）→ 转写空 → null。
        assertNull(withPipeline.captureTurn("u", "chat", "chat:t1",
                List.of(AiMessage.from(""))));
    }

    // ── composeFrozen：逐字节缓存 + LRU ───────────────────────────────────

    @Test
    void composeFrozenCachesPerThreadByteIdenticalUntilEvicted() {
        InMemoryMemoryStore store = new InMemoryMemoryStore();
        Settings s = settings();
        s.memory.contextSnapshotMax = 1; // 极小缓存：第二次 compose 即驱逐第一次
        MemoryService memory = service(store, s);
        store.upsertBlock(new Models.MemoryBlock("u", "*", "persona", "助手人设", 2000, 1, Models.now()));

        String first = memory.composeFrozen("u", "chat", "chat:t1");
        assertTrue(first != null && first.contains("[persona] 助手人设"), first);

        // 同线程缓存命中：改库不影响已缓存内容（逐字节相同 = 前缀缓存地基）。
        store.upsertBlock(new Models.MemoryBlock("u", "*", "persona", "改动人设", 2000, 2, Models.now()));
        assertEquals(first, memory.composeFrozen("u", "chat", "chat:t1"));

        // 换线程 → 缓存驱逐第一线程；再次 compose 反映库内新内容。
        String second = memory.composeFrozen("u", "chat", "chat:t2");
        assertTrue(second.contains("改动人设"), second);
        String recomposed = memory.composeFrozen("u", "chat", "chat:t1");
        assertTrue(recomposed.contains("改动人设"), "t1 缓存被 LRU 驱逐后重组");

        // 摘要进入冻结段。
        store.upsertSummary(new Models.SessionSummary("u", "chat:t3", "对话摘要文本"));
        assertTrue(memory.composeFrozen("u", "chat", "chat:t3").contains("对话摘要文本"));
    }

    @Test
    void composeFrozenReturnsNullWhenNothingToInject() {
        InMemoryMemoryStore store = new InMemoryMemoryStore();
        MemoryService memory = service(store, settings());
        // 无画像/块/摘要 → 无注入段。
        assertNull(memory.composeFrozen("u", "chat", "chat:t1"));
    }

    // ── recallContext ─────────────────────────────────────────────────────

    @Test
    void recallContextComposesDegradedRecall() {
        InMemoryMemoryStore store = new InMemoryMemoryStore();
        MemoryService memory = service(store, settings());
        memory.addMemory("u", "chat", "用户的部署环境是 Ubuntu 22.04", "semantic",
                List.of(), 0.5, "");

        String recall = memory.recallContext("u", "部署环境");
        assertTrue(recall != null && recall.contains("Ubuntu 22.04"), recall);
        // 无命中 → 无召回段。
        assertNull(memory.recallContext("u", "完全无关的查询词"));
    }

    // ── purgeThread：级联清理 ─────────────────────────────────────────────

    @Test
    void purgeThreadClearsSummaryChunksAndSnapshotsKeepsMemories() {
        InMemoryMemoryStore store = new InMemoryMemoryStore();
        MemoryService memory = service(store, settings());
        store.upsertSummary(new Models.SessionSummary("u", "chat:t1", "摘要"));
        memory.ingestDocument("f1", "u", "chat", "线程绑定的文档内容", "chat:t1");
        memory.addMemory("u", "chat", "跨会话保留的记忆", "semantic", List.of(), 0.5, "chat:t1");

        // 冻结段先进入缓存（含摘要），再被 purge 清出。
        String before = memory.composeFrozen("u", "chat", "chat:t1");
        assertTrue(before != null && before.contains("摘要"), before);
        memory.purgeThread("chat:t1");

        assertNull(store.getSummary("u", "chat:t1"), "滚动摘要随线程消亡");
        assertEquals(0, store.chunkCountsByFile("u").getOrDefault("f1", 0), "知识分块随线程消亡");
        // 缓存行已被移除：重组不再包含已删摘要（返回 null——什么都不剩）。
        assertNull(memory.composeFrozen("u", "chat", "chat:t1"));
        // 跨会话记忆保留（M6 语义）。
        assertEquals(1, memory.store()
                .listMemories("u", "chat", false, null, null, 10, false).size());
    }

    // ── 文档摄取/检索/撤销闭环 ────────────────────────────────────────────

    @Test
    void ingestDocumentChunksIdempotentlyAndRevokes() {
        InMemoryMemoryStore store = new InMemoryMemoryStore();
        Settings s = settings();
        s.memory.docChunkChars = 20;
        s.memory.docChunkOverlap = 0;
        MemoryService memory = service(store, s);

        String text = "字".repeat(50);
        int chunks = memory.ingestDocument("f1", "u", "chat", text, "chat:t1");
        assertTrue(chunks >= 2, "50 字按 20 字分块至少 2 块，实际 " + chunks);
        assertEquals(chunks, store.chunkCountsByFile("u").getOrDefault("f1", 0));
        assertEquals(chunks, memory.listFileChunks("u", "f1", 100).size());

        // 幂等：同 file_id 重摄取先清旧分块（数量不翻倍）。
        int again = memory.ingestDocument("f1", "u", "chat", text, "chat:t1");
        assertEquals(chunks, again);
        assertEquals(chunks, store.chunkCountsByFile("u").getOrDefault("f1", 0));

        // 空文本 → 0（不写不炸）。
        assertEquals(0, memory.ingestDocument("f2", "u", "chat", "   ", "chat:t1"));

        // 撤销：分块清零。
        int revoked = memory.revokeDocument("f1", "u");
        assertEquals(chunks, revoked);
        assertEquals(0, store.chunkCountsByFile("u").getOrDefault("f1", 0));
    }

    @Test
    void searchKnowledgeHonorsFileIdPushdown() {
        InMemoryMemoryStore store = new InMemoryMemoryStore();
        Settings s = settings();
        s.memory.docChunkChars = 100;
        s.memory.docChunkOverlap = 0;
        MemoryService memory = service(store, s);
        memory.ingestDocument("f1", "u", "chat", "部署环境的说明文档内容", "chat:t1");
        memory.ingestDocument("f2", "u", "chat", "部署环境的另一份文档", "chat:t1");

        // file_id 下推（#184）：候选集收窄到单文件。
        List<Map<String, Object>> scoped = memory.searchKnowledge("u", null, "部署环境", 5, "f2");
        assertTrue(!scoped.isEmpty());
        for (Map<String, Object> item : scoped) {
            assertEquals("f2", String.valueOf(item.get("file_id")));
        }
        // 全库检索两个文件都可见。
        List<Map<String, Object>> all = memory.searchKnowledge("u", null, "部署环境", 5);
        assertEquals(2, all.stream().map(i -> String.valueOf(i.get("file_id"))).distinct().count());
    }

    /** 恒抛模型桩：只为让管线存在（服务级门控——异常被服务兜底吞掉）。 */
    private static final class NeverMain implements io.github.paopaonb666.agentbase.core.llm.ModelRuntime {
        @Override public dev.langchain4j.model.chat.ChatModel chat() { throw new UnsupportedOperationException(); }
        @Override public dev.langchain4j.model.chat.StreamingChatModel streaming() { return null; }
        @Override public dev.langchain4j.data.message.AiMessage stream(
                List<dev.langchain4j.data.message.ChatMessage> messages,
                List<dev.langchain4j.agent.tool.ToolSpecification> tools,
                java.util.function.Consumer<String> onDelta) {
            throw new UnsupportedOperationException();
        }
        @Override public dev.langchain4j.data.message.AiMessage invoke(
                List<dev.langchain4j.data.message.ChatMessage> messages) {
            throw new UnsupportedOperationException();
        }
    }
}
