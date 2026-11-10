package io.github.paopaonb666.agentbase.memory.service;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.UserMessage;
import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.memory.embeddings.EmbeddingClient;
import io.github.paopaonb666.agentbase.memory.pipeline.MemoryPipeline;
import io.github.paopaonb666.agentbase.memory.retrieval.Retrieval;
import io.github.paopaonb666.agentbase.memory.store.InMemoryMemoryStore;
import io.github.paopaonb666.agentbase.memory.store.Models;
import org.junit.jupiter.api.Test;

import java.util.ArrayDeque;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Supplier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

/**
 * MemoryService 主管线编排测试（roadmap #300）：服务层职责——captureTurn
 * 的服务级门控（无管线/空转写本）、composeFrozen 的同线程逐字节缓存与
 * LRU 驱逐（T2.1 供应商前缀缓存地基）、recallContext 的降级召回、
 * purgeThread 的级联清理（快照/摘要/分块，跨会话记忆保留）、文档摄取的
 * 分块/幂等/撤销闭环、capture 失败重试队列（#182：入队/重放/上限丢弃/
 * 让路回队/关闭排空）。LLM 判分细节在 MemoryPipelineTest（#296），不重复。
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

    // ── capture 失败重试队列（#182） ──────────────────────────────────────

    /**
     * 实时 capture 失败 → 转写快照入有界重试队列；管线恢复后由重试线程
     * 异步重放恰好一次 → 候选落库 + capture_retry ok 审计。
     */
    @Test
    void captureFailureEnqueuesRetryAndReplaysOncePipelineRecovers() {
        InMemoryMemoryStore store = new InMemoryMemoryStore();
        // 脚本：第 1 次 invoke（实时轮抽取）失败，其后依次为抽取/整合/画像
        // 的成功响应（重放走完整管线）。
        ScriptedMain llm = new ScriptedMain(
                new RuntimeException("llm 网络故障（脚本）"),
                "[{\"content\":\"重试落库的记忆\",\"kind\":\"semantic\",\"salience\":0.5,\"tags\":[]}]",
                "{\"op\":\"ADD\",\"content\":\"重试落库的记忆\"}",
                "{\"背景\":\"脚本画像\"}");
        MemoryService memory = scriptedService(store, llm, true);

        assertNull(memory.captureTurn("u", "chat", "chat:t1", List.of(UserMessage.from("你好"))),
                "实时失败仍返回 null（绝不抛异常）");

        // 重试线程异步重放：管线恢复 → addMemory 落库 + capture_retry ok 审计。
        awaitTrue(() -> store.listOps(null, 50).stream().anyMatch(op ->
                "capture_retry".equals(op.op()) && "ok".equals(op.status())), "重放成功审计");
        List<Models.MemoryRecord> rows = memory.store()
                .listMemories("u", "chat", false, null, null, 10, false);
        assertEquals(1, rows.size(), "重放后候选落库");
        assertEquals("重试落库的记忆", rows.get(0).content());
        assertEquals(1, store.listOps(null, 50).stream()
                .filter(op -> "capture_retry".equals(op.op())).count(), "恰好重放一次");
        memory.close();
    }

    /** 实时 + 两次重放全部失败 → attempts=3 达上限丢弃 + error 审计，不再死循环。 */
    @Test
    void captureRetryDropsAfterMaxAttemptsWithAudit() throws Exception {
        InMemoryMemoryStore store = new InMemoryMemoryStore();
        ScriptedMain llm = new ScriptedMain(new RuntimeException("llm 持续故障（脚本）"));
        MemoryService memory = scriptedService(store, llm, true);

        assertNull(memory.captureTurn("u", "chat", "chat:t1", List.of(UserMessage.from("你好"))));

        // 1 次实时 + 2 次重放全部失败 → 达上限丢弃 + capture_retry error 审计。
        awaitTrue(() -> store.listOps(null, 50).stream().anyMatch(op ->
                "capture_retry".equals(op.op()) && "error".equals(op.status())), "达上限丢弃审计");
        Models.MemoryOp drop = store.listOps(null, 50).stream()
                .filter(op -> "capture_retry".equals(op.op())).findFirst().orElseThrow();
        assertEquals(3, drop.detail().get("attempts"), "丢弃时累计失败次数");
        assertEquals(3, llm.invocations.get(), "1 次实时 + 2 次重放");
        assertEquals(0, memory.pendingCaptureRetries(), "丢弃后队列清空");
        Thread.sleep(300);
        assertEquals(3, llm.invocations.get(), "丢弃后不再重试（无死循环）");
        memory.close();
    }

    /**
     * 重试走同一 per-user 锁：实时轮持锁期间 tryLock 失败 → 任务回队尾
     * （不执行也不丢弃）；锁释放后从队尾被重放恰好一次。
     */
    @Test
    void captureRetryYieldsLockAndRequeues() throws Exception {
        InMemoryMemoryStore store = new InMemoryMemoryStore();
        ScriptedMain llm = new ScriptedMain("[]");
        CountDownLatch liveStarted = new CountDownLatch(1);
        CountDownLatch releaseLive = new CountDownLatch(1);
        llm.onFirstInvoke = () -> {
            liveStarted.countDown();
            awaitUnchecked(releaseLive);
        };
        MemoryService memory = scriptedService(store, llm, false);

        Thread live = new Thread(() -> memory.captureTurn("u", "chat", "chat:t1",
                List.of(UserMessage.from("你好"))));
        live.start();
        assertTrue(liveStarted.await(5, TimeUnit.SECONDS), "实时轮已持锁进入管线");

        // 锁被实时轮持有时播种重试任务（同 catch 分支入口）。
        memory.enqueueCaptureRetry("u", "chat", "chat:t1", "用户：你好", 1, false, 1);
        awaitTrue(() -> memory.pendingCaptureRetries() == 1, "任务在队列中（tryLock 失败回队尾）");
        Thread.sleep(300); // > 重试退避（100ms）：让重试线程让路数轮
        assertEquals(1, llm.invocations.get(), "锁被实时轮持有时重试不执行");
        assertEquals(1, memory.pendingCaptureRetries(), "回队尾而非丢弃");

        // 释放实时轮 → 任务被重放恰好一次。
        releaseLive.countDown();
        live.join(5000);
        awaitTrue(() -> store.listOps(null, 50).stream().anyMatch(op ->
                "capture_retry".equals(op.op()) && "ok".equals(op.status())), "解锁后重放成功");
        assertEquals(2, llm.invocations.get(), "解锁后恰好重放一次");
        assertEquals(0, memory.pendingCaptureRetries());
        memory.close();
    }

    /**
     * close：置 stopped + 中断重试线程，剩余任务只计数 WARN 不执行
     * （关闭期不再打 LLM），不产生 capture_retry 审计。
     */
    @Test
    void closeDrainsRetryQueueWithoutExecuting() throws Exception {
        InMemoryMemoryStore store = new InMemoryMemoryStore();
        ScriptedMain llm = new ScriptedMain("[]");
        CountDownLatch liveStarted = new CountDownLatch(1);
        CountDownLatch releaseLive = new CountDownLatch(1);
        llm.onFirstInvoke = () -> {
            liveStarted.countDown();
            awaitUnchecked(releaseLive);
        };
        MemoryService memory = scriptedService(store, llm, false);

        // 实时轮持锁（重试线程只能让路回队）→ 播种任务 → 关闭排空。
        Thread live = new Thread(() -> memory.captureTurn("u", "chat", "chat:t1",
                List.of(UserMessage.from("你好"))));
        live.start();
        assertTrue(liveStarted.await(5, TimeUnit.SECONDS), "实时轮已持锁进入管线");
        memory.enqueueCaptureRetry("u", "chat", "chat:t1", "用户：你好", 1, false, 1);
        awaitTrue(() -> memory.pendingCaptureRetries() == 1, "任务已入队并被重试线程回队");

        memory.close();
        assertEquals(0, memory.pendingCaptureRetries(), "关闭时排空重试队列");

        releaseLive.countDown();
        live.join(5000);
        assertEquals(1, llm.invocations.get(), "重试从未执行（关闭期不再打 LLM）");
        assertTrue(store.listOps(null, 50).stream()
                .noneMatch(op -> "capture_retry".equals(op.op())), "关闭排空不产生重试审计");
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

    // ── #182 重试队列测试的手写桩 ─────────────────────────────────────────

    /**
     * 组装带脚本化管线的服务（MemoryPipeline 为 final 且由服务内部构造，
     * 经包私有注入点传入）。{@code failOnErrorAudit} 见 {@link ScriptedOps}。
     */
    private static MemoryService scriptedService(InMemoryMemoryStore store, ScriptedMain llm,
                                                 boolean failOnErrorAudit) {
        Settings s = settings();
        ScriptedOps ops = new ScriptedOps(failOnErrorAudit);
        MemoryService memory = MemoryService.withPipeline(store, neverEmbedder(), s,
                new MemoryPipeline(llm, null, s, ops));
        ops.bind(memory, store);
        return memory;
    }

    /** 轮询等待异步重试线程的可见效果（M 包无 awaitility，手写小轮询）。 */
    private static void awaitTrue(Supplier<Boolean> condition, String what) {
        long deadline = System.currentTimeMillis() + 5000;
        while (!Boolean.TRUE.equals(condition.get())) {
            if (System.currentTimeMillis() > deadline) {
                fail("等待超时：" + what);
            }
            try {
                Thread.sleep(20);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                fail("等待被中断：" + what);
            }
        }
    }

    private static void awaitUnchecked(CountDownLatch latch) {
        try {
            assertTrue(latch.await(5, TimeUnit.SECONDS), "latch 等待超时");
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            fail("latch 等待被中断");
        }
    }

    /**
     * 脚本化主力模型桩：invoke 依序消费脚本项（Throwable = 抛、String =
     * 响应文本）。响应耗尽后重复最后一个响应；Throwable 仅在它是最后
     * 一项时重复抛（持续失败形态），否则消费一次（首败后恢复形态）。
     */
    private static final class ScriptedMain implements io.github.paopaonb666.agentbase.core.llm.ModelRuntime {
        final AtomicInteger invocations = new AtomicInteger();
        /** 第 1 次 invoke 前的钩子（实时轮持锁栅栏用）。 */
        Runnable onFirstInvoke = () -> { };
        private final ArrayDeque<Object> script = new ArrayDeque<>();
        private String lastResponse = "[]";

        ScriptedMain(Object... steps) {
            Collections.addAll(script, steps);
        }

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
            if (invocations.incrementAndGet() == 1) {
                onFirstInvoke.run();
            }
            Object step = script.poll();
            if (step instanceof Throwable failure) {
                if (script.isEmpty()) {
                    // 单条脚本 = 持续失败形态：同一条重复抛，脚本永不耗尽。
                    script.addFirst(failure);
                }
                if (failure instanceof RuntimeException runtime) {
                    throw runtime;
                }
                throw new RuntimeException(failure);
            }
            if (step instanceof String response) {
                lastResponse = response;
            } else {
                step = lastResponse; // 脚本耗尽：重复最后一个响应
            }
            return dev.langchain4j.data.message.AiMessage.from((String) step);
        }
    }

    /**
     * 脚本化 MemoryOps：全部委托真实服务（构造后 bind）。管线 Step.run 的
     * catch 会吞掉每步业务异常，服务级 catch 兜底（#182 重试入队口）实际
     * 只拦「漏网」异常——测试里让失败步（status=error）的审计写入本身抛，
     * 异常才能穿出 captureTurn 驱动该兜底；成功步照常委托落审计。
     */
    private static final class ScriptedOps implements MemoryPipeline.MemoryOps {
        private final boolean failOnErrorAudit;
        private MemoryService service;
        private InMemoryMemoryStore store;

        ScriptedOps(boolean failOnErrorAudit) {
            this.failOnErrorAudit = failOnErrorAudit;
        }

        void bind(MemoryService service, InMemoryMemoryStore store) {
            this.service = service;
            this.store = store;
        }

        @Override public List<Retrieval.ScoredMemory> search(String userId, String agentId,
                                                             String query, Integer topK) {
            return service.searchScored(userId, agentId, query, topK);
        }

        @Override public void addMemory(String userId, String agentId, String content, String kind,
                                        List<String> tags, double salience, String sourceThreadId) {
            service.addMemory(userId, agentId, content, kind, tags, salience, sourceThreadId);
        }

        @Override public boolean updateMemory(String memoryId, String content) {
            return service.updateMemory(memoryId, content, null, null, null, "update") != null;
        }

        @Override public boolean deleteMemory(String memoryId) {
            return service.deleteMemory(memoryId);
        }

        @Override public Map<String, Object> getProfile(String userId) {
            return service.getProfile(userId);
        }

        @Override public void saveProfile(String userId, Map<String, Object> profile) {
            service.saveProfile(userId, profile);
        }

        @Override public Models.SessionSummary getSummary(String userId, String threadId) {
            return store.getSummary(userId, threadId);
        }

        @Override public void upsertSummary(Models.SessionSummary summary) {
            store.upsertSummary(summary);
        }

        @Override public void recordOp(String op, String userId, String agentId, String threadId,
                                       Map<String, Object> detail, String status, String errorText,
                                       long durationMs) {
            if (failOnErrorAudit && "error".equals(status)) {
                throw new IllegalStateException("脚本：失败步审计通道故障（" + op + "）");
            }
            service.recordOp(op, userId, agentId, threadId, detail, status, errorText, durationMs);
        }
    }
}
