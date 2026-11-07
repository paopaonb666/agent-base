package io.github.paopaonb666.agentbase.entrypoints.server.routes;

import io.github.paopaonb666.agentbase.core.AgentModule;
import io.github.paopaonb666.agentbase.core.AgentRuntime;
import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.core.graph.GraphBuilder;
import io.github.paopaonb666.agentbase.core.graph.InMemoryCheckpointer;
import io.github.paopaonb666.agentbase.core.storage.ThreadIndexRecord;
import io.github.paopaonb666.agentbase.entrypoints.server.AuthBackends;
import io.github.paopaonb666.agentbase.entrypoints.server.HttpError;
import io.github.paopaonb666.agentbase.extensions.costmeter.CostGovernor;
import io.github.paopaonb666.agentbase.extensions.filestore.UploadedFileInfo;
import io.github.paopaonb666.agentbase.extensions.toollog.ToolCallRecord;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.mockito.Mockito;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.http.ResponseEntity;

import java.lang.reflect.Constructor;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * API 11 端点参数矩阵（roadmap #298）：覆盖既有测试组之外的端点参数面
 * ——线程列表（索引视图/降级）、历史序列化、计划状态（planStore 命中/
 * 缺席回退）、审计 limit 钳制、删除的 checkpointer 缺席分支、文件撤销/
 * 预览/切片/原始字节、知识检索与知识库文件列表。invoke 链 422 矩阵在
 * AgentsRoutesTest、上传 413/400 在 FilesRoutesTest，不在此重复。
 */
class ApiEndpointMatrixTest {

    private static final AuthBackends.AuthBackend PASS_THROUGH_AUTH =
            (userId, sig, ts, nonce, settings) ->
                    userId == null || userId.isEmpty() ? "default" : userId;

    private static ObjectProvider<CostGovernor> providerOf(CostGovernor governor) {
        return new ObjectProvider<>() {
            @Override public CostGovernor getObject() { return governor; }
            @Override public CostGovernor getObject(Object... args) { return governor; }
            @Override public CostGovernor getIfAvailable() { return governor; }
            @Override public CostGovernor getIfUnique() { return governor; }
        };
    }

    private static AgentModule graphStub(String name) {
        return new AgentModule() {
            @Override public String name() { return name; }
            @Override public String description() { return "stub module for api matrix"; }
            @Override public io.github.paopaonb666.agentbase.core.graph.CompiledGraph buildGraph(
                    AgentModule.ModuleContext ctx) {
                return GraphBuilder.singleNode(name, (state, nodeCtx) -> Map.of(), ctx.checkpointer());
            }
            @Override public List<io.github.paopaonb666.agentbase.core.tools.AgentTool> getTools() {
                return List.of();
            }
        };
    }

    /** 全组件 mock 的运行时（组件缺省 null，测试按需注入）。 */
    private static AgentRuntime runtime(Map<String, AgentModule> modules, Object checkpointer,
                                        Object threadIndex, Object fileStore, Object memory,
                                        Object recorder, Object planStore) throws Exception {
        Constructor<AgentRuntime> ctor = AgentRuntime.class.getDeclaredConstructor(
                Settings.class, io.github.paopaonb666.agentbase.core.llm.ModelRuntime.class,
                Map.class, io.github.paopaonb666.agentbase.core.tools.ToolPool.class,
                io.github.paopaonb666.agentbase.core.graph.Checkpointer.class,
                io.github.paopaonb666.agentbase.extensions.toollog.ToolCallRecorder.class,
                io.github.paopaonb666.agentbase.extensions.filestore.UploadedFileStore.class,
                AgentModule.MemoryServicePort.class,
                io.github.paopaonb666.agentbase.core.storage.CoreStore.class,
                AgentModule.PlanStorePort.class);
        ctor.setAccessible(true);
        return ctor.newInstance(Settings.load(Map.of("ENV", "development")), null, modules,
                null, checkpointer, recorder, fileStore, memory, threadIndex, planStore);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> mapOf(Map<String, Object> body) {
        return body;
    }

    private static int statusOf(java.util.concurrent.Callable<?> action) {
        try {
            action.call();
            return 200;
        } catch (HttpError e) {
            return e.status();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    // ══ AgentsRoutes：线程端点矩阵 ═══════════════════════════════════════

    @Test
    void listThreadsBuildsIndexViewsAndGuardsModule() throws Exception {
        var index = Mockito.mock(io.github.paopaonb666.agentbase.core.storage.CoreStore.class);
        when(index.listThreadIndexes(eq("alice"), eq("chat"), anyInt())).thenReturn(List.of(
                new ThreadIndexRecord("chat:t1", "alice", "chat", "标题", 100.0, 1234.5),
                new ThreadIndexRecord("legacy-raw", "alice", "", "", 0.0, 0.0)));
        AgentRuntime runtime = runtime(Map.of("chat", graphStub("chat")), null, index,
                null, null, null, null);
        AgentsRoutes routes = new AgentsRoutes(runtime, PASS_THROUGH_AUTH, providerOf(null));

        Map<String, Object> out = mapOf(routes.listThreads("chat", 50, 0, "alice", null, null, null));
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> threads = (List<Map<String, Object>>) out.get("threads");
        assertEquals(2, threads.size());
        // 命名空间行回短名；module/title 空串回退请求模块/整串 id。
        assertEquals("t1", threads.get(0).get("thread_id"));
        assertEquals("chat", threads.get(0).get("module"));
        assertEquals("标题", threads.get(0).get("title"));
        assertEquals(1234500L, threads.get(0).get("updated_at")); // epoch 秒 → 毫秒
        // 非命名空间形态（历史遗留行）整串回退。
        assertEquals("legacy-raw", threads.get(1).get("thread_id"));
        assertEquals("chat", threads.get(1).get("module"));
        assertEquals("legacy-raw", threads.get(1).get("title"));

        // 未知模块 404（鉴权先行，但模块守卫在鉴权之后）。
        assertEquals(404, statusOf(() -> routes.listThreads("nope", 50, 0, "alice", null, null, null)));
    }

    @Test
    void threadListPaginatesWithLimitOffsetOnBothPaths() throws Exception {
        // 索引路径：5 个线程（updated_at 降序 t5..t1），页大小 2。
        var index = Mockito.mock(io.github.paopaonb666.agentbase.core.storage.CoreStore.class);
        List<ThreadIndexRecord> seeded = new ArrayList<>();
        for (int i = 1; i <= 5; i++) {
            seeded.add(new ThreadIndexRecord("chat:t" + i, "alice", "chat", "标题" + i,
                    1000.0 + i, 2000.0 + i));
        }
        when(index.listThreadIndexes(eq("alice"), eq("chat"), anyInt()))
                .thenAnswer(inv -> {
                    // 后端契约：updated_at 降序后取前 fetch 条。
                    List<ThreadIndexRecord> sorted = new ArrayList<>(seeded);
                    sorted.sort(java.util.Comparator.comparingDouble(
                            ThreadIndexRecord::updatedAt).reversed());
                    int fetch = Math.min(sorted.size(), Math.max(1, (int) inv.getArgument(2)));
                    return new ArrayList<>(sorted.subList(0, fetch));
                });
        AgentRuntime runtime = runtime(Map.of("chat", graphStub("chat")), null, index,
                null, null, null, null);
        AgentsRoutes routes = new AgentsRoutes(runtime, PASS_THROUGH_AUTH, providerOf(null));

        // 第一页：2 条 + next_offset=2。
        Map<String, Object> page1 = mapOf(routes.listThreads("chat", 2, 0, "alice", null, null, null));
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> t1 = (List<Map<String, Object>>) page1.get("threads");
        assertEquals(List.of("t5", "t4"), t1.stream().map(m -> m.get("thread_id")).toList());
        assertEquals(2, page1.get("next_offset"));

        // 第二页。
        Map<String, Object> page2 = mapOf(routes.listThreads("chat", 2, 2, "alice", null, null, null));
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> t2 = (List<Map<String, Object>>) page2.get("threads");
        assertEquals(List.of("t3", "t2"), t2.stream().map(m -> m.get("thread_id")).toList());
        assertEquals(4, page2.get("next_offset"));

        // 末页：不满页 → 无 next_offset。
        Map<String, Object> page3 = mapOf(routes.listThreads("chat", 2, 4, "alice", null, null, null));
        assertEquals(1, ((List<?>) page3.get("threads")).size());
        org.junit.jupiter.api.Assertions.assertFalse(page3.containsKey("next_offset"));

        // 钳制：limit=0 → 1 条；offset 越界 → 空页。
        assertEquals(1, ((List<?>) mapOf(routes.listThreads("chat", 0, 0, "alice", null, null, null))
                .get("threads")).size());
        assertTrue(((List<?>) mapOf(routes.listThreads("chat", 2, 99, "alice", null, null, null))
                .get("threads")).isEmpty());

        // 降级路径（索引缺席）：同样支持 limit/offset。
        var checkpointer = Mockito.mock(io.github.paopaonb666.agentbase.core.graph.Checkpointer.class);
        when(checkpointer.listThreadIds()).thenReturn(
                List.of("chat:t1", "chat:t2", "chat:t3", "chat:t4", "chat:t5"));
        when(checkpointer.updatedAt(anyString())).thenAnswer(inv ->
                (long) ((2000.0 + Integer.parseInt(
                        ((String) inv.getArgument(0)).substring("chat:t".length()))) * 1000));
        AgentRuntime degraded = runtime(Map.of("chat", graphStub("chat")), checkpointer, null,
                null, null, null, null);
        AgentsRoutes degradedRoutes = new AgentsRoutes(degraded, PASS_THROUGH_AUTH, providerOf(null));
        Map<String, Object> dPage = mapOf(degradedRoutes.listThreads("chat", 2, 2, "alice", null, null, null));
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> dt = (List<Map<String, Object>>) dPage.get("threads");
        assertEquals(List.of("t3", "t2"), dt.stream().map(m -> m.get("thread_id")).toList());
        assertEquals(4, dPage.get("next_offset"));
    }

    @Test
    void threadHistorySerializesMessagesAndMasksOwnership() throws Exception {
        var index = Mockito.mock(io.github.paopaonb666.agentbase.core.storage.CoreStore.class);
        when(index.getThreadIndex("chat:t1"))
                .thenReturn(ThreadIndexRecord.create("chat:t1", "alice", "chat", "t"));
        AgentRuntime runtime = runtime(Map.of("chat", graphStub("chat")),
                new InMemoryCheckpointer(), index, null, null, null, null);
        AgentsRoutes routes = new AgentsRoutes(runtime, PASS_THROUGH_AUTH, providerOf(null));

        // 无快照 → 空消息列表（形状完整）。
        Map<String, Object> out = mapOf(routes.threadHistory("chat", "t1", "alice", null, null, null));
        assertEquals("t1", out.get("thread_id"));
        assertEquals("chat", out.get("module"));
        assertEquals(List.of(), out.get("messages"));

        // 非属主/不存在 → 404 掩蔽（同一文案）。
        assertEquals(404, statusOf(() -> routes.threadHistory("chat", "t1", "bob", null, null, null)));
        assertEquals(404, statusOf(() -> routes.threadHistory("chat", "ghost", "alice", null, null, null)));
        // thread_id 格式非法 → 422。
        assertEquals(422, statusOf(() -> routes.threadHistory("chat", "bad id!", "alice", null, null, null)));
    }

    @Test
    void threadPlanReadsPlanStoreWithFallbackDefaults() throws Exception {
        var index = Mockito.mock(io.github.paopaonb666.agentbase.core.storage.CoreStore.class);
        when(index.getThreadIndex(anyString()))
                .thenReturn(ThreadIndexRecord.create("chat:t1", "alice", "chat", "t"));
        var planStore = Mockito.mock(AgentModule.PlanStorePort.class);
        when(planStore.getPlan("chat:t1")).thenReturn(Map.of(
                "thread_id", "chat:t1", "module", "chat",
                "tasks", List.of(Map.of("goal", "x")), "cursor", 2, "replans", 1));
        AgentRuntime runtime = runtime(Map.of("chat", graphStub("chat"), "planner", graphStub("planner")),
                new InMemoryCheckpointer(), index, null, null, null, planStore);
        AgentsRoutes routes = new AgentsRoutes(runtime, PASS_THROUGH_AUTH, providerOf(null));

        Map<String, Object> out = mapOf(routes.threadPlan("chat", "t1", "alice", null, null, null));
        assertEquals(2, out.get("cursor"));
        assertEquals(1, out.get("replans"));
        assertEquals(List.of(Map.of("goal", "x")), out.get("tasks"));

        // planner 模块缺席 → 503。
        AgentRuntime noPlanner = runtime(Map.of("chat", graphStub("chat")), null, index,
                null, null, null, planStore);
        AgentsRoutes noPlannerRoutes = new AgentsRoutes(noPlanner, PASS_THROUGH_AUTH, providerOf(null));
        assertEquals(503, statusOf(() -> noPlannerRoutes.threadPlan("chat", "t1", "alice", null, null, null)));

        // planStore 缺席 → 回退图状态：从未跑过 plan 返回空默认值。
        AgentRuntime fallback = runtime(Map.of("chat", graphStub("chat"), "planner", graphStub("planner")),
                new InMemoryCheckpointer(), index, null, null, null, null);
        AgentsRoutes fallbackRoutes = new AgentsRoutes(fallback, PASS_THROUGH_AUTH, providerOf(null));
        Map<String, Object> empty = mapOf(fallbackRoutes.threadPlan("chat", "t1", "alice", null, null, null));
        assertEquals(List.of(), empty.get("tasks"));
        assertEquals(0, empty.get("cursor"));
        assertEquals(0, empty.get("replans"));
    }

    @Test
    void toolCallsClampLimitAndDegradeWithoutRecorder() throws Exception {
        var index = Mockito.mock(io.github.paopaonb666.agentbase.core.storage.CoreStore.class);
        when(index.getThreadIndex(anyString()))
                .thenReturn(ThreadIndexRecord.create("chat:t1", "alice", "chat", "t"));
        var recorder = Mockito.mock(io.github.paopaonb666.agentbase.extensions.toollog.ToolCallRecorder.class);
        when(recorder.listForThread(anyString(), anyInt())).thenReturn(List.of(
                new ToolCallRecord("c1", "web_search", "ok", "{}", "r", "", 12,
                        "chat:t1", "chat", "req").asDict()));
        AgentRuntime runtime = runtime(Map.of("chat", graphStub("chat")), null, index,
                null, null, recorder, null);
        AgentsRoutes routes = new AgentsRoutes(runtime, PASS_THROUGH_AUTH, providerOf(null));

        Map<String, Object> out = mapOf(routes.toolCalls("chat", "t1", 5000, "alice", null, null, null));
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> calls = (List<Map<String, Object>>) out.get("tool_calls");
        assertEquals(1, calls.size());
        ArgumentCaptor<Integer> limit = ArgumentCaptor.forClass(Integer.class);
        verify(recorder).listForThread(eq("chat:t1"), limit.capture());
        assertEquals(500, limit.getValue(), "limit 上钳 500");

        routes.toolCalls("chat", "t1", 0, "alice", null, null, null);
        verify(recorder).listForThread(eq("chat:t1"), eq(1));

        // recorder 缺席 → 空清单（审计旁路缺席不打断读取）。
        AgentRuntime bare = runtime(Map.of("chat", graphStub("chat")), null, index,
                null, null, null, null);
        Map<String, Object> emptyOut = mapOf(
                new AgentsRoutes(bare, PASS_THROUGH_AUTH, providerOf(null))
                        .toolCalls("chat", "t1", 100, "alice", null, null, null));
        assertEquals(List.of(), emptyOut.get("tool_calls"));
    }

    @Test
    void deleteThreadWithoutCheckpointerReportsNotDeleted() throws Exception {
        var index = Mockito.mock(io.github.paopaonb666.agentbase.core.storage.CoreStore.class);
        when(index.getThreadIndex(anyString()))
                .thenReturn(ThreadIndexRecord.create("chat:t1", "alice", "chat", "t"));
        AgentRuntime runtime = runtime(Map.of("chat", graphStub("chat")), null, index,
                null, null, null, null);
        AgentsRoutes routes = new AgentsRoutes(runtime, PASS_THROUGH_AUTH, providerOf(null));
        Map<String, Object> out = mapOf(routes.deleteThread("chat", "t1", "alice", null, null, null));
        assertEquals(false, out.get("deleted"));
        assertEquals("checkpointer unconfigured", out.get("reason"));
    }

    // ══ FilesRoutes：文件/知识端点矩阵 ═══════════════════════════════════

    private static UploadedFileInfo info(String fileId, String userId, String filename,
                                         String format, byte[] content) {
        return new UploadedFileInfo(fileId, "", "chat", userId, filename, format,
                null, null, false, 0, "", "正文", content);
    }

    @Test
    void revokeFileDeletesAndRevokesChunks() throws Exception {
        var fileStore = Mockito.mock(io.github.paopaonb666.agentbase.extensions.filestore.UploadedFileStore.class);
        when(fileStore.getMany(List.of("f1"))).thenReturn(List.of(info("f1", "alice", "a.txt", "txt", new byte[1])));
        when(fileStore.deleteFile("f1")).thenReturn(true);
        var memory = Mockito.mock(AgentModule.MemoryServicePort.class);
        when(memory.revokeDocument("f1", "alice")).thenReturn(3);
        AgentRuntime runtime = runtime(Map.of("chat", graphStub("chat")), null, null,
                fileStore, memory, null, null);
        FilesRoutes routes = new FilesRoutes(runtime, PASS_THROUGH_AUTH);

        Map<String, Object> out = mapOf(routes.revokeFile("chat", "f1", "alice", null, null, null));
        assertEquals(true, out.get("deleted"));
        assertEquals(3, out.get("chunks_removed"));
        verify(fileStore).deleteFile("f1");

        // 非属主/不存在 → 404 掩蔽（不泄露存在性）。
        when(fileStore.getMany(List.of("f2"))).thenReturn(List.of());
        assertEquals(404, statusOf(() -> routes.revokeFile("chat", "f2", "alice", null, null, null)));
        assertEquals(404, statusOf(() -> routes.revokeFile("chat", "f1", "bob", null, null, null)));
        // 未知模块 404。
        assertEquals(404, statusOf(() -> routes.revokeFile("nope", "f1", "alice", null, null, null)));
    }

    @Test
    void previewFileShapesMetaAndOmitsEmptyWarning() throws Exception {
        var fileStore = Mockito.mock(io.github.paopaonb666.agentbase.extensions.filestore.UploadedFileStore.class);
        when(fileStore.getMany(List.of("f1"))).thenReturn(List.of(
                new UploadedFileInfo("f1", "", "chat", "alice", "a.pdf", "pdf",
                        3, 12, true, 456, "扫描件无文本", "正文", new byte[1])));
        AgentRuntime runtime = runtime(Map.of("chat", graphStub("chat")), null, null,
                fileStore, null, null, null);
        FilesRoutes routes = new FilesRoutes(runtime, PASS_THROUGH_AUTH);

        Map<String, Object> out = mapOf(routes.previewFile("chat", "f1", "alice", null, null, null));
        assertEquals("a.pdf", out.get("filename"));
        assertEquals("pdf", out.get("format"));
        assertEquals(3, out.get("pages"));
        assertEquals(456, out.get("text_len"));
        assertEquals(true, out.get("truncated"));
        assertEquals("扫描件无文本", out.get("warning"));
        assertEquals("正文", out.get("extracted_text"));

        // 无警告 → warning 键不出现。
        when(fileStore.getMany(List.of("f2"))).thenReturn(List.of(
                new UploadedFileInfo("f2", "", "chat", "alice", "b.txt", "txt",
                        null, null, false, 5, "", "hello", new byte[1])));
        Map<String, Object> clean = mapOf(routes.previewFile("chat", "f2", "alice", null, null, null));
        assertEquals(false, clean.containsKey("warning"));
    }

    @Test
    void fileChunksClampsLimitAndDegradesWithoutMemory() throws Exception {
        var fileStore = Mockito.mock(io.github.paopaonb666.agentbase.extensions.filestore.UploadedFileStore.class);
        when(fileStore.getMany(List.of("f1"))).thenReturn(List.of(info("f1", "alice", "a.txt", "txt", new byte[1])));
        var memory = Mockito.mock(AgentModule.MemoryServicePort.class);
        when(memory.listFileChunks("alice", "f1", 2000)).thenReturn(List.of(Map.of("chunk_id", "c1")));
        AgentRuntime runtime = runtime(Map.of("chat", graphStub("chat")), null, null,
                fileStore, memory, null, null);
        FilesRoutes routes = new FilesRoutes(runtime, PASS_THROUGH_AUTH);

        Map<String, Object> out = mapOf(routes.fileChunks("chat", "f1", 99999, "alice", null, null, null));
        assertEquals(1, ((List<?>) out.get("chunks")).size());
        verify(memory).listFileChunks("alice", "f1", 2000);

        // memory 缺席 → 空清单。
        AgentRuntime bare = runtime(Map.of("chat", graphStub("chat")), null, null,
                fileStore, null, null, null);
        Map<String, Object> empty = mapOf(
                new FilesRoutes(bare, PASS_THROUGH_AUTH).fileChunks("chat", "f1", 10, "alice", null, null, null));
        assertEquals(List.of(), empty.get("chunks"));
    }

    @Test
    void rawFileServesImagesInlineAndDocumentsWithRfc5987() throws Exception {
        var fileStore = Mockito.mock(io.github.paopaonb666.agentbase.extensions.filestore.UploadedFileStore.class);
        when(fileStore.getMany(List.of("img"))).thenReturn(List.of(
                info("img", "alice", "图.png", "png", new byte[]{1, 2, 3})));
        when(fileStore.getMany(List.of("doc"))).thenReturn(List.of(
                info("doc", "alice", "报告 中文.txt", "txt", new byte[]{4})));
        when(fileStore.getMany(List.of("empty"))).thenReturn(List.of(
                info("empty", "alice", "e.txt", "txt", new byte[0])));
        AgentRuntime runtime = runtime(Map.of("chat", graphStub("chat")), null, null,
                fileStore, null, null, null);
        FilesRoutes routes = new FilesRoutes(runtime, PASS_THROUGH_AUTH);

        // 图片：image/png 直出（前端 <img> 可用）。
        ResponseEntity<byte[]> image = routes.rawFile("chat", "img", "alice", null, null, null);
        assertEquals("image/png", image.getHeaders().getContentType().toString());

        // 文档：octet-stream + RFC 5987 编码（中文文件名不乱码）。
        ResponseEntity<byte[]> doc = routes.rawFile("chat", "doc", "alice", null, null, null);
        assertEquals("application/octet-stream", doc.getHeaders().getContentType().toString());
        String encoded = java.net.URLEncoder.encode("报告 中文.txt",
                java.nio.charset.StandardCharsets.UTF_8).replace("+", "%20");
        assertEquals("attachment; filename*=UTF-8''" + encoded,
                doc.getHeaders().getFirst("Content-Disposition"));

        // 内容缺席（历史遗留行）→ 404。
        assertEquals(404, statusOf(() -> routes.rawFile("chat", "empty", "alice", null, null, null)));
    }

    @Test
    void searchKnowledgeValidatesQueryClampsLimitAndCompletesNames() throws Exception {
        var memory = Mockito.mock(AgentModule.MemoryServicePort.class);
        when(memory.searchKnowledge(anyString(), any(), anyString(), anyInt(), any()))
                .thenReturn(List.of(Map.of("file_id", "f1", "score", 0.9, "chunk_id", "c1",
                        "ordinal", 0, "text", "内容", "offsets", List.of(), "has_embedding", true)));
        var fileStore = Mockito.mock(io.github.paopaonb666.agentbase.extensions.filestore.UploadedFileStore.class);
        when(fileStore.getMany(List.of("f1"))).thenReturn(List.of(info("f1", "alice", "知识.pdf", "pdf", new byte[1])));
        AgentRuntime runtime = runtime(Map.of("chat", graphStub("chat")), null, null,
                fileStore, memory, null, null);
        FilesRoutes routes = new FilesRoutes(runtime, PASS_THROUGH_AUTH);

        Map<String, Object> out = mapOf(routes.searchKnowledge(" 产量 ", "f1", 99, "alice", null, null, null));
        verify(memory).searchKnowledge(eq("alice"), eq(null), eq("产量"), eq(20), eq("f1"));
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> results = (List<Map<String, Object>>) out.get("results");
        assertEquals(1, results.size());
        assertEquals("知识.pdf", results.get(0).get("file_name"));

        // 空 q（含纯空白）→ 422。
        assertEquals(422, statusOf(() -> routes.searchKnowledge("", null, 5, "alice", null, null, null)));
        assertEquals(422, statusOf(() -> routes.searchKnowledge("   ", null, 5, "alice", null, null, null)));

        // memory 缺席 → 503（知识检索直出依赖记忆系统）。
        AgentRuntime bare = runtime(Map.of("chat", graphStub("chat")), null, null,
                null, null, null, null);
        assertEquals(503, statusOf(() -> new FilesRoutes(bare, PASS_THROUGH_AUTH)
                .searchKnowledge("q", null, 5, "alice", null, null, null)));
    }

    @Test
    void knowledgeFilesMergesChunkCountsAndDegradesWithoutStore() throws Exception {
        var fileStore = Mockito.mock(io.github.paopaonb666.agentbase.extensions.filestore.UploadedFileStore.class);
        when(fileStore.listForUser("alice")).thenReturn(List.of(
                info("f1", "alice", "a.pdf", "pdf", new byte[1])));
        var memory = Mockito.mock(AgentModule.MemoryServicePort.class);
        when(memory.chunkCountsByFile("alice")).thenReturn(Map.of("f1", 7));
        AgentRuntime runtime = runtime(Map.of("chat", graphStub("chat")), null, null,
                fileStore, memory, null, null);
        FilesRoutes routes = new FilesRoutes(runtime, PASS_THROUGH_AUTH);

        Map<String, Object> out = mapOf(routes.knowledgeFiles("alice", null, null, null));
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> files = (List<Map<String, Object>>) out.get("files");
        assertEquals(1, files.size());
        assertEquals(7, files.get(0).get("chunks"));
        assertEquals("a.pdf", files.get(0).get("filename"));

        // fileStore 缺席 → 503。
        AgentRuntime bare = runtime(Map.of("chat", graphStub("chat")), null, null,
                null, null, null, null);
        assertEquals(503, statusOf(() -> new FilesRoutes(bare, PASS_THROUGH_AUTH)
                .knowledgeFiles("alice", null, null, null)));

        // memory 缺席 → counts 全 0（列表照常可用）。
        AgentRuntime noMemory = runtime(Map.of("chat", graphStub("chat")), null, null,
                fileStore, null, null, null);
        Map<String, Object> zeroCounts = mapOf(
                new FilesRoutes(noMemory, PASS_THROUGH_AUTH).knowledgeFiles("alice", null, null, null));
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> zeroFiles = (List<Map<String, Object>>) zeroCounts.get("files");
        assertEquals(0, zeroFiles.get(0).get("chunks"));
    }
}
