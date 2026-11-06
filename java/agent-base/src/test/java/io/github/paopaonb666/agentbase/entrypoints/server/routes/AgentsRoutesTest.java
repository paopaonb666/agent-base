package io.github.paopaonb666.agentbase.entrypoints.server.routes;

import io.github.paopaonb666.agentbase.core.AgentModule;
import io.github.paopaonb666.agentbase.core.AgentRuntime;
import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.core.graph.CompiledGraph;
import io.github.paopaonb666.agentbase.core.graph.GraphBuilder;
import io.github.paopaonb666.agentbase.core.graph.InMemoryCheckpointer;
import io.github.paopaonb666.agentbase.core.storage.ThreadIndexRecord;
import io.github.paopaonb666.agentbase.entrypoints.server.AuthBackends;
import io.github.paopaonb666.agentbase.entrypoints.server.HttpError;
import io.github.paopaonb666.agentbase.extensions.costmeter.CostBudgetExceededException;
import io.github.paopaonb666.agentbase.extensions.costmeter.CostGovernor;
import io.github.paopaonb666.agentbase.extensions.filestore.UploadedFileInfo;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.mock.web.MockHttpServletResponse;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.lang.reflect.Constructor;
import java.lang.reflect.Proxy;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * AgentsRoutes invoke 链测试（roadmap #289）：校验链 422 矩阵、503/400/429
 * 分支、附件越权 400 同文案（防枚举佐证 #221 的锁定）、线程端点属主掩蔽
 * 与删除级联顺序。运行时经反射构造（包私有构造器），组件全部用桩——
 * 不起 Spring 上下文，与既有纯 JUnit 风格一致。
 */
class AgentsRoutesTest {

    // ── 桩基建 ────────────────────────────────────────────────────────────

    /** 无鉴权直通：X-User-Id 缺省 default（对齐 AuthBackends 的缺省语义）。 */
    private static final AuthBackends.AuthBackend PASS_THROUGH_AUTH =
            (userId, sig, ts, nonce, settings) ->
                    userId == null || userId.isEmpty() ? "default" : userId;

    private static ObjectProvider<CostGovernor> providerOf(CostGovernor governor) {
        return new ObjectProvider<>() {
            @Override
            public CostGovernor getObject() {
                return governor;
            }

            @Override
            public CostGovernor getObject(Object... args) {
                return governor;
            }

            @Override
            public CostGovernor getIfAvailable() {
                return governor;
            }

            @Override
            public CostGovernor getIfUnique() {
                return governor;
            }
        };
    }

    /** 反射构造 AgentRuntime（包私有构造器，测试专用）。 */
    private static AgentRuntime buildRuntime(Map<String, AgentModule> modules,
                                             Object checkpointer,
                                             Object threadIndex,
                                             Object fileStore,
                                             Object memory,
                                             Object planStore,
                                             Object toolRecorder) throws Exception {
        return buildRuntime(Settings.load(Map.of("ENV", "development")), modules,
                checkpointer, threadIndex, fileStore, memory, planStore, toolRecorder);
    }

    /** 同上，但显式指定 Settings（production 守门测试用）。 */
    private static AgentRuntime buildRuntime(Settings settings,
                                             Map<String, AgentModule> modules,
                                             Object checkpointer,
                                             Object threadIndex,
                                             Object fileStore,
                                             Object memory,
                                             Object planStore,
                                             Object toolRecorder) throws Exception {
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
        return ctor.newInstance(settings, null, modules, null, checkpointer, toolRecorder,
                fileStore, memory, threadIndex, planStore);
    }

    /** 最小模块桩：图 = 单节点空转（不触碰 LLM）。 */
    private static AgentModule stubModule(String name) {
        return new AgentModule() {
            @Override
            public String name() {
                return name;
            }

            @Override
            public String description() {
                return "stub module for route tests";
            }

            @Override
            public CompiledGraph buildGraph(ModuleContext ctx) {
                // 用 ModuleContext 携带的 checkpointer（与线上装配一致——
                // 传 null 会让历史端点永远看不到快照）。
                return GraphBuilder.singleNode(name, (state, nodeCtx) -> Map.of(), ctx.checkpointer());
            }

            @Override
            public List<io.github.paopaonb666.agentbase.core.tools.AgentTool> getTools() {
                return List.of();
            }
        };
    }

    /** 接口动态代理桩：记录调用序列（级联顺序断言用）。 */
    private static Object recordingProxy(Class<?> iface, List<String> calls, String prefix) {
        return Proxy.newProxyInstance(iface.getClassLoader(), new Class<?>[]{iface},
                (proxy, method, args) -> {
                    if (method.getDeclaringClass() == Object.class) {
                        return method.invoke(proxy, args);
                    }
                    calls.add(prefix + ":" + method.getName());
                    return defaultValue(method.getReturnType());
                });
    }

    private static Object defaultValue(Class<?> type) {
        if (type == boolean.class) {
            return false;
        }
        if (type == int.class) {
            return 0;
        }
        if (type == long.class) {
            return 0L;
        }
        if (type == double.class) {
            return 0.0;
        }
        return null;
    }

    private static HttpError httpErrorOf(java.util.concurrent.Callable<?> body) {
        HttpError error = assertThrows(HttpError.class, body::call);
        return error;
    }

    private static AgentsRoutes route(AgentRuntime runtime) {
        return new AgentsRoutes(runtime, PASS_THROUGH_AUTH, providerOf(null));
    }

    private static AgentsRoutes.InvokeRequest invokeBody(String message, String threadId,
                                                         List<String> attachments,
                                                         String mode, String profile) {
        return new AgentsRoutes.InvokeRequest(message, threadId, attachments, mode, profile);
    }

    /** invoke 便捷调用：注入 X-User-Id，其余鉴权头缺省。 */
    private static SseEmitter invoke(AgentsRoutes routes, String module,
                                     AgentsRoutes.InvokeRequest body, String userId) {
        return routes.invoke(module, body, new MockHttpServletResponse(),
                userId, null, null, null);
    }

    // ── invoke 校验链（422 矩阵） ─────────────────────────────────────────

    @Test
    void invokeValidationMatrixGives422() throws Exception {
        AgentRuntime runtime = buildRuntime(Map.of("chat", stubModule("chat")),
                null, null, null, null, null, null);
        AgentsRoutes routes = route(runtime);

        for (AgentsRoutes.InvokeRequest bad : List.of(
                invokeBody(null, null, null, null, null),          // message 缺失
                invokeBody("", null, null, null, null),            // message 空
                invokeBody("hi", "bad id!", null, null, null),     // thread_id 非法字符
                invokeBody("hi", null, List.of("a", "b", "c", "d", "e", "f"), null, null), // >5
                invokeBody("hi", null, null, "research", null),    // mode 非法
                invokeBody("hi", null, null, null, "turbo"))) {    // profile 非法
            HttpError error = httpErrorOf(() -> invoke(routes, "chat", bad, null));
            assertEquals(422, error.status(), error.getMessage());
        }
    }

    @Test
    void invokeThreadLengthLimitsMatchContract() throws Exception {
        AgentRuntime runtime = buildRuntime(Map.of("chat", stubModule("chat")),
                new InMemoryCheckpointer(), null, null, null, null, null);
        AgentsRoutes routes = route(runtime);
        // 128 字符合法，129 字符 422（[\w.-]{1,128} 契约）。
        assertNotNull(invoke(routes, "chat", invokeBody("hi", "a".repeat(128), null, null, null), "u1"));
        HttpError error = httpErrorOf(() -> invoke(routes, "chat",
                invokeBody("hi", "a".repeat(129), null, null, null), "u1"));
        assertEquals(422, error.status());
    }

    @Test
    void planModeWithoutPlannerGives503() throws Exception {
        AgentRuntime runtime = buildRuntime(Map.of("chat", stubModule("chat")),
                null, null, null, null, null, null);
        AgentsRoutes routes = route(runtime);
        HttpError error = httpErrorOf(() ->
                invoke(routes, "chat", invokeBody("hi", null, null, "plan", null), "u1"));
        assertEquals(503, error.status());
        assertTrue(error.getMessage().contains("planner"), error.getMessage());
    }

    @Test
    void autoModeFallsBackSilentlyWithoutPlanner() throws Exception {
        // auto 分流：>200 字符会走 plan，但 planner 未启用时静默降级 chat。
        AgentRuntime runtime = buildRuntime(Map.of("chat", stubModule("chat")),
                new InMemoryCheckpointer(), null, null, null, null, null);
        AgentsRoutes routes = route(runtime);
        String longMessage = "很长的任务描述".repeat(60); // 码点 > 200
        var emitter = invoke(routes, "chat",
                invokeBody(longMessage, null, null, "auto", null), "u1");
        assertNotNull(emitter);
    }

    @Test
    void fastProfileWithoutFastLlmGives400() throws Exception {
        AgentRuntime runtime = buildRuntime(Map.of("chat", stubModule("chat")),
                null, null, null, null, null, null);
        AgentsRoutes routes = route(runtime);
        HttpError error = httpErrorOf(() ->
                invoke(routes, "chat", invokeBody("hi", null, null, null, "fast"), "u1"));
        assertEquals(400, error.status());
        assertTrue(error.getMessage().contains("LLM_FAST"), error.getMessage());
    }

    @Test
    void attachmentsWithoutFileStoreGives503() throws Exception {
        AgentRuntime runtime = buildRuntime(Map.of("chat", stubModule("chat")),
                null, null, null, null, null, null);
        AgentsRoutes routes = route(runtime);
        HttpError error = httpErrorOf(() ->
                invoke(routes, "chat", invokeBody("hi", null, List.of("f1"), null, null), "u1"));
        assertEquals(503, error.status());
    }

    @Test
    void missingAndUnauthorizedAttachmentsShareOne400Text() throws Exception {
        // 防枚举佐证（#221）：缺失与越权同文案同状态码，且按 file_id 排序。
        List<UploadedFileInfo> store = List.of(new UploadedFileInfo(
                "f2", "", "", "someone-else", "b.txt", "txt",
                null, null, false, 10, null, "content", new byte[0]));
        Object fileStore = Proxy.newProxyInstance(
                io.github.paopaonb666.agentbase.extensions.filestore.UploadedFileStore.class.getClassLoader(),
                new Class<?>[]{io.github.paopaonb666.agentbase.extensions.filestore.UploadedFileStore.class},
                (proxy, method, args) -> {
                    if ("getMany".equals(method.getName())) {
                        return store;
                    }
                    return defaultValue(method.getReturnType());
                });
        AgentRuntime runtime = buildRuntime(Map.of("chat", stubModule("chat")),
                null, null, fileStore, null, null, null);
        AgentsRoutes routes = route(runtime);
        HttpError error = httpErrorOf(() -> invoke(routes, "chat",
                invokeBody("hi", null, List.of("f2", "f1"), null, null), "u1"));
        assertEquals(400, error.status());
        // 排序后 f1 在前（请求顺序是 f2,f1）。
        assertEquals("附件不存在或无权访问：f1，f2；请重新上传", error.getMessage());
    }

    @Test
    void budgetExceededGives429BeforeStreamOpens() throws Exception {
        CostGovernor exhausted = Mockito.mock(CostGovernor.class);
        Mockito.doThrow(new CostBudgetExceededException("日预算已用尽"))
                .when(exhausted).check();
        AgentRuntime runtime = buildRuntime(Map.of("chat", stubModule("chat")),
                null, null, null, null, null, null);
        AgentsRoutes routes = new AgentsRoutes(runtime, PASS_THROUGH_AUTH, providerOf(exhausted));
        HttpError error = httpErrorOf(() ->
                invoke(routes, "chat", invokeBody("hi", null, null, null, null), "u1"));
        assertEquals(429, error.status());
        assertTrue(error.getMessage().contains("预算"));
    }

    // ── 线程端点：列表 / 属主掩蔽 / 审计 / 删除级联 ───────────────────────

    @Test
    void listThreadsReturnsOwnedEntriesWithShortIds() throws Exception {
        Object index = Proxy.newProxyInstance(
                io.github.paopaonb666.agentbase.core.storage.CoreStore.class.getClassLoader(),
                new Class<?>[]{io.github.paopaonb666.agentbase.core.storage.CoreStore.class},
                (proxy, method, args) -> {
                    if ("listThreadIndexes".equals(method.getName())) {
                        return List.of(
                                ThreadIndexRecord.create("chat:t1", "u1", "chat", "第一线程"),
                                ThreadIndexRecord.create("chat:t9", "u1", "chat", "另一个"));
                    }
                    return defaultValue(method.getReturnType());
                });
        AgentRuntime runtime = buildRuntime(Map.of("chat", stubModule("chat")),
                null, index, null, null, null, null);
        AgentsRoutes routes = route(runtime);

        Map<String, Object> out = routes.listThreads("chat", "u1", null, null, null);
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> threads = (List<Map<String, Object>>) out.get("threads");
        assertEquals(2, threads.size());
        assertEquals("t1", threads.get(0).get("thread_id"));
        assertEquals("chat", threads.get(0).get("module"));
        assertEquals("第一线程", threads.get(0).get("title"));
    }

    @Test
    void listThreadsUnknownModuleGives404() throws Exception {
        AgentRuntime runtime = buildRuntime(Map.of("chat", stubModule("chat")),
                null, null, null, null, null, null);
        AgentsRoutes routes = route(runtime);
        HttpError error = httpErrorOf(() -> routes.listThreads("ghost", "u1", null, null, null));
        assertEquals(404, error.status());
    }

    @Test
    void degradedListScanRestoresUpdatedAtAndModuleFilter() throws Exception {
        // #142 修复后：索引缺席时 updated_at 取 checkpointer 最后落盘时刻
        //（解析不了才退 0），列表时间语义恢复；只扫描当前模块前缀。
        InMemoryCheckpointer cp = new InMemoryCheckpointer();
        cp.put("chat:a", new io.github.paopaonb666.agentbase.core.graph.GraphState(
                List.of(dev.langchain4j.data.message.UserMessage.from("问我任何问题")),
                Map.of()));
        cp.put("writer:b", new io.github.paopaonb666.agentbase.core.graph.GraphState());
        AgentRuntime runtime = buildRuntime(Map.of("chat", stubModule("chat")),
                cp, null, null, null, null, null);
        AgentsRoutes routes = route(runtime);

        long before = System.currentTimeMillis();
        Map<String, Object> out = routes.listThreads("chat", "u1", null, null, null);
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> threads = (List<Map<String, Object>>) out.get("threads");
        assertEquals(1, threads.size());
        assertEquals("a", threads.get(0).get("thread_id"));
        assertEquals("问我任何问题", threads.get(0).get("title"));
        long updatedAt = (long) threads.get(0).get("updated_at");
        assertTrue(updatedAt > 0, "#142 修复后 updated_at 不再恒 0");
        assertTrue(updatedAt <= System.currentTimeMillis() && updatedAt >= before - 5_000,
                "updated_at 应是最近落盘时刻（epoch 毫秒）: " + updatedAt);
        // 未落盘的线程时间未知 → 0（降级口径，不报错）。
        assertEquals(0L, cp.updatedAt("chat:ghost"));
        // 删除线程后时间随行清除。
        cp.deleteThread("chat:a");
        assertEquals(0L, cp.updatedAt("chat:a"));
    }

    @Test
    void threadHistoryMasksForeignAndMissingThreads() throws Exception {
        Object index = Proxy.newProxyInstance(
                io.github.paopaonb666.agentbase.core.storage.CoreStore.class.getClassLoader(),
                new Class<?>[]{io.github.paopaonb666.agentbase.core.storage.CoreStore.class},
                (proxy, method, args) -> {
                    if ("getThreadIndex".equals(method.getName())) {
                        return "chat:t1".equals(args[0])
                                ? ThreadIndexRecord.create("chat:t1", "owner-a", "chat", "标题")
                                : null;
                    }
                    return defaultValue(method.getReturnType());
                });
        AgentRuntime runtime = buildRuntime(Map.of("chat", stubModule("chat")),
                new InMemoryCheckpointer(), index, null, null, null, null);
        AgentsRoutes routes = route(runtime);

        HttpError foreign = httpErrorOf(() ->
                routes.threadHistory("chat", "t1", "u1", null, null, null));
        assertEquals(404, foreign.status());
        HttpError missing = httpErrorOf(() ->
                routes.threadHistory("chat", "nope", "u1", null, null, null));
        assertEquals(404, missing.status());
        // 属主本人可见（历史快照为空也返回空 messages 而非 404）。
        Map<String, Object> own = routes.threadHistory("chat", "t1", "owner-a", null, null, null);
        assertEquals("t1", own.get("thread_id"));
        assertEquals(List.of(), own.get("messages"));
    }

    @Test
    void chineseThreadIdRoundTripsThroughPathEndpoints() throws Exception {
        // #121 回归：中文 thread id 在 invoke 创建后在历史端点仍可读
        // （路径校验 (?U) 与请求体侧同 unicode 语义，对齐 Python \w）。
        InMemoryCheckpointer cp = new InMemoryCheckpointer();
        AgentRuntime runtime = buildRuntime(Map.of("chat", stubModule("chat")),
                cp, null, null, null, null, null);
        AgentsRoutes routes = route(runtime);
        MockHttpServletResponse response = new MockHttpServletResponse();

        // invoke 校验链接受中文 id（修复前 create 成功是既态——问题在读侧）。
        assertNotNull(invoke(routes, "chat",
                invokeBody("第一条消息", "中文线程", null, null, null), "u1"));
        // SSE 生产者线程依赖真实 servlet 异步上下文才能跑完落盘；测试里直接
        // 同步驱动同一张图写快照（等价于正常完成的一轮对话）。
        runtime.graph("chat").invoke(
                new io.github.paopaonb666.agentbase.core.graph.GraphState(
                        List.of(dev.langchain4j.data.message.UserMessage.from("第一条消息")), Map.of()),
                new io.github.paopaonb666.agentbase.core.graph.RunConfig("chat:中文线程", "u1", 25));

        // 历史/工具审计/删除全部正常返回（修复前路径校验 422——创建后即失联）。
        Map<String, Object> history = routes.threadHistory("chat", "中文线程", "u1", null, null, null);
        assertEquals("中文线程", history.get("thread_id"));
        assertTrue(((List<?>) history.get("messages")).size() >= 1);
        routes.toolCalls("chat", "中文线程", 100, "u1", null, null, null);
        Map<String, Object> deleted = routes.deleteThread("chat", "中文线程", "u1", null, null, null);
        assertEquals(Boolean.TRUE, deleted.get("deleted"));
    }

    @Test
    void toolCallsClampsLimitAndMasksOwner() throws Exception {
        List<String> calls = new ArrayList<>();
        Object recorder = Proxy.newProxyInstance(
                io.github.paopaonb666.agentbase.extensions.toollog.ToolCallRecorder.class.getClassLoader(),
                new Class<?>[]{io.github.paopaonb666.agentbase.extensions.toollog.ToolCallRecorder.class},
                (proxy, method, args) -> {
                    calls.add(method.getName() + List.of(args));
                    return List.of();
                });
        AgentRuntime runtime = buildRuntime(Map.of("chat", stubModule("chat")),
                new InMemoryCheckpointer(), null, null, null, null, recorder);
        AgentsRoutes routes = route(runtime);

        routes.toolCalls("chat", "t1", 600, "u1", null, null, null);
        assertEquals(1, calls.size());
        assertTrue(calls.get(0).contains("500"), calls.get(0));
        // 无 recorder → 空列表（不抛错）。
        AgentRuntime bare = buildRuntime(Map.of("chat", stubModule("chat")),
                new InMemoryCheckpointer(), null, null, null, null, null);
        assertEquals(Map.of("tool_calls", List.of()),
                route(bare).toolCalls("chat", "t1", 100, "u1", null, null, null));
    }

    @Test
    void deleteThreadCascadeRunsInDocumentedOrder() throws Exception {
        // 级联顺序（#213 基线）：checkpointer → fileStore →
        // threadIndex(deleteThreadIndex + deletePlan) → memory.purgeThread。
        List<String> calls = new ArrayList<>();
        Object index = Proxy.newProxyInstance(
                io.github.paopaonb666.agentbase.core.storage.CoreStore.class.getClassLoader(),
                new Class<?>[]{io.github.paopaonb666.agentbase.core.storage.CoreStore.class},
                (proxy, method, args) -> {
                    calls.add("index:" + method.getName());
                    if ("getThreadIndex".equals(method.getName())) {
                        return ThreadIndexRecord.create("chat:t1", "u1", "chat", "标题");
                    }
                    return defaultValue(method.getReturnType());
                });
        Object cp = recordingProxy(io.github.paopaonb666.agentbase.core.graph.Checkpointer.class,
                calls, "cp");
        Object fileStore = recordingProxy(
                io.github.paopaonb666.agentbase.extensions.filestore.UploadedFileStore.class,
                calls, "files");
        Object memory = recordingProxy(AgentModule.MemoryServicePort.class, calls, "memory");
        AgentRuntime runtime = buildRuntime(Map.of("chat", stubModule("chat")),
                cp, index, fileStore, memory, null, null);
        AgentsRoutes routes = route(runtime);

        Map<String, Object> out = routes.deleteThread("chat", "t1", "u1", null, null, null);
        assertEquals(Boolean.TRUE, out.get("deleted"));
        // 第一个 index:getThreadIndex 是属主校验（threadOwnerOr404）。
        assertEquals(List.of(
                "index:getThreadIndex",
                "cp:deleteThread",
                "files:deleteForThread",
                "index:deleteThreadIndex",
                "index:deletePlan",
                "memory:purgeThread"), calls);
    }

    @Test
    void invokeUpsertsThreadIndexExactlyOnce() throws Exception {
        // #141：索引 upsert 收敛到 startStream 单次执行——非附件分支不再
        // 先行写一次（双写放大写入量，两次写之间失败会留下不一致标题）。
        List<String> calls = new ArrayList<>();
        Object index = Proxy.newProxyInstance(
                io.github.paopaonb666.agentbase.core.storage.CoreStore.class.getClassLoader(),
                new Class<?>[]{io.github.paopaonb666.agentbase.core.storage.CoreStore.class},
                (proxy, method, args) -> {
                    calls.add(method.getName());
                    return defaultValue(method.getReturnType());
                });
        AgentRuntime runtime = buildRuntime(Map.of("chat", stubModule("chat")),
                new InMemoryCheckpointer(), index, null, null, null, null);
        AgentsRoutes routes = route(runtime);

        invoke(routes, "chat", invokeBody("第一条消息", "t1", null, null, null), "u1");
        assertEquals(1, calls.stream().filter("upsertThreadIndex"::equals).count(),
                "一次 invoke 恰一次 upsert：" + calls);
    }

    @Test
    void degradedPathsRefuseInProductionAndKeepLocalDevSemantics() throws Exception {
        // #82：索引缺席时 production 拒绝降级服务（无属主校验=越权暴露面，
        // fail-safe 而非 fail-open）；development 保留全库扫描旧行为。
        Settings production = Settings.load(Map.of(
                "ENV", "production",
                "LLM_API_KEY", "sk-test",
                "MEMORY_AUTH_SECRET", "s",
                "MEMORY_ENABLED", "false"));
        AgentRuntime prodRuntime = buildRuntime(production, Map.of("chat", stubModule("chat")),
                new InMemoryCheckpointer(), null, null, null, null, null);
        AgentsRoutes prodRoutes = new AgentsRoutes(prodRuntime, PASS_THROUGH_AUTH, providerOf(null));

        HttpError history = httpErrorOf(() ->
                prodRoutes.threadHistory("chat", "t1", "u1", null, null, null));
        assertEquals(503, history.status());
        assertTrue(history.getMessage().contains("fail-safe"), history.getMessage());
        HttpError list = httpErrorOf(() ->
                prodRoutes.listThreads("chat", "u1", null, null, null));
        assertEquals(503, list.status());
        HttpError delete = httpErrorOf(() ->
                prodRoutes.deleteThread("chat", "t1", "u1", null, null, null));
        assertEquals(503, delete.status());

        // development：降级路径可用（列表能扫到 checkpointer 里的线程）。
        AgentRuntime devRuntime = buildRuntime(Map.of("chat", stubModule("chat")),
                new InMemoryCheckpointer(), null, null, null, null, null);
        devRuntime.checkpointer().put("chat:t1",
                new io.github.paopaonb666.agentbase.core.graph.GraphState(
                        List.of(dev.langchain4j.data.message.UserMessage.from("本地")), Map.of()));
        Map<String, Object> out = new AgentsRoutes(devRuntime, PASS_THROUGH_AUTH,
                providerOf(null)).listThreads("chat", "u1", null, null, null);
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> threads = (List<Map<String, Object>>) out.get("threads");
        assertEquals(1, threads.size());
        assertEquals("t1", threads.get(0).get("thread_id"));
    }

    @Test
    void planEndpointWithoutPlannerGives503() throws Exception {
        AgentRuntime runtime = buildRuntime(Map.of("chat", stubModule("chat")),
                new InMemoryCheckpointer(), null, null, null, null, null);
        AgentsRoutes routes = route(runtime);
        HttpError error = httpErrorOf(() ->
                routes.threadPlan("chat", "t1", "u1", null, null, null));
        assertEquals(503, error.status());
    }
}
