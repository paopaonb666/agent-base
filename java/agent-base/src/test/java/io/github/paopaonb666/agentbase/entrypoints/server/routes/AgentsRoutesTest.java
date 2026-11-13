package io.github.paopaonb666.agentbase.entrypoints.server.routes;

import io.github.paopaonb666.agentbase.core.AgentModule;
import io.github.paopaonb666.agentbase.core.AgentRuntime;
import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.core.graph.CompiledGraph;
import io.github.paopaonb666.agentbase.core.graph.GraphBuilder;
import io.github.paopaonb666.agentbase.core.graph.InMemoryCheckpointer;
import io.github.paopaonb666.agentbase.core.storage.ThreadIndexRecord;
import io.github.paopaonb666.agentbase.entrypoints.server.AuthBackends;
import io.github.paopaonb666.agentbase.entrypoints.server.ErrorCode;
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
import java.util.concurrent.CountDownLatch;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
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

    /** 同上，response 由调用方持有（断言 Retry-After 等响应头用）。 */
    private static SseEmitter invoke(AgentsRoutes routes, String module,
                                     AgentsRoutes.InvokeRequest body, String userId,
                                     MockHttpServletResponse response) {
        return routes.invoke(module, body, response, userId, null, null, null);
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
        // #220：422 文案与常量单源——mode/profile 的全部合法值都出现在文案里。
        HttpError modeError = httpErrorOf(() -> invoke(routes, "chat",
                invokeBody("hi", null, null, "research", null), null));
        for (String mode : AgentsRoutes.MODES) {
            assertTrue(modeError.getMessage().contains(mode), modeError.getMessage());
        }
        HttpError profileError = httpErrorOf(() -> invoke(routes, "chat",
                invokeBody("hi", null, null, null, "turbo"), null));
        for (String profile : io.github.paopaonb666.agentbase.core.llm.LlmFactory.PROFILES) {
            assertTrue(profileError.getMessage().contains(profile), profileError.getMessage());
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
        // #91：结构化错误码与状态码同源——前端只认 code 不认文案。
        assertEquals(ErrorCode.MODULE_UNAVAILABLE.wire(), error.code().wire());
    }

    @Test
    void errorCodesCoverRepresentativeRoutes() throws Exception {
        // #91 抽样锁定：422→validation、400 语义拒绝→bad_request（409/429/503
        // 的码断言分别在 ConsistencyLockTest/限流矩阵/planMode 三处）。
        AgentRuntime runtime = buildRuntime(Map.of("chat", stubModule("chat")),
                null, null, null, null, null, null);
        AgentsRoutes routes = route(runtime);
        HttpError validation = httpErrorOf(() -> invoke(routes, "chat",
                invokeBody("hi", null, null, "research", null), null));
        assertEquals(ErrorCode.VALIDATION.wire(), validation.code().wire());
        HttpError fastMissing = httpErrorOf(() -> invoke(routes, "chat",
                invokeBody("hi", null, null, null, "fast"), "u1"));
        assertEquals(400, fastMissing.status());
        assertEquals(ErrorCode.BAD_REQUEST.wire(), fastMissing.code().wire());
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
    void perUserRateLimitGives429WithRetryAfter() throws Exception {
        // #78：RATE_LIMIT_PER_MINUTE=1 → 同一用户第二发 429 + Retry-After；
        // 限流在认证后、副作用前（不发 SSE、不写索引）。配 secret 走真
        // per-user 分桶（无鉴权部署是共享降级桶，见下一条）。
        AgentRuntime runtime = buildRuntime(Settings.load(Map.of(
                "ENV", "development", "RATE_LIMIT_PER_MINUTE", "1",
                "MEMORY_AUTH_SECRET", "s")), Map.of("chat", stubModule("chat")),
                null, null, null, null, null, null);
        AgentsRoutes routes = route(runtime);
        var first = new org.springframework.mock.web.MockHttpServletResponse();
        assertNotNull(invoke(routes, "chat", invokeBody("hi", null, null, null, null), "u1",
                first));
        var second = new org.springframework.mock.web.MockHttpServletResponse();
        HttpError error = httpErrorOf(() ->
                invoke(routes, "chat", invokeBody("hi", null, null, null, null), "u1", second));
        assertEquals(429, error.status());
        assertTrue(error.getMessage().contains("RATE_LIMIT_PER_MINUTE"), error.getMessage());
        assertEquals("60", second.getHeader("Retry-After"));
        // 不同用户桶独立：bob 首发不受 alice 限流影响。
        var third = new org.springframework.mock.web.MockHttpServletResponse();
        assertNotNull(invoke(routes, "chat", invokeBody("hi", null, null, null, null), "bob", third));
    }

    @Test
    void unauthenticatedDeploymentSharesOneRateLimitBucket() throws Exception {
        // review R1-F2：无鉴权部署下 X-User-Id 客户端可控——按其分桶等于
        // 发无限个桶（无界表 DoS）。共享降级桶：alice 耗尽后 bob 同样 429。
        AgentRuntime runtime = buildRuntime(Settings.load(Map.of(
                "ENV", "development", "RATE_LIMIT_PER_MINUTE", "1")),
                Map.of("chat", stubModule("chat")),
                null, null, null, null, null, null);
        AgentsRoutes routes = route(runtime);
        assertNotNull(invoke(routes, "chat", invokeBody("hi", null, null, null, null), "alice"));
        HttpError bob = httpErrorOf(() ->
                invoke(routes, "chat", invokeBody("hi", null, null, null, null), "bob"));
        assertEquals(429, bob.status(), "共享桶：bob 应受 alice 已耗尽额度约束");
    }

    @Test
    void globalRunGateGives429WhenSaturatedAndReleasesOnCompletion() throws Exception {
        // #79：AGENT_RUN_CONCURRENCY=1 → 在途 run 占满闸门时第二发 429；
        // run 完成后终态钩子释放（produce 的 finally），后续 invoke 恢复。
        java.util.concurrent.CountDownLatch entered = new java.util.concurrent.CountDownLatch(1);
        java.util.concurrent.CountDownLatch release = new java.util.concurrent.CountDownLatch(1);
        AgentModule blocking = new AgentModule() {
            @Override public String name() { return "chat"; }
            @Override public String description() { return "blocking stub for run gate"; }
            @Override public CompiledGraph buildGraph(ModuleContext ctx) {
                return GraphBuilder.singleNode("chat", (state, nodeCtx) -> {
                    entered.countDown();
                    release.await(10, java.util.concurrent.TimeUnit.SECONDS);
                    return Map.of();
                }, ctx.checkpointer());
            }
            @Override public List<io.github.paopaonb666.agentbase.core.tools.AgentTool> getTools() {
                return List.of();
            }
        };
        AgentRuntime runtime = buildRuntime(Settings.load(Map.of(
                "ENV", "development", "AGENT_RUN_CONCURRENCY", "1")),
                Map.of("chat", blocking), new InMemoryCheckpointer(),
                null, null, null, null, null);
        AgentsRoutes routes = route(runtime);

        assertNotNull(invoke(routes, "chat", invokeBody("hi", "r1", null, null, null), "u1"));
        assertTrue(entered.await(5, java.util.concurrent.TimeUnit.SECONDS), "阻塞节点应已进入");
        // 在途 run 占满唯一额度：第二发 429 + Retry-After。
        var second = new org.springframework.mock.web.MockHttpServletResponse();
        HttpError error = httpErrorOf(() ->
                invoke(routes, "chat", invokeBody("hi", "r2", null, null, null), "u1", second));
        assertEquals(429, error.status());
        assertTrue(error.getMessage().contains("AGENT_RUN_CONCURRENCY"), error.getMessage());
        assertEquals("5", second.getHeader("Retry-After"));

        // 释放阻塞节点 → run 完成 → 终态钩子退出闸门 → 后续 invoke 恢复。
        release.countDown();
        HttpError last = null;
        for (int i = 0; i < 200; i++) {
            try {
                invoke(routes, "chat", invokeBody("hi", "r3", null, null, null), "u1");
                last = null;
                break;
            } catch (HttpError e) {
                last = e; // 释放传播有延迟：429 期让位重试
                Thread.sleep(25);
            }
        }
        assertTrue(last == null, "run 完成后闸门应已释放" + (last == null ? "" : "：" + last.getMessage()));
    }

    @Test
    void runGateRejectionDoesNotClaimThreadIndex() throws Exception {
        // review R1-F6：闸门判定提前到线程索引 claim 之前——饱和 429 的请求
        // 不认领/不刷新线程索引（幽灵活跃痕迹消失）；释放后恢复且认领照常。
        // 形态同上一条闸门测试（AGENT_RUN_CONCURRENCY=1 + 阻塞节点占满），
        // 另传线程索引（buildRuntime 第 3 参）：MemoryCoreStore 为 final 不可
        // 子类化，计数桩用 CoreStore 动态代理委托真实 MemoryCoreStore 并统计
        // claimThreadIndex 调用（与文件内既有 CoreStore 桩同形态）。
        java.util.concurrent.CountDownLatch entered = new java.util.concurrent.CountDownLatch(1);
        java.util.concurrent.CountDownLatch release = new java.util.concurrent.CountDownLatch(1);
        AgentModule blocking = new AgentModule() {
            @Override public String name() { return "chat"; }
            @Override public String description() { return "blocking stub for gate claim ordering"; }
            @Override public CompiledGraph buildGraph(ModuleContext ctx) {
                return GraphBuilder.singleNode("chat", (state, nodeCtx) -> {
                    entered.countDown();
                    release.await(10, java.util.concurrent.TimeUnit.SECONDS);
                    return Map.of();
                }, ctx.checkpointer());
            }
            @Override public List<io.github.paopaonb666.agentbase.core.tools.AgentTool> getTools() {
                return List.of();
            }
        };
        var backing = new io.github.paopaonb666.agentbase.core.storage.MemoryCoreStore();
        java.util.concurrent.atomic.AtomicInteger claimCalls = new java.util.concurrent.atomic.AtomicInteger();
        Object threadIndex = Proxy.newProxyInstance(
                io.github.paopaonb666.agentbase.core.storage.CoreStore.class.getClassLoader(),
                new Class<?>[]{io.github.paopaonb666.agentbase.core.storage.CoreStore.class},
                (proxy, method, args) -> {
                    if ("claimThreadIndex".equals(method.getName())) {
                        claimCalls.incrementAndGet();
                    }
                    try {
                        return method.invoke(backing, args);
                    } catch (java.lang.reflect.InvocationTargetException e) {
                        throw e.getCause();
                    }
                });
        AgentRuntime runtime = buildRuntime(Settings.load(Map.of(
                "ENV", "development", "AGENT_RUN_CONCURRENCY", "1")),
                Map.of("chat", blocking), new InMemoryCheckpointer(),
                threadIndex, null, null, null, null);
        AgentsRoutes routes = route(runtime);

        assertNotNull(invoke(routes, "chat", invokeBody("hi", "r1", null, null, null), "u1"));
        assertTrue(entered.await(5, java.util.concurrent.TimeUnit.SECONDS), "阻塞节点应已进入");
        // 首轮已恰一次认领 r1（invoke 返回前 claim 同步完成）。
        assertEquals(1, claimCalls.get(), "首轮应恰一次认领：" + claimCalls.get());
        assertNotNull(backing.getThreadIndex(
                new io.github.paopaonb666.agentbase.core.ThreadId("chat", "r1").toString()));
        // 在途 run 占满唯一额度：第二发 429 + Retry-After。
        var second = new org.springframework.mock.web.MockHttpServletResponse();
        HttpError error = httpErrorOf(() ->
                invoke(routes, "chat", invokeBody("hi", "r2", null, null, null), "u1", second));
        assertEquals(429, error.status());
        assertEquals("5", second.getHeader("Retry-After"));
        // R1-F6 核心断言：429 拒绝未认领——claim 计数不增，索引里没有 r2 的行
        //（幽灵活跃痕迹消失；修复前该请求会先 claim 出 r2 行再撞闸门）。
        assertEquals(1, claimCalls.get(), "闸门 429 不得认领线程索引：" + claimCalls.get());
        assertNull(backing.getThreadIndex(
                new io.github.paopaonb666.agentbase.core.ThreadId("chat", "r2").toString()));

        // 释放阻塞节点 → run 完成 → 终态钩子退出闸门 → r3 正常恢复且认领照常。
        release.countDown();
        HttpError last = null;
        for (int i = 0; i < 200; i++) {
            try {
                invoke(routes, "chat", invokeBody("hi", "r3", null, null, null), "u1");
                last = null;
                break;
            } catch (HttpError e) {
                last = e; // 释放传播有延迟：429 期让位重试
                Thread.sleep(25);
            }
        }
        assertTrue(last == null, "run 完成后闸门应已释放" + (last == null ? "" : "：" + last.getMessage()));
        assertEquals(2, claimCalls.get(), "恢复后的 r3 应认领一次：" + claimCalls.get());
        assertNotNull(backing.getThreadIndex(
                new io.github.paopaonb666.agentbase.core.ThreadId("chat", "r3").toString()));
    }

    @Test
    void runGateSaturated429DoesNotBindAttachments() throws Exception {
        // review C-P2：附件 bindThread 后置于 run gate——闸门 429 的请求不得
        // 残留 file↔thread 绑定行（purgeOrphans 只清未绑定孤儿，绑定行会留到
        // 线程删除为止）。修复前绑定发生在 invoke 主体、早于 startStream 的
        // gate 判定——429 请求已把附件绑上线程（本测试在该序下必红）。
        java.util.concurrent.CountDownLatch entered = new java.util.concurrent.CountDownLatch(1);
        java.util.concurrent.CountDownLatch release = new java.util.concurrent.CountDownLatch(1);
        AgentModule blocking = new AgentModule() {
            @Override public String name() { return "chat"; }
            @Override public String description() { return "blocking stub for attachment gate ordering"; }
            @Override public CompiledGraph buildGraph(ModuleContext ctx) {
                return GraphBuilder.singleNode("chat", (state, nodeCtx) -> {
                    entered.countDown();
                    release.await(10, java.util.concurrent.TimeUnit.SECONDS);
                    return Map.of();
                }, ctx.checkpointer());
            }
            @Override public List<io.github.paopaonb666.agentbase.core.tools.AgentTool> getTools() {
                return List.of();
            }
        };
        List<UploadedFileInfo> store = List.of(new UploadedFileInfo(
                "f1", "", "", "u1", "a.txt", "txt",
                null, null, false, 2, null, "hi", new byte[0]));
        List<String> fileCalls = new ArrayList<>();
        Object fileStore = Proxy.newProxyInstance(
                io.github.paopaonb666.agentbase.extensions.filestore.UploadedFileStore.class.getClassLoader(),
                new Class<?>[]{io.github.paopaonb666.agentbase.extensions.filestore.UploadedFileStore.class},
                (proxy, method, args) -> {
                    if ("getMany".equals(method.getName())) {
                        return store;
                    }
                    fileCalls.add(method.getName());
                    return defaultValue(method.getReturnType());
                });
        AgentRuntime runtime = buildRuntime(Settings.load(Map.of(
                "ENV", "development", "AGENT_RUN_CONCURRENCY", "1")),
                Map.of("chat", blocking), new InMemoryCheckpointer(),
                null, fileStore, null, null, null);
        AgentsRoutes routes = route(runtime);

        // 首轮占满唯一额度（无附件；阻塞节点进入后在途）。
        assertNotNull(invoke(routes, "chat", invokeBody("hi", "r1", null, null, null), "u1"));
        assertTrue(entered.await(5, java.util.concurrent.TimeUnit.SECONDS), "阻塞节点应已进入");
        // 带附件的第二发撞闸门 429：不得发生任何绑定副作用。
        HttpError error = httpErrorOf(() ->
                invoke(routes, "chat", invokeBody("hi", "r2", List.of("f1"), null, null), "u1"));
        assertEquals(429, error.status());
        assertFalse(fileCalls.contains("bindThread"), "闸门 429 不得绑定附件：" + fileCalls);
        release.countDown();
    }

    @Test
    void continuePlanWithoutSnapshotGives400() throws Exception {
        // #66：显式要求延续而快照缺席 → 400（静默重拆反而误导）。
        AgentRuntime runtime = buildRuntime(Map.of("planner", stubModule("planner")),
                null, null, null, null, null, null);
        AgentsRoutes routes = route(runtime);
        HttpError error = httpErrorOf(() -> invoke(routes, "planner",
                new AgentsRoutes.InvokeRequest("继续", "t1", null, "plan", null, true), "u1"));
        assertEquals(400, error.status());
        assertTrue(error.getMessage().contains("没有可延续的计划快照"));
    }

    @Test
    void continuePlanOutsidePlannerGives400() throws Exception {
        AgentRuntime runtime = buildRuntime(Map.of("chat", stubModule("chat")),
                null, null, null, null, null, null);
        AgentsRoutes routes = route(runtime);
        HttpError error = httpErrorOf(() -> invoke(routes, "chat",
                new AgentsRoutes.InvokeRequest("继续", "t1", null, "chat", null, true), "u1"));
        assertEquals(400, error.status());
        assertTrue(error.getMessage().contains("continue_plan 需要与 planner 一起使用"));
    }

    @Test
    void fastProfileInvokeExecutesTheViewGraph() throws Exception {
        // 审查 C-P1：profile=fast 的执行图必须取自快档视图——修复前 view
        // 只做 400 门、图恒取主 runtime（快档静默走主力模型）。判据：模块
        // 的 buildGraph 会被主 runtime 与视图各调一次，两次收到的 ctx.llm()
        // 不同（主档桩为 null、视图为构建出的 fast runtime 非 null）。
        java.util.Set<Object> llmIdentities = java.util.Collections.newSetFromMap(
                java.util.Collections.synchronizedMap(new java.util.IdentityHashMap<>()));
        AgentModule recording = new AgentModule() {
            @Override public String name() { return "chat"; }
            @Override public String description() { return "recording stub for profile routing"; }
            @Override public CompiledGraph buildGraph(ModuleContext ctx) {
                llmIdentities.add(ctx.llm());
                return GraphBuilder.singleNode("chat", (state, nodeCtx) -> Map.of(),
                        ctx.checkpointer());
            }
            @Override public List<io.github.paopaonb666.agentbase.core.tools.AgentTool> getTools() {
                return List.of();
            }
        };
        AgentRuntime runtime = buildRuntime(Settings.load(Map.of(
                "ENV", "development",
                "LLM_API_KEY", "sk-test",
                "LLM_FAST_BASE_URL", "http://fast.example.com/v1",
                "LLM_FAST_MODEL", "fast-model")),
                Map.of("chat", recording), new InMemoryCheckpointer(),
                null, null, null, null, null);
        AgentsRoutes routes = route(runtime);

        // main：走主 runtime 图（ctx.llm() == 注入的 null）。
        assertNotNull(invoke(routes, "chat", invokeBody("hi", "t1", null, null, null), "u1"));
        // fast：走视图图（graphOr404(view, ...) → buildGraph 收到 fast runtime）。
        assertNotNull(invoke(routes, "chat", invokeBody("hi", "t2", null, "chat", "fast"), "u1"));

        assertEquals(2, llmIdentities.size(), "main 与 fast 各建一图：" + llmIdentities.size());
        assertTrue(llmIdentities.stream().anyMatch(java.util.Objects::isNull),
                "主档图 ctx.llm() 为注入桩（null）");
        assertTrue(llmIdentities.stream().anyMatch(java.util.Objects::nonNull),
                "fast 视图图的 ctx.llm() 应为构建出的 fast runtime（非 null）");
    }

    @Test
    void truncatedAttachmentInjectionCarriesGuidance() {        // #250：截断的附件全文注入时，块尾带指路提示（模型不再把半份文档
        // 当全文回答）；未截断的不带。
        List<UploadedFileInfo> infos = List.of(
                new UploadedFileInfo("f1", "", "", "u1", "a.pdf", "pdf",
                        10, null, true, 5000, null, "前半份内容", new byte[0]),
                new UploadedFileInfo("f2", "", "", "u1", "b.txt", "txt",
                        null, null, false, 100, null, "完整内容", new byte[0]),
                new UploadedFileInfo("f3", "", "", "u1", "c.pdf", "pdf",
                        null, null, true, 0, "扫描件无文本层", "", new byte[0]));
        List<String> blocks = AgentsRoutes.attachmentBlocks(infos);
        assertEquals(3, blocks.size());
        assertTrue(blocks.get(0).contains("（注：以上内容因长度已截断，完整内容请用 knowledge_search 检索）"));
        assertFalse(blocks.get(1).contains("knowledge_search"));
        // 空提取走警告分支，不注入截断提示
        assertTrue(blocks.get(2).contains("扫描件无文本层"));
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

        Map<String, Object> out = routes.listThreads("chat", 50, 0, "u1", null, null, null);
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
        HttpError error = httpErrorOf(() -> routes.listThreads("ghost", 50, 0, "u1", null, null, null));
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
        Map<String, Object> out = routes.listThreads("chat", 50, 0, "u1", null, null, null);
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
    void steerEndpointMatrix() throws Exception {
        // #7 运行中插话端点：掩蔽 404（无在途 run/非属主）、422 校验、
        // 429 队列满、成功 queued=true 入队待步边界消费。
        AgentRuntime runtime = buildRuntime(Map.of("chat", stubModule("chat")),
                new InMemoryCheckpointer(), null, null, null, null, null);
        AgentsRoutes routes = route(runtime);

        HttpError idle = httpErrorOf(() -> routes.steerThread("chat", "t1",
                Map.of("content", "插话"), "u1", null, null, null));
        assertEquals(404, idle.status(), "空闲线程掩蔽 404");

        HttpError unknownModule = httpErrorOf(() -> routes.steerThread("nope", "t1",
                Map.of("content", "插话"), "u1", null, null, null));
        assertEquals(404, unknownModule.status());

        io.github.paopaonb666.agentbase.core.graph.RunRegistry.RunHandle handle =
                io.github.paopaonb666.agentbase.core.graph.RunRegistry.register("chat:t1", "owner-a");
        try {
            Map<String, Object> queued = routes.steerThread("chat", "t1",
                    Map.of("content", "插话"), "owner-a", null, null, null);
            assertEquals(Boolean.TRUE, queued.get("queued"));
            assertEquals("t1", queued.get("thread_id"));
            assertEquals(1, handle.pendingSteers());
            var item = handle.drain().get(0);
            assertEquals("user", item.kind());
            assertEquals("插话", item.content());

            // A-#14：kind=system 入队；非法 kind 422。
            assertEquals(Boolean.TRUE, routes.steerThread("chat", "t1",
                    Map.of("content", "提醒", "kind", "system"), "owner-a", null, null, null)
                    .get("queued"));
            assertEquals("system", handle.drain().get(0).kind());
            HttpError badKind = httpErrorOf(() -> routes.steerThread("chat", "t1",
                    Map.of("content", "插话", "kind", "bogus"), "owner-a", null, null, null));
            assertEquals(422, badKind.status());

            HttpError foreign = httpErrorOf(() -> routes.steerThread("chat", "t1",
                    Map.of("content", "插话"), "u1", null, null, null));
            assertEquals(404, foreign.status(), "非属主掩蔽 404（不泄露在途状态）");

            HttpError blank = httpErrorOf(() -> routes.steerThread("chat", "t1",
                    Map.of("content", ""), "owner-a", null, null, null));
            assertEquals(422, blank.status());
            HttpError missing = httpErrorOf(() -> routes.steerThread("chat", "t1",
                    Map.of(), "owner-a", null, null, null));
            assertEquals(422, missing.status());

            for (int i = 0; i < io.github.paopaonb666.agentbase.core.graph.RunRegistry.STEER_QUEUE_CAPACITY; i++) {
                handle.steer("user", "m" + i);
            }
            HttpError full = httpErrorOf(() -> routes.steerThread("chat", "t1",
                    Map.of("content", "插话"), "owner-a", null, null, null));
            assertEquals(429, full.status(), "队列满 429（runaway 防线）");
        } finally {
            io.github.paopaonb666.agentbase.core.graph.RunRegistry.unregister("chat:t1", handle);
        }
    }

    @Test
    void resumeEndpointMatrix() throws Exception {
        // #8 中断恢复端点：无记录/非属主 404 掩蔽、done 400、在途 409、
        // 可恢复 → SSE emitter（空输入重入图）。线程 id 独立（t10）——
        // 其他测试的真实运行会把同 id 的生命周期记录翻 done（跨测试污染）。
        AgentRuntime runtime = buildRuntime(Map.of("chat", stubModule("chat")),
                new InMemoryCheckpointer(), null, null, null, null, null);
        AgentsRoutes routes = route(runtime);
        MockHttpServletResponse response = new MockHttpServletResponse();

        HttpError none = httpErrorOf(() -> routes.resumeThread("chat", "t10", response,
                "u1", null, null, null));
        assertEquals(404, none.status(), "无生命周期记录掩蔽 404");

        HttpError unknownModule = httpErrorOf(() -> routes.resumeThread("nope", "t10", response,
                "u1", null, null, null));
        assertEquals(404, unknownModule.status());

        io.github.paopaonb666.agentbase.core.graph.RunRegistry.recordLastRun(
                "chat:t10", "owner-a", "run-x", "cancelled");
        HttpError foreign = httpErrorOf(() -> routes.resumeThread("chat", "t10", response,
                "u1", null, null, null));
        assertEquals(404, foreign.status(), "非属主掩蔽 404");

        io.github.paopaonb666.agentbase.core.graph.RunRegistry.recordLastRun(
                "chat:t10", "u1", "run-x", "done");
        HttpError done = httpErrorOf(() -> routes.resumeThread("chat", "t10", response,
                "u1", null, null, null));
        assertEquals(400, done.status(), "已完成的运行没有「继续」语义");

        // 可恢复 → emitter 非空（stub 图真实跑到终态）。
        io.github.paopaonb666.agentbase.core.graph.RunRegistry.recordLastRun(
                "chat:t10", "u1", "run-x", "cancelled");
        SseEmitter emitter = routes.resumeThread("chat", "t10", response, "u1", null, null, null);
        assertNotNull(emitter);
    }

    @Test
    void resumeBusyThreadReturns409() throws Exception {
        // 在途 run 持锁：resume 探针 409（与 invoke 同码同文案，权威判定在
        // 图执行 tryLock）。线程 id 独立（t11）。
        CountDownLatch release = new CountDownLatch(1);
        CountDownLatch entered = new CountDownLatch(1);
        AgentModule slow = new AgentModule() {
            @Override public String name() { return "chat"; }
            @Override public String description() { return "slow stub"; }
            @Override public CompiledGraph buildGraph(ModuleContext ctx) {
                return GraphBuilder.begin()
                        .addNode("slow", (state, nodeCtx) -> {
                            entered.countDown();
                            try {
                                release.await(5, java.util.concurrent.TimeUnit.SECONDS);
                            } catch (InterruptedException e) {
                                Thread.currentThread().interrupt();
                            }
                            return Map.of();
                        })
                        .entry("slow")
                        .addEdge("slow", GraphBuilder.END)
                        .compile(ctx.checkpointer(), "chat");
            }
            @Override public List<io.github.paopaonb666.agentbase.core.tools.AgentTool> getTools() {
                return List.of();
            }
        };
        AgentRuntime runtime = buildRuntime(Map.of("chat", slow),
                new InMemoryCheckpointer(), null, null, null, null, null);
        AgentsRoutes routes = route(runtime);
        io.github.paopaonb666.agentbase.core.graph.RunRegistry.recordLastRun(
                "chat:t11", "u1", "run-x", "cancelled");

        java.util.concurrent.ExecutorService pool =
                java.util.concurrent.Executors.newSingleThreadExecutor();
        try {
            var running = pool.submit(() -> routes.resumeThread("chat", "t11",
                    new MockHttpServletResponse(), "u1", null, null, null));
            assertTrue(entered.await(5, java.util.concurrent.TimeUnit.SECONDS),
                    "resume run 应已进入慢节点（持线程锁）");
            MockHttpServletResponse second = new MockHttpServletResponse();
            HttpError busy = httpErrorOf(() -> routes.resumeThread("chat", "t11", second,
                    "u1", null, null, null));
            assertEquals(409, busy.status());
            assertNotNull(running.get(5, java.util.concurrent.TimeUnit.SECONDS));
        } finally {
            release.countDown();
            pool.shutdownNow();
        }
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
        // 同步驱动同一张图写快照（等价于正常完成的一轮对话）。#197 起同
        // thread run 互斥：invoke 触发的异步 producer 与本线程存在获取锁的
        // 竞态（无论谁先），撞 ThreadBusy 就让位重试。
        CompiledGraph graph = runtime.graph("chat");
        io.github.paopaonb666.agentbase.core.graph.GraphState driven = null;
        for (int i = 0; i < 200 && driven == null; i++) {
            try {
                driven = graph.invoke(
                        new io.github.paopaonb666.agentbase.core.graph.GraphState(
                                List.of(dev.langchain4j.data.message.UserMessage.from("第一条消息")), Map.of()),
                        new io.github.paopaonb666.agentbase.core.graph.RunConfig("chat:中文线程", "u1", 25));
            } catch (CompiledGraph.ThreadBusyError e) {
                Thread.sleep(25);
            }
        }
        assertNotNull(driven, "同步驱动应在 producer 让出锁后成功");

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

    /** 记忆端口桩：purgeThread 记录两参重载的 purgeMemories 实参（keep/purge）。 */
    private static Object purgeRecordingMemoryProxy(List<String> calls) {
        return Proxy.newProxyInstance(
                AgentModule.MemoryServicePort.class.getClassLoader(),
                new Class<?>[]{AgentModule.MemoryServicePort.class},
                (proxy, method, args) -> {
                    if (method.getDeclaringClass() == Object.class) {
                        return method.invoke(proxy, args);
                    }
                    if ("purgeThread".equals(method.getName())) {
                        boolean purgeMemories = args != null && args.length == 2
                                && Boolean.TRUE.equals(args[1]);
                        calls.add("memory:purgeThread(" + (purgeMemories ? "purge" : "keep") + ")");
                    } else {
                        calls.add("memory:" + method.getName());
                    }
                    return defaultValue(method.getReturnType());
                });
    }

    @Test
    void deleteThreadPurgeMemoriesDefaultsToKeepAndOptInPurges() throws Exception {
        // #172：删线程默认保留跨会话记忆（purge_memories 缺省 false）；
        // "true"/"1" 才透传给 memory.purgeThread 两参重载（连 memories 一起
        // 删），其余值与缺省一律视为 false。级联顺序不变（#213 基线）。
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
        Object memory = purgeRecordingMemoryProxy(calls);
        AgentRuntime runtime = buildRuntime(Map.of("chat", stubModule("chat")),
                cp, index, fileStore, memory, null, null);
        AgentsRoutes routes = route(runtime);

        // 缺省（六参兼容形态）：保留记忆。
        routes.deleteThread("chat", "t1", "u1", null, null, null);
        assertEquals(List.of(
                "index:getThreadIndex",
                "cp:deleteThread",
                "files:deleteForThread",
                "index:deleteThreadIndex",
                "index:deletePlan",
                "memory:purgeThread(keep)"), calls);

        // purge_memories=true / "1"：透传 true，级联顺序不变。
        for (String raw : new String[]{"true", "1"}) {
            calls.clear();
            routes.deleteThread("chat", "t1", "u1", null, null, null, raw);
            assertEquals(List.of(
                    "index:getThreadIndex",
                    "cp:deleteThread",
                    "files:deleteForThread",
                    "index:deleteThreadIndex",
                    "index:deletePlan",
                    "memory:purgeThread(purge)"), calls, "purge_memories=" + raw);
        }

        // 其余值不误触发（仍是保留语义）。
        calls.clear();
        routes.deleteThread("chat", "t1", "u1", null, null, null, "yes");
        assertEquals("memory:purgeThread(keep)", calls.get(calls.size() - 1));
    }

    @Test
    void invokeClaimsThreadIndexExactlyOnce() throws Exception {
        // #141 语义保持 + P0 认领化：索引写入收敛到 startStream 单次
        // claimThreadIndex——预检的 getThreadIndex 是读不算写，旧 upsert
        // 路径已移除。
        List<String> calls = new ArrayList<>();
        Object index = Proxy.newProxyInstance(
                io.github.paopaonb666.agentbase.core.storage.CoreStore.class.getClassLoader(),
                new Class<?>[]{io.github.paopaonb666.agentbase.core.storage.CoreStore.class},
                (proxy, method, args) -> {
                    calls.add(method.getName());
                    if ("claimThreadIndex".equals(method.getName())) {
                        // 权威行 = 入参行（预检已确保属主一致或缺席）。
                        return args[0];
                    }
                    return defaultValue(method.getReturnType());
                });
        AgentRuntime runtime = buildRuntime(Map.of("chat", stubModule("chat")),
                new InMemoryCheckpointer(), index, null, null, null, null);
        AgentsRoutes routes = route(runtime);

        invoke(routes, "chat", invokeBody("第一条消息", "t1", null, null, null), "u1");
        assertEquals(1, calls.stream().filter("claimThreadIndex"::equals).count(),
                "一次 invoke 恰一次 claim：" + calls);
        assertFalse(calls.contains("upsertThreadIndex"), "旧 upsert 路径应已移除：" + calls);
    }

    @Test
    void invokeMasksForeignThreadsAndSkipsSideEffects() throws Exception {
        // P0 属主校验：他人拿既有 thread_id invoke → 404 掩蔽，且零副作用
        // ——不触发 claim（索引写入）、不绑定附件、不进图（对应 Python
        // test_invoke_rejects_foreign_thread）。缺席行放行（新线程首发）、
        // 属主放行。
        List<String> calls = new ArrayList<>();
        Object index = Proxy.newProxyInstance(
                io.github.paopaonb666.agentbase.core.storage.CoreStore.class.getClassLoader(),
                new Class<?>[]{io.github.paopaonb666.agentbase.core.storage.CoreStore.class},
                (proxy, method, args) -> {
                    calls.add(method.getName());
                    if ("getThreadIndex".equals(method.getName())) {
                        return "chat:t1".equals(args[0])
                                ? ThreadIndexRecord.create("chat:t1", "owner-a", "chat", "标题")
                                : null;
                    }
                    if ("claimThreadIndex".equals(method.getName())) {
                        return args[0];
                    }
                    return defaultValue(method.getReturnType());
                });
        Object fileStore = recordingProxy(
                io.github.paopaonb666.agentbase.extensions.filestore.UploadedFileStore.class,
                calls, "files");
        AgentRuntime runtime = buildRuntime(Map.of("chat", stubModule("chat")),
                new InMemoryCheckpointer(), index, fileStore, null, null, null);
        AgentsRoutes routes = route(runtime);

        HttpError foreign = httpErrorOf(() ->
                invoke(routes, "chat",
                        invokeBody("hi", "t1", List.of("a"), null, null), "u1"));
        assertEquals(404, foreign.status());
        assertTrue(foreign.getMessage().contains("线程不存在"), foreign.getMessage());
        // 零副作用：预检之后没有发生任何索引写入或附件绑定。
        assertFalse(calls.contains("claimThreadIndex"), String.valueOf(calls));
        assertFalse(calls.stream().anyMatch(c -> c.startsWith("files:")), String.valueOf(calls));

        // 缺席行放行（新线程首发），缺席行的 getThreadIndex 返回 null。
        assertNotNull(invoke(routes, "chat", invokeBody("hi", "fresh", null, null, null), "u1"));
        // 属主本人放行。
        assertNotNull(invoke(routes, "chat", invokeBody("hi", "t1", null, null, null), "owner-a"));
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
                prodRoutes.listThreads("chat", 50, 0, "u1", null, null, null));
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
                providerOf(null)).listThreads("chat", 50, 0, "u1", null, null, null);
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> threads = (List<Map<String, Object>>) out.get("threads");
        assertEquals(1, threads.size());
        assertEquals("t1", threads.get(0).get("thread_id"));
    }

    @Test
    void concurrentInvokeOnSameThreadGives409() throws Exception {
        // #197：同 thread 并发 invoke → 409（探针在响应提交前报错），
        // 不同 thread 不受影响。
        CountDownLatch entered = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        AgentModule blocking = new AgentModule() {
            @Override public String name() { return "chat"; }
            @Override public String description() { return "blocking stub"; }
            @Override public io.github.paopaonb666.agentbase.core.graph.CompiledGraph buildGraph(
                    AgentModule.ModuleContext ctx) {
                return io.github.paopaonb666.agentbase.core.graph.GraphBuilder.begin()
                        .addNode("agent", (state, nodeCtx) -> {
                            entered.countDown();
                            release.await();
                            return Map.of();
                        })
                        .entry("agent")
                        .addEdge("agent", io.github.paopaonb666.agentbase.core.graph.GraphBuilder.END)
                        .compile(ctx.checkpointer(), "chat");
            }
            @Override public List<io.github.paopaonb666.agentbase.core.tools.AgentTool> getTools() {
                return List.of();
            }
        };
        AgentRuntime runtime = buildRuntime(Map.of("chat", blocking),
                new InMemoryCheckpointer(), null, null, null, null, null);
        AgentsRoutes routes = route(runtime);

        // 在途 run：直接驱动图（与生产一致的 CompiledGraph 实例）。
        Thread busy = new Thread(() -> runtime.graph("chat").invoke(
                new io.github.paopaonb666.agentbase.core.graph.GraphState(
                        List.of(dev.langchain4j.data.message.UserMessage.from("x")), Map.of()),
                new io.github.paopaonb666.agentbase.core.graph.RunConfig("chat:t1", "u1", 25)));
        busy.start();
        assertTrue(entered.await(5, java.util.concurrent.TimeUnit.SECONDS));

        HttpError conflict = httpErrorOf(() ->
                invoke(routes, "chat", invokeBody("hi", "t1", null, null, null), "u1"));
        assertEquals(409, conflict.status());
        assertTrue(conflict.getMessage().contains("线程忙"), conflict.getMessage());

        release.countDown();
        busy.join(5000);
        // 释放后同 thread 可再次 invoke（不再 409）。
        assertNotNull(invoke(routes, "chat", invokeBody("hi", "t1", null, null, null), "u1"));
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
