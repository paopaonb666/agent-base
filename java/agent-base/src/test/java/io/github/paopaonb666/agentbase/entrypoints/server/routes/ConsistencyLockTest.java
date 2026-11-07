package io.github.paopaonb666.agentbase.entrypoints.server.routes;

import io.github.paopaonb666.agentbase.Version;
import io.github.paopaonb666.agentbase.core.AgentModule;
import io.github.paopaonb666.agentbase.core.AgentRuntime;
import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.core.graph.CompiledGraph;
import io.github.paopaonb666.agentbase.core.graph.InMemoryCheckpointer;
import io.github.paopaonb666.agentbase.entrypoints.server.AuthBackends;
import io.github.paopaonb666.agentbase.entrypoints.server.HttpError;
import io.github.paopaonb666.agentbase.entrypoints.server.WebFilters;
import io.github.paopaonb666.agentbase.extensions.Observability;
import io.github.paopaonb666.agentbase.extensions.costmeter.CostGovernor;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.mock.web.MockHttpServletRequest;
import org.springframework.mock.web.MockHttpServletResponse;

import java.lang.reflect.Constructor;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;

/**
 * 一致性锁定网（roadmap #307，#124/#151/#158 并入）：把既有裁决与佐证
 * 确认（无改动实现）的行为钉成回归网——/v1/modules 的响应形状（#219
 * 佐证确认的唯一 Wave-4 项）、409 忙闲的 Retry-After 头值、X-Request-ID
 * 的回显/再生/回写契约（S1 可审计链路）、版本号形态、保留模块名。
 * 文案级的锁定散在各矩阵测试（Auth/工具回执/端点矩阵）中，这里不重复。
 */
class ConsistencyLockTest {

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

    private static AgentRuntime runtime(Map<String, AgentModule> modules) throws Exception {
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
                null, new InMemoryCheckpointer(), null, null, null, null, null);
    }

    private static AgentModule moduleReturning(CompiledGraph graph) {
        return new AgentModule() {
            @Override public String name() { return "chat"; }
            @Override public String description() { return "chat 模块（一致性锁定桩）"; }
            @Override public CompiledGraph buildGraph(AgentModule.ModuleContext ctx) { return graph; }
            @Override public List<io.github.paopaonb666.agentbase.core.tools.AgentTool> getTools() {
                return List.of();
            }
        };
    }

    // ── #219 佐证锁定：/v1/modules 响应形状 ──────────────────────────────

    @Test
    void modulesEndpointReturnsNameDescriptionWithSupervisorAppended() throws Exception {
        AgentRuntime runtime = runtime(Map.of("chat", moduleReturning(Mockito.mock(CompiledGraph.class))));
        SystemRoutes routes = new SystemRoutes(runtime, providerOf(null));

        @SuppressWarnings("unchecked")
        Map<String, Object> body = (Map<String, Object>) routes.listModules();
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> modules = (List<Map<String, Object>>) body.get("modules");
        assertEquals(2, modules.size());
        // 每个条目都是 name+description 二键视图（缺 description 即前端锚点失效）。
        for (Map<String, Object> entry : modules) {
            assertEquals(Map.of("name", entry.get("name"), "description", entry.get("description")), entry);
        }
        assertEquals("chat", modules.get(0).get("name"));
        // supervisor 追加在尾部，文案锁定（前端模块清单的固定最后一项）。
        assertEquals(AgentRuntime.SUPERVISOR_MODULE, modules.get(1).get("name"));
        assertEquals("supervisor", AgentRuntime.SUPERVISOR_MODULE);
        assertEquals("多 Agent 协作：把所有已注册模块编排为 sub-agent", modules.get(1).get("description"));
    }

    // ── 409 忙闲：Retry-After 头值 ───────────────────────────────────────

    @Test
    void threadBusyInvokeSetsRetryAfterFive() throws Exception {
        CompiledGraph busy = Mockito.mock(CompiledGraph.class);
        when(busy.isThreadBusy(any())).thenReturn(true);
        AgentRuntime runtime = runtime(Map.of("chat", moduleReturning(busy)));
        AgentsRoutes routes = new AgentsRoutes(runtime, PASS_THROUGH_AUTH, providerOf(null));

        MockHttpServletResponse response = new MockHttpServletResponse();
        HttpError error = org.junit.jupiter.api.Assertions.assertThrows(HttpError.class,
                () -> routes.invoke("chat",
                        new AgentsRoutes.InvokeRequest("你好", "t1", null, "chat", "main"),
                        response, "alice", null, null, null));
        assertEquals(409, error.status());
        assertEquals("5", response.getHeader("Retry-After"), "客户端重试节奏锁定为 5s");
        assertTrue(error.getMessage().contains("线程忙"));
    }

    // ── X-Request-ID 契约（S1 可审计链路） ───────────────────────────────

    @Test
    void requestIdFilterEchoesValidRegeneratesInvalidAndRestoresMdc() throws Exception {
        WebFilters.RequestIdFilter filter = new WebFilters.RequestIdFilter();

        // 合法形态原样回显：响应头 + 链内 MDC 都是调用方给的 id。
        MockHttpServletRequest valid = new MockHttpServletRequest("GET", "/health");
        valid.addHeader("X-Request-ID", "abc-123_DEF.456");
        MockHttpServletResponse validResponse = new MockHttpServletResponse();
        String[] seenInChain = new String[1];
        filter.doFilter(valid, validResponse,
                (req, resp) -> seenInChain[0] = Observability.requestId());
        assertEquals("abc-123_DEF.456", validResponse.getHeader("X-Request-ID"));
        assertEquals("abc-123_DEF.456", seenInChain[0]);
        assertNull(org.slf4j.MDC.get(Observability.REQUEST_ID_KEY), "链外 MDC 已卸载");

        // 非法形态（含空格/超长）再生 12 位 hex，不回显攻击载荷。
        MockHttpServletRequest bad = new MockHttpServletRequest("GET", "/health");
        bad.addHeader("X-Request-ID", "bad id!!.." + "x".repeat(70));
        MockHttpServletResponse badResponse = new MockHttpServletResponse();
        String[] regenerated = new String[1];
        filter.doFilter(bad, badResponse,
                (req, resp) -> regenerated[0] = Observability.requestId());
        assertTrue(regenerated[0].matches("[0-9a-f]{12}"), regenerated[0]);
        assertEquals(regenerated[0], badResponse.getHeader("X-Request-ID"));

        // 缺头 → 新 id；链外已有 MDC（嵌套场景）→ 处理后还原旧值。
        MockHttpServletRequest bare = new MockHttpServletRequest("GET", "/health");
        MockHttpServletResponse bareResponse = new MockHttpServletResponse();
        org.slf4j.MDC.put(Observability.REQUEST_ID_KEY, "outer-id");
        try {
            filter.doFilter(bare, bareResponse, (req, resp) -> { });
            assertEquals("outer-id", org.slf4j.MDC.get(Observability.REQUEST_ID_KEY));
        } finally {
            org.slf4j.MDC.remove(Observability.REQUEST_ID_KEY);
        }
        // 缺头且链外无 MDC → 处理后 MDC 干净。
        filter.doFilter(new MockHttpServletRequest("GET", "/health"), new MockHttpServletResponse(),
                (req, resp) -> { });
        assertNull(org.slf4j.MDC.get(Observability.REQUEST_ID_KEY));
        assertNotEquals("bad id!!", regenerated[0]);
    }

    // ── 版本与请求 id 形态 ───────────────────────────────────────────────

    @Test
    void versionAndRequestIdFormatsAreStable() {
        // 语义化版本三段式（运维脚本/发版清单依赖）。
        assertTrue(Version.VERSION.matches("\\d+\\.\\d+\\.\\d+"), Version.VERSION);
        // 请求 id：12 位小写 hex（日志聚合与审计跨语言对齐）。
        for (int i = 0; i < 20; i++) {
            assertTrue(Observability.newRequestId().matches("[0-9a-f]{12}"));
        }
    }
}
