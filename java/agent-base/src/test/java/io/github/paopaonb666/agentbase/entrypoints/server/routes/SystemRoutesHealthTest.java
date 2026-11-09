package io.github.paopaonb666.agentbase.entrypoints.server.routes;

import io.github.paopaonb666.agentbase.core.AgentModule;
import io.github.paopaonb666.agentbase.core.AgentRuntime;
import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.extensions.costmeter.CostGovernor;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.springframework.beans.factory.ObjectProvider;

import java.lang.reflect.Constructor;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * /health 渲染矩阵（roadmap #294）：组件映射（checkpointer 三态、model
 * 四态、memory 可选、cost 可选）、degraded 判定与组件序。运行时经反射
 * 构造（包私有构造器），探活走 SystemRoutes.probeModel 的包内重载预热
 * TTL 缓存——health() 命中缓存后不打真实网络。
 */
class SystemRoutesHealthTest {

    // ── 桩基建 ────────────────────────────────────────────────────────────

    private static AgentRuntime buildRuntime(Settings settings, Object checkpointer,
                                             Object memory) throws Exception {
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
        return ctor.newInstance(settings, null, Map.of(), null, checkpointer, null,
                null, memory, null, null);
    }

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

    @SuppressWarnings("unchecked")
    private static SystemRoutes routes(AgentRuntime runtime, CostGovernor governor) {
        return new SystemRoutes(runtime, providerOf(governor));
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> healthOf(AgentRuntime runtime, CostGovernor governor) {
        return (Map<String, Object>) routes(runtime, governor).health();
    }

    @SuppressWarnings("unchecked")
    private static Map<String, String> components(Map<String, Object> health) {
        return (Map<String, String>) health.get("components");
    }

    private static Settings settingsWith(String apiKey, String baseUrl, boolean probe) {
        return Settings.load(java.util.Map.of(
                "ENV", "development",
                "LLM_API_KEY", apiKey,
                "LLM_BASE_URL", baseUrl,
                "HEALTH_PROBE_MODEL", probe ? "true" : "false"), null);
    }

    // ── 组件矩阵 ──────────────────────────────────────────────────────────

    @Test
    void checkpointerComponentCoversThreeStates() throws Exception {
        // 缺席 ≠ 故障：未配置 checkpointer 报 unconfigured（仍算 degraded？——
        // unconfigured 不是 "ok"，整体状态转 degraded，这是刻意的）。
        Settings s = settingsWith("", "http://health-unconfigured", false);
        Map<String, String> c = components(healthOf(buildRuntime(s, null, null), null));
        assertEquals("unconfigured", c.get("checkpointer"));
        assertEquals("degraded", healthOf(buildRuntime(s, null, null), null).get("status"));

        var healthy = Mockito.mock(io.github.paopaonb666.agentbase.core.graph.Checkpointer.class);
        Mockito.when(healthy.healthy()).thenReturn(true);
        assertEquals("ok", components(
                healthOf(buildRuntime(s, healthy, null), null)).get("checkpointer"));

        var sick = Mockito.mock(io.github.paopaonb666.agentbase.core.graph.Checkpointer.class);
        Mockito.when(sick.healthy()).thenReturn(false);
        Map<String, Object> sickHealth = healthOf(buildRuntime(s, sick, null), null);
        assertEquals("error", components(sickHealth).get("checkpointer"));
        assertEquals("degraded", sickHealth.get("status"));
    }

    @Test
    void modelComponentCoversUnconfiguredMisconfiguredAndProbe() throws Exception {
        // 未配 apiKey → unconfigured（探活也不该打）。
        Settings bare = settingsWith("", "http://health-bare-" + System.nanoTime(), true);
        Map<String, String> c = components(healthOf(buildRuntime(bare, null, null), null));
        assertEquals("unconfigured", c.get("model"));

        // baseUrl 非 http(s) → misconfigured（明示配置错误而不是探活失败）。
        Settings wrong = settingsWith("sk-x", "not-a-url", true);
        assertEquals("misconfigured", components(
                healthOf(buildRuntime(wrong, null, null), null)).get("model"));

        // 探活开启：先用包内重载预热 TTL 缓存（假 fetcher 应答 2xx+data 数组），
        // health() 命中缓存 → ok，不打真实网络。
        String url = "http://health-probe-" + System.nanoTime();
        Settings probed = settingsWith("sk-x", url, true);
        var fetcher = new SystemRoutes.ModelsFetcher() {
            @Override
            public java.net.http.HttpResponse<String> fetch(String u, String apiKey) {
                var r = Mockito.mock(java.net.http.HttpResponse.class);
                Mockito.when(r.statusCode()).thenReturn(200);
                Mockito.when(r.body()).thenReturn("{\"data\": []}");
                return r;
            }
        };
        assertEquals("ok", SystemRoutes.probeModel(probed, fetcher));
        var healthyCp = Mockito.mock(io.github.paopaonb666.agentbase.core.graph.Checkpointer.class);
        Mockito.when(healthyCp.healthy()).thenReturn(true);
        AgentRuntime runtime = buildRuntime(probed, healthyCp, null);
        Map<String, Object> health = healthOf(runtime, null);
        assertEquals("ok", components(health).get("model"));

        // 全组件 ok（memory/cost 缺席不算降级）→ status ok。
        assertEquals("ok", health.get("status"));

        // 探活关闭（默认）：不碰网络，直接 ok。
        Settings noProbe = settingsWith("sk-x", "http://health-noprobe-" + System.nanoTime(), false);
        assertEquals("ok", components(
                healthOf(buildRuntime(noProbe, null, null), null)).get("model"));
    }

    @Test
    void memoryComponentOnlyWhenServiceAttached() throws Exception {
        Settings s = settingsWith("", "http://health-mem-" + System.nanoTime(), false);
        // 服务缺席：组件不出现（缺席 ≠ 降级）。
        Map<String, String> absent = components(healthOf(buildRuntime(s, null, null), null));
        assertFalse(absent.containsKey("memory"));

        // 探针 ok 不改变整体；degraded 拖垮整体状态。
        AgentModule.MemoryServicePort okMemory =
                Mockito.mock(AgentModule.MemoryServicePort.class);
        Mockito.when(okMemory.healthProbe()).thenReturn("ok");
        Map<String, Object> okHealth = healthOf(buildRuntime(s, null, okMemory), null);
        assertEquals("ok", components(okHealth).get("memory"));

        AgentModule.MemoryServicePort sickMemory =
                Mockito.mock(AgentModule.MemoryServicePort.class);
        Mockito.when(sickMemory.healthProbe()).thenReturn("degraded");
        Map<String, Object> sickHealth = healthOf(buildRuntime(s, null, sickMemory), null);
        assertEquals("degraded", components(sickHealth).get("memory"));
        assertEquals("degraded", sickHealth.get("status"));
    }

    @Test
    void costGovernorOnlyWhenPresentAndCanDegrade() throws Exception {
        Settings s = settingsWith("", "http://health-cost-" + System.nanoTime(), false);
        // governor 缺席（COST_ENABLED=false）：组件不出现。
        assertFalse(components(healthOf(buildRuntime(s, null, null), null)).containsKey("cost"));

        // blocked → 组件报 blocked 且整体 degraded。
        CostGovernor blocked = Mockito.mock(CostGovernor.class);
        Mockito.when(blocked.status()).thenReturn("blocked");
        Map<String, Object> blockedHealth = healthOf(buildRuntime(s, null, null), blocked);
        assertEquals("blocked", components(blockedHealth).get("cost"));
        assertEquals("degraded", blockedHealth.get("status"));

        // warn 同样拖垮整体（放行但可见）。
        CostGovernor warn = Mockito.mock(CostGovernor.class);
        Mockito.when(warn.status()).thenReturn("warn");
        assertEquals("degraded", healthOf(buildRuntime(s, null, null), warn).get("status"));

        // ok governor 不降级（其余组件 unconfigured 仍会 degraded，只看 cost 键）。
        CostGovernor ok = Mockito.mock(CostGovernor.class);
        Mockito.when(ok.status()).thenReturn("ok");
        assertEquals("ok", components(healthOf(buildRuntime(s, null, null), ok)).get("cost"));
    }

    @Test
    void componentOrderIsStableForHumans() throws Exception {
        // 渲染顺序锁定（LinkedHashMap）：checkpointer → model → memory → cost
        // → pools（#81 观测数值段恒在尾部）。
        Settings s = settingsWith("", "http://health-order-" + System.nanoTime(), false);
        var checkpointer = Mockito.mock(io.github.paopaonb666.agentbase.core.graph.Checkpointer.class);
        Mockito.when(checkpointer.healthy()).thenReturn(true);
        AgentModule.MemoryServicePort memory =
                Mockito.mock(AgentModule.MemoryServicePort.class);
        Mockito.when(memory.healthProbe()).thenReturn("ok");
        CostGovernor governor = Mockito.mock(CostGovernor.class);
        Mockito.when(governor.status()).thenReturn("ok");
        Map<String, String> c = components(healthOf(buildRuntime(s, checkpointer, memory), governor));
        assertEquals("[checkpointer, model, memory, cost, pools]", c.keySet().toString());
        assertTrue(c.containsKey("memory"));
    }

    @Test
    void poolsComponentReportsGaugesWithoutFlippingStatus() throws Exception {
        // #81：池水位是观测数值段（tool 在飞/排队/泄漏 + bg_pending +
        // cost_dropped），不参与 ok/degraded 判定；tools 缺席时省 tool 段。
        Settings s = settingsWith("", "http://health-pools", false);
        Map<String, String> c = components(healthOf(buildRuntime(s, null, null), null));
        String pools = c.get("pools");
        assertTrue(pools != null && pools.startsWith("bg_pending:0")
                && pools.contains("cost_dropped:0"), String.valueOf(pools));
        assertFalse(pools.contains("tool:"), "tools 缺席应省 tool 段：" + pools);

        // 带真身工具池 + 健康 checkpointer + 已配 model：tool 段出现且数值为
        // 运行计数；其余组件全 ok 时整体状态仍 ok（pools 不参与判定）。
        var pool = io.github.paopaonb666.agentbase.core.tools.ToolPool.build(
                java.util.List.of(), java.util.List.of(), 30.0, java.util.Map.of(), null);
        var checkpointer = Mockito.mock(io.github.paopaonb666.agentbase.core.graph.Checkpointer.class);
        Mockito.when(checkpointer.healthy()).thenReturn(true);
        Settings allOkSettings = settingsWith("sk-test", "http://health-pools-ok", false);
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
        AgentRuntime withPool = ctor.newInstance(allOkSettings, null, Map.of(), pool, checkpointer, null,
                null, null, null, null);
        Map<String, Object> health = healthOf(withPool, null);
        String withTool = components(health).get("pools");
        assertTrue(withTool.startsWith("tool:0/0,leaked:0,"), withTool);
        assertEquals("ok", health.get("status"), "pools 数值段不得翻转整体状态");
    }
}
