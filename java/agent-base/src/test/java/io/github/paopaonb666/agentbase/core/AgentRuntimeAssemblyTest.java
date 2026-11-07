package io.github.paopaonb666.agentbase.core;

import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.core.graph.CompiledGraph;
import io.github.paopaonb666.agentbase.core.graph.GraphBuilder;
import io.github.paopaonb666.agentbase.core.graph.InMemoryCheckpointer;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.verify;

/**
 * AgentRuntime 装配面测试（roadmap #295）：保留名守门、graph() 的未知
 * 模块报错文案/缓存语义、close() 的资源收敛（含全 null 安全）、profileView
 * 的 main/fast/未知三态与快档视图共享面。构造器包私有——core 包测试直连。
 */
class AgentRuntimeAssemblyTest {

    // ── 桩基建 ────────────────────────────────────────────────────────────

    /** 可计数的桩模块：buildGraph 调用次数进 counter。 */
    private record CountingModule(String moduleName, AtomicInteger counter) implements AgentModule {

        @Override
        public String name() {
            return moduleName;
        }

        @Override
        public String description() {
            return "counting stub module";
        }

        @Override
        public CompiledGraph buildGraph(ModuleContext ctx) {
            counter.incrementAndGet();
            return GraphBuilder.singleNode(moduleName, (state, nodeCtx) -> Map.of(),
                    ctx.checkpointer() != null ? ctx.checkpointer() : null);
        }

        @Override
        public List<io.github.paopaonb666.agentbase.core.tools.AgentTool> getTools() {
            return List.of();
        }
    }

    private static AgentRuntime runtime(Settings settings, Map<String, AgentModule> modules,
                                        Object checkpointer, Object tools, Object recorder,
                                        Object fileStore, Object memory, Object threadIndex) {
        return new AgentRuntime(settings, null, modules,
                (io.github.paopaonb666.agentbase.core.tools.ToolPool) tools,
                (io.github.paopaonb666.agentbase.core.graph.Checkpointer) checkpointer,
                (io.github.paopaonb666.agentbase.extensions.toollog.ToolCallRecorder) recorder,
                (io.github.paopaonb666.agentbase.extensions.filestore.UploadedFileStore) fileStore,
                (AgentModule.MemoryServicePort) memory,
                (io.github.paopaonb666.agentbase.core.storage.CoreStore) threadIndex,
                null);
    }

    private static Settings devSettings() {
        return Settings.load(Map.of("ENV", "development"), null);
    }

    // ── 保留名与未知模块 ──────────────────────────────────────────────────

    @Test
    void supervisorIsAReservedModuleName() {
        Map<String, AgentModule> modules = Map.of(AgentRuntime.SUPERVISOR_MODULE,
                new CountingModule(AgentRuntime.SUPERVISOR_MODULE, new AtomicInteger()));
        ModuleRegistry.RegistryError e = assertThrows(ModuleRegistry.RegistryError.class,
                () -> AgentRuntime.validateModuleNames(modules));
        assertTrue(e.getMessage().contains("reserved"));
    }

    @Test
    void unknownModuleErrorListsSortedAvailableNames() {
        AgentRuntime runtime = runtime(devSettings(),
                Map.of("writer", new CountingModule("writer", new AtomicInteger()),
                        "chat", new CountingModule("chat", new AtomicInteger())),
                null, null, null, null, null, null);
        AgentRuntime.UnknownModuleError e = assertThrows(AgentRuntime.UnknownModuleError.class,
                () -> runtime.graph("planner"));
        // available 按 sorted 排序，supervisor 追加在尾部。
        assertTrue(e.getMessage().contains("available: chat, writer, supervisor"), e.getMessage());
        assertTrue(e.getMessage().contains("Add it to AGENT_MODULES"));
    }

    @Test
    void emptyModuleListFallsBackToNoneInErrorText() {
        AgentRuntime runtime = runtime(devSettings(), Map.of(), null, null, null, null, null, null);
        AgentRuntime.UnknownModuleError e = assertThrows(AgentRuntime.UnknownModuleError.class,
                () -> runtime.graph("chat"));
        assertTrue(e.getMessage().contains("available: (none)"), e.getMessage());
    }

    // ── 图缓存与共享池覆写 ────────────────────────────────────────────────

    @Test
    void moduleGraphsAreBuiltOnceAndCached() {
        AtomicInteger builds = new AtomicInteger();
        AgentRuntime runtime = runtime(devSettings(),
                Map.of("m1", new CountingModule("m1", builds)),
                new InMemoryCheckpointer(), null, null, null, null, null);
        CompiledGraph first = runtime.graph("m1");
        CompiledGraph second = runtime.graph("m1");
        assertSame(first, second);
        assertEquals(1, builds.get(), "第二次 graph() 命中缓存，buildGraph 不再调用");
    }

    @Test
    void supervisorGraphIsBuiltLazilyAndCached() {
        AgentRuntime runtime = runtime(devSettings(),
                Map.of("chat", new CountingModule("chat", new AtomicInteger())),
                new InMemoryCheckpointer(), null, null, null, null, null);
        CompiledGraph first = runtime.supervisorGraph();
        assertSame(first, runtime.supervisorGraph());
        assertSame(first, runtime.graph(AgentRuntime.SUPERVISOR_MODULE));
    }

    // ── close() 资源收敛 ──────────────────────────────────────────────────

    @Test
    void closeReleasesEveryAttachedResource() {
        var tools = Mockito.mock(io.github.paopaonb666.agentbase.core.tools.ToolPool.class);
        Mockito.when(tools.registeredTools()).thenReturn(List.of());
        var checkpointer = Mockito.mock(io.github.paopaonb666.agentbase.core.graph.Checkpointer.class);
        var recorder = Mockito.mock(io.github.paopaonb666.agentbase.extensions.toollog.ToolCallRecorder.class);
        var fileStore = Mockito.mock(io.github.paopaonb666.agentbase.extensions.filestore.UploadedFileStore.class);
        var memory = Mockito.mock(AgentModule.MemoryServicePort.class);
        var threadIndex = Mockito.mock(io.github.paopaonb666.agentbase.core.storage.CoreStore.class);

        AgentRuntime runtime = runtime(devSettings(), Map.of(), checkpointer, tools,
                recorder, fileStore, memory, threadIndex);
        runtime.close();

        // 全部收敛，一个不落（基座存储与记忆存储相互独立，两个都要关）。
        verify(tools).close();
        verify(checkpointer).close();
        verify(recorder).close();
        verify(fileStore).close();
        verify(memory).close();
        verify(threadIndex).close();
    }

    @Test
    void closeIsSafeWhenNothingIsAttached() {
        // 组件全 null（最小装配/降级部署）：close() 不得抛。
        runtime(devSettings(), Map.of(), null, null, null, null, null, null).close();
    }

    // ── profileView（成本治理 T1.2） ──────────────────────────────────────

    @Test
    void mainProfileReturnsSelfAndUnknownProfileIsRejected() {
        AgentRuntime runtime = runtime(devSettings(), Map.of(), null, null, null, null, null, null);
        assertSame(runtime, runtime.profileView("main"));
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> runtime.profileView("turbo"));
        assertTrue(e.getMessage().contains("expected 'main' or 'fast'"));
    }

    @Test
    void fastProfileReturnsNullWhenUnconfigured() {
        // 快档是优化不是依赖：LLM_FAST_* 未配置 → null（调用方显式处理）。
        AgentRuntime runtime = runtime(devSettings(), Map.of(), null, null, null, null, null, null);
        assertNull(runtime.profileView("fast"));
    }

    @Test
    void fastProfileSharesStateButOwnsItsLlm() {
        Settings settings = Settings.load(Map.of(
                "ENV", "development",
                "LLM_FAST_API_KEY", "sk-fast",
                "LLM_FAST_BASE_URL", "http://llm-fast.invalid/v1",
                "LLM_FAST_MODEL", "fast-model"), null);
        var checkpointer = new InMemoryCheckpointer();
        var memory = Mockito.mock(AgentModule.MemoryServicePort.class);
        var modules = Map.<String, AgentModule>of("chat", new CountingModule("chat", new AtomicInteger()));
        AgentRuntime runtime = runtime(settings, modules, checkpointer, null, null, null, memory, null);

        AgentRuntime view = runtime.profileView("fast");
        assertNotSame(runtime, view);
        // 线程历史跨档连续：checkpointer/memory 共享。
        assertSame(checkpointer, view.checkpointer());
        assertSame(memory, view.memory());
        // LLM 独立（快档自己的 runtime），且第二次取的是缓存视图。
        assertNotSame(runtime.llm(), view.llm());
        assertSame(view, runtime.profileView("fast"));
        // 模块清单共享（视图只换 LLM 与图缓存）。
        assertSame(runtime.modules(), view.modules());
    }
}
