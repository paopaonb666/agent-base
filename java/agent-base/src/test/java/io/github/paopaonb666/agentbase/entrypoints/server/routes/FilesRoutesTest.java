package io.github.paopaonb666.agentbase.entrypoints.server.routes;

import io.github.paopaonb666.agentbase.core.AgentModule;
import io.github.paopaonb666.agentbase.core.AgentRuntime;
import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.entrypoints.server.AuthBackends;
import io.github.paopaonb666.agentbase.entrypoints.server.HttpError;
import org.junit.jupiter.api.Test;
import org.springframework.mock.web.MockMultipartFile;

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
 * FilesRoutes 上传端点测试（roadmap #144）：413 检查点在流式读取中
 * 边读边判——声明 size 与实际不符（multipart 声明值不可信）时超限内容
 * 不再整读进内存；空上传 400；正常路径元信息完整。
 */
class FilesRoutesTest {

    private static final AuthBackends.AuthBackend PASS_THROUGH_AUTH =
            (userId, sig, ts, nonce, settings) ->
                    userId == null || userId.isEmpty() ? "default" : userId;

    /** 反射构造 AgentRuntime（包私有构造器，与 AgentsRoutesTest 同法）。 */
    private static AgentRuntime buildRuntime(Settings settings, Object fileStore) throws Exception {
        Map<String, AgentModule> modules = Map.of("chat", moduleStub());
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
        return ctor.newInstance(settings, null, modules, null, null, null,
                fileStore, null, null, null);
    }

    /** 最小模块桩（FilesRoutes 只查 modules.containsKey）。 */
    private static AgentModule moduleStub() {
        return new AgentModule() {
            @Override public String name() { return "chat"; }
            @Override public String description() { return "stub"; }
            @Override public io.github.paopaonb666.agentbase.core.graph.CompiledGraph buildGraph(
                    AgentModule.ModuleContext ctx) {
                throw new UnsupportedOperationException();
            }
            @Override public List<io.github.paopaonb666.agentbase.core.tools.AgentTool> getTools() {
                return List.of();
            }
        };
    }

    private static Object recordingFileStore(List<String> calls) {
        return Proxy.newProxyInstance(
                io.github.paopaonb666.agentbase.extensions.filestore.UploadedFileStore.class
                        .getClassLoader(),
                new Class<?>[]{io.github.paopaonb666.agentbase.extensions.filestore.UploadedFileStore.class},
                (proxy, method, args) -> {
                    calls.add(method.getName());
                    // purgeOrphans 等返回基本型的方法不能回 null（后台线程
                    // 拆箱会 NPE——任务失败日志噪音）。
                    if (method.getReturnType() == int.class) {
                        return 0;
                    }
                    if (method.getReturnType() == long.class) {
                        return 0L;
                    }
                    if (method.getReturnType() == boolean.class) {
                        return false;
                    }
                    return null;
                });
    }

    private static HttpError httpErrorOf(java.util.concurrent.Callable<?> body) {
        return assertThrows(HttpError.class, body::call);
    }

    /** 声明 size 说谎的 multipart（声明 1 字节，实际内容任意大）。 */
    private static final class LyingSizeMultipartFile extends MockMultipartFile {
        LyingSizeMultipartFile(String filename, byte[] content) {
            super("file", filename, "application/octet-stream", content);
        }

        @Override
        public long getSize() {
            return 1;
        }
    }

    @Test
    void oversizeUploadRejectsMidStreamWith413EvenWhenDeclaredSizeLies() throws Exception {
        // #144：getSize 声明 1 字节、实际 2KB > 上限 1KB——流式边读边判
        // 在读满前拦下（修复前会整读进内存后才在解析侧报 400）。
        Settings settings = Settings.load(Map.of(
                "ENV", "development",
                "DOC_PARSE_MAX_INPUT_BYTES", "1024"));
        List<String> calls = new java.util.concurrent.CopyOnWriteArrayList<>();
        AgentRuntime runtime = buildRuntime(settings, recordingFileStore(calls));
        FilesRoutes routes = new FilesRoutes(runtime, PASS_THROUGH_AUTH);

        HttpError error = httpErrorOf(() -> routes.uploadFile("chat",
                new LyingSizeMultipartFile("big.bin", new byte[2048]),
                "u1", null, null, null));
        assertEquals(413, error.status());
        assertTrue(error.getMessage().contains("1024"), error.getMessage());
        // 413 先于任何入库。
        assertTrue(!calls.contains("save"), calls.toString());
    }

    @Test
    void emptyUploadGives400() throws Exception {
        Settings settings = Settings.load(Map.of("ENV", "development"));
        AgentRuntime runtime = buildRuntime(settings, recordingFileStore(new ArrayList<>()));
        FilesRoutes routes = new FilesRoutes(runtime, PASS_THROUGH_AUTH);

        HttpError error = httpErrorOf(() -> routes.uploadFile("chat",
                new MockMultipartFile("file", "empty.txt", "text/plain", new byte[0]),
                "u1", null, null, null));
        assertEquals(400, error.status());
        assertTrue(error.getMessage().contains("为空"), error.getMessage());
    }

    @Test
    void honestUploadUnderLimitReturnsMeta() throws Exception {
        Settings settings = Settings.load(Map.of("ENV", "development"));
        List<String> calls = new java.util.concurrent.CopyOnWriteArrayList<>();
        AgentRuntime runtime = buildRuntime(settings, recordingFileStore(calls));
        FilesRoutes routes = new FilesRoutes(runtime, PASS_THROUGH_AUTH);

        Map<String, Object> meta = routes.uploadFile("chat",
                new MockMultipartFile("file", "notes.txt", "text/plain",
                        "hello agent-base".getBytes(java.nio.charset.StandardCharsets.UTF_8)),
                "u1", null, null, null);
        assertNotNull(meta.get("file_id"));
        assertEquals("notes.txt", meta.get("filename"));
        // 正常路径流式读取后落库（save）；purgeOrphans 在后台线程触发，
        // 不参与时序断言。
        assertTrue(calls.contains("save"), calls.toString());
    }
}
