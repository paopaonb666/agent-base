package io.github.paopaonb666.agentbase.entrypoints.server.routes;

import io.github.paopaonb666.agentbase.core.AgentModule;
import io.github.paopaonb666.agentbase.core.AgentRuntime;
import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.entrypoints.server.AuthBackends;
import io.github.paopaonb666.agentbase.entrypoints.server.Background;
import io.github.paopaonb666.agentbase.entrypoints.server.HttpError;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.PDPage;
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
 * FilesRoutes 上传端点测试（roadmap #144/#226/#290）：413 检查点在流式
 * 读取中边读边判——声明 size 与实际不符（multipart 声明值不可信）时超限
 * 内容不再整读进内存；空上传 400；正常路径元信息完整；上传响应携带
 * knowledge_status 三态（disabled / skipped_empty_text / queued）。
 */
class FilesRoutesTest {

    private static final AuthBackends.AuthBackend PASS_THROUGH_AUTH =
            (userId, sig, ts, nonce, settings) ->
                    userId == null || userId.isEmpty() ? "default" : userId;

    /** 反射构造 AgentRuntime（包私有构造器，与 AgentsRoutesTest 同法）。 */
    private static AgentRuntime buildRuntime(Settings settings, Object fileStore) throws Exception {
        return buildRuntime(settings, fileStore, null);
    }

    /** 同上，另可注入 memory（AgentModule.MemoryServicePort 的桩/代理）。 */
    private static AgentRuntime buildRuntime(Settings settings, Object fileStore,
                                             Object memory) throws Exception {
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
                fileStore, memory, null, null);
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

    /** 记录调用名的 memory 桩（AgentModule.MemoryServicePort 代理，#226）。 */
    private static Object recordingMemoryPort(List<String> calls) {
        return Proxy.newProxyInstance(
                AgentModule.MemoryServicePort.class.getClassLoader(),
                new Class<?>[]{AgentModule.MemoryServicePort.class},
                (proxy, method, args) -> {
                    calls.add(method.getName());
                    // 基本型回默认值防拆箱 NPE（ingestDocument/revokeDocument 返 int）。
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

    /** 等待后台任务排空（BackgroundTest 同款轮询），给摄取断言确定性时点。 */
    private static void awaitBackgroundDrain() throws InterruptedException {
        long deadline = System.currentTimeMillis() + 5_000;
        while (Background.pending() > 0 && System.currentTimeMillis() < deadline) {
            Thread.sleep(20);
        }
        assertEquals(0, Background.pending(), "后台任务应在 5s 内排空");
    }

    /** 单页无内容流的 PDF（扫描件形态）：解析成功但提取文本为空。 */
    private static byte[] emptyTextPdf() throws java.io.IOException {
        try (PDDocument document = new PDDocument()) {
            document.addPage(new PDPage());
            java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream();
            document.save(out);
            return out.toByteArray();
        }
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
    void imageUploadsOutsideWhitelistGetExplicit400() throws Exception {
        // #216/#247：图片白名单契约的 400 矩阵——每个失败分支都有可读
        // 原因（而不是静默降级或解析器兜底）。
        Settings settings = Settings.load(Map.of("ENV", "development"));
        AgentRuntime runtime = buildRuntime(settings, recordingFileStore(new ArrayList<>()));
        FilesRoutes routes = new FilesRoutes(runtime, PASS_THROUGH_AUTH);

        // 1) 图片扩展名 + 内容不是该类型（magic 校验失败）→ 400。
        HttpError mismatch = httpErrorOf(() -> routes.uploadFile("chat",
                new MockMultipartFile("file", "fake.png", "image/png",
                        new byte[]{1, 2, 3, 4}),
                "u1", null, null, null));
        assertEquals(400, mismatch.status());
        assertTrue(mismatch.getMessage().contains("magic 校验失败"), mismatch.getMessage());

        // 2) 图片形态扩展名但不在扩展白名单（bmp）→ 解析器 400（不是
        //    文本兜底吞掉二进制垃圾）。
        HttpError unsupported = httpErrorOf(() -> routes.uploadFile("chat",
                new MockMultipartFile("file", "pic.bmp", "image/bmp",
                        java.nio.ByteBuffer.allocate(52).put((byte) 'B').put((byte) 'M')
                                .array()),
                "u1", null, null, null));
        assertEquals(400, unsupported.status());
        assertTrue(unsupported.getMessage().contains("无法识别的文档扩展名 'bmp'"),
                unsupported.getMessage());

        // 3) 无扩展名 → 推断失败 400。
        HttpError noExt = httpErrorOf(() -> routes.uploadFile("chat",
                new MockMultipartFile("file", "image", "application/octet-stream",
                        "GIF89a".getBytes(java.nio.charset.StandardCharsets.US_ASCII)),
                "u1", null, null, null));
        assertEquals(400, noExt.status());
        assertTrue(noExt.getMessage().contains("无法识别的文档扩展名 ''"), noExt.getMessage());

        // 4) 白名单内且 magic 相符（jpg 扩展名）→ 入库 format 归一为
        //    嗅探 mime（jpeg），png 魔数扩展名说谎同样按真实类型入库。
        java.util.List<String> calls = new java.util.concurrent.CopyOnWriteArrayList<>();
        AgentRuntime saving = buildRuntime(settings, recordingFileStore(calls));
        Map<String, Object> meta = (Map<String, Object>) new FilesRoutes(saving, PASS_THROUGH_AUTH)
                .uploadFile("chat",
                        new MockMultipartFile("file", "photo.jpg", "image/jpeg",
                                new byte[]{(byte) 0xFF, (byte) 0xD8, (byte) 0xFF, 0x00}),
                        "u1", null, null, null);
        assertEquals("jpeg", meta.get("format"), "扩展名 jpg 归一为嗅探 mime jpeg");
        assertTrue(calls.contains("save"));

        // 5) PNG 魔数 + .png 扩展名 → format png。
        Map<String, Object> pngMeta = (Map<String, Object>) new FilesRoutes(saving, PASS_THROUGH_AUTH)
                .uploadFile("chat",
                        new MockMultipartFile("file", "pixel.png", "image/png",
                                new byte[]{(byte) 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A}),
                        "u1", null, null, null);
        assertEquals("png", pngMeta.get("format"));
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

    @Test
    void uploadResponseCarriesKnowledgeStatusThreeStates() throws Exception {
        // #226：摄取是后台任务，同步响应拿不到 chunks_created 实数，但
        // 必须明确告知前端知识库摄取的三态（disabled / skipped_empty_text
        // / queued）——meta() 键集保持 Python 对齐，状态只在响应组装层附加。
        Settings settings = Settings.load(Map.of("ENV", "development"));

        // 1) memory 缺席 → disabled（入库照常，只是知识库摄取关闭）。
        List<String> storeCalls = new java.util.concurrent.CopyOnWriteArrayList<>();
        AgentRuntime noMemory = buildRuntime(settings, recordingFileStore(storeCalls));
        Map<String, Object> disabled = new FilesRoutes(noMemory, PASS_THROUGH_AUTH)
                .uploadFile("chat",
                        new MockMultipartFile("file", "notes.txt", "text/plain",
                                "正文".getBytes(java.nio.charset.StandardCharsets.UTF_8)),
                        "u1", null, null, null);
        assertEquals("disabled", disabled.get("knowledge_status"));

        // 2) 非空文本 + memory 在 → queued；后台排空后 ingestDocument
        //    确被调用（状态不是空头支票）。
        List<String> memCalls = new java.util.concurrent.CopyOnWriteArrayList<>();
        AgentRuntime withMemory = buildRuntime(settings, recordingFileStore(new ArrayList<>()),
                recordingMemoryPort(memCalls));
        Map<String, Object> queued = new FilesRoutes(withMemory, PASS_THROUGH_AUTH)
                .uploadFile("chat",
                        new MockMultipartFile("file", "notes.txt", "text/plain",
                                "正文".getBytes(java.nio.charset.StandardCharsets.UTF_8)),
                        "u1", null, null, null);
        assertEquals("queued", queued.get("knowledge_status"));
        awaitBackgroundDrain();
        assertTrue(memCalls.contains("ingestDocument"), memCalls.toString());

        // 3) 空提取（单页无内容流的 PDF，扫描件形态）→ skipped_empty_text
        //    + warning 文案；后台排空后 memory 侧零摄取调用（「不摄取」
        //    半边与「明确状态」半边一起锁定）。
        List<String> emptyMemCalls = new java.util.concurrent.CopyOnWriteArrayList<>();
        AgentRuntime scanned = buildRuntime(settings, recordingFileStore(new ArrayList<>()),
                recordingMemoryPort(emptyMemCalls));
        Map<String, Object> skipped = new FilesRoutes(scanned, PASS_THROUGH_AUTH)
                .uploadFile("chat",
                        new MockMultipartFile("file", "scan.pdf", "application/pdf",
                                emptyTextPdf()),
                        "u1", null, null, null);
        assertEquals("skipped_empty_text", skipped.get("knowledge_status"));
        assertEquals("未能从文件中提取到文本：可能是扫描件或纯图片 PDF",
                skipped.get("warning"));
        awaitBackgroundDrain();
        assertTrue(!emptyMemCalls.contains("ingestDocument"), emptyMemCalls.toString());

        // 4) 图片路径同口径：无提取文本，memory 在 → skipped_empty_text。
        List<String> imgMemCalls = new java.util.concurrent.CopyOnWriteArrayList<>();
        AgentRuntime images = buildRuntime(settings, recordingFileStore(new ArrayList<>()),
                recordingMemoryPort(imgMemCalls));
        Map<String, Object> image = new FilesRoutes(images, PASS_THROUGH_AUTH)
                .uploadFile("chat",
                        new MockMultipartFile("file", "pixel.png", "image/png",
                                new byte[]{(byte) 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A}),
                        "u1", null, null, null);
        assertEquals("skipped_empty_text", image.get("knowledge_status"));
        awaitBackgroundDrain();
        assertTrue(!imgMemCalls.contains("ingestDocument"), imgMemCalls.toString());
    }
}
