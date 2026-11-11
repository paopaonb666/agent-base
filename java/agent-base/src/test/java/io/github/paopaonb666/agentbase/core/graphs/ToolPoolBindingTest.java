package io.github.paopaonb666.agentbase.core.graphs;

import io.github.paopaonb666.agentbase.core.AgentModule;
import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.core.graph.CompiledGraph;
import io.github.paopaonb666.agentbase.core.graph.GraphBuilder;
import io.github.paopaonb666.agentbase.core.graph.GraphState;
import io.github.paopaonb666.agentbase.core.graph.InMemoryCheckpointer;
import io.github.paopaonb666.agentbase.core.tools.AgentTool;
import io.github.paopaonb666.agentbase.core.tools.ToolPool;
import io.github.paopaonb666.agentbase.extensions.events.Events;
import io.github.paopaonb666.agentbase.extensions.filestore.UploadedFileInfo;
import io.github.paopaonb666.agentbase.extensions.filestore.UploadedFileStore;
import io.github.paopaonb666.agentbase.extensions.filestore.UploadedFileStores;
import io.github.paopaonb666.agentbase.modules.planner.graph.PlannerGraphs;
import io.github.paopaonb666.agentbase.toolkit.mcp.McpClient;
import io.github.paopaonb666.agentbase.toolkit.mcp.McpTool;
import io.github.paopaonb666.agentbase.toolkit.mcp.McpToolDef;
import io.github.paopaonb666.agentbase.toolkit.mcp.McpTransport;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 工具池绑定收敛（roadmap #238 + #131）：池在 compile 期定死——经
 * ModuleContext 从装配方传入共享池，不再有编译后 volatile 覆写（竞态面
 * 与装配顺序陷阱）；缺省兜底池保留默认超时（#131 的"保留超时"分支）。
 */
class ToolPoolBindingTest {

    @Test
    void toolPoolIsBoundAtCompileAndImmutableAfterwards() {
        // 缺省编译：无池（NodeContext.callTool 走"工具不可用"文案）。
        CompiledGraph bare = GraphBuilder.singleNode("n", (state, ctx) -> Map.of(),
                new InMemoryCheckpointer());
        assertNull(bare.toolPool());

        // 显式池：编译期绑定，getter 返回同一实例。
        ToolPool pool = ToolPool.build(List.of(), List.of(),
                ToolPool.DEFAULT_TOOL_TIMEOUT_SECONDS, Map.of(), null);
        CompiledGraph bound = GraphBuilder.begin()
                .addNode("n", (state, ctx) -> Map.of())
                .entry("n")
                .addEdge("n", GraphBuilder.END)
                .compile(new InMemoryCheckpointer(), "bound", pool);
        assertSame(pool, bound.toolPool());
    }

    @Test
    void moduleContextCarriesSharedPoolIntoBuiltGraph() {
        ToolPool shared = ToolPool.build(List.of(), List.of(),
                ToolPool.DEFAULT_TOOL_TIMEOUT_SECONDS, Map.of(), null);
        AgentModule.ModuleContext ctx = new AgentModule.ModuleContext(
                Settings.load(Map.of("ENV", "development")), null, new InMemoryCheckpointer(),
                List.of(), null, null, shared);
        // 共享池经 ctx 到达 buildSingleAgentGraph → compile 参数（同一实例）。
        CompiledGraph graph = Graphs.buildSingleAgentGraph(ctx, "chat", "系统提示", null);
        assertSame(shared, graph.toolPool());

        // 六参便利构造 = 无池（测试直构形态不受影响）。
        AgentModule.ModuleContext legacy = new AgentModule.ModuleContext(
                Settings.load(Map.of("ENV", "development")), null, new InMemoryCheckpointer(),
                List.of(), null, null);
        assertNull(legacy.toolPool());
        CompiledGraph fallbackBuilt = Graphs.buildSingleAgentGraph(legacy, "chat", "系统提示", null);
        // 兜底池：非 null（#131：保留默认超时，不做无超时执行），但不是共享池。
        assertNull(legacy.toolPool());
        org.junit.jupiter.api.Assertions.assertNotSame(shared, fallbackBuilt.toolPool());
        org.junit.jupiter.api.Assertions.assertNotNull(fallbackBuilt.toolPool());
    }

    @Test
    void plannerGraphBindsSharedPoolFromModuleContext() {
        // planner 图（modules/planner/graph）同走 compile 期绑定：编译图绑定
        // 的池 == ctx 的共享池（同一实例，Graphs 条目同款 assertSame 手段）。
        ToolPool shared = ToolPool.build(List.of(), List.of(),
                ToolPool.DEFAULT_TOOL_TIMEOUT_SECONDS, Map.of(), null);
        AgentModule.ModuleContext ctx = new AgentModule.ModuleContext(
                Settings.load(Map.of("ENV", "development")), null, new InMemoryCheckpointer(),
                List.of(), null, null, shared);
        CompiledGraph graph = PlannerGraphs.buildPlannerGraph(ctx);
        assertSame(shared, graph.toolPool());

        // 六参便利构造 = 无池：planner 图同样不绑池（execute 调工具回
        // "工具不可用"文案——直构测试的既有兜底形态）。
        AgentModule.ModuleContext legacy = new AgentModule.ModuleContext(
                Settings.load(Map.of("ENV", "development")), null, new InMemoryCheckpointer(),
                List.of(), null, null);
        assertNull(PlannerGraphs.buildPlannerGraph(legacy).toolPool());
    }

    @Test
    void untrustedMarkerWrapsExternalToolResultsInContextOnly() {
        // #84①：开关开 → 外部数据工具（web_search）的 ToolMessage 带标记，
        // 豁免工具不带；开关关 → 逐字节旧行为。事件面（ToolCallEvent）在
        // 池内用原文——本图无 sink 侧断言，分离性由 ToolPool 侧测试覆盖
        // （事件发射先于 Graphs 组装，结构性保证）。
        AgentTool webSearch = new AgentTool() {
            @Override public String name() { return "web_search"; }
            @Override public String description() { return "搜索（测试桩）"; }
            @Override public ToolSpecification specification() {
                return ToolSpecification.builder().name(name()).description(description())
                        .parameters(JsonObjectSchema.builder().build()).build();
            }
            @Override public String call(String argsJson, ToolContext ctx) {
                return "网页结果正文";
            }
        };
        // 两段式 stub LLM：第一次回 tool_calls（触发 tools 节点）、第二次收尾
        // ——工厂形态，两次图运行各用独立实例（计数器互不串扰）。
        java.util.function.Supplier<io.github.paopaonb666.agentbase.core.llm.ModelRuntime>
                scriptedLlmFactory = () -> {
            java.util.concurrent.atomic.AtomicInteger rounds = new java.util.concurrent.atomic.AtomicInteger();
            return new io.github.paopaonb666.agentbase.core.llm.ModelRuntime() {
                    @Override
                    public dev.langchain4j.model.chat.ChatModel chat() {
                        throw new UnsupportedOperationException("测试桩不走向线");
                    }

                    @Override
                    public dev.langchain4j.model.chat.StreamingChatModel streaming() {
                        throw new UnsupportedOperationException("测试桩不走向线");
                    }

                    @Override
                    public dev.langchain4j.data.message.AiMessage stream(
                            List<dev.langchain4j.data.message.ChatMessage> messages,
                            List<ToolSpecification> tools,
                            java.util.function.Consumer<String> onDelta) {
                        if (rounds.incrementAndGet() == 1) {
                            return dev.langchain4j.data.message.AiMessage.from("",
                                    List.of(dev.langchain4j.agent.tool.ToolExecutionRequest.builder()
                                            .id("c1").name("web_search").arguments("{}").build()));
                        }
                        return dev.langchain4j.data.message.AiMessage.from("done");
                    }

                    @Override
                    public dev.langchain4j.data.message.AiMessage invoke(
                            List<dev.langchain4j.data.message.ChatMessage> messages) {
                        return dev.langchain4j.data.message.AiMessage.from("done");
                    }
            };
        };

        // 开关开：ToolMessage 头部带标记。
        Settings on = Settings.load(java.util.Map.of(
                "ENV", "development", "TOOL_RESULT_UNTRUSTED_MARKER", "true"), null);
        AgentModule.ModuleContext ctxOn = new AgentModule.ModuleContext(
                on, scriptedLlmFactory.get(), new InMemoryCheckpointer(), List.of(), null, null, null);
        CompiledGraph graphOn = Graphs.buildSingleAgentGraph(ctxOn, "chat", null, List.of(webSearch));
        GraphState withCalls = new GraphState(List.of(
                dev.langchain4j.data.message.AiMessage.from(
                    "", List.of(dev.langchain4j.agent.tool.ToolExecutionRequest.builder()
                        .id("c1").name("web_search").arguments("{}").build()))),
                Map.of());
        GraphState outOn = graphOn.invoke(withCalls, io.github.paopaonb666.agentbase.core.graph.RunConfig.of("chat:t84", "u1"));
        String toolText = outOn.messages().stream()
                .filter(m -> m instanceof dev.langchain4j.data.message.ToolExecutionResultMessage)
                .map(m -> ((dev.langchain4j.data.message.ToolExecutionResultMessage) m).text())
                .findFirst().orElseThrow();
        assertTrue(toolText.startsWith(
                io.github.paopaonb666.agentbase.core.context.ContextEngine.UNTRUSTED_TOOL_RESULT_PREFIX),
                "开关开时外部工具结果应带标记：" + toolText);

        // 开关关（默认）：逐字节旧行为。
        Settings off = Settings.load(Map.of("ENV", "development"), null);
        AgentModule.ModuleContext ctxOff = new AgentModule.ModuleContext(
                off, scriptedLlmFactory.get(), new InMemoryCheckpointer(), List.of(), null, null, null);
        CompiledGraph graphOff = Graphs.buildSingleAgentGraph(ctxOff, "chat", null, List.of(webSearch));
        GraphState outOff = graphOff.invoke(withCalls, io.github.paopaonb666.agentbase.core.graph.RunConfig.of("chat:t85", "u1"));
        String plainText = outOff.messages().stream()
                .filter(m -> m instanceof dev.langchain4j.data.message.ToolExecutionResultMessage)
                .map(m -> ((dev.langchain4j.data.message.ToolExecutionResultMessage) m).text())
                .findFirst().orElseThrow();
        assertEquals("网页结果正文", plainText);
    }

    // ── 条目 6：工具结果外置 + read_tool_result 按需读取 ────────────────────

    /** 两段式 stub LLM（工厂形态）：第一轮回指定工具的 tool_calls、第二轮回纯文本。 */
    private static io.github.paopaonb666.agentbase.core.llm.ModelRuntime scriptedToolCallLlm(String toolName) {
        java.util.concurrent.atomic.AtomicInteger rounds = new java.util.concurrent.atomic.AtomicInteger();
        return new io.github.paopaonb666.agentbase.core.llm.ModelRuntime() {
            @Override public dev.langchain4j.model.chat.ChatModel chat() {
                throw new UnsupportedOperationException("测试桩不走向线");
            }

            @Override public dev.langchain4j.model.chat.StreamingChatModel streaming() {
                throw new UnsupportedOperationException("测试桩不走向线");
            }

            @Override public dev.langchain4j.data.message.AiMessage stream(
                    List<dev.langchain4j.data.message.ChatMessage> messages,
                    List<ToolSpecification> tools,
                    java.util.function.Consumer<String> onDelta) {
                if (rounds.incrementAndGet() == 1) {
                    return dev.langchain4j.data.message.AiMessage.from("",
                            List.of(dev.langchain4j.agent.tool.ToolExecutionRequest.builder()
                                    .id("c1").name(toolName).arguments("{}").build()));
                }
                return dev.langchain4j.data.message.AiMessage.from("done");
            }

            @Override public dev.langchain4j.data.message.AiMessage invoke(
                    List<dev.langchain4j.data.message.ChatMessage> messages) {
                return dev.langchain4j.data.message.AiMessage.from("done");
            }
        };
    }

    /** 返回固定文本的工具桩。 */
    private static AgentTool fixedTool(String name, String reply) {
        return new AgentTool() {
            @Override public String name() { return name; }
            @Override public String description() { return "外置接线测试桩：返回固定文本。"; }
            @Override public ToolSpecification specification() {
                return ToolSpecification.builder().name(name()).description(description())
                        .parameters(JsonObjectSchema.builder().build()).build();
            }
            @Override public String call(String argsJson, ToolContext ctx) {
                return reply;
            }
        };
    }

    /** 带事件收集的单轮图运行：AI(tool_calls) → tools 节点 → agent 收尾。 */
    private record GraphRun(GraphState state, List<Object> events) {}

    private static GraphRun runToolsGraph(AgentModule.ModuleContext ctx, String threadId,
                                          List<AgentTool> tools) {
        CompiledGraph graph = Graphs.buildSingleAgentGraph(ctx, "chat", null, tools);
        GraphState withCalls = new GraphState(List.of(dev.langchain4j.data.message.AiMessage.from("",
                List.of(dev.langchain4j.agent.tool.ToolExecutionRequest.builder()
                        .id("c1").name(tools.get(0).name()).arguments("{}").build()))), Map.of());
        List<Object> events = new java.util.ArrayList<>();
        GraphState out = graph.stream(withCalls,
                io.github.paopaonb666.agentbase.core.graph.RunConfig.of(threadId, "u1"), events::add);
        return new GraphRun(out, events);
    }

    private static String toolText(GraphState state) {
        return state.messages().stream()
                .filter(m -> m instanceof dev.langchain4j.data.message.ToolExecutionResultMessage)
                .map(m -> ((dev.langchain4j.data.message.ToolExecutionResultMessage) m).text())
                .findFirst().orElseThrow();
    }

    private static String endEventResult(List<Object> events, String toolName) {
        return events.stream()
                .filter(e -> e instanceof Events.ToolCallEvent call
                        && "end".equals(call.phase()) && toolName.equals(call.name()))
                .map(e -> ((Events.ToolCallEvent) e).result())
                .findFirst().orElseThrow();
    }

    /** 外置 save 必抛的坏 store 桩（MemoryUploadedFileStore 是 final，全量实现接口）。 */
    private static UploadedFileStore brokenStore() {
        return new UploadedFileStore() {
            @Override public void save(UploadedFileInfo info) { throw new IllegalStateException("disk full"); }
            @Override public List<UploadedFileInfo> getMany(List<String> fileIds) { return List.of(); }
            @Override public List<UploadedFileInfo> listForUser(String userId, String module, int limit) { return List.of(); }
            @Override public void bindThread(List<String> fileIds, String threadId) { }
            @Override public void deleteForThread(String threadId) { }
            @Override public boolean deleteFile(String fileId) { return false; }
            @Override public int purgeOrphans(int maxAgeHours) { return 0; }
        };
    }

    @Test
    void oversizeToolResultSpillsToStoreAndToolMessageCarriesSummary() {
        String big = "x".repeat(500);
        AgentModule.ModuleContext ctx = new AgentModule.ModuleContext(
                spillSettings(100, false), scriptedToolCallLlm("web_search"),
                new InMemoryCheckpointer(), List.of(), null, null, null, null,
                new UploadedFileStores.MemoryUploadedFileStore());
        GraphRun run = runToolsGraph(ctx, "chat:t-spill", List.of(fixedTool("web_search", big)));
        String text = toolText(run.state());
        // ToolMessage 是摘要块：前缀 + 元信息 + file_id 引用。
        assertTrue(text.startsWith(io.github.paopaonb666.agentbase.core.context.ContextEngine
                .SPILL_SUMMARY_PREFIX), text);
        assertTrue(text.contains("工具 web_search 返回共 500 字符（超过阈值 100）"), text);
        int start = text.indexOf("file_id=") + "file_id=".length();
        int end = text.indexOf("。首屏前");
        String fileId = text.substring(start, end);
        // store.extractedText = 全文（外置行 module="tool_result"、thread/user 绑定）。
        var store = ctx.fileStore();
        List<UploadedFileInfo> rows = store.getMany(List.of(fileId));
        assertEquals(1, rows.size());
        assertEquals(big, rows.get(0).extractedText());
        assertEquals("tool_result", rows.get(0).module());
        assertEquals("chat:t-spill", rows.get(0).threadId());
        assertEquals("u1", rows.get(0).userId());
        assertEquals(500, rows.get(0).textLen());
        // 事件/审计原文分离：ToolCallEvent 里是池内原文（无摘要、无标记）。
        assertEquals(big, endEventResult(run.events(), "web_search"));
    }

    @Test
    void spillThenWrapKeepsMarkerAsFirstLineAndStoresRawText() {
        String big = "y".repeat(500);
        // 开关开：先 spill 后 wrap——#84 标记仍是 ToolMessage 第一行，摘要块随后。
        AgentModule.ModuleContext ctx = new AgentModule.ModuleContext(
                spillSettings(100, true), scriptedToolCallLlm("web_search"),
                new InMemoryCheckpointer(), List.of(), null, null, null, null,
                new UploadedFileStores.MemoryUploadedFileStore());
        GraphRun run = runToolsGraph(ctx, "chat:t-spill-wrap", List.of(fixedTool("web_search", big)));
        String text = toolText(run.state());
        assertTrue(text.startsWith(io.github.paopaonb666.agentbase.core.context.ContextEngine
                .UNTRUSTED_TOOL_RESULT_PREFIX), "标记必须是第一行：" + text);
        assertTrue(text.contains(io.github.paopaonb666.agentbase.core.context.ContextEngine
                .SPILL_SUMMARY_PREFIX), text);
        // 外置行存原文（标记是上下文层概念，不进存储）。
        List<UploadedFileInfo> rows = ctx.fileStore().listForUser("u1");
        assertEquals(1, rows.size());
        assertEquals(big, rows.get(0).extractedText());
        assertEquals("y".repeat(500), endEventResult(run.events(), "web_search"));
    }

    @Test
    void spillDisabledByZeroThresholdKeepsLegacyBehavior() {
        String big = "x".repeat(500);
        AgentModule.ModuleContext ctx = new AgentModule.ModuleContext(
                spillSettings(0, false), scriptedToolCallLlm("web_search"),
                new InMemoryCheckpointer(), List.of(), null, null, null, null,
                new UploadedFileStores.MemoryUploadedFileStore());
        GraphRun run = runToolsGraph(ctx, "chat:t-off", List.of(fixedTool("web_search", big)));
        assertEquals(big, toolText(run.state()));
        assertEquals(0, ctx.fileStore().listForUser("u1").size());
    }

    @Test
    void spillSaveFailureFallsBackToRawOutput() {
        String big = "x".repeat(500);
        // 坏 store 桩：save 抛异常 → 原文直回（绝不因外置失败打断图）。
        AgentModule.ModuleContext ctx = new AgentModule.ModuleContext(
                spillSettings(100, false), scriptedToolCallLlm("web_search"),
                new InMemoryCheckpointer(), List.of(), null, null, null, null, brokenStore());
        GraphRun run = runToolsGraph(ctx, "chat:t-broken", List.of(fixedTool("web_search", big)));
        assertEquals(big, toolText(run.state()));
    }

    @Test
    void thresholdExactlyEqualDoesNotSpill() {
        String exact = "x".repeat(100);
        AgentModule.ModuleContext ctx = new AgentModule.ModuleContext(
                spillSettings(100, false), scriptedToolCallLlm("web_search"),
                new InMemoryCheckpointer(), List.of(), null, null, null, null,
                new UploadedFileStores.MemoryUploadedFileStore());
        GraphRun run = runToolsGraph(ctx, "chat:t-exact", List.of(fixedTool("web_search", exact)));
        assertEquals(exact, toolText(run.state()));
        assertEquals(0, ctx.fileStore().listForUser("u1").size());
    }

    /** 外置测试用的 Settings：阈值 + #84 开关（其余默认）。 */
    private static Settings spillSettings(int threshold, boolean untrustedMarker) {
        Settings settings = Settings.load(Map.of("ENV", "development"), null);
        settings.toolkit.toolResultSpillThreshold = threshold;
        settings.toolkit.toolResultUntrustedMarker = untrustedMarker;
        return settings;
    }

    // ── 条目 6 / 汇总 5：MCP 判据顺修（非 mcp_ 前缀的真实 MCP 命名也判外部） ──

    /** 行式 JSON-RPC 脚本桩：initialize 回握手结果、tools/call 回固定文本。 */
    private static final class ScriptedMcpTransport implements McpTransport {
        private final java.util.concurrent.BlockingQueue<String> lines =
                new java.util.concurrent.LinkedBlockingQueue<>();
        private final String toolResult;
        private volatile boolean closed;
        private final com.fasterxml.jackson.databind.ObjectMapper mapper =
                new com.fasterxml.jackson.databind.ObjectMapper();

        ScriptedMcpTransport(String toolResult) {
            this.toolResult = toolResult;
        }

        @Override public void start() { }

        @Override public void sendLine(String line) {
            try {
                var request = mapper.readTree(line);
                String method = request.path("method").asText("");
                if (!request.has("id")) {
                    return; // 通知不回包
                }
                var response = mapper.createObjectNode();
                response.put("jsonrpc", "2.0");
                response.set("id", request.get("id"));
                var result = response.putObject("result");
                if ("initialize".equals(method)) {
                    result.put("protocolVersion", McpClient.PROTOCOL_VERSION);
                    result.putObject("capabilities");
                } else if ("tools/call".equals(method)) {
                    result.put("isError", false);
                    result.putArray("content").addObject()
                            .put("type", "text").put("text", toolResult);
                } else {
                    result.putObject("capabilities");
                }
                lines.put(mapper.writeValueAsString(response));
            } catch (Exception e) {
                throw new IllegalStateException("scripted transport failed", e);
            }
        }

        @Override public String readLine() {
            try {
                String line = lines.poll(2, java.util.concurrent.TimeUnit.SECONDS);
                return closed ? null : line;
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return null;
            }
        }

        @Override public void close() {
            closed = true;
        }
    }

    /** 反射构造 McpTool（构造器包私有在 toolkit.mcp）：池内名 = "{server}_{tool}"。 */
    private static AgentTool mcpToolViaReflection(String serverName, String rawName, String resultText)
            throws Exception {
        McpClient client = new McpClient(serverName, new ScriptedMcpTransport(resultText));
        client.connect("tool-pool-binding-test", "1.0", 5.0);
        var constructor = McpTool.class.getDeclaredConstructor(
                String.class, McpToolDef.class, McpClient.class, double.class);
        constructor.setAccessible(true);
        return (AgentTool) constructor.newInstance(serverName,
                new McpToolDef(rawName, "MCP 判据测试桩",
                        new com.fasterxml.jackson.databind.ObjectMapper().createObjectNode()),
                client, 30.0);
    }

    @Test
    void mcpToolWithNonPrefixedNameIsJudgedExternalViaPoolRegistry() throws Exception {
        // server=github、tool=search → 池内名 "github_search"（非 mcp_ 前缀，
        // 旧前缀判据漏掉、新名单判据命中）。
        AgentTool githubSearch = mcpToolViaReflection("github", "search", "mcp 原始结果");
        ToolPool pool = ToolPool.build(List.of(), List.of(githubSearch), 30.0, Map.of(), null);
        assertTrue(pool.isMcpTool("github_search"));
        assertFalse(pool.isMcpTool("web_search"));
        assertFalse(pool.isMcpTool("mcp_github_search"));
        assertFalse(pool.isMcpTool(null));

        // 端到端：开关开 + 共享池绑定 → "github_search" 结果带不可信标记
        // （判外部数据），标记内容 = MCP 工具原始返回。
        Settings on = Settings.load(Map.of("ENV", "development", "TOOL_RESULT_UNTRUSTED_MARKER", "true"), null);
        AgentModule.ModuleContext ctx = new AgentModule.ModuleContext(
                on, scriptedToolCallLlm("github_search"), new InMemoryCheckpointer(),
                List.of(), null, null, pool);
        CompiledGraph graph = Graphs.buildSingleAgentGraph(ctx, "chat", null, List.of(githubSearch));
        GraphState withCalls = new GraphState(List.of(dev.langchain4j.data.message.AiMessage.from("",
                List.of(dev.langchain4j.agent.tool.ToolExecutionRequest.builder()
                        .id("c1").name("github_search").arguments("{}").build()))), Map.of());
        GraphState out = graph.invoke(withCalls,
                io.github.paopaonb666.agentbase.core.graph.RunConfig.of("chat:t-mcp", "u1"));
        String text = toolText(out);
        assertTrue(text.startsWith(io.github.paopaonb666.agentbase.core.context.ContextEngine
                .UNTRUSTED_TOOL_RESULT_PREFIX), "非 mcp_ 前缀的 MCP 工具也应包标记：" + text);
        assertTrue(text.endsWith("mcp 原始结果"), text);
    }
}
