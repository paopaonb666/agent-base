package io.github.paopaonb666.agentbase.memory.tools;

import io.github.paopaonb666.agentbase.core.MemoryScope;
import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.core.tools.AgentTool;
import io.github.paopaonb666.agentbase.memory.service.MemoryService;
import io.github.paopaonb666.agentbase.memory.store.InMemoryMemoryStore;
import io.github.paopaonb666.agentbase.memory.store.Models.DocChunk;
import io.github.paopaonb666.agentbase.extensions.events.Events;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.concurrent.CopyOnWriteArrayList;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 五个记忆工具的参数矩阵与输出契约（roadmap #299）：约束对应 Python
 * Pydantic 参数模型（越界抛错走 ToolPool 归一而不是静默钳制）、作用域
 * 传递（MemoryScope MDC → user/agent）、块追加/替换/保尾截断、删除的
 * 属主掩蔽与知识检索的 [n] 引用锚 + sources 事件。真实 MemoryService +
 * InMemoryMemoryStore（无嵌入 = 关键词打分路径）。
 */
class MemoryToolsTest {

    private final InMemoryMemoryStore store = new InMemoryMemoryStore();
    private final MemoryService memory =
            new MemoryService(store, neverEmbedder(), Settings.load(Map.of(), null), null, null);

    /** 恒不向量化：MemoryService 各路径对 null 嵌入的降级分支。 */
    private static io.github.paopaonb666.agentbase.memory.embeddings.EmbeddingClient neverEmbedder() {
        return new io.github.paopaonb666.agentbase.memory.embeddings.EmbeddingClient() {
            @Override public Integer dims() { return null; }
            @Override public java.util.List<java.util.List<Float>> embed(java.util.List<String> texts) {
                return null;
            }
            @Override public boolean probe() { return false; }
        };
    }
    private final List<AgentTool> tools = MemoryTools.build(memory);

    private AgentTool tool(String name) {
        return tools.stream().filter(t -> t.name().equals(name)).findFirst().orElseThrow();
    }

    @AfterEach
    void clearScope() {
        MemoryScope.clear();
    }

    private interface ThrowingRunnable {
        void run() throws Exception;
    }

    private static IllegalArgumentException errorOf(ThrowingRunnable action) {
        try {
            action.run();
            throw new AssertionError("expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            return e;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /** AgentTool.call 抛受检异常：测试统一经此收口。 */
    private static String call(AgentTool tool, String argsJson) {
        return call(tool, argsJson, null);
    }

    private static String call(AgentTool tool, String argsJson, AgentTool.ToolContext ctx) {
        try {
            return tool.call(argsJson, ctx);
        } catch (IllegalArgumentException e) {
            throw e; // 参数校验断言直接看它
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    // ── memory_search ─────────────────────────────────────────────────────

    @Test
    void memorySearchValidatesParamsAndReportsMisses() {
        MemoryScope.set("alice", "chat:t1");
        assertEquals("（没有找到相关记忆）", call(tool("memory_search"), 
                "{\"query\": \"不存在的记忆内容\"}", null));
        // query 空/超长 → 抛错（不静默钳制）。
        assertEquals("query 必须是 1..2000 字符",
                errorOf(() -> call(tool("memory_search"), "{\"query\": \"\"}", null)).getMessage());
        assertEquals("query 必须是 1..2000 字符", errorOf(
                () -> call(tool("memory_search"), "{\"query\": \"" + "字".repeat(2001) + "\"}", null))
                .getMessage());
        assertEquals("limit 必须在 1..20 之间",
                errorOf(() -> call(tool("memory_search"), "{\"query\": \"q\", \"limit\": 21}", null))
                        .getMessage());
        assertEquals("limit 必须在 1..20 之间",
                errorOf(() -> call(tool("memory_search"), "{\"query\": \"q\", \"limit\": 0}", null))
                        .getMessage());
    }

    @Test
    void memorySearchIsUserScopedAndReturnsShortIds() {
        memory.addMemory("alice", "chat", "用户的部署环境是 Ubuntu 22.04", "semantic",
                List.of(), 0.5, "");
        memory.addMemory("bob", "chat", "用户的部署环境是 Ubuntu 22.04", "semantic",
                List.of(), 0.5, "");
        MemoryScope.set("alice", "chat:t1");
        String out = call(tool("memory_search"), "{\"query\": \"部署环境 Ubuntu\"}", null);
        assertTrue(out.startsWith("找到的相关记忆："), out);
        assertTrue(out.contains("（相关度 "), out);
        // id 截 12 字符输出（memory_id= 前缀形态，供 memory_delete 使用）。
        assertTrue(out.matches("(?s).*\\[id=[0-9a-f]{12}\\].*"), out);
        // bob 的同文记忆不可见（user 隔离是硬边界）。
        assertEquals(1, out.split("\n-").length - 1, out);
    }

    // ── memory_save ───────────────────────────────────────────────────────

    @Test
    void memorySaveValidatesAndWritesWithScope() {
        MemoryScope.set("alice", "chat:t1");
        assertEquals("content 必须是 1..2000 字符",
                errorOf(() -> call(tool("memory_save"), "{\"content\": \"\"}", null)).getMessage());
        assertEquals("tags 最多 10 个", errorOf(() -> call(tool("memory_save"),
                "{\"content\": \"x\", \"tags\": [\"a\",\"b\",\"c\",\"d\",\"e\",\"f\",\"g\",\"h\",\"i\",\"j\",\"k\"]}",
                null)).getMessage());

        String out = call(tool("memory_save"),
                "{\"content\": \"用户的部署环境是 Ubuntu 22.04\", \"kind\": \"semantic\", "
                        + "\"tags\": [\"env\"]}", null);
        assertTrue(out.matches("(?s)^已记住（memory_id=[0-9a-f]{12}…，kind=semantic）$"), out);
        // agent_id 取 thread 前缀的模块名；thread_id 随记忆落库。
        var record = memory.store().listMemories("alice", "chat", false, null, null, 10, false).get(0);
        assertEquals("用户的部署环境是 Ubuntu 22.04", record.content());
        assertEquals("chat:t1", record.sourceThreadId());
        assertEquals("env", record.tags().get(0));

        // 无 thread 作用域：agent_id 回退 "*"（全局）；user 保持 alice。
        MemoryScope.set("alice", "");
        call(tool("memory_save"), "{\"content\": \"全局偏好：简洁回复\"}", null);
        assertEquals(1, memory.store()
                .listMemories("alice", "*", false, null, null, 10, false).size());
    }

    @Test
    void memorySaveRejectsSensitiveContentWithDesensitizedNotice() {
        // #48（Java 先行）：命中密钥形态 → 回脱敏提示文案（不抛异常、
        // 不打断对话），内容绝不落库。
        MemoryScope.set("alice", "chat:t1");
        String out = call(tool("memory_save"),
                "{\"content\": \"用户的密钥是 sk-abcdefghijklmnop123456\"}", null);
        assertEquals("记忆保存被拒绝：内容命中敏感信息规则（api_key），请移除后重试", out);
        assertTrue(memory.store().listMemories("alice", "chat", false, null, null, 10, false).isEmpty());

        // 多规则命中：规则名全部列出。
        String multi = call(tool("memory_save"),
                "{\"content\": \"key 是 sk-abcdefghijklmnop123456，手机 13812345678\"}", null);
        assertEquals("记忆保存被拒绝：内容命中敏感信息规则（api_key、phone_number），请移除后重试",
                multi);

        // 关闭开关（build 重载）：同一内容照常落库。
        MemoryService unfiltered = new MemoryService(store, neverEmbedder(),
                Settings.load(Map.of(), null), null, null);
        AgentTool save = MemoryTools.build(unfiltered, false).stream()
                .filter(t -> t.name().equals("memory_save")).findFirst().orElseThrow();
        String saved = call(save, "{\"content\": \"用户的密钥是 sk-abcdefghijklmnop123456\"}", null);
        assertTrue(saved.startsWith("已记住（memory_id="), saved);
    }

    @Test
    void sensitivePolicyRuleMatrixAndFalsePositives() {
        var policy = io.github.paopaonb666.agentbase.memory.textkit.SensitiveContentPolicy.INSTANCE;
        // 命中矩阵：每条规则一个高置信样例。
        assertTrue(policy.violations("密钥 sk-abcdefghijklmnop123456").contains("api_key"));
        assertTrue(policy.violations("AWS key AKIAIOSFODNN7EXAMPLE").contains("aws_access_key"));
        assertTrue(policy.violations("token ghp_0123456789abcdefghij").contains("github_token"));
        assertTrue(policy.violations("slack xoxb-1234567890-abcd").contains("slack_token"));
        assertTrue(policy.violations("Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9")
                .contains("bearer_token"));
        assertTrue(policy.violations("手机号 13812345678").contains("phone_number"));
        assertTrue(policy.violations("身份证 11010119900307751X").contains("id_card"));
        // 多规则：按声明顺序返回。
        assertEquals(List.of("api_key", "phone_number"),
                policy.violations("sk-abcdefghijklmnop123456 和 13812345678"));

        // 误报边界（短数字串/词内子串不报）：
        assertTrue(policy.violations("任务编号 task-1234567890123456").isEmpty(),
                "risk-/task- 等词内 sk- 子串不应命中");
        assertTrue(policy.violations("sk-abc").isEmpty(), "密钥体过短不命中");
        assertTrue(policy.violations("工单号 1381234567").isEmpty(), "10 位数字不是手机号");
        assertTrue(policy.violations("订单 1381234567890").isEmpty(), "两侧贴数字的长数字串不命中");
        assertTrue(policy.violations("1234567890123456789").isEmpty(), "19 位数字不是 18 位证件号");
        assertTrue(policy.violations("Bearer short").isEmpty(), "Bearer 后短 token 不命中");
        // null/空白视为干净。
        assertTrue(policy.violations(null).isEmpty());
        assertTrue(policy.violations("   ").isEmpty());
    }

    @Test
    void sensitiveContentFilterSettingParses() {
        // Settings 解析（#48）：默认开；env 布尔形式走 requireBool 全集。
        assertTrue(Settings.load(Map.of(), null).memory.sensitiveContentFilter);
        assertFalse(Settings.load(Map.of("MEMORY_SENSITIVE_CONTENT_FILTER", "false"), null)
                .memory.sensitiveContentFilter);
        assertFalse(Settings.load(Map.of("MEMORY_SENSITIVE_CONTENT_FILTER", "0"), null)
                .memory.sensitiveContentFilter);
        assertTrue(Settings.load(Map.of("MEMORY_SENSITIVE_CONTENT_FILTER", "on"), null)
                .memory.sensitiveContentFilter);
    }

    // ── memory_update_block ───────────────────────────────────────────────

    @Test
    void memoryUpdateBlockValidatesLabelAndContent() {
        MemoryScope.set("alice", "chat:t1");
        assertEquals("label 必须是 1..64 字符且匹配 [\\w.-]+",
                errorOf(() -> call(tool("memory_update_block"), 
                        "{\"label\": \"\", \"content\": \"x\"}", null)).getMessage());
        assertEquals("label 必须是 1..64 字符且匹配 [\\w.-]+",
                errorOf(() -> call(tool("memory_update_block"), 
                        "{\"label\": \"bad label!\", \"content\": \"x\"}", null)).getMessage());
        assertEquals("label 必须是 1..64 字符且匹配 [\\w.-]+",
                errorOf(() -> call(tool("memory_update_block"), 
                        "{\"label\": \"" + "长".repeat(65) + "\", \"content\": \"x\"}", null)).getMessage());
        assertEquals("content 必须是 1..4000 字符",
                errorOf(() -> call(tool("memory_update_block"), 
                        "{\"label\": \"persona\", \"content\": \"\"}", null)).getMessage());
        assertEquals("content 必须是 1..4000 字符",
                errorOf(() -> call(tool("memory_update_block"), 
                        "{\"label\": \"persona\", \"content\": \"" + "字".repeat(4001) + "\"}", null))
                        .getMessage());
        // 中文 label 合法（(?U) unicode \w，与 Python re \w 同语义）。
        assertTrue(call(tool("memory_update_block"), 
                "{\"label\": \"人设\", \"content\": \"简洁\"}", null).contains("已更新"));
    }

    @Test
    void memoryUpdateBlockAppendReplaceAndTailKeep() {
        MemoryScope.set("alice", "chat:t1");
        // append：默认操作；尾部空白折叠后拼接。
        String first = call(tool("memory_update_block"), 
                "{\"label\": \"persona\", \"content\": \"第一行\"}", null);
        assertTrue(first.matches("(?s)^记忆块 'persona' 已更新（v1，3/2000 字符）$"), first);
        String second = call(tool("memory_update_block"), 
                "{\"label\": \"persona\", \"content\": \"第二行\", \"operation\": \"append\"}", null);
        assertTrue(second.contains("v2"), second);
        var block = memory.store().getBlock("alice", "chat", "persona");
        assertEquals("第一行\n第二行", block.content());

        // replace：整块替换，版本继续累加。
        call(tool("memory_update_block"), 
                "{\"label\": \"persona\", \"content\": \"全新内容\", \"operation\": \"replace\"}", null);
        assertEquals("全新内容", memory.store().getBlock("alice", "chat", "persona").content());
        assertEquals(3, memory.store().getBlock("alice", "chat", "persona").version());

        // 未知 operation：提示文案（不是异常）。
        assertEquals("未知 operation 'upsert'（只支持 append / replace）",
                call(tool("memory_update_block"), 
                        "{\"label\": \"persona\", \"content\": \"x\", \"operation\": \"upsert\"}", null));

        // 超限保尾：note 前缀 + 最近内容保留在 charLimit 内。
        call(tool("memory_update_block"),
                "{\"label\": \"persona\", \"content\": \"" + "旧".repeat(1990) + "\", \"operation\": \"replace\"}",
                null);
        call(tool("memory_update_block"),
                "{\"label\": \"persona\", \"content\": \"" + "新".repeat(100) + "\"}", null);
        var capped = memory.store().getBlock("alice", "chat", "persona");
        assertTrue(capped.content().startsWith("……（较早内容已截断）"), capped.content());
        assertTrue(capped.content().endsWith("新".repeat(100)), "保尾：最近写入保留");
        assertTrue(io.github.paopaonb666.agentbase.core.text.CharMetrics.len(capped.content()) <= 2000);
    }

    @Test
    void memoryUpdateBlockAcceptsContentAtToolLimit() {
        // #178 三路径上限收口：工具入参上限 BlockLimits.CONTENT_MAX_TOOL
        // 边界值成功——块声明容量预置 CHAR_LIMIT_MAX，保尾截断不触发。
        MemoryScope.set("alice", "chat:t1");
        store.upsertBlock(new io.github.paopaonb666.agentbase.memory.store.Models.MemoryBlock(
                "alice", "chat", "persona", "seed", BlockLimits.CHAR_LIMIT_MAX, 1,
                io.github.paopaonb666.agentbase.memory.store.Models.now()));
        String out = call(tool("memory_update_block"),
                "{\"label\": \"persona\", \"content\": \""
                        + "字".repeat(BlockLimits.CONTENT_MAX_TOOL)
                        + "\", \"operation\": \"replace\"}", null);
        assertTrue(out.matches("(?s)^记忆块 'persona' 已更新（v2，"
                + BlockLimits.CONTENT_MAX_TOOL + "/" + BlockLimits.CHAR_LIMIT_MAX + " 字符）$"), out);
        assertEquals(BlockLimits.CONTENT_MAX_TOOL,
                memory.store().getBlock("alice", "chat", "persona").content().length());
    }

    // ── memory_delete ─────────────────────────────────────────────────────

    @Test
    void memoryDeleteValidatesMasksOwnershipAndDeletes() {
        MemoryScope.set("alice", "chat:t1");
        var record = memory.addMemory("alice", "chat", "旧的部署环境是 CentOS 7",
                "semantic", List.of(), 0.5, "");

        assertEquals("memory_id 必须是 4..64 字符且匹配 [\\w.-]+",
                errorOf(() -> call(tool("memory_delete"), "{\"memory_id\": \"ab\"}", null)).getMessage());
        assertEquals("memory_id 必须是 4..64 字符且匹配 [\\w.-]+",
                errorOf(() -> call(tool("memory_delete"), "{\"memory_id\": \"bad id!\"}", null)).getMessage());
        assertEquals("记忆 missing-id 不存在（可能已删除，id 以 memory_search 结果为准）",
                call(tool("memory_delete"), "{\"memory_id\": \"missing-id\"}", null));

        // 跨用户掩蔽（不泄露存在性以外的信息，统一文案）。
        MemoryScope.set("bob", "chat:t1");
        assertEquals("无法删除：该记忆不属于当前用户。",
                call(tool("memory_delete"), "{\"memory_id\": \"" + record.memoryId() + "\"}", null));

        // 属主删除成功：回执带内容前 60 码点。
        MemoryScope.set("alice", "chat:t1");
        String out = call(tool("memory_delete"), "{\"memory_id\": \"" + record.memoryId() + "\"}", null);
        assertEquals("已删除：" + io.github.paopaonb666.agentbase.core.text.CharMetrics
                .head("旧的部署环境是 CentOS 7", 60), out);
    }

    // ── knowledge_search ──────────────────────────────────────────────────

    @Test
    void knowledgeSearchValidatesAndRendersAnchorsWithSourcesEvent() {
        store.putChunks(List.of(
                new DocChunk("c1", "f1", "t", "alice", "chat", 0,
                        "部署".repeat(250) + "——超长分块尾部被截断", List.of(),
                        new byte[0], null, 1.0),
                new DocChunk("c2", "f1", "t", "alice", "chat", 1,
                        "另一个分块：测试环境变量", List.of(), new byte[0], null, 1.0)));
        MemoryScope.set("alice", "chat:t1");

        // 无 NodeContext：emit 无操作但不影响文本渲染。
        String out = call(tool("knowledge_search"), "{\"query\": \"部署\"}", null);
        assertTrue(out.startsWith("知识库检索结果（回答引用以上内容时请用 [n] 标注来源序号）："), out);
        assertTrue(out.contains("[1]（相关度 "), out);
        assertTrue(out.contains(" · 文件 f1 第 1 段）"), out);
        // 单块回喂截断 400 码点（KNOWLEDGE_SNIPPET_CHARS）。
        assertTrue(io.github.paopaonb666.agentbase.core.text.CharMetrics.len(out) < 1000, out);

        assertEquals("（知识库中没有相关内容）",
                call(tool("knowledge_search"), "{\"query\": \"完全无关的查询\"}", null));
        assertEquals("query 必须是 1..2000 字符",
                errorOf(() -> call(tool("knowledge_search"), "{\"query\": \"\"}", null)).getMessage());
        assertEquals("limit 必须在 1..20 之间",
                errorOf(() -> call(tool("knowledge_search"), "{\"query\": \"q\", \"limit\": -1}", null))
                        .getMessage());

        // 带 NodeContext：SourcesEvent 与 [n] 序号同序对齐（#338 契约）。
        var emitted = new CopyOnWriteArrayList<Object>();
        var nodeContext = new io.github.paopaonb666.agentbase.core.graph.NodeContext() {
            @Override public io.github.paopaonb666.agentbase.core.graph.RunConfig config() { return null; }
            @Override public void emit(Object event) { emitted.add(event); }
            @Override public String callTool(String name, String argsJson) { return ""; }
            @Override public io.github.paopaonb666.agentbase.core.tools.ToolPool tools() { return null; }
            @Override public boolean cancelled() { return false; }
        };
        var ctx = new AgentTool.ToolContext(null, nodeContext);
        call(tool("knowledge_search"), "{\"query\": \"部署\"}", ctx);
        Events.SourcesEvent sources = emitted.stream()
                .filter(Events.SourcesEvent.class::isInstance)
                .map(Events.SourcesEvent.class::cast)
                .findFirst().orElseThrow();
        assertEquals(1, sources.sources().size());
        assertEquals("文件 f1 第 1 段", sources.sources().get(0).title());
    }
}
