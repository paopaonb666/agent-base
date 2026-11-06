package io.github.paopaonb666.agentbase.memory.tools;

import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.model.chat.request.json.JsonArraySchema;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import dev.langchain4j.model.chat.request.json.JsonStringSchema;

import java.util.List;
import java.util.Map;

import io.github.paopaonb666.agentbase.core.MemoryScope;
import io.github.paopaonb666.agentbase.core.ThreadId;
import io.github.paopaonb666.agentbase.core.text.CharMetrics;
import io.github.paopaonb666.agentbase.core.tools.AgentTool;
import io.github.paopaonb666.agentbase.memory.service.MemoryService;
import io.github.paopaonb666.agentbase.memory.store.Models.MemoryBlock;
import io.github.paopaonb666.agentbase.memory.store.Models.MemoryRecord;

/**
 * Agent 记忆工具（M6e）：agent 自服务的记忆读写与知识库检索。
 *
 * <p>五个工具（Letta 式"agent 管理自己的记忆"）：memory_search /
 * memory_save / memory_update_block / memory_delete / knowledge_search。
 *
 * <p><b>作用域传递</b>：user_id / thread_id 经 core/MemoryScope（MDC，
 * server 在拉起图执行前设置，工具线程继承）——CLI 等未设置的场景优雅
 * 回退到 default / 全局作用域。
 *
 * <p><b>检索作用域</b>：memory_search / knowledge_search 是<b>用户级</b>
 * （跨模块可见，user 隔离是唯一边界）；memory_save / 常驻块写入继续带
 * 模块归属（agent_id），供管理端按模块溯源与过滤。
 *
 * <p>对应 Python 侧 {@code agent_base_memory/tools.py}；输出文案逐字节对齐。
 */
public final class MemoryTools {

    private MemoryTools() {}

    public static final int DEFAULT_BLOCK_CHAR_LIMIT = 2000;
    /** knowledge_search 单条结果回给模型的文本截断（防止单块撑爆上下文）。 */
    public static final int KNOWLEDGE_SNIPPET_CHARS = 400;

    /** 读取作用域 → (user_id, agent_id, full_thread_id)。 */
    private record Scope(String userId, String agentId, String threadId) {

        static Scope current() {
            String userId = MemoryScope.currentUserId();
            String threadId = MemoryScope.currentThreadId();
            ThreadId parsed = ThreadId.tryParse(threadId);
            String agentId = parsed != null ? parsed.module() : "*";
            return new Scope(userId, agentId, threadId);
        }
    }

    /** 参数读取工具（argsJson → 字段，缺省对齐 Python Field 默认值）。 */
    private static final com.fasterxml.jackson.databind.ObjectMapper MAPPER =
            new com.fasterxml.jackson.databind.ObjectMapper();

    private static Map<String, Object> args(String argsJson) {
        try {
            return MAPPER.readValue(argsJson == null || argsJson.isEmpty() ? "{}" : argsJson,
                    new com.fasterxml.jackson.core.type.TypeReference<Map<String, Object>>() {});
        } catch (Exception e) {
            return Map.of();
        }
    }

    private static String str(Map<String, Object> args, String key, String fallback) {
        Object value = args.get(key);
        return value == null ? fallback : String.valueOf(value);
    }

    private static int intOf(Map<String, Object> args, String key, int fallback) {
        try {
            return Integer.parseInt(String.valueOf(args.get(key)));
        } catch (Exception e) {
            return fallback;
        }
    }

    /** 构建绑定到 MemoryService 的五个记忆工具。 */
    public static List<AgentTool> build(MemoryService memory) {
        return List.of(
                new AgentTool() {
                    @Override public String name() { return "memory_search"; }

                    @Override
                    public String description() {
                        return "检索关于当前用户的长期记忆（偏好、背景、过往约定等）。"
                                + "当需要回忆用户提过的事情或偏好时调用。";
                    }

                    @Override
                    public ToolSpecification specification() {
                        return ToolSpecification.builder().name(name()).description(description())
                                .parameters(JsonObjectSchema.builder()
                                        .addStringProperty("query", "要检索的记忆内容：关键词、人名、型号或语义描述")
                                        .addIntegerProperty("limit", "最多返回的条数")
                                        .required("query")
                                        .build())
                                .build();
                    }

                    @Override
                    public String call(String argsJson, ToolContext ctx) {
                        Map<String, Object> a = args(argsJson);
                        // 参数约束对应 Python _MemorySearchArgs（Pydantic）：
                        // 越界抛错走 ToolPool 归一（Python 侧 ValidationError
                        // 同样变成错误反馈），而不是静默钳制。
                        String query = str(a, "query", "");
                        if (query.isEmpty() || CharMetrics.len(query) > 2000) {
                            throw new IllegalArgumentException("query 必须是 1..2000 字符");
                        }
                        int limit = intOf(a, "limit", 5);
                        if (limit < 1 || limit > 20) {
                            throw new IllegalArgumentException("limit 必须在 1..20 之间");
                        }
                        // 检索是用户级的（S1）：跨模块可见；user 隔离是硬边界。
                        String userId = Scope.current().userId();
                        var scored = memory.searchScored(userId, null, query, limit);
                        if (scored.isEmpty()) {
                            return "（没有找到相关记忆）";
                        }
                        // 带 memory_id：agent 可据此调用 memory_delete 更正过时记忆。
                        StringBuilder sb = new StringBuilder("找到的相关记忆：");
                        for (var item : scored) {
                            MemoryRecord record = item.record();
                            sb.append("\n-（相关度 ").append(String.format(java.util.Locale.ROOT, "%.2f", item.score()))
                                    .append("）[id=")
                                    .append(record.memoryId(), 0, Math.min(12, record.memoryId().length()))
                                    .append("] ").append(record.content());
                        }
                        return sb.toString();
                    }
                },
                new AgentTool() {
                    @Override public String name() { return "memory_save"; }

                    @Override
                    public String description() {
                        return "把值得跨会话记住的信息写入长期记忆。用户说'记住……'、"
                                + "或对话中出现持久的偏好/约束/事实时调用。宁缺毋滥。";
                    }

                    @Override
                    public ToolSpecification specification() {
                        return ToolSpecification.builder().name(name()).description(description())
                                .parameters(JsonObjectSchema.builder()
                                        .addStringProperty("content",
                                                "要记住的信息，第三人称、自包含的一句话（如'用户的部署环境是 Ubuntu 22.04'）")
                                        .addStringProperty("kind",
                                                "记忆类别：semantic（事实/偏好）、episodic（事件）、procedural（规则/流程）")
                                        .addProperty("tags", JsonArraySchema.builder()
                                                .items(JsonStringSchema.builder().build())
                                                .description("检索用标签")
                                                .build())
                                        .required("content")
                                        .build())
                                .build();
                    }

                    @Override
                    public String call(String argsJson, ToolContext ctx) {
                        Map<String, Object> a = args(argsJson);
                        // 约束对应 Python _MemorySaveArgs：content 1..2000、tags ≤ 10。
                        String content = str(a, "content", "");
                        if (content.isEmpty() || CharMetrics.len(content) > 2000) {
                            throw new IllegalArgumentException("content 必须是 1..2000 字符");
                        }
                        List<String> tags = a.get("tags") instanceof List<?> raw
                                ? raw.stream().map(String::valueOf).toList() : List.of();
                        if (tags.size() > 10) {
                            throw new IllegalArgumentException("tags 最多 10 个");
                        }
                        Scope scope = Scope.current();
                        MemoryRecord record = memory.addMemory(scope.userId(), scope.agentId(),
                                content, str(a, "kind", "semantic"), tags, 0.5,
                                scope.threadId());
                        return "已记住（memory_id="
                                + record.memoryId().substring(0, Math.min(12, record.memoryId().length()))
                                + "…，kind=" + record.kind() + "）";
                    }
                },
                new AgentTool() {
                    @Override public String name() { return "memory_update_block"; }

                    @Override
                    public String description() {
                        return "更新你自己的常驻记忆块：persona（你的人设与工作方式）、"
                                + "human（对当前用户的画像摘要）或自定义标签。这些块每轮"
                                + "对话都会出现在你的上下文里。";
                    }

                    @Override
                    public ToolSpecification specification() {
                        return ToolSpecification.builder().name(name()).description(description())
                                .parameters(JsonObjectSchema.builder()
                                        .addStringProperty("label",
                                                "记忆块标签：persona（助手人设）、human（用户画像摘要）或自定义标签")
                                        .addStringProperty("operation", "append（追加，默认）或 replace（整块替换）")
                                        .addStringProperty("content", "块内容")
                                        .required("label", "content")
                                        .build())
                                .build();
                    }

                    @Override
                    public String call(String argsJson, ToolContext ctx) {
                        Map<String, Object> a = args(argsJson);
                        // 约束对应 Python _MemoryBlockArgs：label 1..64 且
                        // ^[\w.-]+$（(?U) 使 \w 支持unicode——Python re 的 \w
                        // 默认 unicode 语义），content 1..4000。
                        String label = str(a, "label", "");
                        if (label.isEmpty() || CharMetrics.len(label) > 64
                                || !label.matches("(?U)^[\\w.-]+$")) {
                            throw new IllegalArgumentException(
                                    "label 必须是 1..64 字符且匹配 [\\w.-]+");
                        }
                        String content = str(a, "content", "");
                        if (content.isEmpty() || CharMetrics.len(content) > 4000) {
                            throw new IllegalArgumentException("content 必须是 1..4000 字符");
                        }
                        String operation = str(a, "operation", "append");
                        Scope scope = Scope.current();
                        String agent = scope.agentId();
                        if (!"append".equals(operation) && !"replace".equals(operation)) {
                            return "未知 operation '" + operation + "'（只支持 append / replace）";
                        }
                        MemoryBlock existing = memory.store().getBlock(scope.userId(), agent, label);
                        int charLimit = existing != null ? existing.charLimit() : DEFAULT_BLOCK_CHAR_LIMIT;
                        String merged;
                        if ("append".equals(operation) && existing != null) {
                            merged = (existing.content().replaceAll("\\s+$", "") + "\n" + content).strip();
                        } else {
                            merged = content;
                        }
                        if (CharMetrics.len(merged) > charLimit) {
                            // 保尾部：最近写入的内容通常更重要（Letta 同语义）。
                            // 前缀计入预算，保证结果不超过 char_limit。
                            // 截断按码点（Python len/切片语义），代理对不劈半。
                            String note = "……（较早内容已截断）\n";
                            merged = note + CharMetrics.tail(merged, charLimit - note.length());
                        }
                        int version = (existing != null ? existing.version() : 0) + 1;
                        memory.store().upsertBlock(new MemoryBlock(scope.userId(), agent, label,
                                merged, charLimit, version,
                                io.github.paopaonb666.agentbase.memory.store.Models.now()));
                        return "记忆块 '" + label + "' 已更新（v" + version + "，"
                                + CharMetrics.len(merged) + "/" + charLimit + " 字符）";
                    }
                },
                new AgentTool() {
                    @Override public String name() { return "memory_delete"; }

                    @Override
                    public String description() {
                        return "删除一条已过时或错误的长期记忆（先用 memory_search 找到"
                                + " id）。当用户更正了此前的事实（如换了项目代号、搬了城市）"
                                + "时，配合 memory_save 使用：删旧、存新。";
                    }

                    @Override
                    public ToolSpecification specification() {
                        return ToolSpecification.builder().name(name()).description(description())
                                .parameters(JsonObjectSchema.builder()
                                        .addStringProperty("memory_id",
                                                "要删除的记忆 id（从 memory_search 结果的 [id=…] 中获取）")
                                        .required("memory_id")
                                        .build())
                                .build();
                    }

                    @Override
                    public String call(String argsJson, ToolContext ctx) {
                        // 约束对应 Python _MemoryDeleteArgs：4..64 且 ^[\w.-]+$。
                        String memoryId = str(args(argsJson), "memory_id", "");
                        if (CharMetrics.len(memoryId) < 4 || CharMetrics.len(memoryId) > 64
                                || !memoryId.matches("(?U)^[\\w.-]+$")) {
                            throw new IllegalArgumentException(
                                    "memory_id 必须是 4..64 字符且匹配 [\\w.-]+");
                        }
                        // user 隔离：只能删自己的记忆。
                        MemoryRecord record = memory.getMemory(memoryId);
                        if (record == null) {
                            return "记忆 " + memoryId + " 不存在（可能已删除，id 以 memory_search 结果为准）";
                        }
                        if (!record.userId().equals(Scope.current().userId())) {
                            return "无法删除：该记忆不属于当前用户。";
                        }
                        boolean deleted = memory.deleteMemory(memoryId);
                        return deleted
                                ? "已删除：" + CharMetrics.head(record.content(), 60)
                                : "删除失败（记录已不存在）";
                    }
                },
                new AgentTool() {
                    @Override public String name() { return "knowledge_search"; }

                    @Override
                    public String description() {
                        return "在用户上传过的文档构建的知识库里检索（PDF/DOCX/TXT 等）。"
                                + "当用户的问题可能在他们上传的文档中有答案时调用。";
                    }

                    @Override
                    public ToolSpecification specification() {
                        return ToolSpecification.builder().name(name()).description(description())
                                .parameters(JsonObjectSchema.builder()
                                        .addStringProperty("query", "要在用户上传的文档里检索的内容")
                                        .addIntegerProperty("limit", "最多返回的分块数")
                                        .required("query")
                                        .build())
                                .build();
                    }

                    @Override
                    public String call(String argsJson, ToolContext ctx) {
                        Map<String, Object> a = args(argsJson);
                        // 约束对应 Python _KnowledgeSearchArgs：query 1..2000、
                        // limit 1..20（越界报错而非钳制）。
                        String query = str(a, "query", "");
                        if (query.isEmpty() || CharMetrics.len(query) > 2000) {
                            throw new IllegalArgumentException("query 必须是 1..2000 字符");
                        }
                        int limit = intOf(a, "limit", 5);
                        if (limit < 1 || limit > 20) {
                            throw new IllegalArgumentException("limit 必须在 1..20 之间");
                        }
                        // 同 memory_search：知识库检索为用户级——文档是用户的知识资产。
                        String userId = Scope.current().userId();
                        var scored = memory.searchKnowledge(userId, null, query, limit);
                        if (scored.isEmpty()) {
                            return "（知识库中没有相关内容）";
                        }
                        StringBuilder sb = new StringBuilder("知识库检索结果：");
                        for (Map<String, Object> item : scored) {
                            String text = String.valueOf(item.get("text"));
                            sb.append("\n-（相关度 ")
                                    .append(String.format(java.util.Locale.ROOT, "%.2f", ((Number) item.get("score")).doubleValue()))
                                    .append("）").append(CharMetrics.head(text, KNOWLEDGE_SNIPPET_CHARS));
                        }
                        return sb.toString();
                    }
                });
    }
}
