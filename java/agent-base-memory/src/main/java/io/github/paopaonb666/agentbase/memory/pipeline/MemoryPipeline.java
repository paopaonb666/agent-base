package io.github.paopaonb666.agentbase.memory.pipeline;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.BiConsumer;

import io.github.paopaonb666.agentbase.core.AgentModule;
import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.core.llm.ModelRuntime;
import io.github.paopaonb666.agentbase.extensions.Metrics;

/**
 * LLM 驱动的记忆形成管线；所有步骤失败安全（绝不打断主对话）。
 *
 * <p>模型分派（成本治理）：抽取/画像合并/摘要属于"简单、大量、重复"的
 * 批量工作，走快档（fast_llm——LLM_FAST_* 装配的限流回退包装）；整合
 * 裁决是唯一会不可逆改动记忆库的破坏性决策步，默认永远走主力模型
 * （{@code pipeline_profile="split"}）。快档未装配时一切回落主力——
 * 快档是优化不是依赖。
 *
 * <p>对应 Python 侧 {@code agent_base_memory/pipeline.py}；四个 prompt
 * 常量逐字节复制。
 */
public final class MemoryPipeline {

    private static final Logger LOG = LoggerFactory.getLogger(MemoryPipeline.class);

    // ───────────────────────────── Prompt 常量 ─────────────────────────────

    public static final String EXTRACTION_PROMPT = """
            你是记忆抽取器。从下面的对话转写中提取"值得跨会话记住"的信息。

            只提取这些类别：
            - 用户的身份、背景、长期目标（semantic）
            - 用户的偏好、习惯、约束（semantic）
            - 对话中发生的重要事件与结论（episodic）
            - 用户明确要求记住的规则或工作方式（procedural）

            不要提取：闲聊、寒暄、一次性的临时请求、对话本身的字面内容。
            没有值得记的就返回空数组。宁缺毋滥，最多 5 条。

            输出纯 JSON 数组（不要 markdown 围栏、不要解释），每条：
            {"content": "第三人称、自包含的一句话",
              "kind": "semantic|episodic|procedural", "salience": 0.0到1.0, "tags": ["标签"]}

            对话转写：
            {transcript}""";

    public static final String CONSOLIDATION_PROMPT = """
            你是记忆整合器。新候选记忆要与既有记忆保持一致：去重、纠错、剔除失效。

            新候选：
            {candidate}

            既有相似记忆（id: 内容）：
            {existing}

            决策规则：
            - 既有记忆已覆盖候选信息 → NONE（或内容更完整时 UPDATE 该条）
            - 候选是既有记忆的更新/修正 → UPDATE（给出修正后的完整 content）
            - 候选与某条既有记忆直接矛盾（信息已失效，如"已换工作"）→ DELETE 该条；
              只 DELETE 还是"DELETE+ADD"按信息价值判断
            - 全新的信息 → ADD

            输出纯 JSON（不要 markdown 围栏、不要解释）：
            {"op": "ADD|UPDATE|DELETE|NONE",
              "memory_id": "仅 UPDATE/DELETE 需要", "content": "仅 UPDATE/ADD 的最终内容"}""";

    public static final String PROFILE_MERGE_PROMPT = """
            你是用户画像维护器。把新事实合并进既有画像 JSON。

            规则：
            - 保留仍然有效的既有内容；与新事实矛盾的以新事实为准
            - 画像条目要简短、第三人称、可执行
            - 新事实与画像无关时保持画像不变（只格式化）
            - 总长度控制在 600 字以内

            既有画像 JSON：
            {profile}

            本轮新事实（逐条）：
            {facts}

            输出合并后的完整画像 JSON（不要 markdown 围栏、不要解释）。""";

    public static final String SUMMARY_PROMPT = """
            你是会话摘要器。把既有摘要与新对话合并为一份更新的滚动摘要。

            规则：
            - 摘要覆盖"到目前为止这个会话聊过什么"：主题、结论、未决事项、
              用户表达的关键诉求
            - 既有摘要中仍然成立的内容保留；被新对话推翻的以新对话为准
            - 第三人称、简洁，不超过 300 字

            既有摘要（可能为空）：
            {prior}

            最近对话转写：
            {transcript}

            输出新摘要的纯文本（不要 JSON、不要标题、不要解释）。""";

    /** 整合决策的合法操作集。 */
    public static final List<String> DECISION_OPS = List.of("ADD", "UPDATE", "DELETE", "NONE");

    /** 画像分节（Python PROFILE_SECTIONS）。 */
    public static final List<String> PROFILE_SECTIONS =
            List.of("基本信息", "偏好", "工作背景", "沟通风格", "长期目标");

    /** 一条抽取候选。 */
    public record Candidate(String content, String kind, double salience, List<String> tags) {}

    /** 通配 Map → String 键 Map（Jackson 解析产物；键原样转 String）。 */
    private static Map<String, Object> asStringMap(Map<?, ?> map) {
        Map<String, Object> out = new LinkedHashMap<>();
        for (Map.Entry<?, ?> e : map.entrySet()) {
            out.put(String.valueOf(e.getKey()), e.getValue());
        }
        return out;
    }

    private final MemoryOps ops;
    private final Settings settings;
    private final ModelRuntime llm;
    private final AgentModule.ChatInvokerPort fastLlm;

    /** 管线对服务的操作面（对应 Python 侧传进来的 service 引用的受控子集）。 */
    public interface MemoryOps {
        List<io.github.paopaonb666.agentbase.memory.retrieval.Retrieval.ScoredMemory> search(
                String userId, String agentId, String query, Integer topK);

        void addMemory(String userId, String agentId, String content, String kind,
                       List<String> tags, double salience, String sourceThreadId);

        boolean updateMemory(String memoryId, String content);

        boolean deleteMemory(String memoryId);

        Map<String, Object> getProfile(String userId);

        void saveProfile(String userId, Map<String, Object> profile);

        io.github.paopaonb666.agentbase.memory.store.Models.SessionSummary getSummary(
                String userId, String threadId);

        void upsertSummary(io.github.paopaonb666.agentbase.memory.store.Models.SessionSummary summary);

        void recordOp(String op, String userId, String agentId, String threadId,
                      Map<String, Object> detail, String status, String errorText, long durationMs);
    }

    public MemoryPipeline(ModelRuntime llm, AgentModule.ChatInvokerPort fastLlm,
                          Settings settings, MemoryOps ops) {
        this.llm = llm;
        this.fastLlm = fastLlm;
        this.settings = settings;
        this.ops = ops;
    }

    /** 按档位取模型；快档缺席时回落主力（不报错）。 */
    private String complete(String prompt, boolean fast) {
        if (fast && fastLlm != null) {
            String response = fastLlm.invoke(List.of(UserMessage.from(prompt)));
            return response == null ? "" : response;
        }
        AiMessage response = llm.invoke(List.of(UserMessage.from(prompt)));
        return response == null || response.text() == null ? "" : response.text();
    }

    /** 抽取/画像/摘要的档位：split 与 all_fast 都交给快档侧。 */
    private boolean bulkFast() {
        return !"all_main".equals(settings.memory.pipelineProfile);
    }

    /** 整合裁决的档位：只有显式 all_fast 才交给快档侧。 */
    private boolean decisionFast() {
        return "all_fast".equals(settings.memory.pipelineProfile);
    }

    /**
     * 从容解析 LLM 输出里的 JSON：剥 markdown 围栏、取首个平衡块
     * （对应 Python _parse_json_block）。
     */
    public static Object parseJsonBlock(String text) {
        String cleaned = text == null ? "" : text.strip();
        java.util.regex.Matcher fence = java.util.regex.Pattern
                .compile("```(?:json)?\\s*(.*?)```", java.util.regex.Pattern.DOTALL)
                .matcher(cleaned);
        if (fence.find()) {
            cleaned = fence.group(1).strip();
        }
        com.fasterxml.jackson.databind.ObjectMapper mapper = new com.fasterxml.jackson.databind.ObjectMapper();
        try {
            return mapper.readValue(cleaned, Object.class);
        } catch (Exception ignored) {
            // 兜底：截取第一个 [ 或 { 到与之匹配的最后一个 ] 或 }。
        }
        for (String[] pair : new String[][]{{"[", "]"}, {"{", "}"}}) {
            int start = cleaned.indexOf(pair[0]);
            int end = cleaned.lastIndexOf(pair[1]);
            if (start != -1 && end > start) {
                try {
                    return mapper.readValue(cleaned.substring(start, end + 1), Object.class);
                } catch (Exception ignored) {
                    // 尝试下一对括号
                }
            }
        }
        throw new IllegalArgumentException(
                "无法从 LLM 输出解析 JSON：" + cleaned.substring(0, Math.min(cleaned.length(), 200)));
    }

    /** 把消息渲染成抽取/摘要用的对话转写（保尾部，超限截断）。 */
    public static String renderTranscript(List<ChatMessage> messages, int maxChars) {
        List<String> lines = new ArrayList<>();
        for (ChatMessage message : messages) {
            String text = textOf(message).strip();
            if (text.isEmpty()) {
                continue;
            }
            if (message instanceof UserMessage) {
                lines.add("用户：" + text);
            } else if (message instanceof AiMessage) {
                // 纯 tool_calls 的 AI 消息（无文本）没有转写价值。
                lines.add("助手：" + text);
            } else if (message instanceof ToolExecutionResultMessage tool) {
                lines.add("工具[" + tool.toolName() + "]："
                        + text.substring(0, Math.min(text.length(), 300)));
            }
        }
        String transcript = String.join("\n", lines);
        if (transcript.length() <= maxChars) {
            return transcript;
        }
        // 保尾部：最近的内容对记忆形成更重要。
        return "……（前文截断）\n" + transcript.substring(transcript.length() - maxChars);
    }

    private static String textOf(ChatMessage message) {
        if (message instanceof UserMessage user) {
            StringBuilder sb = new StringBuilder();
            for (dev.langchain4j.data.message.Content c : user.contents()) {
                if (c instanceof dev.langchain4j.data.message.TextContent t) {
                    sb.append(t.text());
                }
            }
            return sb.toString();
        }
        if (message instanceof SystemMessage s) return s.text();
        if (message instanceof AiMessage a) return a.text() == null ? "" : a.text();
        if (message instanceof ToolExecutionResultMessage t) return t.text();
        return String.valueOf(message);
    }

    // ── 抽取 ─────────────────────────────────────────────────────────────

    public List<Candidate> extractCandidates(String transcript) {
        String raw = complete(EXTRACTION_PROMPT.replace("{transcript}", transcript), bulkFast());
        Object parsed = parseJsonBlock(raw);
        if (!(parsed instanceof List<?> list)) {
            throw new IllegalArgumentException("抽取输出不是 JSON 数组");
        }
        List<Candidate> candidates = new ArrayList<>();
        for (Object item : list.subList(0, Math.min(list.size(), 5))) {
            if (!(item instanceof Map<?, ?> rawMap)) {
                continue;
            }
            Map<String, Object> map = asStringMap(rawMap);
            String content = String.valueOf(map.getOrDefault("content", "")).strip();
            if (content.isEmpty() || "null".equals(content)) {
                continue;
            }
            String kind = String.valueOf(map.getOrDefault("kind", "semantic"));
            if (!io.github.paopaonb666.agentbase.memory.store.Models.KNOWN_MEMORY_KINDS.contains(kind)) {
                kind = "semantic";
            }
            double salience;
            try {
                salience = Math.min(Math.max(Double.parseDouble(
                        String.valueOf(map.getOrDefault("salience", "0.5"))), 0.0), 1.0);
            } catch (NumberFormatException e) {
                salience = 0.5;
            }
            List<String> tags = new ArrayList<>();
            if (map.get("tags") instanceof List<?> rawTags) {
                for (Object tag : rawTags) {
                    if (tags.size() >= 10) {
                        break;
                    }
                    tags.add(String.valueOf(tag));
                }
            }
            candidates.add(new Candidate(content, kind, salience, tags));
        }
        return candidates;
    }

    // ── 整合 ─────────────────────────────────────────────────────────────

    /** 对一条候选拿到整合决策并落库；返回实际执行的 op。 */
    public String consolidate(Candidate candidate, String userId, String agentId, String threadId) {
        List<io.github.paopaonb666.agentbase.memory.retrieval.Retrieval.ScoredMemory> similar =
                ops.search(userId, agentId, candidate.content(), 3);
        StringBuilder existing = new StringBuilder();
        for (var item : similar) {
            if (existing.length() > 0) {
                existing.append('\n');
            }
            existing.append(item.record().memoryId()).append(": ").append(item.record().content());
        }
        String candidateJson;
        try {
            // json.dumps({...}, ensure_ascii=False) 形态（默认分隔符，键序
            // content→kind 与 Python dict 字面量一致）。
            java.util.Map<String, Object> candidateMap = new java.util.LinkedHashMap<>();
            candidateMap.put("content", candidate.content());
            candidateMap.put("kind", candidate.kind());
            candidateJson = io.github.paopaonb666.agentbase.memory.textkit.PyJson.dumps(candidateMap);
        } catch (Exception e) {
            candidateJson = "{\"content\": \"" + candidate.content() + "\"}";
        }
        String raw = complete(CONSOLIDATION_PROMPT
                .replace("{candidate}", candidateJson)
                .replace("{existing}", existing.isEmpty() ? "（无）" : existing.toString()), decisionFast());
        Object parsed = parseJsonBlock(raw);
        if (!(parsed instanceof Map<?, ?> rawParsed)) {
            throw new IllegalArgumentException("整合输出不是 JSON 对象");
        }
        Map<String, Object> map = asStringMap(rawParsed);
        String op = String.valueOf(map.getOrDefault("op", "NONE")).toUpperCase(java.util.Locale.ROOT);
        if (!DECISION_OPS.contains(op)) {
            op = "NONE";
        }
        Object memoryId = map.get("memory_id");
        Object content = map.get("content");
        switch (op) {
            case "ADD" -> ops.addMemory(userId, agentId,
                    content == null || "null".equals(String.valueOf(content)) || String.valueOf(content).isEmpty()
                            ? candidate.content() : String.valueOf(content),
                    candidate.kind(), candidate.tags(), candidate.salience(), threadId);
            case "UPDATE" -> {
                if (memoryId == null) {
                    op = "NONE";
                } else if (!ops.updateMemory(String.valueOf(memoryId),
                        content == null || String.valueOf(content).isEmpty()
                                ? candidate.content() : String.valueOf(content))) {
                    op = "NONE";
                }
            }
            case "DELETE" -> {
                if (memoryId == null || !ops.deleteMemory(String.valueOf(memoryId))) {
                    op = "NONE";
                }
            }
            default -> { /* NONE */ }
        }
        return op;
    }

    // ── 画像 ─────────────────────────────────────────────────────────────

    /** 把新事实合并进用户画像；画像不存在时先建骨架。返回是否更新。 */
    public boolean mergeProfile(String userId, List<String> facts) {
        if (facts == null || facts.isEmpty()) {
            return false;
        }
        Map<String, Object> profile = ops.getProfile(userId);
        if (profile == null || profile.isEmpty()) {
            profile = new LinkedHashMap<>();
            for (String section : PROFILE_SECTIONS) {
                profile.put(section, new ArrayList<String>());
            }
        }
        StringBuilder factLines = new StringBuilder();
        for (String fact : facts) {
            factLines.append("- ").append(fact).append('\n');
        }
        String profileJson;
        try {
            // json.dumps(profile, ensure_ascii=False, indent=1) 形态（1 空格
            // 缩进 + ",\n" + ": "），与 Python pipeline.py 的 merge prompt 逐位一致。
            profileJson = io.github.paopaonb666.agentbase.memory.textkit.PyJson.dumpsIndent(profile, 1);
        } catch (Exception e) {
            profileJson = "{}";
        }
        String raw = complete(PROFILE_MERGE_PROMPT
                .replace("{profile}", profileJson)
                .replace("{facts}", factLines.toString()), bulkFast());
        Object merged = parseJsonBlock(raw);
        if (!(merged instanceof Map<?, ?> rawMerged) || rawMerged.isEmpty()) {
            throw new IllegalArgumentException("画像合并输出不是 JSON 对象");
        }
        ops.saveProfile(userId, asStringMap(rawMerged));
        return true;
    }

    // ── 滚动摘要 ─────────────────────────────────────────────────────────

    /** 把既有摘要与最新对话合并为新的滚动摘要。 */
    public boolean updateSummary(String userId, String agentId, String threadId, String transcript) {
        var prior = ops.getSummary(userId, threadId);
        String raw = complete(SUMMARY_PROMPT
                .replace("{prior}", prior == null ? "（空）" : prior.summary())
                .replace("{transcript}", transcript), bulkFast());
        String text = raw.strip();
        if (text.isEmpty()) {
            return false;
        }
        var priorCount = prior == null ? 0 : prior.coveredMessageCount();
        ops.upsertSummary(new io.github.paopaonb666.agentbase.memory.store.Models.SessionSummary(
                userId, threadId, text.substring(0, Math.min(text.length(), 1000)),
                agentId, priorCount, io.github.paopaonb666.agentbase.memory.store.Models.now()));
        return true;
    }

    // ── 编排 ─────────────────────────────────────────────────────────────

    /**
     * 一轮对话结束后的完整形成流程；每步独立审计、失败安全。
     * {@code force=true} 旁路 MEMORY_CAPTURE_ENABLED 门控（夜间批脚本
     * T3.2 的替代形成路径）；服务端每轮路径永远用默认 False。
     */
    public Map<String, Object> captureTurn(String userId, String agentId, String threadId,
                                           String transcript, int humanCount, boolean force) {
        Map<String, Object> detail = new LinkedHashMap<>();
        detail.put("candidates", 0);
        Map<String, Object> opsMap = new LinkedHashMap<>();
        detail.put("ops", opsMap);
        detail.put("profile_updated", false);
        detail.put("summary_updated", false);
        detail.put("forced", force);

        // 单步包装：计时 + 失败审计 + 指标，异常绝不向上抛（对应 _run）。
        class Step<T> {
            T run(String op, java.util.function.Supplier<T> body) {
                long started = System.nanoTime();
                try {
                    T result = body.get();
                    long durationMs = (System.nanoTime() - started) / 1_000_000;
                    ops.recordOp(op, userId, agentId, threadId,
                            Map.of("human_count", humanCount), "ok", "", durationMs);
                    Metrics.memory().observe(op, "ok", durationMs / 1000.0);
                    return result;
                } catch (Exception exc) {
                    long durationMs = (System.nanoTime() - started) / 1_000_000;
                    LOG.warn("memory: {} 失败（不影响对话）：{}", op, exc.toString());
                    ops.recordOp(op, userId, agentId, threadId,
                            Map.of("human_count", humanCount), "error",
                            (exc.getClass().getSimpleName() + ": " + exc.getMessage())
                                    .substring(0, Math.min(300,
                                            (exc.getClass().getSimpleName() + ": " + exc.getMessage()).length())),
                            durationMs);
                    Metrics.memory().observe(op, "error", durationMs / 1000.0);
                    return null;
                }
            }
        }
        Step<Object> step = new Step<>();

        // 1) 抽取 + 逐条整合（合并计为一次 extract 审计 + 各 consolidate）。
        if (force || settings.memory.captureEnabled) {
            @SuppressWarnings("unchecked")
            List<Candidate> extracted = (List<Candidate>) step.run("extract",
                    () -> extractCandidates(transcript));
            if (extracted != null && !extracted.isEmpty()) {
                detail.put("candidates", extracted.size());
                List<String> accepted = new ArrayList<>();
                for (Candidate candidate : extracted) {
                    String op = (String) step.run("consolidate",
                            () -> consolidate(candidate, userId, agentId, threadId));
                    if (op != null) {
                        opsMap.put(candidate.content().substring(0,
                                Math.min(candidate.content().length(), 50)), op);
                        if ("ADD".equals(op) || "UPDATE".equals(op)) {
                            accepted.add(candidate.content());
                        }
                    }
                }
                // 2) 画像合并（用被接受的记忆作为事实源）。
                if (settings.memory.profileEnabled && !accepted.isEmpty()) {
                    Object updated = step.run("profile", () -> mergeProfile(userId, accepted));
                    detail.put("profile_updated", Boolean.TRUE.equals(updated));
                }
            }
        }

        // 3) 滚动摘要（独立于抽取：即使本轮不抽取，长会话也需要摘要）。
        if (settings.memory.summaryEnabled
                && humanCount >= settings.memory.summaryTriggerMessages) {
            Object updated = step.run("summary",
                    () -> updateSummary(userId, agentId, threadId, transcript));
            detail.put("summary_updated", Boolean.TRUE.equals(updated));
        }
        return detail;
    }
}
