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
import io.github.paopaonb666.agentbase.core.text.CharMetrics;
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
 * 常量逐字节复制——例外：SUMMARY_PROMPT 已改造为「三键结构化模板 +
 * 角色参数」（#329，Java 单边先行，Python 对齐需求登记差异登记簿）：
 * rolling（会话滚动）与 compaction（历史折叠，#328 消费）两角色共享同一
 * 三键 JSON 输出契约（task_state / key_entities / open_questions），
 * 解析失败回退原文保证旧行为可回退。
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

    /**
     * 抽取候选二次 LLM 自检提示（A-43）：批量审查候选表——列表内去重/
     * 特异性剔除，输出=要剔除的下标数组（全部通过输出空数组）。
     */
    public static final String SELF_CHECK_PROMPT = """
            你是记忆抽取质检员。下面是从对话中抽取的候选记忆列表（编号从 0 开始）。请审查每一条：
            1. 与列表内其他条目内容重复或几乎相同 → 只保留信息量最大的一条，其余剔除；
            2. 泛泛而谈、无具体信息量（寒暄、空洞总结、"用户聊了天"之类）→ 剔除；
            3. 其余通过。
            只输出要剔除的下标 JSON 数字数组（如 [0,2]）；全部通过输出 []。不要输出其他内容。

            候选列表：
            {candidates}""";

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

    /**
     * 摘要三键输出契约（#329 单源）：rolling 与 compaction 两角色拼接共用，
     * 键集 {@link #SUMMARY_KEYS}。三键体单源——改这里 = 两个角色同步改。
     */
    public static final String SUMMARY_KEYS_BODY = """
            输出纯 JSON 对象（不要 markdown 围栏、不要解释），三键固定：
            {"task_state": "任务态：进行到哪一步、已有哪些结论、正在做什么",
              "key_entities": "关键实体：人名/项目/数字与用户表达的关键诉求（保留具体值）",
              "open_questions": "未决问题：悬而未决的事项、待确认的约定、下一步计划"}
            三个键都是字符串，缺内容给空串；不要新增键、不要嵌套。""";

    /** 滚动摘要角色（#329）：300 字引导维持、存储 1000 硬截维持。 */
    public static final String SUMMARY_PROMPT = """
            你是会话摘要器。把既有摘要与新对话合并为一份更新的滚动摘要。

            规则：
            - 摘要覆盖"到目前为止这个会话聊过什么"：主题、结论、未决事项、
              用户表达的关键诉求
            - 既有摘要中仍然成立的内容保留；被新对话推翻的以新对话为准
            - 第三人称、简洁，总字数不超过 300 字
            """ + SUMMARY_KEYS_BODY + """


            既有摘要（可能为空）：
            {prior}

            最近对话转写：
            {transcript}""";

    /**
     * 历史折叠角色（#328 压缩任务消费）：把待折叠段转写压成 {target} 字
     * 以内的三键摘要。{target} 由调用方按预算参数化（targetChars）。
     */
    public static final String COMPACTION_SUMMARY_PROMPT = """
            你是会话历史压缩器。把下面这段较长的历史对话折叠为一份紧凑摘要，
            供后续对话替代被折叠的原文使用。

            规则：
            - 覆盖整段历史：任务进展与结论、关键实体与数字、未决事项
            - 第三人称、信息密集；总字数控制在 {target} 字以内
            """ + SUMMARY_KEYS_BODY + """


            历史对话转写：
            {transcript}""";

    /** 整合决策的合法操作集。 */
    public static final List<String> DECISION_OPS = List.of("ADD", "UPDATE", "DELETE", "NONE");

    /** 摘要模板三键（#329）：渲染顺序即持久化顺序（任务态/关键实体/未决问题）。 */
    public static final List<String> SUMMARY_KEYS = List.of("task_state", "key_entities", "open_questions");

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

        /**
         * 候选内容向量（#38 预筛用）：{@code embedder.embed} 的单文本形态，
         * 降级（无 embedding）返回 null——预筛跳过、照旧进 LLM。default
         * 实现返回 null，对手写桩向后兼容（预筛自动旁路）。
         */
        default List<Float> embedCandidate(String content) {
            return null;
        }
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

    /**
     * 三键摘要 → 可读文本（#329 渲染契约）：注入出口的持久化格式，一旦
     * 定死不得再改（改 = 冻结段字节变化 = 破坏供应商前缀缓存，须一次性
     * 切换）。三节定序渲染，缺值占位「（无）」。
     */
    public static String renderStructuredSummary(Map<?, ?> keys) {
        return "【任务态】" + summarySection(keys.get("task_state")) + '\n'
                + "【关键实体】" + summarySection(keys.get("key_entities")) + '\n'
                + "【未决问题】" + summarySection(keys.get("open_questions"));
    }

    private static String summarySection(Object value) {
        String text = value == null ? "" : String.valueOf(value).strip();
        return text.isEmpty() ? "（无）" : text;
    }

    /**
     * LLM 原始输出 → 摘要文本（#329）：按三键契约解析（{@link #parseJsonBlock}
     * 容围栏/尾注），命中至少一个三键才渲染；解析失败或非三键输出回退原文
     * ——rolling 角色的旧行为（自由文本原样存储）由此可回退。空输出返回
     * 空串（调用方按无摘要处理）。
     */
    public static String structuredSummaryText(String raw) {
        String text = raw == null ? "" : raw.strip();
        if (text.isEmpty()) {
            return "";
        }
        try {
            Object parsed = parseJsonBlock(text);
            if (parsed instanceof Map<?, ?> map) {
                for (String key : SUMMARY_KEYS) {
                    if (map.containsKey(key)) {
                        return renderStructuredSummary(map);
                    }
                }
            }
        } catch (Exception ignored) {
            // 解析失败回退原文（consolidate 的 fallback 先例）。
        }
        return text;
    }

    /** 把消息渲染成抽取/摘要用的对话转写（保尾部，超限截断）。 */
    public static String renderTranscript(List<ChatMessage> messages, int maxChars) {
        List<String> lines = new ArrayList<>();
        for (ChatMessage message : messages) {
            // #328/#330：压缩摘要的 SystemMessage 是历史折叠产物，跳过——
            // 摘要内容不重复进抽取转写（否则会被再抽取/再摘要）。SystemMessage
            // 本就不进转写行（下方无分支），此处显式守卫防未来加分支时漏掉。
            if (message instanceof SystemMessage system
                    && system.text().startsWith(
                            io.github.paopaonb666.agentbase.core.context.ContextEngine
                                    .COMPACTION_SUMMARY_PREFIX)) {
                continue;
            }
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
                        + CharMetrics.head(text, 300));
            }
        }
        String transcript = String.join("\n", lines);
        if (CharMetrics.len(transcript) <= maxChars) {
            return transcript;
        }
        // 保尾部：最近的内容对记忆形成更重要。截断按码点（Python len/
        // 切片语义），代理对不劈半。
        return "……（前文截断）\n" + CharMetrics.tail(transcript, maxChars);
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

    // ── 抽取自检（A-43）──────────────────────────────────────────────

    /**
     * 抽取候选二次 LLM 自检（A-43，{@code MEMORY_EXTRACTION_SELF_CHECK}
     * 默认关）：对候选表做一遍去重/特异性复核——批量单次 LLM 调用（快档，
     * 与抽取同 {@link #bulkFast()} 口径），返回剔除后的候选表。三态：
     * 判定剔除（下标命中）/ 通过（原表返回）/ LLM 失败或输出不可解析
     * <b>回退原判全保留</b>——自检是质量增强层，绝不阻断主管线；越界/
     * 非数字下标忽略。由 captureTurn 在 extract 与 consolidate 之间调用
     * （开关关时整层跳过，零额外调用）。
     */
    public List<Candidate> selfCheckCandidates(List<Candidate> candidates) {
        if (candidates.isEmpty()) {
            return candidates;
        }
        StringBuilder listing = new StringBuilder();
        for (int i = 0; i < candidates.size(); i++) {
            listing.append(i).append(": ").append(candidates.get(i).content()).append('\n');
        }
        String raw;
        try {
            raw = complete(SELF_CHECK_PROMPT.replace("{candidates}", listing.toString()),
                    bulkFast());
        } catch (Exception e) {
            LOG.warn("memory: 抽取自检 LLM 失败，回退原判（全保留 {} 条）", candidates.size(), e);
            return candidates;
        }
        Object parsed;
        try {
            parsed = parseJsonBlock(raw);
        } catch (Exception e) {
            LOG.warn("memory: 抽取自检输出不可解析，回退原判（全保留 {} 条）", candidates.size());
            return candidates;
        }
        if (!(parsed instanceof List<?> drops)) {
            return candidates;
        }
        java.util.Set<Integer> dropIdx = new java.util.HashSet<>();
        for (Object drop : drops) {
            try {
                int idx = Integer.parseInt(String.valueOf(drop).strip());
                if (idx >= 0 && idx < candidates.size()) {
                    dropIdx.add(idx);
                }
            } catch (NumberFormatException ignored) {
                // 非数字下标忽略（宽容解析，回退=保留该候选）。
            }
        }
        if (dropIdx.isEmpty()) {
            return candidates;
        }
        if (dropIdx.size() >= candidates.size()) {
            // 防御：LLM 要求剔除全部候选按理会发生在输出错乱时——全保留
            // 回退原判（空表=整轮无记忆可落，等价静默丢弃，宁可保守）。
            LOG.warn("memory: 抽取自检要求剔除全部 {} 条，视为输出错乱回退原判", candidates.size());
            return candidates;
        }
        List<Candidate> kept = new ArrayList<>(candidates.size());
        for (int i = 0; i < candidates.size(); i++) {
            if (!dropIdx.contains(i)) {
                kept.add(candidates.get(i));
            }
        }
        return kept;
    }

    // ── 整合 ─────────────────────────────────────────────────────────────

    /**
     * 对一条候选拿到整合决策并落库；返回实际执行的 op。
     *
     * <p>敏感内容红线（#48）：候选内容命中
     * {@link io.github.paopaonb666.agentbase.memory.textkit.SensitiveContentPolicy}
     * 规则即拒绝——不入检索、不入库，只记一条 {@code status=rejected}
     * 审计（detail 带规则名，不写原文，避免把密钥落进审计表）并返回
     * {@code REJECTED}。可经 {@code MEMORY_SENSITIVE_CONTENT_FILTER=false}
     * 关闭。Java 先行单边增强（Python 侧无此层）。
     */
    public String consolidate(Candidate candidate, String userId, String agentId, String threadId) {
        if (settings.memory.sensitiveContentFilter) {
            long started = System.nanoTime();
            List<String> rules = io.github.paopaonb666.agentbase.memory.textkit.SensitiveContentPolicy
                    .INSTANCE.violations(candidate.content());
            if (!rules.isEmpty()) {
                long durationMs = (System.nanoTime() - started) / 1_000_000;
                ops.recordOp("consolidate", userId, agentId, threadId,
                        Map.of("rules", String.join(",", rules), "op", "REJECTED"),
                        "rejected", "", durationMs);
                LOG.info("memory: 候选记忆命中敏感内容红线（{}），已拒绝入库", String.join(",", rules));
                return "REJECTED";
            }
        }
        List<io.github.paopaonb666.agentbase.memory.retrieval.Retrieval.ScoredMemory> similar =
                ops.search(userId, agentId, candidate.content(), 3);
        // 去重硬阈值预筛（#38）：候选与既有记忆最高余弦 > 阈值 → 内容几乎
        // 相同，直接 NONE 不烧 LLM 裁决；预筛命中落审计（op=consolidate +
        // detail.prefilter=dedup_skip，写 max_sim 数值不落原文）。降级路径
        // （任一侧无向量）跳过预筛、照旧进 LLM——行为与旧一致。
        if (settings.memory.dedupPrefilter && !similar.isEmpty()) {
            List<Float> candidateVector = ops.embedCandidate(candidate.content());
            if (candidateVector != null) {
                double maxSim = -1.0;
                String matchedId = null;
                for (var item : similar) {
                    Double sim = io.github.paopaonb666.agentbase.memory.retrieval.Retrieval
                            .cosineSimilarity(candidateVector,
                                    io.github.paopaonb666.agentbase.memory.store.Models
                                            .decodeEmbedding(item.record().embedding()));
                    if (sim != null && sim > maxSim) {
                        maxSim = sim;
                        matchedId = item.record().memoryId();
                    }
                }
                if (maxSim > settings.memory.dedupSkipCosine) {
                    ops.recordOp("consolidate", userId, agentId, threadId,
                            Map.of("op", "NONE", "prefilter", "dedup_skip",
                                    "max_sim", String.format(java.util.Locale.ROOT, "%.4f", maxSim),
                                    "matched_memory_id", matchedId == null ? "" : matchedId),
                            "ok", "", 0);
                    return "NONE";
                }
            }
        }
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
                    contentOrFallback(content, candidate),
                    candidate.kind(), candidate.tags(), candidate.salience(), threadId);
            case "UPDATE" -> {
                if (memoryId == null) {
                    op = "NONE";
                } else if (!ops.updateMemory(String.valueOf(memoryId),
                        contentOrFallback(content, candidate))) {
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

    /**
     * 整合落库 content 的统一回落（#136）：LLM 把 null 序列化成字面
     * "null" 或空串时与缺失等同，两个落库口（ADD/UPDATE）同一规则——
     * 之前 ADD 多回落字面 "null"、UPDATE 不查，会产出与 Python 不同的
     * 记忆行。非字符串值按 str() 口径转字符串。
     */
    private static String contentOrFallback(Object content, Candidate candidate) {
        if (content == null) {
            return candidate.content();
        }
        String text = String.valueOf(content);
        return (text.isEmpty() || "null".equals(text)) ? candidate.content() : text;
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

    /**
     * 把既有摘要与最新对话合并为新的滚动摘要（#329 模板化）：LLM 响应 →
     * 三键解析 → 渲染 → 1000 码点截断。解析失败回退原文——旧行为
     * （自由文本原样存储、空响应 no-op）逐字节可回退。
     */
    public boolean updateSummary(String userId, String agentId, String threadId, String transcript) {
        var prior = ops.getSummary(userId, threadId);
        String raw = complete(SUMMARY_PROMPT
                .replace("{prior}", prior == null ? "（空）" : prior.summary())
                .replace("{transcript}", transcript), bulkFast());
        String text = structuredSummaryText(raw);
        if (text.isEmpty()) {
            return false;
        }
        var priorCount = prior == null ? 0 : prior.coveredMessageCount();
        ops.upsertSummary(new io.github.paopaonb666.agentbase.memory.store.Models.SessionSummary(
                userId, threadId, CharMetrics.head(text, 1000),
                agentId, priorCount, io.github.paopaonb666.agentbase.memory.store.Models.now()));
        return true;
    }

    /**
     * 历史折叠摘要（#328 压缩任务消费，compaction 角色）：把待折叠段转写
     * 压成 {@code targetChars} 字以内的三键摘要。档位随 pipelineProfile
     * （{@link #bulkFast} 单源，R4-a F9：all_main=主力），与抽取/画像/滚动
     * 摘要一致；快档未装配时回落主力（既有逻辑）。返回渲染后的摘要文本；
     * 模型输出不可用（空）时返回空串，调用方按放弃本轮处理。
     */
    public String compactSummary(String transcript, int targetChars) {
        String raw = complete(COMPACTION_SUMMARY_PROMPT
                .replace("{target}", String.valueOf(targetChars))
                .replace("{transcript}", transcript), bulkFast());
        return structuredSummaryText(raw);
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
        // 审计 detail 对齐 Python（#135）：成功分支按结果形状记录——dict
        // 原样、list 取 content 前 120 字摘要、其余包一层 {"result": …}；
        // 之前恒写 {"human_count": N}，审计端点无法回答「这轮抽出了什么/
        // 整合了什么」。失败分支维持 human_count。
        class Step<T> {
            T run(String op, java.util.function.Supplier<T> body) {
                long started = System.nanoTime();
                try {
                    T result = body.get();
                    long durationMs = (System.nanoTime() - started) / 1_000_000;
                    ops.recordOp(op, userId, agentId, threadId,
                            detailOf(result), "ok", "", durationMs);
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

            /** 结果 → 审计 detail（对齐 Python _run 的三种形状）。 */
            private Map<String, Object> detailOf(Object result) {
                if (result instanceof Map<?, ?> map) {
                    @SuppressWarnings("unchecked")
                    Map<String, Object> cast = (Map<String, Object>) map;
                    return cast;
                }
                if (result instanceof List<?> list) {
                    // 候选记忆等数据列表：取可读摘要（content 前 120 码点）。
                    List<String> summary = new ArrayList<>();
                    for (Object item : list) {
                        String text = item instanceof Candidate c
                                ? c.content() : String.valueOf(item);
                        summary.add(io.github.paopaonb666.agentbase.core.text.CharMetrics
                                .head(text, 120));
                    }
                    Map<String, Object> wrapped = new LinkedHashMap<>();
                    wrapped.put("result", summary);
                    return wrapped;
                }
                // singletonMap 允许 null 值（result 本身可能为 null）。
                return java.util.Collections.singletonMap("result", result);
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
                // A-43 抽取自检（默认关）：extract 与 consolidate 之间的
                // 质量闸——拒绝/通过/失败回退三态见 selfCheckCandidates；
                // 审计走独立 self_check op（detail=保留候选摘要表）。
                if (settings.memory.extractionSelfCheck) {
                    // extracted 随后会被重赋值，lambda 捕获走 final 局部。
                    final List<Candidate> preCheck = extracted;
                    @SuppressWarnings("unchecked")
                    List<Candidate> checked = (List<Candidate>) step.run("self_check",
                            () -> selfCheckCandidates(preCheck));
                    if (checked != null) {
                        extracted = checked;
                    }
                }
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
