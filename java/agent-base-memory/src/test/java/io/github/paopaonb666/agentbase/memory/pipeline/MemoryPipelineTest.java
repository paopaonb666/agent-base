package io.github.paopaonb666.agentbase.memory.pipeline;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import io.github.paopaonb666.agentbase.core.AgentModule;
import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.core.llm.ModelRuntime;
import io.github.paopaonb666.agentbase.memory.retrieval.Retrieval;
import io.github.paopaonb666.agentbase.memory.store.Models;
import org.junit.jupiter.api.Test;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * MemoryPipeline 四步编排测试（roadmap #296）：EXTRACTION/CONSOLIDATION/
 * PROFILE/SUMMARY 的脚本化编排、parseJsonBlock 容错、null content 处理、
 * 审计调用面。抽取/整合输出按生产 prompt 的 JSON 契约脚本化。
 *
 * <p>#134/#136 的裁决基线在此锁定：字面 "null" 候选在抽取侧被跳过（Java
 * 语义保留）；落库回落规则在 ADD/UPDATE 两口统一——null/空串/字面
 * "null" content 一律回落 candidate.content。
 */
class MemoryPipelineTest {

    /** 脚本化主力模型：弹出预置回复，记录收到的 prompt。 */
    private static final class ScriptedMain implements ModelRuntime {
        final Deque<String> responses = new ArrayDeque<>();
        final List<String> prompts = new ArrayList<>();

        @Override
        public dev.langchain4j.model.chat.ChatModel chat() {
            throw new UnsupportedOperationException("not needed in tests");
        }

        @Override
        public dev.langchain4j.model.chat.StreamingChatModel streaming() {
            return null;
        }

        @Override
        public AiMessage stream(List<ChatMessage> messages,
                                List<dev.langchain4j.agent.tool.ToolSpecification> tools,
                                java.util.function.Consumer<String> onDelta) {
            throw new UnsupportedOperationException("not needed in tests");
        }

        @Override
        public AiMessage invoke(List<ChatMessage> messages) {
            prompts.add(((UserMessage) messages.get(0)).singleText());
            String next = responses.poll();
            if (next == null) {
                throw new IllegalStateException("scripted main exhausted");
            }
            return AiMessage.from(next);
        }
    }

    /** 脚本化快档（ChatInvokerPort）。 */
    private static final class ScriptedFast implements AgentModule.ChatInvokerPort {
        final Deque<String> responses = new ArrayDeque<>();
        final List<String> prompts = new ArrayList<>();

        @Override
        public String invoke(List<ChatMessage> messages) {
            prompts.add(((UserMessage) messages.get(0)).singleText());
            String next = responses.poll();
            if (next == null) {
                throw new IllegalStateException("scripted fast exhausted");
            }
            return next;
        }
    }

    /** MemoryOps 记录桩。 */
    private static final class RecordingOps implements MemoryPipeline.MemoryOps {
        final List<String> calls = new ArrayList<>();
        final Map<String, Map<String, Object>> opDetails = new LinkedHashMap<>();
        Map<String, Object> profileToReturn = null;
        Models.SessionSummary summaryToReturn = null;
        boolean updateResult = true;
        boolean deleteResult = true;
        // #38 预筛测试注入口（默认值 = 旧行为：空结果、无候选向量）。
        List<Retrieval.ScoredMemory> searchResults = List.of();
        List<Float> candidateVector = null;

        @Override
        public List<Retrieval.ScoredMemory> search(String userId, String agentId,
                                                   String query, Integer topK) {
            calls.add("search");
            return searchResults;
        }

        @Override
        public List<Float> embedCandidate(String content) {
            return candidateVector;
        }

        @Override
        public void addMemory(String userId, String agentId, String content, String kind,
                              List<String> tags, double salience, String sourceThreadId) {
            calls.add("addMemory:" + content + "|" + kind + "|" + tags);
        }

        @Override
        public boolean updateMemory(String memoryId, String content) {
            calls.add("updateMemory:" + memoryId + "|" + content);
            return updateResult;
        }

        @Override
        public boolean deleteMemory(String memoryId) {
            calls.add("deleteMemory:" + memoryId);
            return deleteResult;
        }

        @Override
        public Map<String, Object> getProfile(String userId) {
            calls.add("getProfile");
            return profileToReturn;
        }

        @Override
        public void saveProfile(String userId, Map<String, Object> profile) {
            calls.add("saveProfile:" + profile.keySet());
        }

        @Override
        public Models.SessionSummary getSummary(String userId, String threadId) {
            calls.add("getSummary");
            return summaryToReturn;
        }

        @Override
        public void upsertSummary(Models.SessionSummary summary) {
            calls.add("upsertSummary:" + summary.summary());
        }

        @Override
        public void recordOp(String op, String userId, String agentId, String threadId,
                             Map<String, Object> detail, String status, String errorText,
                             long durationMs) {
            calls.add("op:" + op + "|" + status);
            opDetails.put(op + "|" + status, detail);
        }
    }

    private static Retrieval.ScoredMemory scored(String memoryId, String content) {
        return new Retrieval.ScoredMemory(
                new Models.MemoryRecord(memoryId, "u1", "chat", "semantic", content),
                0.9, Map.of());
    }

    // ── parseJsonBlock 容错 ──────────────────────────────────────────────

    @Test
    void parseJsonBlockHandlesFencesAndProse() {
        assertEquals(List.of(), MemoryPipeline.parseJsonBlock("[ ]"));
        assertEquals("x", MemoryPipeline.parseJsonBlock("```json\n\"x\"\n```"));
        Object fromProse = MemoryPipeline.parseJsonBlock("好的，以下是结果：\n[{\"op\": \"ADD\"}]");
        assertTrue(fromProse instanceof List<?> list && list.size() == 1);
        // 反引号包裹优先：剥围栏后取内容。
        Object fenced = MemoryPipeline.parseJsonBlock("```json\n{\"op\": \"NONE\"}\n```");
        assertTrue(fenced instanceof Map<?, ?> map && "NONE".equals(map.get("op")));
    }

    @Test
    void parseJsonBlockThrowsOnGarbage() {
        assertThrows(IllegalArgumentException.class,
                () -> MemoryPipeline.parseJsonBlock("完全不是 JSON"));
        assertThrows(IllegalArgumentException.class, () -> MemoryPipeline.parseJsonBlock(""));
        assertThrows(IllegalArgumentException.class, () -> MemoryPipeline.parseJsonBlock(null));
    }

    // ── renderTranscript ─────────────────────────────────────────────────

    @Test
    void renderTranscriptLabelsTurnsAndTruncatesToolLines() {
        String transcript = MemoryPipeline.renderTranscript(List.of(
                UserMessage.from("你好"),
                AiMessage.from("你好！"),
                ToolExecutionResultMessage.from("c1", "web_search", "t".repeat(400))),
                10_000);
        List<String> lines = List.of(transcript.split("\n"));
        assertEquals(3, lines.size());
        assertEquals("用户：你好", lines.get(0));
        assertEquals("助手：你好！", lines.get(1));
        // 工具行截 300 字符。
        assertEquals("工具[web_search]：" + "t".repeat(300), lines.get(2));
    }

    @Test
    void renderTranscriptKeepsTailWhenOverBudget() {
        List<ChatMessage> messages = List.of(
                UserMessage.from("u".repeat(600)),
                UserMessage.from("尾巴"));
        String transcript = MemoryPipeline.renderTranscript(messages, 50);
        assertTrue(transcript.startsWith("……（前文截断）\n"));
        assertTrue(transcript.endsWith("尾巴"));
    }

    @Test
    void renderTranscriptCountsCodePointsNotUtf16Units() {
        // #138：emoji（代理对）按 1 个码点计，截断不劈半——代理对恰好
        // 骑在 300 码点边界上时，UTF-16 substring 会留下半个代理对。
        String emoji = "😀";
        String body = "a".repeat(299) + emoji + "b".repeat(10);
        String transcript = MemoryPipeline.renderTranscript(List.of(
                ToolExecutionResultMessage.from("c1", "web_search", body)),
                10_000);
        String line = List.of(transcript.split("\n")).get(0);
        String prefix = "工具[web_search]：";
        assertEquals(300, line.codePointCount(0, line.length()) - prefix.length());
        assertTrue(line.endsWith(emoji), "第 300 个码点是完整 emoji，UTF-16 截断会劈半");
        // 尾部保码点：budget 4 → 保留最后 4 个码点（UTF-16 截断会只留 2 前 + emoji）。
        String tail = MemoryPipeline.renderTranscript(List.of(
                UserMessage.from("前".repeat(300) + emoji)),
                4);
        assertEquals("……（前文截断）\n前前前" + emoji, tail);
    }

    // ── extractCandidates ────────────────────────────────────────────────

    private static MemoryPipeline pipelineWith(ScriptedMain main, ScriptedFast fast,
                                               RecordingOps ops) {
        Settings settings = Settings.load(Map.of());
        return new MemoryPipeline(main, fast, settings, ops);
    }

    @Test
    void extractCandidatesParsesAndNormalizes() {
        ScriptedFast fast = new ScriptedFast();
        fast.responses.add("""
                ```json
                [
                  {"content": "用户喜欢咖啡", "kind": "preference", "salience": "0.7", "tags": ["饮食"]},
                  {"content": "null"},
                  {"content": "   "},
                  {"content": "用户是工程师", "kind": "episodic", "salience": 9.9},
                  {"kind": "semantic"}
                ]
                ```""");
        RecordingOps ops = new RecordingOps();
        MemoryPipeline pipeline = pipelineWith(new ScriptedMain(), fast, ops);

        List<MemoryPipeline.Candidate> candidates = pipeline.extractCandidates("用户：我喜欢咖啡");

        // 字面 "null" 与空白内容被跳过（#134 基线）；非法 kind 归 semantic；
        // salience 越界钳 [0,1]；无 content 的条目跳过。
        assertEquals(2, candidates.size());
        assertEquals("用户喜欢咖啡", candidates.get(0).content());
        assertEquals("semantic", candidates.get(0).kind()); // 非法 kind 兜底
        assertEquals(0.7, candidates.get(0).salience(), 1e-9); // 数字串可解析
        assertEquals(List.of("饮食"), candidates.get(0).tags());
        assertEquals(1.0, candidates.get(1).salience(), 1e-9); // 9.9 → 钳 1.0
    }

    @Test
    void extractCandidatesCapsAtFiveAndDefaultsSalience() {
        ScriptedFast fast = new ScriptedFast();
        StringBuilder items = new StringBuilder();
        for (int i = 0; i < 8; i++) {
            if (i > 0) {
                items.append(',');
            }
            items.append("{\"content\": \"c").append(i).append("\"}");
        }
        fast.responses.add("[" + items + "]");
        MemoryPipeline pipeline = pipelineWith(new ScriptedMain(), fast, new RecordingOps());
        List<MemoryPipeline.Candidate> candidates = pipeline.extractCandidates("t");
        assertEquals(5, candidates.size());
        assertEquals(0.5, candidates.get(0).salience(), 1e-9);
    }

    @Test
    void extractRejectsNonArrayOutput() {
        ScriptedFast fast = new ScriptedFast();
        fast.responses.add("{\"op\": \"NONE\"}");
        MemoryPipeline pipeline = pipelineWith(new ScriptedMain(), fast, new RecordingOps());
        assertThrows(IllegalArgumentException.class, () -> pipeline.extractCandidates("t"));
    }

    // ── consolidate ──────────────────────────────────────────────────────

    @Test
    void consolidateDedupPrefilterSkipsLlmAboveThreshold() {
        // #38：候选与既有记忆最高余弦 > 0.92 → 直接 NONE，LLM 不被调用；
        // 审计 detail 带 prefilter=dedup_skip + max_sim 数值（不落原文）。
        ScriptedMain main = new ScriptedMain();
        RecordingOps ops = new RecordingOps();
        // 同向单位向量 → cosine = 1.0 > 0.92。
        ops.searchResults = List.of(new Retrieval.ScoredMemory(
                new Models.MemoryRecord("m1", "u1", "chat", "semantic", "几乎相同的内容",
                        List.of(), Models.encodeEmbedding(List.of(0.6f, 0.8f)), 2,
                        0.5, "active", "", List.of(), 1000.0, 1000.0, null, 0),
                0.9, Map.of("keyword", 0.9)));
        ops.candidateVector = List.of(0.6f, 0.8f);
        MemoryPipeline pipeline = pipelineWith(main, new ScriptedFast(), ops);

        assertEquals("NONE", pipeline.consolidate(
                new MemoryPipeline.Candidate("几乎相同的内容", "semantic", 0.5, List.of()),
                "u1", "chat", "chat:tt"));
        assertTrue(main.prompts.isEmpty(), "预筛命中不得进 LLM：" + main.prompts);
        Map<String, Object> detail = ops.opDetails.get("consolidate|ok");
        assertEquals("dedup_skip", detail.get("prefilter"));
        assertEquals("m1", detail.get("matched_memory_id"));
        // max_sim 是数值字符串（审计可读、不落原文）。
        assertTrue(String.valueOf(detail.get("max_sim")).startsWith("1.0"),
                String.valueOf(detail.get("max_sim")));
    }

    @Test
    void consolidateDedupPrefilterPassesMidBandToLlm() {
        // 余弦低于阈值（0.707 < 0.92）→ 照旧进 LLM 裁决（中间带与低带同路）。
        ScriptedMain main = new ScriptedMain();
        main.responses.add("{\"op\": \"ADD\", \"content\": \"新事实\"}");
        RecordingOps ops = new RecordingOps();
        ops.searchResults = List.of(new Retrieval.ScoredMemory(
                new Models.MemoryRecord("m1", "u1", "chat", "semantic", "相关但不同",
                        List.of(), Models.encodeEmbedding(List.of(0.5f, 0.5f)), 2,
                        0.5, "active", "", List.of(), 1000.0, 1000.0, null, 0),
                0.9, Map.of("keyword", 0.9)));
        ops.candidateVector = List.of(1.0f, 0.0f);
        MemoryPipeline pipeline = pipelineWith(main, new ScriptedFast(), ops);

        assertEquals("ADD", pipeline.consolidate(
                new MemoryPipeline.Candidate("候选内容", "semantic", 0.5, List.of()),
                "u1", "chat", "chat:tt"));
        assertEquals(1, main.prompts.size(), "中间带应照旧进 LLM");
    }

    @Test
    void consolidateDedupPrefilterSkippedWithoutEmbedding() {
        // 降级路径：候选无向量（embedCandidate null）→ 预筛旁路、行为与旧一致。
        ScriptedMain main = new ScriptedMain();
        main.responses.add("{\"op\": \"NONE\"}");
        RecordingOps ops = new RecordingOps();
        ops.searchResults = List.of(new Retrieval.ScoredMemory(
                new Models.MemoryRecord("m1", "u1", "chat", "semantic", "同文本",
                        List.of(), null, null, 0.5, "active", "", List.of(),
                        1000.0, 1000.0, null, 0),
                0.9, Map.of("keyword", 0.9)));
        // candidateVector 保持 null。
        MemoryPipeline pipeline = pipelineWith(main, new ScriptedFast(), ops);

        assertEquals("NONE", pipeline.consolidate(
                new MemoryPipeline.Candidate("同文本", "semantic", 0.5, List.of()),
                "u1", "chat", "chat:tt"));
        assertEquals(1, main.prompts.size(), "无向量应跳过预筛照旧进 LLM");
    }

    @Test
    void consolidateAddFallsBackToCandidateContent() {
        ScriptedMain main = new ScriptedMain();
        main.responses.add("{\"op\": \"ADD\", \"content\": null}");
        main.responses.add("{\"op\": \"ADD\", \"content\": \"\"}");
        main.responses.add("{\"op\": \"ADD\", \"content\": \"null\"}");
        RecordingOps ops = new RecordingOps();
        MemoryPipeline pipeline = pipelineWith(main, new ScriptedFast(), ops);

        for (int i = 0; i < 3; i++) {
            assertEquals("ADD", pipeline.consolidate(
                    new MemoryPipeline.Candidate("候选内容", "semantic", 0.5, List.of("t")),
                    "u1", "chat", "chat:tt"));
        }
        // #136 统一回落：ADD 的 null/空串/字面 "null" content 一律回落
        // candidate.content（LLM null 序列化产物与缺失等同）。
        assertEquals(3, ops.calls.stream().filter(c -> c.equals("addMemory:候选内容|semantic|[t]")).count());
    }

    @Test
    void consolidateUpdateFallsBackToCandidateContent() {
        // #136：UPDATE 与 ADD 同一回落规则（修复前 UPDATE 不查字面 "null"，
        // 会把 "null" 原样落库）。
        ScriptedMain main = new ScriptedMain();
        main.responses.add("{\"op\": \"UPDATE\", \"memory_id\": \"m1\", \"content\": \"null\"}");
        main.responses.add("{\"op\": \"UPDATE\", \"memory_id\": \"m1\", \"content\": \"\"}");
        RecordingOps ops = new RecordingOps();
        MemoryPipeline pipeline = pipelineWith(main, new ScriptedFast(), ops);

        for (int i = 0; i < 2; i++) {
            assertEquals("UPDATE", pipeline.consolidate(
                    new MemoryPipeline.Candidate("候选内容", "semantic", 0.5, List.of()),
                    "u1", "chat", "t"));
        }
        assertEquals(2, ops.calls.stream()
                .filter(c -> c.equals("updateMemory:m1|候选内容")).count());
    }

    @Test
    void consolidateUpdateRequiresMemoryIdAndHonorsFailure() {
        ScriptedMain main = new ScriptedMain();
        main.responses.add("{\"op\": \"UPDATE\", \"memory_id\": \"m1\", \"content\": \"新内容\"}");
        main.responses.add("{\"op\": \"UPDATE\"}");
        main.responses.add("{\"op\": \"UPDATE\", \"memory_id\": \"m1\"}");
        RecordingOps ops = new RecordingOps();
        MemoryPipeline pipeline = pipelineWith(main, new ScriptedFast(), ops);

        assertEquals("UPDATE", pipeline.consolidate(
                new MemoryPipeline.Candidate("c", "semantic", 0.5, List.of()), "u1", "chat", "t"));
        // 缺 memory_id → NONE（不落库）。
        assertEquals("NONE", pipeline.consolidate(
                new MemoryPipeline.Candidate("c", "semantic", 0.5, List.of()), "u1", "chat", "t"));
        // updateMemory 落库失败 → NONE。
        ops.updateResult = false;
        assertEquals("NONE", pipeline.consolidate(
                new MemoryPipeline.Candidate("c", "semantic", 0.5, List.of()), "u1", "chat", "t"));
        assertTrue(ops.calls.contains("updateMemory:m1|新内容"));
    }

    @Test
    void consolidateDeleteAndUnknownOps() {
        ScriptedMain main = new ScriptedMain();
        main.responses.add("{\"op\": \"delete\", \"memory_id\": \"m9\"}");
        main.responses.add("{\"op\": \"DELETE\"}");
        main.responses.add("{\"op\": \"_destroy\"}");
        RecordingOps ops = new RecordingOps();
        MemoryPipeline pipeline = pipelineWith(main, new ScriptedFast(), ops);

        assertEquals("DELETE", pipeline.consolidate(
                new MemoryPipeline.Candidate("c", "semantic", 0.5, List.of()), "u1", "chat", "t"));
        // memory_id 缺失 → NONE。
        assertEquals("NONE", pipeline.consolidate(
                new MemoryPipeline.Candidate("c", "semantic", 0.5, List.of()), "u1", "chat", "t"));
        // 未知名归一 NONE。
        assertEquals("NONE", pipeline.consolidate(
                new MemoryPipeline.Candidate("c", "semantic", 0.5, List.of()), "u1", "chat", "t"));
    }

    // ── 画像与摘要 ────────────────────────────────────────────────────────

    @Test
    void mergeProfileBuildsSkeletonWhenAbsent() {
        ScriptedFast fast = new ScriptedFast();
        fast.responses.add("{\"基本信息\": [\"工程师\"], \"偏好\": []}");
        RecordingOps ops = new RecordingOps();
        ops.profileToReturn = null;
        MemoryPipeline pipeline = pipelineWith(new ScriptedMain(), fast, ops);

        assertTrue(pipeline.mergeProfile("u1", List.of("是工程师")));
        // 画像缺失时骨架先建（PROFILE_SECTIONS 五节进 prompt），LLM 合并输出
        // 原样落库（本桩返回两节）。
        assertTrue(fast.prompts.get(0).contains("长期目标"), "prompt 应含骨架分节");
        assertTrue(ops.calls.contains("saveProfile:[基本信息, 偏好]"), ops.calls.toString());
    }

    @Test
    void mergeProfileSkipsEmptyFacts() {
        RecordingOps ops = new RecordingOps();
        MemoryPipeline pipeline = pipelineWith(new ScriptedMain(), new ScriptedFast(), ops);
        assertFalse(pipeline.mergeProfile("u1", List.of()));
        assertFalse(pipeline.mergeProfile("u1", null));
        assertEquals(List.of(), ops.calls);
    }

    @Test
    void updateSummaryCapsAtThousandChars() {
        ScriptedFast fast = new ScriptedFast();
        fast.responses.add("s".repeat(1200));
        RecordingOps ops = new RecordingOps();
        ops.summaryToReturn = new Models.SessionSummary("u1", "t", "旧摘要", "chat", 5, 1.0);
        MemoryPipeline pipeline = pipelineWith(new ScriptedMain(), fast, ops);

        assertTrue(pipeline.updateSummary("u1", "chat", "t", "transcript"));
        assertTrue(ops.calls.get(ops.calls.size() - 1).startsWith("upsertSummary:"));
        // 硬截 1000（#179 的双上限基线：prompt 说 300，硬截 1000）。
        Models.SessionSummary saved = null;
        for (String call : ops.calls) {
            if (call.startsWith("upsertSummary:")) {
                String summary = call.substring("upsertSummary:".length());
                assertEquals(1000, summary.length());
            }
        }
    }

    @Test
    void updateSummaryEmptyResponseIsNoop() {
        ScriptedFast fast = new ScriptedFast();
        fast.responses.add("   ");
        RecordingOps ops = new RecordingOps();
        MemoryPipeline pipeline = pipelineWith(new ScriptedMain(), fast, ops);
        assertFalse(pipeline.updateSummary("u1", "chat", "t", "transcript"));
        assertTrue(ops.calls.contains("getSummary"));
        assertFalse(ops.calls.stream().anyMatch(c -> c.startsWith("upsertSummary")));
    }

    // ── captureTurn 编排 ─────────────────────────────────────────────────

    @Test
    void captureTurnHappyPathRunsAllFourSteps() {
        ScriptedFast fast = new ScriptedFast();
        // 抽取（快档）→ 整合（主力）→ 画像（快档）→ 摘要（快档）。
        fast.responses.add("[{\"content\": \"用户在杭州\", \"kind\": \"semantic\", \"salience\": 0.8}]");
        ScriptedMain main = new ScriptedMain();
        main.responses.add("{\"op\": \"ADD\", \"content\": \"用户在杭州\"}");
        fast.responses.add("{\"基本信息\": [\"在杭州\"]}");
        fast.responses.add("第一轮摘要");
        RecordingOps ops = new RecordingOps();

        MemoryPipeline pipeline = pipelineWith(main, fast, ops);
        // humanCount=12 恰达摘要触发线（默认 summaryTriggerMessages=12）。
        Map<String, Object> detail = pipeline.captureTurn("u1", "chat", "chat:t", "用户：我在杭州", 12, false);

        assertEquals(1, ((Number) detail.get("candidates")).intValue());
        assertEquals(Map.of("用户在杭州", "ADD"), detail.get("ops"));
        assertEquals(Boolean.TRUE, detail.get("profile_updated"), "detail=" + detail);
        assertEquals(Boolean.TRUE, detail.get("summary_updated"), "detail=" + detail);
        assertEquals(Boolean.FALSE, detail.get("forced"));
        // 审计四步全部 ok。
        for (String step : List.of("extract", "consolidate", "profile", "summary")) {
            assertTrue(ops.calls.contains("op:" + step + "|ok"), ops.calls.toString());
        }
        // #135：成功审计 detail 按结果形状记录，不再恒写 human_count——
        // 抽取=list 摘要（content 前 120 码点）、整合=op 字符串、
        // 画像/摘要=布尔，全部包一层 {"result": …}。
        assertEquals(Map.of("result", List.of("用户在杭州")),
                ops.opDetails.get("extract|ok"));
        assertEquals(Map.of("result", "ADD"), ops.opDetails.get("consolidate|ok"));
        assertEquals(Map.of("result", Boolean.TRUE), ops.opDetails.get("profile|ok"));
        assertEquals(Map.of("result", Boolean.TRUE), ops.opDetails.get("summary|ok"));
    }

    @Test
    void captureTurnExtractFailureDoesNotBlockSummary() {
        ScriptedFast fast = new ScriptedFast();
        fast.responses.add("不是 JSON");          // 抽取失败
        fast.responses.add("摘要内容");          // 摘要仍执行
        RecordingOps ops = new RecordingOps();
        MemoryPipeline pipeline = pipelineWith(new ScriptedMain(), fast, ops);

        Map<String, Object> detail = pipeline.captureTurn(
                "u1", "chat", "chat:t", "用户：你好", 20, false);
        assertEquals(0, ((Number) detail.get("candidates")).intValue());
        assertEquals(Boolean.FALSE, detail.get("profile_updated"));
        assertEquals(Boolean.TRUE, detail.get("summary_updated"));
        assertTrue(ops.calls.contains("op:extract|error"));
        assertTrue(ops.calls.contains("op:summary|ok"));
        // 失败分支维持 {"human_count": N}；成功分支不受失败影响。
        assertEquals(Map.of("human_count", 20), ops.opDetails.get("extract|error"));
        assertEquals(Map.of("result", Boolean.TRUE), ops.opDetails.get("summary|ok"));
        // 失败审计绝不向上抛：captureTurn 正常返回。
    }

    @Test
    void captureTurnHonorsCaptureGateAndForceBypass() {
        Settings settings = Settings.load(Map.of("MEMORY_CAPTURE_ENABLED", "false"));
        ScriptedFast fast = new ScriptedFast();
        RecordingOps ops = new RecordingOps();
        MemoryPipeline pipeline = new MemoryPipeline(new ScriptedMain(), fast, settings, ops);

        // 门控关闭：不抽取（无快档消耗）；摘要照常。
        fast.responses.add("摘要");
        Map<String, Object> detail = pipeline.captureTurn("u1", "chat", "t", "transcript", 20, false);
        assertEquals(0, ((Number) detail.get("candidates")).intValue());
        assertTrue(ops.calls.contains("op:summary|ok"));

        // force=true 旁路门控（夜间批路径）。
        ScriptedMain scriptedMain = new ScriptedMain();
        scriptedMain.responses.add("{\"op\": \"NONE\"}");
        fast.responses.add("[{\"content\": \"事实\"}]");
        MemoryPipeline forced = new MemoryPipeline(scriptedMain, fast, settings, ops);
        Map<String, Object> forcedDetail = forced.captureTurn("u1", "chat", "t", "transcript", 1, true);
        assertEquals(1, ((Number) forcedDetail.get("candidates")).intValue());
        assertEquals(Boolean.TRUE, forcedDetail.get("forced"));
    }

    @Test
    void captureTurnSkipsSummaryBelowTrigger() {
        RecordingOps ops = new RecordingOps();
        ScriptedFast fast = new ScriptedFast();
        MemoryPipeline pipeline = pipelineWith(new ScriptedMain(), fast, ops);
        Map<String, Object> detail = pipeline.captureTurn("u1", "chat", "t", "transcript", 1, false);
        assertFalse(ops.calls.stream().anyMatch(c -> c.startsWith("op:summary")));
        assertEquals(Boolean.FALSE, detail.get("summary_updated"));
    }

    @Test
    void consolidateUsesMainModelWhileExtractionUsesFast() {
        // 档位分派：pipelineProfile=split → 抽取/画像/摘要走快档，整合走主力。
        ScriptedFast fast = new ScriptedFast();
        ScriptedMain main = new ScriptedMain();
        fast.responses.add("[]");
        MemoryPipeline pipeline = pipelineWith(main, fast, new RecordingOps());
        pipeline.extractCandidates("t");
        assertEquals(1, fast.prompts.size());
        assertTrue(main.prompts.isEmpty());
        assertTrue(fast.prompts.get(0).contains("转写") || fast.prompts.get(0).contains("transcript")
                        || fast.prompts.get(0).contains("抽取"),
                "extraction prompt should hit the fast lane");
    }

    // ── 敏感内容红线（#48，Java 先行） ───────────────────────────────────

    @Test
    void consolidateRejectsSensitiveContentWithRejectedAudit() {
        // 默认开启（sensitiveContentFilter=true）：直接 consolidate 命中
        // 即拒——REJECTED、rejected 审计（detail 带规则名）、不消耗 LLM。
        ScriptedMain main = new ScriptedMain();
        RecordingOps ops = new RecordingOps();
        MemoryPipeline pipeline = new MemoryPipeline(main, new ScriptedFast(),
                Settings.load(Map.of(), null), ops);

        assertEquals("REJECTED", pipeline.consolidate(
                new MemoryPipeline.Candidate("密钥是 sk-abcdefghijklmnop123456",
                        "semantic", 0.5, List.of()),
                "u1", "chat", "t"));
        assertTrue(ops.calls.contains("op:consolidate|rejected"), ops.calls.toString());
        assertEquals(Map.of("rules", "api_key", "op", "REJECTED"),
                ops.opDetails.get("consolidate|rejected"));
        assertTrue(main.prompts.isEmpty(), "命中红线不应消耗 LLM 调用");
        // 不落库：无任何 addMemory/updateMemory 调用。
        assertFalse(ops.calls.stream().anyMatch(c ->
                c.startsWith("addMemory") || c.startsWith("updateMemory")));
    }

    @Test
    void consolidateHonorsSensitiveFilterToggle() {
        // 开关关闭：命中内容照常走整合（LLM 消耗一个决策），无 rejected 审计。
        ScriptedMain main = new ScriptedMain();
        main.responses.add("{\"op\": \"NONE\"}");
        RecordingOps ops = new RecordingOps();
        MemoryPipeline off = new MemoryPipeline(main, new ScriptedFast(),
                Settings.load(Map.of("MEMORY_SENSITIVE_CONTENT_FILTER", "false"), null), ops);

        assertEquals("NONE", off.consolidate(
                new MemoryPipeline.Candidate("密钥是 sk-abcdefghijklmnop123456",
                        "semantic", 0.5, List.of()),
                "u1", "chat", "t"));
        assertFalse(ops.calls.contains("op:consolidate|rejected"));
        assertEquals(1, main.prompts.size());
    }

    @Test
    void captureTurnRejectsSensitiveCandidateButKeepsCleanOnes() {
        ScriptedFast fast = new ScriptedFast();
        // 两条候选：一条含 sk- 密钥（应被红线拦下），一条干净（照常整合）。
        fast.responses.add("""
                [
                  {"content": "用户的 API key 是 sk-abcdefghijklmnop123456", "kind": "semantic", "salience": 0.8},
                  {"content": "用户在杭州", "kind": "semantic", "salience": 0.8}
                ]
                """);
        ScriptedMain main = new ScriptedMain();
        main.responses.add("{\"op\": \"ADD\", \"content\": \"用户在杭州\"}");
        RecordingOps ops = new RecordingOps();
        MemoryPipeline pipeline = new MemoryPipeline(main, fast, Settings.load(Map.of(), null), ops);

        // humanCount=1 不触发摘要；关掉画像避免无关 fast 响应消耗。
        Map<String, Object> detail = pipeline.captureTurn("u1", "chat", "chat:t",
                "用户：我把 key 发你\n用户：我在杭州", 1, false);

        // 脏候选被拒（REJECTED 不入 ops 落库），干净候选照常 ADD。
        @SuppressWarnings("unchecked")
        Map<String, Object> opsMap = (Map<String, Object>) detail.get("ops");
        assertEquals("REJECTED", opsMap.get("用户的 API key 是 sk-abcdefghijklmnop123456"));
        assertEquals("ADD", opsMap.get("用户在杭州"));
        assertTrue(ops.calls.contains("op:consolidate|rejected"), ops.calls.toString());
        assertEquals("api_key", ((Map<?, ?>) ops.opDetails.get("consolidate|rejected")).get("rules"));
        // 脏内容绝不落库，也绝不进整合 LLM prompt（不把密钥回喂给模型）。
        assertFalse(ops.calls.stream().anyMatch(c -> c.startsWith("addMemory:用户的 API key")));
        assertEquals(1, main.prompts.size());
        assertFalse(main.prompts.get(0).contains("sk-"), "密钥不应出现在整合 prompt 里");
        assertTrue(main.prompts.get(0).contains("用户在杭州"));
    }
}
