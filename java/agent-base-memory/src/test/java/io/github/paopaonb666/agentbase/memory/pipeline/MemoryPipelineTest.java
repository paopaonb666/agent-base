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
 * 语义保留），ADD 侧字面 "null" 回落 candidate.content。
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
        Map<String, Object> profileToReturn = null;
        Models.SessionSummary summaryToReturn = null;
        boolean updateResult = true;
        boolean deleteResult = true;

        @Override
        public List<Retrieval.ScoredMemory> search(String userId, String agentId,
                                                   String query, Integer topK) {
            calls.add("search");
            return List.of();
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
    void consolidateAddFallsBackToCandidateContent() {
        ScriptedMain main = new ScriptedMain();
        main.responses.add("{\"op\": \"ADD\", \"content\": null}");
        RecordingOps ops = new RecordingOps();
        MemoryPipeline pipeline = pipelineWith(main, new ScriptedFast(), ops);

        String op = pipeline.consolidate(
                new MemoryPipeline.Candidate("候选内容", "semantic", 0.5, List.of("t")),
                "u1", "chat", "chat:tt");
        assertEquals("ADD", op);
        // #136 基线：ADD 的字面 null/空串 content 回落 candidate.content。
        assertTrue(ops.calls.contains("addMemory:候选内容|semantic|[t]"), ops.calls.toString());
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
}
