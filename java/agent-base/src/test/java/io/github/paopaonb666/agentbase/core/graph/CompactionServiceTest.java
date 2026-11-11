package io.github.paopaonb666.agentbase.core.graph;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import io.github.paopaonb666.agentbase.core.AgentModule;
import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.core.context.ContextEngine;
import io.github.paopaonb666.agentbase.core.text.EstimatingTokenCounter;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.function.BooleanSupplier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

/**
 * 会话历史压缩语义 golden（#328/#330，引擎侧）：脚本化摘要通道驱动完整
 * 任务体——折叠段转写、摘要消息置顶、近 N 轮原文保留（UserMessage 边界）、
 * 压缩后模型输入在预算内、标记前缀不被注入去重误剔、extra 元数据齐全、
 * 每线程在途至多 1、基准不一致放弃。
 *
 * <p>token 计量用字符系数估算器（确定性口径），结束恢复默认——所有手算
 * （15 CJK 字 = 9 token，+4 开销 = 13/条）都锚定在该口径上。
 */
class CompactionServiceTest {

    @AfterEach
    void restoreDefaultCounter() {
        ContextEngine.setTokenCounter(null); // null → 默认（JTokkit cl100k）
    }

    private static void useEstimator() {
        ContextEngine.setTokenCounter(EstimatingTokenCounter.INSTANCE);
    }

    /** 等待异步评估落地（单守护线程执行器；轮询 + 超时失败）。 */
    private static void waitFor(BooleanSupplier condition) {
        long deadline = System.currentTimeMillis() + 5000;
        while (!condition.getAsBoolean()) {
            if (System.currentTimeMillis() > deadline) {
                fail("条件超时未满足");
            }
            try {
                Thread.sleep(20);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                fail("等待被中断");
            }
        }
    }

    /**
     * 「零动作」断言的静默窗：无操作路径没有可观测完成点（不摘要、不写回、
     * 不计数），留出固定窗口让专属 worker 走完整个评估后再断言快照原样——
     * 评估本身是微秒级读+算，300ms 足够冗余。
     */
    private static void settle() {
        try {
            Thread.sleep(300);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            fail("静默窗被中断");
        }
    }

    /** 脚本化摘要通道：记录转写与 target，返回预置摘要；可注入响应前钩子。 */
    private static final class ScriptedSummarizer implements CompactionService.Summarizer {
        final Deque<String> responses = new ArrayDeque<>();
        final List<String> transcripts = new CopyOnWriteArrayList<>();
        final List<Integer> targets = new CopyOnWriteArrayList<>();
        final CountDownLatch entered = new CountDownLatch(1);
        Runnable beforeRespond;

        @Override
        public String summarize(String transcript, int targetChars) {
            transcripts.add(transcript);
            targets.add(targetChars);
            entered.countDown();
            if (beforeRespond != null) {
                beforeRespond.run();
            }
            String next = responses.poll();
            if (next == null) {
                throw new IllegalStateException("scripted summarizer exhausted");
            }
            return next;
        }
    }

    /**
     * 摘要通道返回的最终文本（桥契约：生产装配 memoryPort::compactTranscript
     * → MemoryPipeline.compactSummary 内部完成三键 JSON 解析+渲染，Summarizer
     * 拿到的已是渲染成品——桩直接给成品，解析/回退语义由 MemoryPipelineTest
     * 侧用例覆盖）。渲染后 ≈ 25 token，含前缀。
     */
    private static final String SUMMARY_TEXT = "【任务态】任务态\n【关键实体】实体\n【未决问题】问题";

    /** 第 i 轮用户消息：15 CJK + 1 ASCII = 9 token（+4 开销 = 13/条）。 */
    private static UserMessage turnUser(int i) {
        return UserMessage.from("第" + i + "轮用户提问甲乙丙丁戊己庚辛壬");
    }

    private static AiMessage turnAi(int i) {
        return AiMessage.from("第" + i + "轮助手回答甲乙丙丁戊己庚辛壬");
    }

    /** turns 轮完整历史（每轮 用户+助手 两条）。 */
    private static List<ChatMessage> history(int turns) {
        List<ChatMessage> messages = new ArrayList<>();
        for (int i = 1; i <= turns; i++) {
            messages.add(turnUser(i));
            messages.add(turnAi(i));
        }
        return messages;
    }

    private static Settings.MemorySettings settings() {
        Settings.MemorySettings memory = new Settings().memory;
        memory.compactionEnabled = true;
        memory.compactionRatio = 0.7;
        memory.compactionKeepRecentTurns = 3;
        return memory;
    }

    // ── 压缩语义 golden ──────────────────────────────────────────────────

    @Test
    void goldenCompactionSemantics() {
        useEstimator();
        InMemoryCheckpointer cp = new InMemoryCheckpointer();
        // 8 轮 × 2 条 × 13 = 208 token > 250×0.7=175 → 触发。
        cp.put("chat:t1", new GraphState(history(8), Map.of()));
        ScriptedSummarizer summarizer = new ScriptedSummarizer();
        summarizer.responses.add(SUMMARY_TEXT);
        CompactionService service = new CompactionService(cp, settings(), 250, summarizer);
        try {
            service.onRunTerminated(RunConfig.of("chat:t1", "u1"), null);
            waitFor(() -> service.compactedRounds() == 1);

            // 摘要通道收到的 target = clamp(250/10, 400, 2000) = 400（下限兜底）。
            assertEquals(List.of(400), summarizer.targets);
            // 折叠段转写 = 前 5 轮（8 轮 - 保留 3 轮）；保留窗的 6/7/8 轮不进转写。
            String transcript = summarizer.transcripts.get(0);
            assertTrue(transcript.contains("用户：第1轮用户提问"), transcript);
            assertTrue(transcript.contains("助手：第5轮助手回答"), transcript);
            assertFalse(transcript.contains("第6轮"), transcript);
            assertFalse(transcript.contains("第7轮"), transcript);
            assertFalse(transcript.contains("第8轮"), transcript);

            GraphState compacted = cp.get("chat:t1");
            // 新历史 = [压缩摘要 SystemMessage] + 近 3 轮原文（6 条）。
            assertEquals(7, compacted.messages().size());
            ChatMessage head = compacted.messages().get(0);
            assertTrue(head instanceof SystemMessage);
            assertTrue(((SystemMessage) head).text().startsWith(
                    ContextEngine.COMPACTION_SUMMARY_PREFIX));
            assertEquals("【会话历史摘要（自动压缩生成）】\n【任务态】任务态"
                            + "\n【关键实体】实体\n【未决问题】问题",
                    ((SystemMessage) head).text(),
                    "渲染格式即契约（329 三键渲染单源）");
            for (int i = 0; i < 6; i++) {
                assertEquals(history(8).get(10 + i), compacted.messages().get(1 + i),
                        "近 3 轮原文逐条保留（UserMessage 边界）");
            }
            // extra 元数据齐全：original_message_count / original_token_estimate /
            // folded_turns / at。
            Map<String, Object> meta = castMap(compacted.extra().get("compaction"));
            assertEquals(16, ((Number) meta.get("original_message_count")).intValue());
            assertEquals(208, ((Number) meta.get("original_token_estimate")).intValue(),
                    "13/条 × 16（估算器口径锁定）");
            assertEquals(5, ((Number) meta.get("folded_turns")).intValue(),
                    "折叠段（前 5 轮）内的 UserMessage 数（F7 读侧加回门控的口径）");
            assertFalse(String.valueOf(meta.get("at")).isBlank());
            // 既有 extra 通道内容保留（本用例无，仅形状核对）。
            assertTrue(compacted.extra().containsKey("compaction"));

            // 标记前缀不是注入前缀：injectedKind 必须放行 null——否则
            // buildModelInput 会把它当注入块去重/提升，摘要到不了模型面前。
            assertNull(ContextEngine.injectedKind(head));
            // 压缩后全量历史在预算内：buildModelInput 不再修剪（保全部）。
            List<ChatMessage> modelInput = ContextEngine.buildModelInput(
                    compacted.messages(), 250);
            assertEquals(compacted.messages().size(), modelInput.size(),
                    "压缩后 buildModelInput 输出应完整在预算内");
        } finally {
            service.close();
        }
    }

    @Test
    void belowThresholdOrShortHistoryDoesNotCompact() {
        useEstimator();
        InMemoryCheckpointer cp = new InMemoryCheckpointer();
        // 208 token ≤ 400×0.7=280 → 不触发。
        cp.put("chat:t1", new GraphState(history(8), Map.of()));
        ScriptedSummarizer summarizer = new ScriptedSummarizer();
        CompactionService service = new CompactionService(cp, settings(), 400, summarizer);
        try {
            service.onRunTerminated(RunConfig.of("chat:t1", "u1"), null);
            // 静默窗后确认零动作（若摘要被调用会因脚本耗尽抛错进放弃计数）。
            settle();
            assertEquals(0, summarizer.transcripts.size());
            assertEquals(0, service.abandonedRounds());
            assertEquals(0, service.compactedRounds());
            assertEquals(8 * 2, cp.get("chat:t1").messages().size(), "快照原样");
        } finally {
            service.close();
        }

        // 历史不足 keepRecentTurns 轮：无可折叠段（52 > 50×0.7=35 触发检测
        // 仍为真，但折叠不发生）。
        InMemoryCheckpointer cp2 = new InMemoryCheckpointer();
        cp2.put("chat:t2", new GraphState(history(2), Map.of()));
        ScriptedSummarizer summarizer2 = new ScriptedSummarizer();
        CompactionService service2 = new CompactionService(cp2, settings(), 50, summarizer2);
        try {
            service2.onRunTerminated(RunConfig.of("chat:t2", "u1"), null);
            settle();
            assertEquals(0, summarizer2.transcripts.size());
            assertEquals(0, service2.abandonedRounds());
            assertEquals(4, cp2.get("chat:t2").messages().size());
        } finally {
            service2.close();
        }
    }

    @Test
    void baselineMismatchAbandonsWithoutClobberingConcurrentWrite() {
        // #330 方案 B：摘要期间新消息落盘 → 写回前重读基准不一致 → 放弃，
        // 并发写入完整保留（不产生半写、不丢消息）。
        useEstimator();
        InMemoryCheckpointer cp = new InMemoryCheckpointer();
        cp.put("chat:t1", new GraphState(history(8), Map.of()));
        ScriptedSummarizer summarizer = new ScriptedSummarizer();
        summarizer.responses.add(SUMMARY_TEXT);
        summarizer.beforeRespond = () -> {
            // 模拟并发 run 落盘新消息（快照从 16 条变 17 条）。
            GraphState concurrent = new GraphState(cp.get("chat:t1").messages(), Map.of());
            concurrent.apply(Map.of(GraphState.MESSAGES, List.of(UserMessage.from("并发新消息"))));
            cp.put("chat:t1", concurrent);
        };
        CompactionService service = new CompactionService(cp, settings(), 250, summarizer);
        try {
            service.onRunTerminated(RunConfig.of("chat:t1", "u1"), null);
            waitFor(() -> service.abandonedRounds() == 1);
            assertEquals(0, service.compactedRounds());
            // 并发写入完整保留：17 条且最后一条是新消息。
            GraphState state = cp.get("chat:t1");
            assertEquals(17, state.messages().size());
            assertEquals("并发新消息", ((UserMessage) state.messages()
                    .get(state.messages().size() - 1)).singleText());
            assertFalse(state.messages().get(0) instanceof SystemMessage,
                    "放弃路径不得写入摘要消息（无半写）");
        } finally {
            service.close();
        }
    }

    @Test
    void inFlightGuardDropsDuplicateSubmissionForSameThread() {
        // #328：每线程在途至多 1 个评估——同线程连续终态丢弃而非积压队列。
        useEstimator();
        InMemoryCheckpointer cp = new InMemoryCheckpointer();
        cp.put("chat:t1", new GraphState(history(8), Map.of()));
        ScriptedSummarizer summarizer = new ScriptedSummarizer();
        summarizer.responses.add(SUMMARY_TEXT);
        CountDownLatch hold = new CountDownLatch(1);
        // summarize 记账后挂起：首个任务在途。
        summarizer.beforeRespond = () -> {
            try {
                hold.await(5, TimeUnit.SECONDS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        };
        CompactionService service = new CompactionService(cp, settings(), 250, summarizer);
        try {
            RunConfig config = RunConfig.of("chat:t1", "u1");
            service.onRunTerminated(config, null);
            waitFor(() -> summarizer.entered.getCount() == 0);
            assertEquals(0, service.pendingEvaluations(), "在途任务不在队列");
            // 同线程第二个终态：在途守卫丢弃（不入队）。
            service.onRunTerminated(config, null);
            assertEquals(0, service.pendingEvaluations(), "守卫应丢弃而非入队");
            // 放行 → 首个任务完成写回。
            hold.countDown();
            waitFor(() -> service.compactedRounds() == 1);
            waitFor(() -> service.pendingEvaluations() == 0);
            assertEquals(1, summarizer.transcripts.size(), "被丢弃的评估不得二次摘要");
        } finally {
            service.close();
        }
    }

    @Test
    void unavailableSummaryAbandonsTheRound() {
        // 摘要通道返回 null（端口未配 LLM 等价形态）→ 放弃，快照原样。
        useEstimator();
        InMemoryCheckpointer cp = new InMemoryCheckpointer();
        cp.put("chat:t1", new GraphState(history(8), Map.of()));
        CompactionService service = new CompactionService(cp, settings(), 250,
                (transcript, targetChars) -> null);
        try {
            service.onRunTerminated(RunConfig.of("chat:t1", "u1"), null);
            waitFor(() -> service.abandonedRounds() == 1);
            assertEquals(0, service.compactedRounds());
            assertEquals(16, cp.get("chat:t1").messages().size());
        } finally {
            service.close();
        }
    }

    @Test
    void moduleContextWithoutHookIsTheZeroChangeShape() {
        // #328 开关关/端口缺席 = 钩子 null（七参/六参兼容构造零破坏）。
        AgentModule.ModuleContext ctx = new AgentModule.ModuleContext(
                new Settings(), null, null, null, null, null, null);
        assertNull(ctx.compactionHook());
        assertFalse(new Settings().memory.compactionEnabled, "默认关（vectorPushdown 先例）");
    }

    @Test
    void queueFullSubmissionResetsInFlightGuardForThatThread() throws Exception {
        // R4-a F1 回归：队列满被拒的提交必须复位该线程的在途守卫——否则该
        // 线程的压缩静默永久失效（自定义吞掉式 handler 使 catch 不可达的
        // 缺陷已在产品码修复，本用例锁守卫复位语义）。
        useEstimator();
        InMemoryCheckpointer cp = new InMemoryCheckpointer();
        for (int i = 1; i <= 10; i++) {
            cp.put("chat:t" + i, new GraphState(history(8), Map.of()));
        }
        ScriptedSummarizer summarizer = new ScriptedSummarizer();
        for (int i = 0; i < 11; i++) {
            summarizer.responses.add(SUMMARY_TEXT);
        }
        CountDownLatch hold = new CountDownLatch(1);
        java.util.concurrent.atomic.AtomicBoolean gate =
                new java.util.concurrent.atomic.AtomicBoolean(true);
        summarizer.beforeRespond = () -> {
            // 仅首个任务阻塞（占住单 worker），其余照常返回。
            if (gate.compareAndSet(true, false)) {
                try {
                    hold.await(5, TimeUnit.SECONDS);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
        };
        CompactionService service = new CompactionService(cp, settings(), 250, summarizer);
        try {
            service.onRunTerminated(RunConfig.of("chat:t1", "u1"), null);
            assertTrue(summarizer.entered.await(5, TimeUnit.SECONDS), "t1 已占住 worker");
            // t2..t9 填满 8 槽队列，t10 被拒（AbortPolicy 抛出 → 提交侧复位守卫）。
            for (int i = 2; i <= 10; i++) {
                service.onRunTerminated(RunConfig.of("chat:t" + i, "u1"), null);
            }
            hold.countDown();
            waitFor(() -> service.compactedRounds() == 9);
            // 被拒的 t10 守卫已复位：再次终态可正常评估并执行。
            service.onRunTerminated(RunConfig.of("chat:t10", "u1"), null);
            waitFor(() -> service.compactedRounds() == 10);
        } finally {
            service.close();
        }
    }

    @Test
    void secondCompactionFoldsPriorSummaryIntoTranscript() throws Exception {
        // R4-a F2 回归：二次压缩必须把上一轮压缩摘要并入本轮转写——否则旧
        // 摘要承载的更早历史在新状态（新摘要+近 N 轮）中被静默销毁。
        useEstimator();
        InMemoryCheckpointer cp = new InMemoryCheckpointer();
        List<ChatMessage> prior = new ArrayList<>();
        prior.add(SystemMessage.from(ContextEngine.COMPACTION_SUMMARY_PREFIX
                + "旧摘要内容：用户偏好深色主题，项目代号夜莺"));
        prior.addAll(history(8));
        cp.put("chat:t1", new GraphState(prior, Map.of()));
        ScriptedSummarizer summarizer = new ScriptedSummarizer();
        summarizer.responses.add(SUMMARY_TEXT);
        CompactionService service = new CompactionService(cp, settings(), 250, summarizer);
        try {
            service.onRunTerminated(RunConfig.of("chat:t1", "u1"), null);
            waitFor(() -> service.compactedRounds() == 1);
            // 旧摘要内容进转写（折叠链不断档）。
            assertTrue(summarizer.transcripts.get(0)
                    .contains("此前会话摘要：旧摘要内容：用户偏好深色主题"),
                    summarizer.transcripts.get(0));
            // 新状态头部 = 新摘要（旧摘要消息被新摘要替换，不叠加）。
            assertEquals(7, cp.get("chat:t1").messages().size());
            assertEquals(ContextEngine.COMPACTION_SUMMARY_PREFIX + SUMMARY_TEXT,
                    ((SystemMessage) cp.get("chat:t1").messages().get(0)).text());
        } finally {
            service.close();
        }
    }

    @Test
    void tokenFactorMovesTriggerLineAndScalesEstimate() {
        // R4-a F5：估算侧乘 tokenFactor（对齐 #340/buildModelInput 口径）——
        // 208 token 在 budget=300×0.7=210 下不触发（factor=1.0）；factor=1.3
        // 时触发线随系数下移（(300/1.3)×0.7≈161）→ 触发。估算侧偏保守的
        // 修正不再让压缩触发晚于运行期修剪。
        useEstimator();
        InMemoryCheckpointer cp = new InMemoryCheckpointer();
        cp.put("chat:t1", new GraphState(history(8), Map.of()));
        ScriptedSummarizer idle = new ScriptedSummarizer();
        // 旧四参构造 = factor 1.0（委托旧口径，行为逐字节不变）。
        CompactionService unitFactor = new CompactionService(cp, settings(), 300, idle);
        try {
            unitFactor.onRunTerminated(RunConfig.of("chat:t1", "u1"), null);
            settle();
            assertEquals(0, idle.transcripts.size(), "factor=1.0 不触发（208 ≤ 210）");
            assertEquals(0, unitFactor.abandonedRounds());
            assertEquals(0, unitFactor.compactedRounds());
            assertEquals(16, cp.get("chat:t1").messages().size(), "快照原样");
        } finally {
            unitFactor.close();
        }

        // 同一快照换 factor=1.3：触发线移动 → 压缩发生，元数据估算按系数放大。
        InMemoryCheckpointer cp2 = new InMemoryCheckpointer();
        cp2.put("chat:t1", new GraphState(history(8), Map.of()));
        ScriptedSummarizer summarizer = new ScriptedSummarizer();
        summarizer.responses.add(SUMMARY_TEXT);
        CompactionService factored = new CompactionService(cp2, settings(), 300, 1.3, summarizer);
        try {
            factored.onRunTerminated(RunConfig.of("chat:t1", "u1"), null);
            waitFor(() -> factored.compactedRounds() == 1);
            // targetChars 仍按原始预算裁决（与系数无关）：clamp(300/10,400,2000)=400。
            assertEquals(List.of(400), summarizer.targets);
            GraphState compacted = cp2.get("chat:t1");
            assertEquals(7, compacted.messages().size());
            // 每条出口乘系数（与 buildModelInput 同乘法点）：round(13×1.3)=17 × 16。
            Map<String, Object> meta = castMap(compacted.extra().get("compaction"));
            assertEquals(272, ((Number) meta.get("original_token_estimate")).intValue(),
                    "13/条 × 1.3 → 17/条 × 16（估算侧乘 factor）");
            assertEquals(5, ((Number) meta.get("folded_turns")).intValue());
        } finally {
            factored.close();
        }
    }

    @Test
    void staleExtraSnapshotIsNotWrittenBackWhenMessagesUnchanged() {
        // R4-a F6：摘要期间「消息不变」的 extra-only 落盘（attachments 通道
        // 更新、#199 cancelled 标记等）不被旧快照整行回退——replacement 的
        // extra 以锁内重读的 fresh.extra() 为 prior，不是摘要前的旧快照。
        useEstimator();
        InMemoryCheckpointer cp = new InMemoryCheckpointer();
        cp.put("chat:t1", new GraphState(history(8),
                Map.of("attachments", Map.of("0", "首轮附件"))));
        ScriptedSummarizer summarizer = new ScriptedSummarizer();
        summarizer.responses.add(SUMMARY_TEXT);
        summarizer.beforeRespond = () -> {
            // 摘要期间 extra-only 落盘：消息序列不变（基准校验仍过），
            // attachments 追加一条（附件元数据整表替换的常规形态）。
            GraphState mutated = new GraphState(cp.get("chat:t1").messages(),
                    Map.of("attachments", Map.of("0", "首轮附件", "1", "摘要期间新增附件")));
            cp.put("chat:t1", mutated);
        };
        CompactionService service = new CompactionService(cp, settings(), 250, summarizer);
        try {
            service.onRunTerminated(RunConfig.of("chat:t1", "u1"), null);
            waitFor(() -> service.compactedRounds() == 1);
            GraphState compacted = cp.get("chat:t1");
            // 压缩照常发生（消息基准一致）。
            assertEquals(7, compacted.messages().size());
            assertTrue(compacted.messages().get(0) instanceof SystemMessage);
            // 摘要期间的 extra 更新保留（不被旧快照回退）。
            Map<String, Object> attachments = castMap(compacted.extra().get("attachments"));
            assertEquals("首轮附件", attachments.get("0"));
            assertEquals("摘要期间新增附件", attachments.get("1"),
                    "摘要期间落盘的 extra-only 更新必须保留（F6）");
            // 压缩元数据仍在（prior 之上单键覆盖）。
            Map<String, Object> meta = castMap(compacted.extra().get("compaction"));
            assertEquals(16, ((Number) meta.get("original_message_count")).intValue());
            assertEquals(5, ((Number) meta.get("folded_turns")).intValue());
        } finally {
            service.close();
        }
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> castMap(Object value) {
        assertTrue(value instanceof Map, "compaction 元数据应为 Map：" + value);
        return (Map<String, Object>) value;
    }
}
