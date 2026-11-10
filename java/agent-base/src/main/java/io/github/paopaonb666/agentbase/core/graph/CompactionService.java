package io.github.paopaonb666.agentbase.core.graph;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.core.context.ContextEngine;
import io.github.paopaonb666.agentbase.core.text.CharMetrics;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

/**
 * 会话历史压缩任务体（#328，引擎侧）：操作 checkpointer、不碰记忆库——
 * 记忆库只经 {@code Summarizer} 端口桥提供快档摘要。
 *
 * <p><b>背景</b>：buildModelInput 的预算修剪只影响"本轮给模型看什么"，
 * checkpointer 里的历史本体永不缩——长会话每步整行覆盖写全量 JSON，历史
 * 无界增长。本服务在轮次终态后（{@link CompiledGraph.CompactionHook} 挂点，
 * 覆盖 invoke/stream 两条路径）异步评估：历史 token 估算超预算×ratio 即把
 * 历史折叠为「压缩摘要 SystemMessage（{@link ContextEngine#COMPACTION_SUMMARY_PREFIX}
 * 标记前缀）+ 近 N 轮原文」写回。
 *
 * <p><b>执行模型</b>：自带单守护线程执行器（有界队列，池满丢弃 WARN——
 * 对齐 #212/#182 有界放弃风格），绝不占用 Background 捕获池、绝不阻塞运行
 * 线程；每线程在途至多 1 个评估（{@code ConcurrentHashMap.newKeySet} 守卫，
 * 热线程的连续终态不积压队列）。
 *
 * <p><b>写回互斥（#330 方案 B）</b>：经 {@link CompiledGraph#compactThread}
 * 持 per-thread 锁「重读快照 → 基准校验 → 整行覆盖」——基准（消息数 +
 * 最后一条消息相等性）不一致 = 有并发写入，放弃本轮（WARN + 计数可观测，
 * 不重试轰炸）。放弃是安全默认：下一轮终态会重新评估。
 *
 * <p>对应 Python 侧无此机制（langgraph 历史无界）——Java 单边增强，登记
 * 差异登记簿。
 */
public final class CompactionService implements AutoCloseable {

    private static final Logger LOG = LoggerFactory.getLogger(CompactionService.class);

    /** 压缩评估任务队列容量（有界；满则丢弃本轮评估，可观测不阻塞）。 */
    private static final int EVALUATION_QUEUE_CAPACITY = 8;

    /**
     * 摘要通道（#328）：待折叠段转写 → 摘要文本；返回 null/空白 = 摘要
     * 不可用（本轮放弃）。实现方（记忆端口桥）负责 LLM 调用与失败安全。
     */
    @FunctionalInterface
    public interface Summarizer {
        String summarize(String transcript, int targetChars);
    }

    private final Checkpointer checkpointer;
    private final double ratio;
    private final int keepRecentTurns;
    /** 折叠段转写的字符上限（保尾部）：超长历史不把快档 LLM 输入撑爆。 */
    private final int transcriptMaxChars;
    private final int budget;
    private final Summarizer summarizer;
    private final ThreadPoolExecutor executor;
    /** 在途评估守卫：每线程至多 1 个（提交侧判定，热线程不积压队列）。 */
    private final Set<String> inFlight = ConcurrentHashMap.newKeySet();
    /** 放弃轮累计（忙/基准不一致/摘要不可用）——压缩静默失效的可观测面。 */
    private final AtomicLong abandonedRounds = new AtomicLong();
    private final AtomicLong compactedRounds = new AtomicLong();

    /**
     * @param budget    输入预算（装配期按主力 runtime 经
     *                  {@code ContextEngine.resolveInputBudget} 同口径推导）
     * @param summarizer 摘要通道（生产装配 = {@code memoryPort::compactTranscript}）
     */
    public CompactionService(Checkpointer checkpointer, Settings.MemorySettings settings,
                             int budget, Summarizer summarizer) {
        this.checkpointer = checkpointer;
        this.ratio = settings.compactionRatio;
        this.keepRecentTurns = settings.compactionKeepRecentTurns;
        this.transcriptMaxChars = settings.extractionMaxInputChars;
        this.budget = budget;
        this.summarizer = summarizer;
        // 单守护线程 + 有界队列（#328）：压缩属批量成本型工作，与 Background
        // 捕获池隔离（LLM 耗时长，不挤占 capture）；池满丢弃可观测。
        this.executor = new ThreadPoolExecutor(1, 1, 0L, TimeUnit.MILLISECONDS,
                new java.util.concurrent.ArrayBlockingQueue<>(EVALUATION_QUEUE_CAPACITY),
                r -> {
                    Thread thread = new Thread(r, "compaction-worker");
                    thread.setDaemon(true);
                    return thread;
                });
        // 拒绝策略保持默认 AbortPolicy（R4-a F1）：队列满抛
        // RejectedExecutionException——提交侧 catch 负责 inFlight 守卫复位，
        // 自定义吞掉式 handler 会让守卫条目永久泄漏、该线程压缩静默失效。
    }

    /** 放弃轮累计（#330：压缩静默失效不可诊断——对齐 #198 欠账计数风格）。 */
    public long abandonedRounds() {
        return abandonedRounds.get();
    }

    /** 完成压缩的轮次累计。 */
    public long compactedRounds() {
        return compactedRounds.get();
    }

    /** 排队中的评估任务数（可观测/测试：在途守卫丢弃而非入队的判定面）。 */
    int pendingEvaluations() {
        return executor.getQueue().size();
    }

    /**
     * 终态钩子入口（#328）：提交侧先做在途守卫（同线程已有评估在途即跳过
     * ——终态后快照随时可再评估，不积压），再转自带执行器；{@code finalState}
     * 可能为 null（失败/触顶轮），快照一律以 checkpointer 为准。
     */
    public void onRunTerminated(RunConfig config, GraphState finalState) {
        String threadId = config.threadId();
        if (!inFlight.add(threadId)) {
            return;
        }
        try {
            executor.execute(() -> {
                try {
                    compactOnce(threadId);
                } catch (Exception e) {
                    // 评估失败安全（绝不影响对话主链路）；放弃可观测。
                    abandonedRounds.incrementAndGet();
                    LOG.warn("graph: 压缩评估失败（放弃本轮）：thread={}，{}", threadId, e.toString());
                } finally {
                    inFlight.remove(threadId);
                }
            });
        } catch (RejectedExecutionException e) {
            // 队列满（有界丢弃）或服务已关闭（shutdownNow 后提交被拒）：
            // 守卫条目必须在这里复位——否则该线程的压缩静默永久失效
            // （R4-a F1）；放弃可观测，计一次放弃轮。
            inFlight.remove(threadId);
            abandonedRounds.incrementAndGet();
            LOG.warn("graph: 压缩评估提交被拒（{}），放弃本轮：thread={}",
                    executor.isShutdown() ? "服务已关闭" : "评估队列已满 " + EVALUATION_QUEUE_CAPACITY,
                    threadId);
        }
    }

    /** 释放执行器（AgentRuntime.close 调用）；在途任务中断，队列丢弃。 */
    @Override
    public void close() {
        executor.shutdownNow();
    }

    // ── 任务体 ───────────────────────────────────────────────────────────

    private void compactOnce(String threadId) {
        GraphState snapshot = checkpointer.get(threadId);
        if (snapshot == null || snapshot.messages().isEmpty()) {
            return;
        }
        List<ChatMessage> messages = List.copyOf(snapshot.messages());
        if (!ContextEngine.compactionNeeded(messages, budget, ratio)) {
            return;
        }
        int keepFrom = recentTurnsStart(messages, keepRecentTurns);
        if (keepFrom <= 0) {
            // 全部历史都在保留窗内（不足 N 轮）：没有可折叠段。
            LOG.debug("graph: 历史不足 {} 轮，无可折叠段：thread={}", keepRecentTurns, threadId);
            return;
        }
        // 基准（#330 方案 B）：消息数 + 最后一条消息相等性。摘要期间新 run
        // 可能落盘新消息——写回前在锁内重读校验，不一致即放弃。
        int baseCount = messages.size();
        ChatMessage baseLast = messages.get(baseCount - 1);
        int tokenEstimate = estimateTokens(messages);

        String transcript = renderTranscript(messages.subList(0, keepFrom));
        if (transcript.isBlank()) {
            // 折叠段没有可摘要的对话内容（只剩注入块/压缩摘要等 system 消息）
            // ——不烧 LLM，直接返回（下一轮终态再评估）。
            return;
        }
        int targetChars = targetCharsFor(budget);
        String summary = summarizer.summarize(transcript, targetChars);
        if (summary == null || summary.isBlank()) {
            abandonedRounds.incrementAndGet();
            LOG.warn("graph: 压缩摘要不可用（放弃本轮）：thread={}", threadId);
            return;
        }
        // 新状态 = [压缩摘要（标记前缀 SystemMessage）] + [近 N 轮原文]；
        // extra 保留并追加压缩元数据（读侧不解析，GraphStateJson.VERSION 不 bump）。
        List<ChatMessage> compactedMessages = new ArrayList<>();
        compactedMessages.add(SystemMessage.from(
                ContextEngine.COMPACTION_SUMMARY_PREFIX + summary.strip()));
        compactedMessages.addAll(messages.subList(keepFrom, messages.size()));
        GraphState replacement = new GraphState(compactedMessages,
                compactionExtra(snapshot.extra(), baseCount, tokenEstimate));

        boolean written = CompiledGraph.compactThread(checkpointer, threadId, fresh -> {
            List<ChatMessage> current = fresh.messages();
            if (current.size() != baseCount
                    || !current.get(current.size() - 1).equals(baseLast)) {
                // 基准不一致 = 有并发写入（#330 方案 B）：放弃，不产生半写。
                return Optional.empty();
            }
            return Optional.of(replacement);
        });
        if (written) {
            compactedRounds.incrementAndGet();
            LOG.info("graph: 历史压缩完成：thread={}，{} 条 → {} 条（摘要 {} 字，目标 {} 字）",
                    threadId, baseCount, compactedMessages.size(),
                    summary.strip().length(), targetChars);
        } else {
            abandonedRounds.incrementAndGet();
            LOG.warn("graph: 压缩写回放弃（忙或基准不一致）：thread={}，累计放弃 {} 轮",
                    threadId, abandonedRounds.get());
        }
    }

    /**
     * 近 N 轮原文的窗口起点（#328）：「轮」= 从一条 UserMessage 到下一条
     * UserMessage 之前。从尾部收集 UserMessage 起点，锚定在第 N 个（从尾数）
     * UserMessage；起点落上 ToolMessage 时向前扩窗（工具配对不拆散——同
     * buildModelInput 边界守卫语义，方向相反：扩窗保对而不是丢消息）。
     * 不足 N 轮返回 0（无可折叠段）。
     */
    static int recentTurnsStart(List<ChatMessage> messages, int keepTurns) {
        int seen = 0;
        for (int i = messages.size() - 1; i >= 0; i--) {
            if (messages.get(i) instanceof UserMessage) {
                seen++;
                if (seen == keepTurns) {
                    int start = i;
                    while (start > 0 && messages.get(start) instanceof ToolExecutionResultMessage) {
                        start--;
                    }
                    return start;
                }
            }
        }
        return 0;
    }

    /** targetChars 裁决（#328）：clamp(budget/10, 400, 2000)。 */
    static int targetCharsFor(int budget) {
        return Math.max(400, Math.min(2000, budget / 10));
    }

    /** 压缩元数据（#328）：extra "compaction" 键；读侧不解析，VERSION 不 bump。 */
    private static Map<String, Object> compactionExtra(Map<String, Object> prior,
                                                       int originalCount, int tokenEstimate) {
        Map<String, Object> meta = new LinkedHashMap<>();
        meta.put("original_message_count", originalCount);
        meta.put("original_token_estimate", tokenEstimate);
        meta.put("at", java.time.Instant.now().toString());
        Map<String, Object> extra = new LinkedHashMap<>(prior);
        extra.put("compaction", meta);
        return extra;
    }

    /** 历史估算（#328 口径）：countMessageTokens + 每条 +4，与 compactionNeeded 一致。 */
    private static int estimateTokens(List<ChatMessage> messages) {
        int total = 0;
        for (ChatMessage message : messages) {
            total += ContextEngine.countMessageTokens(message) + 4;
        }
        return total;
    }

    /**
     * 折叠段转写（#328）：格式与 MemoryPipeline.renderTranscript 同构
     * （用户：/助手：/工具[..]：、工具行 300 码点截断、保尾部）——core 侧
     * 自实现（M 包不可反向依赖）。注入型 SystemMessage（历史注入块随
     * checkpoint 轮轮累积）与压缩摘要 SystemMessage 不进转写：前者不是
     * 对话内容、后者是上一轮折叠产物（不重复摘要）。
     */
    private String renderTranscript(List<ChatMessage> messages) {
        List<String> lines = new ArrayList<>();
        for (ChatMessage message : messages) {
            // SystemMessage 分流（R4-a F2）：注入型 system（历史注入块）不是
            // 对话内容，跳过——随折叠一并消失是设计语义（模型输入侧它们本就
            // 被注入去重剔除，每轮仍有新注入）；压缩摘要是上一轮折叠产物，
            // **必须并入本轮转写**——否则二次压缩会把旧摘要内容从折叠链中
            // 销毁（新状态只含新摘要+近 N 轮，旧摘要承载的更早历史静默丢失）。
            if (message instanceof SystemMessage system) {
                if (system.text().startsWith(ContextEngine.COMPACTION_SUMMARY_PREFIX)) {
                    lines.add("此前会话摘要：" + system.text()
                            .substring(ContextEngine.COMPACTION_SUMMARY_PREFIX.length()).strip());
                }
                continue;
            }
            String text = ContextEngine.contentText(message).strip();
            if (text.isEmpty()) {
                continue;
            }
            if (message instanceof UserMessage) {
                lines.add("用户：" + text);
            } else if (message instanceof AiMessage) {
                // 纯 tool_calls 的 AI 消息（无文本）没有转写价值。
                lines.add("助手：" + text);
            } else if (message instanceof ToolExecutionResultMessage tool) {
                lines.add("工具[" + tool.toolName() + "]：" + CharMetrics.head(text, 300));
            }
        }
        String transcript = String.join("\n", lines);
        if (CharMetrics.len(transcript) <= transcriptMaxChars) {
            return transcript;
        }
        // 保尾部：最近的内容对折叠摘要更重要。
        return "……（前文截断）\n" + CharMetrics.tail(transcript, transcriptMaxChars);
    }
}
