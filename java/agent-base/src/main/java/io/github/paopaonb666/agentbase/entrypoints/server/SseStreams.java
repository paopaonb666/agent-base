package io.github.paopaonb666.agentbase.entrypoints.server;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.atomic.AtomicBoolean;

import io.github.paopaonb666.agentbase.core.MemoryScope;
import io.github.paopaonb666.agentbase.core.graph.CompiledGraph;
import io.github.paopaonb666.agentbase.core.graph.EventSink;
import io.github.paopaonb666.agentbase.core.graph.GraphState;
import io.github.paopaonb666.agentbase.core.graph.RunConfig;
import io.github.paopaonb666.agentbase.extensions.events.Events;

/**
 * SSE 流式契约的生产端：生产者 / 心跳 / 事件流 / 记忆形成钩子。
 *
 * <p>取消传播：客户端断开会取消生产者任务，从而取消正在进行的图执行
 * 与随之而来的 LLM 请求——token 消耗随即停止，而不是在后台跑完。
 *
 * <p>对应 Python 侧 {@code entrypoints/server/sse.py}：
 * <ul>
 *   <li>事件队列有界（256）——慢客户端背压自然传导，ping 允许丢弃；</li>
 *   <li>首帧立即发 ping（代理保活）；心跳 15s；</li>
 *   <li>step running 按节点首见去重（对应 messages 模式的 running 集合），
 *       completed 每节点结束各一条（updates 模式）；</li>
 *   <li>自定义事件（tool_call/plan/sources）经契约类型校验，未知/畸形
 *       丢弃——坏模块不能把未校验数据透给浏览器；</li>
 *   <li>异常转 ErrorEvent（URL 脱敏 + 300 字符封顶）；每个终态都以
 *       done(termination) 收尾——统一终止标注由图层判定（#9）。</li>
 * </ul>
 */
public final class SseStreams {

    private static final Logger LOG = LoggerFactory.getLogger(SseStreams.class);

    /** 心跳周期（秒）。 */
    public static final double HEARTBEAT_SECONDS = 15.0;
    /** SSE 事件队列上限：背压有界，ping 允许丢弃。 */
    public static final int SSE_QUEUE_MAXSIZE = 256;

    /**
     * 流终止哨兵：ArrayBlockingQueue 不接受 null——null 哨兵会在 put 时抛
     * NPE 且被守护生产者线程静默吞掉，导致消费者永久阻塞在 take()、
     * complete() 永不执行、SSE 连接不关闭（前端状态停在"正在对话"）。
     */
    private static final Object SENTINEL = new Object();

    /** 给客户端的错误摘要：类型名 + 消息，URL 脱敏、长度封顶。 */
    public static String safeErrorText(Throwable exc) {
        String message = exc.getMessage() == null ? "" : exc.getMessage();
        String text = "%s: %s".formatted(exc.getClass().getSimpleName(), message)
                .replaceAll("[a-zA-Z][a-zA-Z0-9+.\\-]*://\\S+", "<redacted-url>");
        return text.length() > 300 ? text.substring(0, 300) : text;
    }

    private SseStreams() {}

    /**
     * 打开一次对话轮的 SSE 流（对应 _event_stream 的 StreamingResponse）。
     *
     * @param graph         已编译图
     * @param messagesInput 本轮输入消息
     * @param config        运行配置
     * @param userThreadId  用户侧 thread id（DoneEvent 载荷）
     * @param onComplete    轮次正常完成后的回调（记忆形成管线）；可为 null
     */
    public static SseEmitter open(CompiledGraph graph, java.util.List<dev.langchain4j.data.message.ChatMessage> messagesInput,
                                  java.util.Map<String, Object> extra, RunConfig config, String userThreadId, Runnable onComplete) {
        return open(graph, messagesInput, extra, config, userThreadId, onComplete,
                new SseEmitter(0L)); // 不超时（Python 无超时语义）
    }

    /** 可注入 emitter 的内部重载（测试替身用；生产方用上面的默认形态）。 */
    static SseEmitter open(CompiledGraph graph, java.util.List<dev.langchain4j.data.message.ChatMessage> messagesInput,
                           java.util.Map<String, Object> extra, RunConfig config, String userThreadId,
                           Runnable onComplete, SseEmitter emitter) {
        return open(graph, messagesInput, extra, config, userThreadId, onComplete, null, emitter);
    }

    /**
     * 生产形态（#79）：终态钩子 + 默认 emitter。onTerminate 在 produce 的
     * finally 里恰好执行一次（正常/失败/取消全路径），并发闸门在此退出。
     */
    public static SseEmitter open(CompiledGraph graph, java.util.List<dev.langchain4j.data.message.ChatMessage> messagesInput,
                                  java.util.Map<String, Object> extra, RunConfig config, String userThreadId,
                                  Runnable onComplete, Runnable onTerminate) {
        return open(graph, messagesInput, extra, config, userThreadId, onComplete, onTerminate,
                new SseEmitter(0L)); // 不超时（Python 无超时语义）
    }

    /**
     * 全参形态（#79）：{@code onTerminate} 是 run 终态钩子——produce 的
     * finally 里恰好执行一次（正常/失败/取消全路径），并发闸门在此退出。
     */
    public static SseEmitter open(CompiledGraph graph, java.util.List<dev.langchain4j.data.message.ChatMessage> messagesInput,
                                  java.util.Map<String, Object> extra, RunConfig config, String userThreadId,
                                  Runnable onComplete, Runnable onTerminate, SseEmitter emitter) {
        BlockingQueue<Object> queue = new ArrayBlockingQueue<>(SSE_QUEUE_MAXSIZE);
        AtomicBoolean cancelled = new AtomicBoolean(false);
        Thread producer = new Thread(() -> produce(queue, graph, messagesInput, extra, config,
                userThreadId, onComplete, onTerminate, cancelled), "sse-producer");
        producer.setDaemon(true);
        Thread heartbeat = new Thread(() -> heartbeat(queue, cancelled), "sse-heartbeat");
        heartbeat.setDaemon(true);

        Runnable shutdown = () -> {
            cancelled.set(true);
            producer.interrupt();
            heartbeat.interrupt();
        };
        emitter.onCompletion(shutdown);
        emitter.onTimeout(shutdown);
        emitter.onError(t -> shutdown.run());

        producer.start();
        heartbeat.start();

        Thread consumer = new Thread(() -> consume(emitter, queue, cancelled, shutdown),
                "sse-consumer");
        consumer.setDaemon(true);
        consumer.start();
        return emitter;
    }

    /** 生产者：把图事件推入队列；哨兵 null 终止（对应 _produce）。 */
    private static void produce(BlockingQueue<Object> queue, CompiledGraph graph,
                                java.util.List<dev.langchain4j.data.message.ChatMessage> messagesInput,
                                java.util.Map<String, Object> extra, RunConfig config,
                                String userThreadId, Runnable onComplete, Runnable onTerminate,
                                AtomicBoolean cancelled) {
        // 记忆作用域（M6e）：图节点与记忆工具在本线程运行，据此拿到
        // user/thread（工具线程经 MDC 转发继承）。
        MemoryScope.set(config.userId(), config.threadId());
        Set<String> running = new HashSet<>();
        // #338：本轮来源登记簿——各工具的 sources 事件按发射序收集，最终
        // 回复的 [n] 引用据此解析（容错：解析不出就不发）。
        java.util.List<io.github.paopaonb666.agentbase.extensions.events.Events.Source>
                sourceRegistry = new java.util.ArrayList<>();
        try {
            EventSink sink = new EventSink() {
                @Override
                public void accept(Object event) {
                    if (cancelled.get()) {
                        throw new java.util.concurrent.CancellationException("client disconnected");
                    }
                    if (event instanceof Events.StepEvent step && "running".equals(step.status())) {
                        if (!running.add(step.name())) {
                            return; // 首见去重（对应 Python running 集合）
                        }
                    }
                    if (event instanceof Events.SourcesEvent sources) {
                        sourceRegistry.addAll(sources.sources());
                    }
                    putUninterruptibly(queue, event, cancelled);
                }

                @Override
                public boolean cancelled() {
                    return cancelled.get();
                }
            };
            // 输入 extras（附件元数据等）随首轮输入进入状态通道：extras 按
            // 键合并（apply 语义），缺席的键保留历史值——对应 Python 输入
            // dict 只提供 messages 时其余通道保持原值。
            // #9：终止标注的权威在图层——streamWithTermination 把失败也归一
            // 为终态（不抛出），SSE 层只负责渲染 done 的 termination 字段。
            CompiledGraph.RunOutcome outcome = graph.streamWithTermination(
                    new GraphState(messagesInput,
                            extra == null ? java.util.Map.of() : extra), config, sink);
            // 终态投递前清掉挂起的中断标志：中断的使命（打断图执行）已完成，
            // 保留它会让下面每个 put 立即抛 CancellationException——done 与
            // 哨兵全部丢失、消费者永久阻塞在 take()（全量套件实测踩中）。
            // 清除只影响「过期」标志；put 期间新到的中断仍按 cancelled 语义处理。
            Thread.interrupted();
            Events.RunTermination termination = outcome.termination();
            if (outcome.error() == null) {
                // 正常返回：DONE（或断连 break 的 CANCELLED——照常走完引用
                // 解析与记忆钩子，连接已死则入队无害）。
                // #338：最终回复 [n] 引用解析 → cited sources 事件（在 done 前）。
                // 与工具执行期的全量 sources 事件互补：这里只发回复真正引用的子集。
                List<Events.Source> cited = Citations.resolve(
                        lastReplyText(outcome.state()), Citations.dedup(sourceRegistry));
                if (!cited.isEmpty()) {
                    putUninterruptibly(queue, new Events.SourcesEvent(cited), cancelled);
                }
                putUninterruptibly(queue, new Events.DoneEvent(userThreadId, termination), cancelled);
                if (onComplete != null) {
                    // 独立任务：随后的哨兵与 finally 取消都不影响它跑完。
                    Background.spawn(onComplete, "memory-capture");
                }
            } else if (termination == Events.RunTermination.CANCELLED) {
                // 断连经 sink 取消传播：连接已死，仍补发标注（#9）——尽力入队。
                putBestEffort(queue, new Events.DoneEvent(userThreadId, termination), cancelled);
            } else {
                // 失败终态：ErrorEvent（既有形状不动）+ done 补 termination
                // （#9）——前端不再从 error 文本反推结束原因。
                LOG.error("sse: stream failed", outcome.error());
                putUninterruptibly(queue, new Events.ErrorEvent(safeErrorText(outcome.error())), cancelled);
                putUninterruptibly(queue, new Events.DoneEvent(userThreadId, termination), cancelled);
            }
        } catch (java.util.concurrent.CancellationException e) {
            // 客户端断开（入队被打断）：补发取消标注（#9），入不进队列就放弃。
            Thread.interrupted(); // 同上：终态投递前清过期中断标志
            putBestEffort(queue, new Events.DoneEvent(userThreadId, Events.RunTermination.CANCELLED), cancelled);
        } catch (Exception exc) {
            // 图层之后的失败（引用解析/入队等）：归 INTERNAL_ERROR（#9）。
            LOG.error("sse: stream failed", exc);
            Thread.interrupted(); // 同上：终态投递前清过期中断标志
            putUninterruptibly(queue, new Events.ErrorEvent(safeErrorText(exc)), cancelled);
            putUninterruptibly(queue, new Events.DoneEvent(userThreadId, Events.RunTermination.INTERNAL_ERROR), cancelled);
        } finally {
            // 终态钩子（#79 并发闸门退出）最先执行：保证后续 put 抛异常时
            // 配对释放不丢（finally 里后续语句会被新异常跳过）。
            if (onTerminate != null) {
                try {
                    onTerminate.run();
                } catch (Exception e) {
                    LOG.debug("sse: terminate hook failed", e);
                }
            }
            Thread.interrupted(); // 哨兵必须送达：同上，清过期中断标志
            putUninterruptibly(queue, SENTINEL, cancelled);
            MemoryScope.clear();
        }
    }

    /** 尽力入队（取消路径专用）：入不进去就放弃——连接已死，标注无从送达。 */
    private static void putBestEffort(BlockingQueue<Object> queue, Object event,
                                      AtomicBoolean cancelled) {
        try {
            putUninterruptibly(queue, event, cancelled);
        } catch (java.util.concurrent.CancellationException ignored) {
            // 断连后满队列上的 put 永远无人接收（见 putUninterruptibly 的取消分支）。
        }
    }

    private static void putUninterruptibly(BlockingQueue<Object> queue, Object event,
                                           AtomicBoolean cancelled) {
        boolean interrupted = false;
        try {
            while (true) {
                try {
                    queue.put(event);
                    return;
                } catch (InterruptedException e) {
                    interrupted = true;
                    if (cancelled.get()) {
                        // 消费者已死（断连/超时/错误触发 shutdown）：满队列上
                        // 的 put 永远无人接收——继续重试会让生产者线程卡死、
                        // graph.stream 不返回、在途的 LLM 工作被拖住（对应
                        // Python 侧断连即取消生产者任务）。
                        throw new java.util.concurrent.CancellationException(
                                "client disconnected");
                    }
                }
            }
        } finally {
            if (interrupted) {
                Thread.currentThread().interrupt();
            }
        }
    }

    /** 心跳：每 15s 一个 ping，队列满（慢客户端）直接丢。 */
    private static void heartbeat(BlockingQueue<Object> queue, AtomicBoolean cancelled) {
        while (!cancelled.get()) {
            try {
                Thread.sleep((long) (HEARTBEAT_SECONDS * 1000));
            } catch (InterruptedException e) {
                return;
            }
            // ping 是可丢弃的保活信号：offer 不等待（满即丢），不与关键事件争抢。
            if (!queue.offer(new Events.PingEvent())) {
                // 丢弃可观测（#210）：慢客户端背压下 ping 被丢的次数进 /metrics。
                io.github.paopaonb666.agentbase.extensions.Metrics.events().observeDrop("ping_full");
            }
        }
    }

    /** 消费者：发首帧 ping，循环消费队列直到哨兵（对应 _event_stream 主体）。 */
    /** 最终回复文本：状态里最后一条带文本的 AiMessage；没有则 null（不解析不 emit）。 */
    private static String lastReplyText(io.github.paopaonb666.agentbase.core.graph.GraphState state) {
        if (state == null) {
            return null;
        }
        var messages = state.messages();
        for (int i = messages.size() - 1; i >= 0; i--) {
            if (messages.get(i) instanceof dev.langchain4j.data.message.AiMessage ai) {
                String text = io.github.paopaonb666.agentbase.core.context.ContextEngine
                        .contentText(ai);
                if (text != null && !text.isBlank()) {
                    return text;
                }
            }
        }
        return null;
    }

    private static void consume(SseEmitter emitter, BlockingQueue<Object> queue,
                                AtomicBoolean cancelled, Runnable shutdown) {
        try {
            send(emitter, new Events.PingEvent());
            while (true) {
                Object event = queue.take();
                if (event == SENTINEL) {
                    break;
                }
                send(emitter, event);
            }
            emitter.complete();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            emitter.completeWithError(new IllegalStateException("consumer interrupted"));
        } catch (Exception e) {
            // 客户端断开（IOException）：取消生产者，LLM 请求中止。
            cancelled.set(true);
            shutdown.run();
            emitter.completeWithError(e);
        }
    }

    private static void send(SseEmitter emitter, Object event) throws java.io.IOException {
        emitter.send(SseEmitter.event()
                .name(Events.typeOf(event))
                .data(Events.dataJson(event)));
    }
}
