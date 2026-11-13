package io.github.paopaonb666.agentbase.core.graph;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import io.github.paopaonb666.agentbase.extensions.events.Events;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 编译后的图：invoke（批量）与 stream（三通道流式）的执行引擎。
 *
 * <p>对应 LangGraph {@code CompiledStateGraph} 在本项目里的实际使用面：
 * {@code ainvoke}（CLI）、{@code astream(stream_mode=["messages","updates","custom"])}
 * （SSE）、{@code aget_state}（历史/计划端点）、线程删除。
 *
 * <p>执行语义：从入口节点开始，每步执行一个节点——推送
 * {@code StepEvent(running)} → 节点函数（期间的自定义事件与 token 增量
 * 直通 sink）→ 应用局部更新 → {@code StepEvent(completed)} → 按条件路由
 * 下一步；每步结束后 upsert checkpointer（superstep 粒度，重启可恢复）。
 * 步数超过 {@code RunConfig.recursionLimit} 抛 {@link GraphRecursionError}。
 */
public final class CompiledGraph {

    private static final Logger LOG = LoggerFactory.getLogger(CompiledGraph.class);

    /**
     * per-thread 执行互斥（#197）：同 thread 并发 run 会各自 loadState
     * 起步、每步整行覆盖写快照——last-writer-wins 让历史错乱且无法诊断。
     * 第二个 run 以 {@link ThreadBusyError} 快速失败（路由层转 409）。
     * 锁对象按 threadId 常驻（不回收：ReentrantLock 很小，换取无竞态的
     * 正确性——回收 map 条目会在 unlock/remove 窗口放进两个持有者）。
     */
    private static final java.util.concurrent.ConcurrentHashMap<String, java.util.concurrent.locks.ReentrantLock>
            THREAD_LOCKS = new java.util.concurrent.ConcurrentHashMap<>();

    /** 同 thread 已有进行中的 run（数据完整性保护，非暂时性错误）。 */
    public static final class ThreadBusyError extends IllegalStateException {
        public ThreadBusyError(String threadId) {
            super("thread busy: " + threadId);
        }
    }

    /**
     * 运行终态钩子（#328）：图运行结束（per-thread 锁已释放）后对快照做
     * 异步评估的挂点——覆盖 invoke 与 stream 两条路径（CLI 与 SSE 共用）。
     *
     * <p>契约：实现内部必须自转线程/池（如 {@code CompactionService} 的
     * 单守护线程执行器），<b>绝不阻塞运行线程</b>；仅 CANCELLED 终态不调用
     * （done/failed/partial 都触发——失败轮的快照同样会无界增长）；失败/
     * 触顶轮的 {@code finalState} 可能为 null（快照以 checkpointer 为准）。
     * 钩子自身的异常被吞掉记 WARN——绝不能反过来影响运行结果。
     */
    @FunctionalInterface
    public interface CompactionHook {
        void onRunTerminated(RunConfig config, GraphState finalState);
    }

    private final String name;
    private final Map<String, GraphBuilder.Node> nodes;
    private final Map<String, GraphBuilder.Edge> edges;
    private final String entry;
    private final Checkpointer checkpointer;
    /** 运行终态钩子（#328）：null = 未挂（开关关/端口缺席，零行为变化）。 */
    private final CompactionHook compactionHook;

    CompiledGraph(String name, Map<String, GraphBuilder.Node> nodes,
                  Map<String, GraphBuilder.Edge> edges, String entry, Checkpointer checkpointer,
                  io.github.paopaonb666.agentbase.core.tools.ToolPool toolPool,
                  CompactionHook compactionHook) {
        this.name = name;
        this.nodes = nodes;
        this.edges = edges;
        this.entry = entry;
        this.checkpointer = checkpointer;
        this.toolPoolRef = toolPool;
        this.compactionHook = compactionHook;
    }

    public String name() {
        return name;
    }

    public Checkpointer checkpointer() {
        return checkpointer;
    }

    /**
     * 探针：该 thread 当前是否在跑（路由层用它把并发 invoke 提前转成
     * 409，而不是等 SSE 流开始后才报错）。探针与真正加锁之间存在窗口，
     * 权威判定在 {@link #run} 的 tryLock——探针只优化报错时机。
     */
    public boolean isThreadBusy(String threadId) {
        java.util.concurrent.locks.ReentrantLock lock = THREAD_LOCKS.get(threadId);
        return lock != null && lock.isLocked();
    }

    /**
     * 批量执行到终点，返回最终状态（对应 ainvoke）。
     * checkpointer 存在时从存储加载历史并回写。
     */
    public GraphState invoke(GraphState input, RunConfig config) {
        return run(input, config, null);
    }

    /**
     * 流式执行到终点，返回最终状态（对应 astream）。sink 为 null 时与
     * invoke 等价。
     */
    public GraphState stream(GraphState input, RunConfig config, EventSink sink) {
        return run(input, config, sink);
    }

    /**
     * 流式执行并携带权威终止标注（#9）：终止原因由本层在运行结束处判定，
     * SSE 层据此渲染 done 事件的 termination 字段，不再从异常文本反推。
     *
     * <p>失败不抛出——终态以 outcome 携带：ThreadBusyError →
     * THREAD_BUSY、GraphRecursionError → RECURSION_LIMIT、经 sink 取消
     * 传播 → CANCELLED、取消态下的其余异常 → CANCELLED（断连 teardown
     * 期间的中断/关闭失败，客户端已不在，不是服务内部故障）、其余异常 →
     * INTERNAL_ERROR；原始异常随 error 携带供上层渲染文本。
     * {@link #invoke}/{@link #stream} 的既有抛出契约保持不变。
     */
    public RunOutcome streamWithTermination(GraphState input, RunConfig config, EventSink sink) {
        try {
            return runExit(input, config, sink);
        } catch (ThreadBusyError e) {
            return new RunOutcome(null, Events.RunTermination.THREAD_BUSY, e);
        } catch (GraphRecursionError e) {
            return new RunOutcome(null, Events.RunTermination.RECURSION_LIMIT, e);
        } catch (java.util.concurrent.CancellationException e) {
            return new RunOutcome(null, Events.RunTermination.CANCELLED, e);
        } catch (RuntimeException e) {
            // 节点内的 InterruptedException 会被上方节点包装点转成
            // IllegalStateException(cause=interrupt) 的 RuntimeException——
            // 取消态下（shutdown 先置位再中断）按 CANCELLED 归类，否则
            // 客户端断开会被误报成服务内部故障。
            Events.RunTermination termination = sink != null && sink.cancelled()
                    ? Events.RunTermination.CANCELLED
                    : Events.RunTermination.INTERNAL_ERROR;
            return new RunOutcome(null, termination, e);
        }
    }

    /**
     * 一次图运行的终态（#9）：统一终止标注的权威携带。state 为最终（或
     * 中断时已累积的）状态，失败终态下为 null；error 携带导致终态的原始
     * 异常（DONE 与图层 break 的 CANCELLED 为 null）。
     */
    public record RunOutcome(GraphState state, Events.RunTermination termination, Throwable error) {
    }

    private GraphState run(GraphState input, RunConfig config, EventSink sink) {
        return runExit(input, config, sink).state();
    }

    /** 运行出口（#9）：终止标注在本层判定——正常走到 END → DONE、断连 break → CANCELLED。 */
    private RunOutcome runExit(GraphState input, RunConfig config, EventSink sink) {
        // per-thread 互斥（#197）：拿不到锁说明同 thread 已有 run 在途——
        // 快速失败（last-writer-wins 的快照互踩从根上消除）。THREAD_BUSY
        // 的 run 根本没执行，不触发终态钩子。
        java.util.concurrent.locks.ReentrantLock lock =
                THREAD_LOCKS.computeIfAbsent(config.threadId(), k -> new java.util.concurrent.locks.ReentrantLock());
        if (!lock.tryLock()) {
            throw new ThreadBusyError(config.threadId());
        }
        // #7 run 生命周期注册（steering/resume 的共享地基）：per-thread 锁内
        // 注册——同线程至多一个在途 run，句柄即权威。
        RunRegistry.RunHandle handle = null;
        RunOutcome outcome = null;
        RuntimeException failure = null;
        try {
            handle = RunRegistry.register(config.threadId(), config.userId());
            outcome = runLocked(input, config, sink);
        } catch (RuntimeException e) {
            failure = e;
        } finally {
            if (handle != null) {
                RunRegistry.unregister(config.threadId(), handle);
                // #8 生命周期记录：末次终态进 RunRegistry（resume 端点的
                // 可恢复性判定依据；wire 小写形态与 #9 终态枚举一致）。
                RunRegistry.recordLastRun(config.threadId(), handle.userId(), handle.runId(),
                        terminationLabel(outcome, failure, sink));
            }
            lock.unlock();
        }
        // 压缩触发挂点（#328）：锁释放之后调用（hook 实现内部异步，绝不阻塞
        // 运行线程）。仅 CANCELLED 跳过——done/failed/partial 都触发；节点
        // 异常路径的终态按 sink 取消态归类（与 streamWithTermination 同口径）。
        // F10①（R4-a）：节点抛 CancellationException 且 sink 未取消时同样
        // 视为取消——streamWithTermination 已把它归 CANCELLED（#9 既有口径），
        // 钩子契约「仅 CANCELLED 不调用」在 failure 路径不再有缝。
        boolean cancelled = failure != null
                ? (sink != null && sink.cancelled())
                        || failure instanceof java.util.concurrent.CancellationException
                : outcome.termination() == Events.RunTermination.CANCELLED
                        || outcome.termination() == Events.RunTermination.PAUSED;
        if (!cancelled) {
            notifyCompactionHook(config, outcome == null ? null : outcome.state());
        }
        if (failure != null) {
            throw failure;
        }
        return outcome;
    }

    /** 终态钩子调用（#328）：钩子缺席零开销；钩子异常吞掉记 WARN。 */
    private void notifyCompactionHook(RunConfig config, GraphState finalState) {
        if (compactionHook == null) {
            return;
        }
        try {
            compactionHook.onRunTerminated(config, finalState);
        } catch (RuntimeException e) {
            LOG.warn("graph: 压缩终态钩子失败（不影响运行结果）：thread={}，{}",
                    config.threadId(), e.toString());
        }
    }

    /**
     * 生命周期记录的终态标注（#8）：与 {@link #streamWithTermination} 的
     * 归类同口径——failure 按 sink 取消态区分 cancelled/internal_error，
     * 递归触顶记 recursion_limit，正常出口用终态枚举的 wire 小写形态。
     */
    private static String terminationLabel(RunOutcome outcome, RuntimeException failure,
                                           EventSink sink) {
        if (failure != null) {
            if ((sink != null && sink.cancelled())
                    || failure instanceof java.util.concurrent.CancellationException) {
                return Events.RunTermination.CANCELLED.name()
                        .toLowerCase(java.util.Locale.ROOT);
            }
            if (failure instanceof GraphRecursionError) {
                return Events.RunTermination.RECURSION_LIMIT.name()
                        .toLowerCase(java.util.Locale.ROOT);
            }
            return Events.RunTermination.INTERNAL_ERROR.name()
                    .toLowerCase(java.util.Locale.ROOT);
        }
        return outcome == null
                ? Events.RunTermination.INTERNAL_ERROR.name().toLowerCase(java.util.Locale.ROOT)
                : outcome.termination().name().toLowerCase(java.util.Locale.ROOT);
    }

    /**
     * #14 system-reminder 注入前缀：运行中注入的 system 形态消息的标识
     * 前缀——与 #328 的 {@code COMPACTION_SUMMARY_PREFIX}（会话历史摘要）
     * 刻意不同字面量：压缩转写折叠只认摘要前缀，注入提醒不会被当成摘要
     * 折叠（两者语义不冲突，注册在案）。
     */
    public static final String SYSTEM_REMINDER_PREFIX = "【系统提醒（运行中注入）】\n";

    /**
     * #7/#14 steering 步边界注入：把待消费的用户插话（{@link RunRegistry}
     * 句柄的队列）注入状态（messages append-only，A-8 契约）并落盘，随后
     * 发一条 {@code steer} step 事件（detail 标注入条数与形态——wire 形态
     * additive：复用既有 StepEvent，无新事件类型）。kind=user → UserMessage
     * 原样；kind=system → {@link #SYSTEM_REMINDER_PREFIX} 包裹的
     * SystemMessage。队列空=零开销直返（事件流逐字节不变）。
     */
    private void drainSteering(RunConfig config, GraphState state, EventSink sink) {
        RunRegistry.RunHandle handle = RunRegistry.lookup(config.threadId());
        if (handle == null || handle.pendingSteers() == 0) {
            return;
        }
        List<RunRegistry.SteerItem> injected = handle.drain();
        boolean hasSystem = false;
        for (RunRegistry.SteerItem item : injected) {
            if ("system".equals(item.kind())) {
                hasSystem = true;
                state.apply(Map.of(GraphState.MESSAGES, List.of(
                        dev.langchain4j.data.message.SystemMessage.from(
                                SYSTEM_REMINDER_PREFIX + item.content()))));
            } else {
                state.apply(Map.of(GraphState.MESSAGES, List.of(
                        new dev.langchain4j.data.message.UserMessage(item.content()))));
            }
        }
        persistState(config, state);
        if (sink != null) {
            sink.accept(new Events.StepEvent("steer", "completed",
                    "injected_messages=" + injected.size()
                            + (hasSystem ? ";injected_kind=system" : "")));
        }
        LOG.debug("steering: 注入 {} 条插话（thread={}）", injected.size(), config.threadId());
    }

    private RunOutcome runLocked(GraphState input, RunConfig config, EventSink sink) {
        GraphState state = loadState(config);
        state.apply(input.updatePayload());
        // A-#26/#8：入口覆盖（审批暂停续跑指向 resume_node——如 tools 节点
        // ，批准后从它继续执行而不重跑 chat）。未知节点由运行期兜底报错。
        String current = config.entryOverride() != null ? config.entryOverride() : entry;
        int steps = 0;
        while (!GraphBuilder.END.equals(current)) {
            if (sink != null && sink.cancelled()) {
                // 循环顶取消（审查 A-P1 步间窗）：上一步的完成态可能已带着
                // AiMessage(tool_calls) 落盘（路由指向 tools 节点、工具结果
                // 还没配对）——快照悬挂调用会让该线程下一轮起每轮被 provider
                // 400 拒收。与节点中途取消分支同款善后：剥除悬挂调用 +
                // cancelled 标记 + 重落盘（残余窗口收窄为「上一步 persist
                // 之后到本检查之前」的进程死亡，与一切落盘窗口同级）。
                stripDanglingToolCalls(state);
                state.cancelledRound(true);
                persistState(config, state);
                LOG.debug("图执行被取消（客户端断开）：thread={}，上步悬挂调用已剥除并带标记落盘",
                        config.threadId());
                return new RunOutcome(state, Events.RunTermination.CANCELLED, null);
            }
            // #7 steering 步边界（取消检查之后——取消竞态窗口的插话随取消
            // 丢弃，注入矩阵窗口 c）。
            drainSteering(config, state, sink);
            if (steps >= config.recursionLimit()) {
                throw new GraphRecursionError(recursionMessage(config.recursionLimit()));
            }
            GraphBuilder.Node node = nodes.get(current);
            if (node == null) {
                // #202：运行期兜底（未声明目标集的条件边仍可能到这里）——
                // 错误带节点名 + 可用节点列表，可直接定位路由笔误。
                throw new IllegalStateException("未知节点 '%s'（可用：%s）"
                        .formatted(current, String.join(", ", nodes.keySet())));
            }
            if (sink != null) {
                sink.accept(Events.StepEvent.running(current));
            }
            Map<String, Object> update;
            try {
                update = executeNode(node, current, state, config, sink);
            } catch (PauseForApprovalException pause) {
                // A-#26 审批暂停：状态带 pending_approval 标记落盘——悬挂
                // tool_calls 有意保留（剥除守卫跳过），终态 PAUSED（wire
                // "paused"，additive 值）；审批端点按 call_id 决策后续跑。
                state.apply(Map.of("pending_approval", Map.of(
                        "resume_node", pause.resumeNode(),
                        "detail", pause.detail())));
                persistState(config, state);
                if (sink != null) {
                    sink.accept(new Events.StepEvent("approval", "running", pause.detail()));
                }
                LOG.info("graph: 工具审批暂停（thread={}，{}）", config.threadId(), pause.detail());
                return new RunOutcome(state, Events.RunTermination.PAUSED, null);
            }
            if (sink != null && sink.cancelled()) {
                // 节点已在途执行完（无法回滚）：半步产出照常进状态并落盘
                // （#199 翻转基线——取消的轮次不再从历史里凭空消失），快照
                // 带 cancelled 标记；对应 Python asyncio 取消语义的显式偏离
                // （中断的 superstep 不进 checkpoint），登记为 Java 单边增强。
                // 不再 emit StepEvent.completed：断连 sink 的入队在取消态是
                // 有界放弃（R1-F1），事件也只会被消费端丢弃。
                state.apply(update);
                stripDanglingToolCalls(state);
                state.cancelledRound(true);
                persistState(config, state);
                LOG.debug("图执行被取消（客户端断开）：thread={}，半步 {} 已带标记落盘",
                        config.threadId(), current);
                return new RunOutcome(state, Events.RunTermination.CANCELLED, null);
            }
            state.apply(update);
            // A-#26：续跑的审批节点完成——待审批调用已全部消费（执行或回喂
            // 拒绝文本），pending_approval 标记退场（此后取消路径照常剥除）。
            // A-#62 marker 型暂停：resume_node 之外节点产出的标记（如 plan
            // 节点产 tasks）保留并在下方触发暂停；resume_node 自身完成则清。
            if (state.extra().get("pending_approval") instanceof Map<?, ?> pending
                    && current.equals(String.valueOf(pending.get("resume_node")))) {
                state.extra().remove("pending_approval");
            } else if (state.extra().containsKey("pending_approval")) {
                // A-#62：计划产出后暂停（tasks 已随 update 进状态并落盘），
                // 等审批端点决定后以 entryOverride=resume_node 续跑。
                persistState(config, state);
                if (sink != null) {
                    sink.accept(new Events.StepEvent("approval", "running",
                            String.valueOf(((Map<?, ?>) state.extra().get("pending_approval"))
                                    .get("detail"))));
                }
                LOG.info("graph: 审批暂停（thread={}，等待 {} 批准）", config.threadId(),
                        state.extra().get("pending_approval"));
                return new RunOutcome(state, Events.RunTermination.PAUSED, null);
            }
            if (sink != null) {
                sink.accept(Events.StepEvent.completed(current));
            }
            persistState(config, state);
            steps++;
            current = nextOf(current, state);
        }
        return new RunOutcome(state, Events.RunTermination.DONE, null);
    }

    private Map<String, Object> executeNode(GraphBuilder.Node node, String nodeName,
                                            GraphState state,
                                            RunConfig config, EventSink sink) {
        NodeContext ctx = new NodeContext() {
            @Override
            public RunConfig config() {
                return config;
            }

            @Override
            public void emit(Object event) {
                if (sink != null && !sink.cancelled()) {
                    sink.accept(event);
                } else if (sink != null) {
                    // 断连后的发射是面板缺口（#210）：计入丢弃，可诊断。
                    io.github.paopaonb666.agentbase.extensions.Metrics.events()
                            .observeDrop("cancelled");
                }
            }

            @Override
            public String callTool(String name, String argsJson) {
                // 工具执行通道由装配方注入（AgentRuntime 组合根设置）；
                // 无通道时返回错误文本（与 handle_tool_error 的归一语义一致）。
                io.github.paopaonb666.agentbase.core.tools.ToolPool pool = toolPool();
                if (pool == null) {
                    return "工具不可用：工具池未装配";
                }
                return pool.call(name, argsJson, config, this);
            }

            @Override
            public io.github.paopaonb666.agentbase.core.tools.ToolPool tools() {
                return toolPool();
            }

            @Override
            public boolean cancelled() {
                return sink != null && sink.cancelled();
            }
        };
        try {
            Map<String, Object> update = node.apply(state, ctx);
            return update == null ? Map.of() : update;
        } catch (GraphRecursionError e) {
            throw e;
        } catch (Exception e) {
            // 节点内部异常归一抛出（SSE 层转 ErrorEvent；URL 脱敏在出口做）。
            // A-#239：错误文本附节点名——「哪个节点炸了」从日志/事件文本可
            // 直接定位（既有断言面无此文案，零断言同步）。
            if (e instanceof RuntimeException re) {
                throw re;
            }
            throw new IllegalStateException(
                    "节点执行失败[node=" + nodeName + "]: " + e.getMessage(), e);
        }
    }

    // 工具池在 compile 期定死（#238：post-compile 的 volatile 覆写是竞态面
    // 也是装配顺序陷阱）；null = 本图未接工具。
    private final io.github.paopaonb666.agentbase.core.tools.ToolPool toolPoolRef;

    public io.github.paopaonb666.agentbase.core.tools.ToolPool toolPool() {
        return toolPoolRef;
    }

    private String nextOf(String current, GraphState state) {
        GraphBuilder.Edge edge = edges.get(current);
        if (edge == null) {
            return GraphBuilder.END;
        }
        if (edge.router() != null) {
            String next = edge.router().apply(state);
            return next == null ? GraphBuilder.END : next;
        }
        return edge.target();
    }

    /** 读取线程状态快照（对应 aget_state）；checkpointer 缺席返回 null。 */
    public GraphState getState(RunConfig config) {
        return checkpointer == null ? null : checkpointer.get(config.threadId());
    }

    /** 删除线程（checkpointer 缺席为无操作）。 */
    public void deleteThread(RunConfig config) {
        if (checkpointer != null) {
            checkpointer.deleteThread(config.threadId());
        }
    }

    private GraphState loadState(RunConfig config) {
        if (checkpointer != null) {
            GraphState persisted = checkpointer.get(config.threadId());
            if (persisted != null) {
                // #199：cancelled 标记是快照行元数据不是状态通道——新 round
                // 加载时复位（首个正常 persistState 整行覆盖后标记消失）。
                persisted.cancelledRound(false);
                return persisted;
            }
        }
        return new GraphState();
    }

    /**
     * 半截轮的悬挂 tool_calls 剥除（#199 兜底）：取消落在「chat 节点已产出
     * 工具调用请求、tools 节点未执行」的窗口时，快照若保留未配对的
     * tool_calls，下一轮模型输入以「assistant(tool_calls) 无配对 tool 结果」
     * 结尾——OpenAI 兼容 provider 普遍拒收（400）。剥掉后以纯文本
     * assistant 消息续接（协议要求 tool 消息配对；调用明细在
     * tool_call_records 审计不受影响）。
     */
    private static void stripDanglingToolCalls(GraphState state) {
        // A-#26：审批暂停快照的悬挂 tool_calls 是**有意保留**（等批准后续跑
        // 执行）——pending_approval 标记在场时剥除守卫跳过。
        if (state.extra().containsKey("pending_approval")) {
            return;
        }
        List<ChatMessage> messages = state.messages();
        if (messages.isEmpty()) {
            return;
        }
        ChatMessage last = messages.get(messages.size() - 1);
        if (last instanceof AiMessage ai && ai.hasToolExecutionRequests()) {
            messages.set(messages.size() - 1, AiMessage.builder()
                    .text(ai.text())
                    .thinking(ai.thinking())
                    .attributes(ai.attributes() == null ? Map.of() : ai.attributes())
                    .build());
        }
    }

    private void persistState(RunConfig config, GraphState state) {
        persistSnapshot(checkpointer, config.threadId(), state);
    }

    /**
     * 快照落盘的共享路径（#330 从 persistState 提取）：run 每步与压缩写回
     * 同一套 put+重试欠账语义（序列化在各后端 put 内部）。put 失败重试
     * 1 次（#198），仍失败记欠账——调用方内存续跑，绝不因快照欠账报废
     * 已产出内容。
     */
    private static void persistSnapshot(Checkpointer checkpointer, String threadId, GraphState state) {
        if (checkpointer == null) {
            return;
        }
        try {
            checkpointer.put(threadId, state.copy());
            recoverPersistDebt();
            return;
        } catch (RuntimeException first) {
            // put 失败重试 1 次（#198）：连接抖动/锁竞争多为一过性。
            try {
                checkpointer.put(threadId, state.copy());
                recoverPersistDebt();
                LOG.warn("graph: 快照落盘重试成功：thread={}", threadId);
                return;
            } catch (RuntimeException second) {
                second.addSuppressed(first);
                recordPersistDebt(second);
            }
        }
        // 内存续跑：run 循环里的 state 本就是内存对象，这里只跳过持久化
        // ——绝不因快照欠账把整轮已产出的 token 报废（#198）。
    }

    /**
     * 压缩写回互斥入口（#330 方案 B）：tryLock {@link #THREAD_LOCKS}——
     * 失败快速返回 false（与 run 同语义：不排队，忙 = 放弃本轮写回）；
     * 持锁内 re-read checkpointer 最新状态交给 {@code rewriter}——返回
     * empty = 基准不一致/放弃（不产生半写），有值 = 持久化（复用 run 的
     * put+重试欠账路径）后返回 true。
     *
     * <p>「进程内所有写者被同一把锁串行后重读校验实际恒过，是防御性双保险」
     * （#330 方案 B 原文）：单进程里 run 与压缩写回被 {@link #THREAD_LOCKS}
     * 串行，重读校验兜底的是锁外的第三类写者（如运维脚本直写 checkpointer）。
     *
     * <p>{@code rewrite} 在锁内执行——必须快（纯内存变换），绝不做 IO/LLM
     * 调用；摘要已在 {@link CompactionService} 的任务体里完成。
     */
    public boolean compactThread(String threadId,
                                 java.util.function.Function<GraphState, java.util.Optional<GraphState>> rewrite) {
        return compactThread(checkpointer, threadId, rewrite);
    }

    /**
     * 静态形态（#330）：{@code CompactionService} 装配面只有 checkpointer
     * 没有图实例——THREAD_LOCKS 是进程级静态锁，同 threadId 与所有图的
     * run/写回天然互斥。图实例方法委托到这里。
     */
    public static boolean compactThread(Checkpointer checkpointer, String threadId,
                                        java.util.function.Function<GraphState, java.util.Optional<GraphState>> rewrite) {
        if (checkpointer == null) {
            return false;
        }
        java.util.concurrent.locks.ReentrantLock lock =
                THREAD_LOCKS.computeIfAbsent(threadId, k -> new java.util.concurrent.locks.ReentrantLock());
        if (!lock.tryLock()) {
            return false;
        }
        try {
            GraphState fresh = checkpointer.get(threadId);
            if (fresh == null) {
                fresh = new GraphState();
            }
            java.util.Optional<GraphState> rewritten = rewrite.apply(fresh);
            if (rewritten.isEmpty()) {
                // 基准不一致/放弃：不动快照（不产生半写）。
                return false;
            }
            persistSnapshot(checkpointer, threadId, rewritten.get());
            return true;
        } finally {
            lock.unlock();
        }
    }

    /** 快照欠账（#198）：进程级计数——/health 据此报告快照积压。 */
    private static final java.util.concurrent.atomic.AtomicLong PERSIST_DEBT_STEPS =
            new java.util.concurrent.atomic.AtomicLong();
    private static final java.util.concurrent.atomic.AtomicBoolean PERSIST_IN_DEBT =
            new java.util.concurrent.atomic.AtomicBoolean();

    private static void recordPersistDebt(RuntimeException cause) {
        PERSIST_IN_DEBT.set(true);
        PERSIST_DEBT_STEPS.incrementAndGet();
        LOG.error("graph: 快照落盘失败（重试后仍失败）——本轮内存续跑，快照欠账累计 {} 步: {}",
                PERSIST_DEBT_STEPS.get(), cause.toString());
    }

    private static void recoverPersistDebt() {
        if (PERSIST_IN_DEBT.compareAndSet(true, false)) {
            LOG.warn("graph: 快照落盘恢复（欠账期间被跳过的历史步不会补写）");
        }
    }

    /** /health 组件（#198）：true = 存在未持久化的快照步（内存续跑中）。 */
    public static boolean persistInDebt() {
        return PERSIST_IN_DEBT.get();
    }

    /** /health 组件（#198）：启动以来被跳过的快照步累计数。 */
    public static long persistDebtSteps() {
        return PERSIST_DEBT_STEPS.get();
    }

    /** 图步数超限（对应 LangGraph GraphRecursionError；SSE 层转 ErrorEvent）。 */
    public static final class GraphRecursionError extends RuntimeException {
        public GraphRecursionError(String message) {
            super(message);
        }
    }

    /**
     * 触顶错误带可定位信息（#229）：图名/节点数，planner 图附 H5 最坏
     * 步数公式与建议值——报错直接给出「该调哪个配置」而不是让用户翻
     * 文档反推。
     */
    private String recursionMessage(int limit) {
        String advice = "planner".equals(name)
                ? "；planner 最坏步数 ≈ 8 + 2*PLANNER_MAX_SUBTASKS*(PLANNER_SUBTASK_TOOL_ROUNDS+1)"
                        + "——建议 AGENT_RECURSION_LIMIT ≥ 该值，或调低 PLANNER_* 预算"
                : "";
        return "Recursion limit of %d reached on graph '%s' (%d nodes)%s"
                .formatted(limit, name, nodes.size(), advice);
    }
}
