package io.github.paopaonb666.agentbase.core.graph;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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

    private final String name;
    private final Map<String, GraphBuilder.Node> nodes;
    private final Map<String, GraphBuilder.Edge> edges;
    private final String entry;
    private final Checkpointer checkpointer;

    CompiledGraph(String name, Map<String, GraphBuilder.Node> nodes,
                  Map<String, GraphBuilder.Edge> edges, String entry, Checkpointer checkpointer) {
        this.name = name;
        this.nodes = nodes;
        this.edges = edges;
        this.entry = entry;
        this.checkpointer = checkpointer;
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

    private GraphState run(GraphState input, RunConfig config, EventSink sink) {
        // per-thread 互斥（#197）：拿不到锁说明同 thread 已有 run 在途——
        // 快速失败（last-writer-wins 的快照互踩从根上消除）。
        java.util.concurrent.locks.ReentrantLock lock =
                THREAD_LOCKS.computeIfAbsent(config.threadId(), k -> new java.util.concurrent.locks.ReentrantLock());
        if (!lock.tryLock()) {
            throw new ThreadBusyError(config.threadId());
        }
        try {
            return runLocked(input, config, sink);
        } finally {
            lock.unlock();
        }
    }

    private GraphState runLocked(GraphState input, RunConfig config, EventSink sink) {
        GraphState state = loadState(config);
        state.apply(input.updatePayload());
        String current = entry;
        int steps = 0;
        while (!GraphBuilder.END.equals(current)) {
            if (sink != null && sink.cancelled()) {
                LOG.debug("图执行被取消（客户端断开）：thread={}", config.threadId());
                break;
            }
            if (steps >= config.recursionLimit()) {
                throw new GraphRecursionError(recursionMessage(config.recursionLimit()));
            }
            GraphBuilder.Node node = nodes.get(current);
            if (node == null) {
                throw new IllegalStateException("未知节点 '%s'".formatted(current));
            }
            if (sink != null) {
                sink.accept(io.github.paopaonb666.agentbase.extensions.events.Events.StepEvent.running(current));
            }
            Map<String, Object> update = executeNode(node, state, config, sink);
            if (sink != null && sink.cancelled()) {
                // 节点已在途执行完（无法回滚），但取消后不再把这一步落盘：
                // 对应 Python asyncio 任务在节点 await 点被取消——中断的
                // superstep 不会进 checkpoint，恢复后看不到这半步。
                LOG.debug("图执行被取消（客户端断开）：thread={}，丢弃步骤 {} 的快照",
                        config.threadId(), current);
                break;
            }
            state.apply(update);
            if (sink != null) {
                sink.accept(io.github.paopaonb666.agentbase.extensions.events.Events.StepEvent.completed(current));
            }
            persistState(config, state);
            steps++;
            current = nextOf(current, state);
        }
        return state;
    }

    private Map<String, Object> executeNode(GraphBuilder.Node node, GraphState state,
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
            if (e instanceof RuntimeException re) {
                throw re;
            }
            throw new IllegalStateException("节点执行失败: " + e.getMessage(), e);
        }
    }

    private volatile io.github.paopaonb666.agentbase.core.tools.ToolPool toolPoolRef;

    /** 装配方（AgentRuntime）注入工具池；null = 本图未接工具。 */
    public void setToolPool(io.github.paopaonb666.agentbase.core.tools.ToolPool pool) {
        this.toolPoolRef = pool;
    }

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
                return persisted;
            }
        }
        return new GraphState();
    }

    private void persistState(RunConfig config, GraphState state) {
        if (checkpointer == null) {
            return;
        }
        try {
            checkpointer.put(config.threadId(), state.copy());
            recoverPersistDebt();
            return;
        } catch (RuntimeException first) {
            // put 失败重试 1 次（#198）：连接抖动/锁竞争多为一过性。
            try {
                checkpointer.put(config.threadId(), state.copy());
                recoverPersistDebt();
                LOG.warn("graph: 快照落盘重试成功：thread={}", config.threadId());
                return;
            } catch (RuntimeException second) {
                second.addSuppressed(first);
                recordPersistDebt(second);
            }
        }
        // 内存续跑：run 循环里的 state 本就是内存对象，这里只跳过持久化
        // ——绝不因快照欠账把整轮已产出的 token 报废（#198）。
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
