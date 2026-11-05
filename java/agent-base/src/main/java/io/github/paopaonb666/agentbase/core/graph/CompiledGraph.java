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
                throw new GraphRecursionError(
                        "Recursion limit of %d reached".formatted(config.recursionLimit()));
            }
            GraphBuilder.Node node = nodes.get(current);
            if (node == null) {
                throw new IllegalStateException("未知节点 '%s'".formatted(current));
            }
            if (sink != null) {
                sink.accept(io.github.paopaonb666.agentbase.extensions.events.Events.StepEvent.running(current));
            }
            Map<String, Object> update = executeNode(node, state, config, sink);
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
        if (checkpointer != null) {
            checkpointer.put(config.threadId(), state.copy());
        }
    }

    /** 图步数超限（对应 LangGraph GraphRecursionError；SSE 层转 ErrorEvent）。 */
    public static final class GraphRecursionError extends RuntimeException {
        public GraphRecursionError(String message) {
            super(message);
        }
    }
}
