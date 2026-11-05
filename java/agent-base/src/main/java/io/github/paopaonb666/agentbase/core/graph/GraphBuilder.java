package io.github.paopaonb666.agentbase.core.graph;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.Function;

/**
 * 图构建器（对应 LangGraph StateGraph 的受控子集）。
 *
 * <pre>{@code
 * CompiledGraph graph = GraphBuilder.begin()
 *     .addNode("agent", (state, ctx) -> Map.of("messages", List.of(aiMessage)))
 *     .addNode("tools", toolNode)
 *     .addEdge(START, "agent")
 *     .addConditionalEdges("agent", state -> hasToolCalls ? "tools" : END)
 *     .addEdge("tools", "agent")
 *     .compile(checkpointer, "chat");
 * }</pre>
 *
 * <p>边语义：无条件边固定后继；条件边由路由函数在节点执行后选择后继
 * （可返回 {@link #END} 终止）。单入口（START），节点后继唯一。
 */
public final class GraphBuilder {

    /** 入口哨兵（对应 LangGraph START）。 */
    public static final String START = "__start__";
    /** 终点哨兵（对应 LangGraph END）。 */
    public static final String END = "__end__";

    /** 节点函数：输入当前状态，返回局部更新（messages 追加、其余替换）。 */
    @FunctionalInterface
    public interface Node {
        Map<String, Object> apply(GraphState state, NodeContext ctx) throws Exception;
    }

    /** 条件路由函数：返回下一个节点名或 {@link #END}。 */
    @FunctionalInterface
    public interface Router {
        String apply(GraphState state);
    }

    record Edge(String target, Router router) {}

    final Map<String, Node> nodes = new LinkedHashMap<>();
    final Map<String, Edge> edges = new LinkedHashMap<>();
    String entry;

    private GraphBuilder() {}

    public static GraphBuilder begin() {
        return new GraphBuilder();
    }

    /** 注册节点（重名抛异常——与 LangGraph 一致的快速失败）。 */
    public GraphBuilder addNode(String name, Node node) {
        if (nodes.containsKey(name)) {
            throw new IllegalArgumentException("重复节点 '%s'".formatted(name));
        }
        nodes.put(name, node);
        return this;
    }

    /** 注册无条件边（重复出边抛异常）。 */
    public GraphBuilder addEdge(String from, String to) {
        return addEdge(from, new Edge(to, null));
    }

    /** 注册条件边（对应 add_conditional_edges）。 */
    public GraphBuilder addConditionalEdges(String from, Router router) {
        return addEdge(from, new Edge(null, router));
    }

    private GraphBuilder addEdge(String from, Edge edge) {
        if (edges.containsKey(from)) {
            throw new IllegalArgumentException("节点 '%s' 已有出边".formatted(from));
        }
        edges.put(from, edge);
        return this;
    }

    /** 设置入口边（对应 add_edge(START, name)）。 */
    public GraphBuilder entry(String node) {
        this.entry = node;
        return this;
    }

    /** 编译：校验拓扑完整性。checkpointer 可为 null（测试手动构造）。 */
    public CompiledGraph compile(Checkpointer checkpointer, String name) {
        if (entry == null || START.equals(entry) || END.equals(entry)) {
            throw new IllegalStateException("必须设置入口节点（builder.entry(...)）");
        }
        if (!nodes.containsKey(entry)) {
            throw new IllegalStateException("入口节点 '%s' 未注册".formatted(entry));
        }
        for (Map.Entry<String, Edge> e : edges.entrySet()) {
            Edge edge = e.getValue();
            if (edge.target() != null && !END.equals(edge.target()) && !nodes.containsKey(edge.target())) {
                throw new IllegalStateException("边 '%s' 的后继 '%s' 未注册".formatted(e.getKey(), edge.target()));
            }
        }
        // 所有节点都必须有出边（终点由显式路由到 END 表达）。
        for (String node : nodes.keySet()) {
            if (!edges.containsKey(node)) {
                throw new IllegalStateException("节点 '%s' 缺少出边".formatted(node));
            }
        }
        return new CompiledGraph(name, Map.copyOf(nodes), Map.copyOf(edges), entry, checkpointer);
    }

    /** 便捷重载：单节点直线图（START → node → END）。 */
    public static CompiledGraph singleNode(String name, Node node, Checkpointer checkpointer) {
        return GraphBuilder.begin()
                .addNode(name, node)
                .entry(name)
                .addEdge(name, END)
                .compile(checkpointer, name);
    }

    /** 工具函数：把状态经由 Function 包装成无状态节点（用于复用小函数）。 */
    public static Node nodeOf(Function<GraphState, Map<String, Object>> fn) {
        return (state, ctx) -> fn.apply(state);
    }
}
