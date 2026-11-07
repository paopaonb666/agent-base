package io.github.paopaonb666.agentbase.core.graph;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
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

    record Edge(String target, Router router, Set<String> possibleTargets) {
        /** 旧两参形态（possibleTargets=null = 未声明目标集，不做编译期校验）。 */
        Edge(String target, Router router) {
            this(target, router, null);
        }
    }

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

    /**
     * 注册条件边并声明路由的全部可能目标（#202）：router 返回值应为封闭
     * 集合时用它——compile 校验目标 ⊆ 已注册节点 + {@link #END}，笔误在
     * 编译期暴露而不是首次命中分支才炸。不声明目标集的旧重载保留运行期
     * 兜底语义。
     */
    public GraphBuilder addConditionalEdges(String from, Router router, Set<String> possibleTargets) {
        return addEdge(from, new Edge(null, router, Set.copyOf(possibleTargets)));
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
        return compile(checkpointer, name, null);
    }

    /**
     * 编译并绑定工具池（#238：池是编译参数，不是编译后覆写）；
     * {@code pool} 为 null = 本图未接工具。
     */
    public CompiledGraph compile(Checkpointer checkpointer, String name,
                                 io.github.paopaonb666.agentbase.core.tools.ToolPool toolPool) {
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
            // #202：条件边声明了目标集时编译期校验（⊆ 节点 + END）。
            if (edge.possibleTargets() != null) {
                for (String target : edge.possibleTargets()) {
                    if (!END.equals(target) && !nodes.containsKey(target)) {
                        throw new IllegalStateException("图 '%s' 条件边 '%s' 声明的目标 '%s' 未注册（可用：%s）"
                                .formatted(name, e.getKey(), target, declaredTargetsHelp()));
                    }
                }
            }
        }
        // 所有节点都必须有出边（终点由显式路由到 END 表达）。
        for (String node : nodes.keySet()) {
            if (!edges.containsKey(node)) {
                throw new IllegalStateException("节点 '%s' 缺少出边".formatted(node));
            }
        }
        return new CompiledGraph(name, Map.copyOf(nodes), Map.copyOf(edges), entry, checkpointer, toolPool);
    }

    /** 可用目标提示（#202 错误文案）：已注册节点 + END。 */
    private String declaredTargetsHelp() {
        return String.join(", ", nodes.keySet()) + ", " + END;
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
