package io.github.paopaonb666.agentbase.extensions.collab;

import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import io.github.paopaonb666.agentbase.core.AgentModule;
import io.github.paopaonb666.agentbase.core.context.ContextEngine;
import io.github.paopaonb666.agentbase.core.graph.CompiledGraph;
import io.github.paopaonb666.agentbase.core.graph.EventSink;
import io.github.paopaonb666.agentbase.core.graph.GraphBuilder;
import io.github.paopaonb666.agentbase.core.graph.GraphState;
import io.github.paopaonb666.agentbase.core.graph.NodeContext;
import io.github.paopaonb666.agentbase.core.graph.RunConfig;
import io.github.paopaonb666.agentbase.core.prompt.PromptComposer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * 多 Agent 协作：supervisor 模板（阶段 4）——对应 Python 侧
 * {@code extensions/collab.py}。
 *
 * <p>Python 侧包装 LangGraph 原生的 {@code create_supervisor}
 * （langgraph-supervisor）：每个已注册的模块都成为一个 sub-agent，由一个
 * supervisor 节点读取每个模块的 {@code name} + {@code description}（与用于
 * 发现的契约字段相同）来路由。Java 侧按 langgraph-supervisor 的实际行为
 * 自实现同一模板（handoff 模式），基座自身不添加任何编排逻辑——这正是
 * "薄基座"原则的全部意义所在。
 *
 * <p>与 langgraph-supervisor 的行为对齐（output_mode="last_message" 默认、
 * {@code create_handoff_tool} 语义）：
 * <ul>
 *   <li>handoff 工具名 {@code transfer_to_<module>}（库默认前缀
 *       {@code transfer_to_} + 模块名）；description = 模块 description
 *       （Python 侧 collab.py 显式传 {@code description=module.description}，
 *       覆盖库默认的 "Ask agent 'X' for help"）；参数 schema 为空对象。</li>
 *   <li>supervisor 节点 = LLM + handoff 工具绑定 + {@link #DEFAULT_PROMPT}
 *       作为开头追加的 SystemMessage（库的 {@code prompt: str} 语义）。</li>
 *   <li>最后一条 AiMessage 带 tool_calls → 执行 handoff；否则 END（对应库的
 *       react agent 的 tools_condition）。</li>
 *   <li>handoff 执行：把当前 messages 交给目标模块自己的编译图（每个模块图
 *       只 build 一次），子图最终 messages 的最后一条 AiMessage 文本作为
 *       工具结果回填——对应库的 output_mode="last_message"（只把子 agent
 *       的最后一条消息并入历史）。</li>
 *   <li>无匹配模块 → 工具结果文本说明未知目标（模型可自愈重路由）。</li>
 * </ul>
 *
 * <p>线程命名空间：Java 侧 checkpointer 以平铺字符串为键，子图若复用
 * supervisor 的 threadId 会与父图快照互相覆盖。langgraph 用子图 checkpoint
 * 命名空间隔离，这里的等价物是派生 threadId {@code <supThreadId>@<module>}
 * （仅作存储键，不经 ThreadId 协议解析）。
 */
public final class SupervisorGraphs {

    /** 图名（compile(name=...)）；Python 侧 create_supervisor 默认 supervisor_name。 */
    public static final String MODULE_NAME = "supervisor";

    /** handoff 工具名前缀（langgraph-supervisor 默认，未传 prefix 时的形态）。 */
    public static final String HANDOFF_TOOL_PREFIX = "transfer_to_";

    // ── handoff 预算（#68）：run 级计数器（init 节点每轮清零） ──────
    /** 状态通道：本轮已执行的 handoff 次数。 */
    static final String HANDOFF_COUNT = "handoff_count";
    /** 状态通道：最近一次 handoff 的目标模块（乒乓检测留观，登记不强制）。 */
    static final String LAST_HANDOFF = "last_handoff";

    /** 预算耗尽时回给 supervisor 的结构化收敛提示（超限后不再执行子图）。 */
    static String handoffBudgetExhaustedText(int count, int limit) {
        return ("handoff 预算耗尽（本轮已 %d 次，上限 %d）：请基于已有信息直接给出最终回答，"
                + "不要再调用 transfer_to_* 工具。").formatted(count, limit);
    }

    /**
     * supervisor 的默认系统提示词（Python 侧 DEFAULT_PROMPT 逐字复制）。
     */
    public static final String DEFAULT_PROMPT =
            "You are a team supervisor coordinating specialist agents. "
            + "Break the user request into steps and dispatch them to suitable agents "
            + "in sequence using the handoff tools; call as many agents as the task "
            + "requires. Relay intermediate results between agents when one step "
            + "depends on another, then synthesize a final answer. "
            + "The supervisor itself has no memory tools: if the user asks to recall, "
            + "save or update long-term memory, hand off to the conversational agent "
            + "(e.g. chat) that owns the memory_search/memory_save tools, then relay "
            + "the tool results verbatim.";

    private SupervisorGraphs() {}

    /** 把每个已注册的模块编排为由 supervisor 主导的 sub-agent。 */
    public static CompiledGraph buildSupervisorGraph(AgentModule.ModuleContext ctx,
                                                     Map<String, AgentModule> modules) {
        if (modules == null || modules.isEmpty()) {
            throw new IllegalArgumentException("supervisor requires at least one module in AGENT_MODULES");
        }

        // 每个 sub-agent 都是该模块自己的编译图；名字来自模块（并且与它们
        // 的 AGENT_MODULES 条目一致）。只 build 一次并缓存复用——handoff
        // 执行时不重新 build（Python: agents = [module.build_graph(ctx)]）。
        Map<String, CompiledGraph> subgraphs = new LinkedHashMap<>();
        List<ToolSpecification> handoffSpecs = new ArrayList<>();
        for (AgentModule module : modules.values()) {
            subgraphs.put(module.name(), module.buildGraph(ctx));
            handoffSpecs.add(handoffSpec(module));
        }
        List<ToolSpecification> boundSpecs = List.copyOf(handoffSpecs);
        // 条目 15 / R2-F1 收口：SYSTEM 域扣减——窗口生效时 DEFAULT_PROMPT 计入
        // 预算（window=0 原样，逐字节兼容）；窗口=0 的旧口径不变。
        int maxTokens = ContextEngine.budgetAfterSystem(ctx.settings(), ctx.llm(), DEFAULT_PROMPT);

        return GraphBuilder.begin()
                // #68：run 级 handoff 预算的清零点——每轮 invoke 从 entry
                // 进入，计数器随快照持久化但每轮重置为 0。
                .addNode("init", (GraphState state, NodeContext nodeCtx) -> Map.<String, Object>of(
                        HANDOFF_COUNT, 0,
                        LAST_HANDOFF, ""))
                .addNode("supervisor", (GraphState state, NodeContext nodeCtx) -> {
                    // prompt 作为 SystemMessage 加在消息列表开头（库的
                    // prompt: str 语义）；上下文工程对所有模块生效。
                    // 条目 13：收口 collab.supervisor 层（逐字节等价；层
                    // id/版本/token 只进日志）。
                    List<ChatMessage> input = PromptComposer.assemble(
                            PromptComposer.Layer.collabSupervisor(DEFAULT_PROMPT), MODULE_NAME,
                            ContextEngine.buildModelInput(state.messages(), maxTokens, ctx.llm().tokenFactor()));
                    AiMessage message = ctx.llm().stream(input, boundSpecs, nodeCtx::emitDelta);
                    if (message == null) {
                        return Map.of();
                    }
                    return Map.<String, Object>of(GraphState.MESSAGES, List.of(message));
                })
                .entry("init")
                .addEdge("init", "supervisor")
                // 最后一条 AiMessage 带 tool_calls → handoffs；否则 END。
                // #202：封闭目标集显式声明，compile 校验笔误。
                .addConditionalEdges("supervisor",
                        state -> lastAiHasToolCalls(state) ? "handoffs" : GraphBuilder.END,
                        Set.of("handoffs", GraphBuilder.END))
                .addNode("handoffs", (GraphState state, NodeContext nodeCtx) -> {
                    AiMessage last = lastAi(state);
                    if (last == null || !last.hasToolExecutionRequests()) {
                        return Map.of();
                    }
                    // #68：run 级预算——超限后不再执行子图，回结构化提示
                    // 让 supervisor 收敛出最终回答（乒乓循环止于预算而非
                    // recursionLimit 触顶的裸错误）。
                    int limit = ctx.settings().agentHandoffMaxPerRun;
                    int count = state.get(HANDOFF_COUNT) instanceof Number n ? n.intValue() : 0;
                    // String.valueOf(state.get(...)) 会让泛型 T 推断成 char[]
                    // （valueOf(char[]) 比 valueOf(Object) 更具体）——先收窄到 Object
                    Object lastRaw = state.get(LAST_HANDOFF);
                    String lastTarget = lastRaw == null ? "" : String.valueOf(lastRaw);
                    List<ChatMessage> results = new ArrayList<>();
                    // handoff 按非只读语义串行执行（#337：callAll 对池外工具
                    // 同样串行；预算判定要求按序，直接内联）。
                    boolean budgetExhausted = false;
                    int executed = 0;
                    for (ToolExecutionRequest request : last.toolExecutionRequests()) {
                        String output;
                        if (budgetExhausted) {
                            output = handoffBudgetExhaustedText(count, limit);
                        } else if (limit > 0 && count + executed >= limit) {
                            budgetExhausted = true;
                            output = handoffBudgetExhaustedText(count + executed, limit);
                        } else {
                            output = runHandoff(request, state.messages(), subgraphs, nodeCtx);
                            executed++;
                            lastTarget = request.name() == null ? lastTarget
                                    : request.name().substring(
                                            request.name().startsWith(HANDOFF_TOOL_PREFIX)
                                                    ? HANDOFF_TOOL_PREFIX.length() : 0);
                        }
                        results.add(ToolExecutionResultMessage.from(
                                request.id() == null ? "" : request.id(),
                                request.name() == null ? "" : request.name(),
                                output));
                    }
                    Map<String, Object> update = new java.util.LinkedHashMap<>();
                    update.put(GraphState.MESSAGES, results);
                    update.put(HANDOFF_COUNT, count + executed);
                    update.put(LAST_HANDOFF, lastTarget);
                    return update;
                })
                .addEdge("handoffs", "supervisor")
                // 压缩终态钩子（#328）随 ModuleContext 带入（#330 写回互斥）。
                .compile(ctx.checkpointer(), MODULE_NAME, ctx.toolPool(), ctx.compactionHook());
    }

    /**
     * handoff 工具规格：名字 {@code transfer_to_<module>}，description =
     * 模块 description + capabilities 附加段（#69——supervisor 模型据此
     * 路由与判断能力边界），参数 schema 为空对象（无参数——与
     * create_handoff_tool 的零参工具一致）。
     */
    private static ToolSpecification handoffSpec(AgentModule module) {
        return ToolSpecification.builder()
                .name(HANDOFF_TOOL_PREFIX + module.name())
                .description(handoffDescription(module))
                .parameters(JsonObjectSchema.builder().build())
                .build();
    }

    /**
     * handoff 工具描述（#69）：capabilities 缺省（空串）时与
     * {@code description()} 逐字节一致（向后兼容）；非空时追加
     * 「适用场景」段——模块声明的能力边界是 supervisor 路由的判据素材。
     */
    static String handoffDescription(AgentModule module) {
        String capabilities = module.capabilities();
        return capabilities == null || capabilities.isBlank()
                ? module.description()
                : module.description() + "\n\n适用场景：" + capabilities.strip();
    }

    /**
     * 执行一次 handoff：解析目标模块 → 运行其编译图 → 返回子图最后一条
     * AiMessage 文本（output_mode="last_message"）。子图事件的三个通道
     * （StepEvent / DeltaEvent / 自定义契约事件）经转发 sink 透传到当前
     * 节点的事件出口。
     */
    private static String runHandoff(ToolExecutionRequest request, List<ChatMessage> messages,
                                     Map<String, CompiledGraph> subgraphs, NodeContext nodeCtx) {
        String toolName = request.name() == null ? "" : request.name();
        String target = toolName.startsWith(HANDOFF_TOOL_PREFIX)
                ? toolName.substring(HANDOFF_TOOL_PREFIX.length())
                : toolName;
        CompiledGraph subgraph = subgraphs.get(target);
        if (subgraph == null) {
            // 与工具池的 unknown tool 归一语义同款：文本进 ToolMessage，
            // 模型看到错误并能恢复（自愈重路由），绝不打断图执行。
            return "tool execution failed: unknown handoff target '%s'（可用：%s）"
                    .formatted(target, String.join(", ", subgraphs.keySet()));
        }
        // 子图命名空间（见类注释）：threadId 加 @module 后缀，避免与
        // supervisor 自身的快照互相覆盖。只带当前 messages 的状态进子图。
        RunConfig config = nodeCtx.config();
        RunConfig subConfig = new RunConfig(
                config.threadId() + "@" + target,
                config.userId(),
                config.recursionLimit());
        EventSink forwardSink = nodeCtx::emit;
        GraphState finalState = subgraph.stream(new GraphState(messages, Map.of()), subConfig, forwardSink);
        AiMessage reply = lastAi(finalState);
        if (reply == null) {
            return "";
        }
        String text = ContextEngine.contentText(reply);
        return text == null ? "" : text;
    }

    /** 最后一条 AiMessage（不存在返回 null）。 */
    private static AiMessage lastAi(GraphState state) {
        List<ChatMessage> messages = state.messages();
        for (int i = messages.size() - 1; i >= 0; i--) {
            if (messages.get(i) instanceof AiMessage ai) {
                return ai;
            }
        }
        return null;
    }

    private static boolean lastAiHasToolCalls(GraphState state) {
        AiMessage last = lastAi(state);
        return last != null && last.hasToolExecutionRequests();
    }
}
