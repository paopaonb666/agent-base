package io.github.paopaonb666.agentbase.core.graphs;

import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import io.github.paopaonb666.agentbase.core.AgentModule;
import io.github.paopaonb666.agentbase.core.context.ContextEngine;
import io.github.paopaonb666.agentbase.core.graph.CompiledGraph;
import io.github.paopaonb666.agentbase.core.graph.GraphBuilder;
import io.github.paopaonb666.agentbase.core.graph.GraphState;
import io.github.paopaonb666.agentbase.core.graph.NodeContext;
import io.github.paopaonb666.agentbase.core.tools.AgentTool;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * 共享图工厂（P1-4 / H3）：单 LLM 节点 + 可选工具循环的公共装配。
 *
 * <p>chat / writer / hello 这类"单 LLM 节点（可选工具循环）"模块过去各自
 * 复制同一套 {@code StateGraph → ToolNode → compile} 样板，且上下文工程
 * （注入去重 + token 预算修剪）只在 chat 里生效——writer/hello 的历史里
 * 注入块逐轮累积，模型输入持续膨胀。工厂把样板与上下文工程一并收敛：
 *
 * <ul>
 *   <li>模块只声明 {@code name} / {@code systemPrompt} / {@code tools}，
 *       几行即可接入；</li>
 *   <li>发给模型的输入统一经过 {@link ContextEngine#buildModelInput}
 *       （注入去重 + 预算修剪，工具配对不拆散）——<b>对所有模块生效</b>；</li>
 *   <li>模块级 {@code systemPrompt} 是临时拼装（不落库、不参与修剪去重）；</li>
 *   <li>流式累积保持不变：SSE 层看到 token 级 delta，取消传播到 LLM 请求。</li>
 * </ul>
 *
 * <p>对应 Python 侧 {@code core/graphs.py::build_single_agent_graph}。
 * 需要自定义拓扑的模块保留 escape hatch：直接使用 core.context 与
 * 图运行时原生 API 自行组装（planner 即如此）。
 */
public final class Graphs {

    private Graphs() {}

    /**
     * planner 最坏步数（H5 口径，#229 编译期推导）：固定壳 8 步 +
     * 每子任务（执行 1 + 工具轮 rounds）× 重规划 2 轮。默认配置下
     * 8+2*5*(8+1)=98 —— 远超 AGENT_RECURSION_LIMIT 默认 25，组合根在
     * 启动时据此给出显式建议（全局上限仍只做兜底，不自动放宽）。
     */
    public static int plannerWorstSteps(io.github.paopaonb666.agentbase.core.config.Settings settings) {
        var p = settings.planner;
        return 8 + 2 * p.maxSubtasks * (p.subtaskToolRounds + 1);
    }

    /**
     * 构建并编译"单 LLM 节点 + 可选工具循环"的图。
     *
     * <p>{@code tools} 缺省用共享工具池（{@code ctx.tools()}）；传空列表
     * 可显式禁用工具循环。图以 {@code name=<模块名>} 编译——supervisor
     * 要求每个 sub-agent 图携带其模块名。
     */
    public static CompiledGraph buildSingleAgentGraph(
            AgentModule.ModuleContext ctx, String name, String systemPrompt, List<AgentTool> tools) {
        List<AgentTool> pool = tools == null ? ctx.tools() : tools;
        // 条目 15 / R2-F1 收口：SYSTEM 域扣减——窗口生效时把 systemPrompt 计入
        // 预算（window=0 原样，逐字节兼容）；窗口=0 的旧口径不变。
        int maxTokens = ContextEngine.budgetAfterSystem(ctx.settings(), ctx.llm(), systemPrompt);
        // #84①：不可信标记开关在图构建期固化（build 期读配置，运行期零开销）。
        boolean untrustedMarker = ctx.settings() != null
                && ctx.settings().toolkit.toolResultUntrustedMarker;
        // 条目 15：TOOL_RESULTS 域观测上限在构建期固化进闭包（0 = 完全跳过，
        // 零开销零日志；>0 只告警不截断——外置由条目 6 承接）。
        int toolResultCapTokens = ctx.settings() == null
                ? 0 : ctx.settings().memory.contextBudgetToolResultTokens;

        GraphBuilder builder = GraphBuilder.begin();
        builder.addNode("agent", (GraphState state, NodeContext nodeCtx) -> {
            // 上下文工程（H3）：注入去重 + token 预算修剪，对所有模块生效。
            List<ChatMessage> modelInput =
                    ContextEngine.buildModelInput(state.messages(), maxTokens, ctx.llm().tokenFactor());
            if (systemPrompt != null) {
                // 模块提示词临时拼装在最前：不落库，也不属于"注入型系统消息"。
                List<ChatMessage> withPrompt = new ArrayList<>();
                withPrompt.add(SystemMessage.from(systemPrompt));
                withPrompt.addAll(modelInput);
                modelInput = withPrompt;
            }
            AiMessage finalMessage = ctx.llm().stream(modelInput, specs(pool),
                    nodeCtx::emitDelta);
            if (finalMessage == null) {
                return Map.of();
            }
            return Map.<String, Object>of(GraphState.MESSAGES, List.of(finalMessage));
        });

        if (pool.isEmpty()) {
            builder.entry("agent");
            builder.addEdge("agent", GraphBuilder.END);
        } else {
            // 条目 5：工具循环检测阈值在构建期读取（运行期零开销；0 = 关闭）。
            int loopThreshold = ctx.settings() == null
                    ? ToolLoopDetector.DEFAULT_THRESHOLD : ctx.settings().agentToolLoopThreshold;
            // 工具失败在池内归一成模型可读文本反馈（坏掉的工具绝不打断对话）。
            builder.addNode("tools", (GraphState state, NodeContext nodeCtx) -> {
                AiMessage last = lastAi(state);
                if (last == null || !last.hasToolExecutionRequests()) {
                    return Map.of();
                }
                List<ToolExecutionRequest> requests = last.toolExecutionRequests();
                // 条目 5：循环检测在 callAll 之前分桶——被拦截的请求不执行
                // （省成本的前提），每个拦截请求回喂一条结构化提示 ToolMessage
                // （配对完整、协议安全，#68 handoff 预算先例形态）。
                int[] loopRepeats = ToolLoopDetector.consecutiveRepeatCounts(state.messages(), requests);
                boolean[] loopIntercepted = new boolean[requests.size()];
                boolean anyIntercepted = false;
                for (int i = 0; i < requests.size(); i++) {
                    loopIntercepted[i] = ToolLoopDetector.isIntercepted(loopRepeats[i], loopThreshold);
                    anyIntercepted |= loopIntercepted[i];
                    if (loopIntercepted[i]) {
                        ToolLoopDetector.logIntercepted(
                                requests.get(i).name() == null ? "" : requests.get(i).name(),
                                loopRepeats[i], loopThreshold);
                    }
                }
                List<String> outputs;
                if (anyIntercepted) {
                    // 分桶：拦截槽位不执行、先填提示文本；正常请求子列表照常
                    // 执行后按请求序回填（ToolPool.callAll 槽位重组先例）。
                    List<String> slots = new ArrayList<>(requests.size());
                    List<ToolExecutionRequest> executable = new ArrayList<>();
                    List<Integer> executableSlots = new ArrayList<>();
                    for (int i = 0; i < requests.size(); i++) {
                        if (loopIntercepted[i]) {
                            slots.add(ToolLoopDetector.loopHintText(
                                    requests.get(i).name() == null ? "" : requests.get(i).name(),
                                    loopRepeats[i]));
                        } else {
                            slots.add(null); // 待执行回填
                            executable.add(requests.get(i));
                            executableSlots.add(i);
                        }
                    }
                    // #337：只读并发、有副作用串行；池未接（测试直构）退化为串行。
                    List<String> executed = nodeCtx.tools() == null
                            ? executable.stream().map(r -> nodeCtx.callTool(r.name(), r.arguments())).toList()
                            : nodeCtx.tools().callAll(executable, nodeCtx.config(), nodeCtx,
                                    request -> nodeCtx.callTool(request.name(), request.arguments()));
                    for (int k = 0; k < executed.size(); k++) {
                        slots.set(executableSlots.get(k), executed.get(k));
                    }
                    outputs = slots;
                } else {
                    // 无拦截：既有执行通道逐字节不变（#337）。
                    outputs = nodeCtx.tools() == null
                            ? requests.stream().map(r -> nodeCtx.callTool(r.name(), r.arguments())).toList()
                            : nodeCtx.tools().callAll(requests, nodeCtx.config(), nodeCtx,
                                    request -> nodeCtx.callTool(request.name(), request.arguments()));
                }
                List<ChatMessage> results = new ArrayList<>();
                int index = 0;
                for (ToolExecutionRequest request : requests) {
                    String toolName = request.name() == null ? "" : request.name();
                    String output = loopIntercepted[index]
                            ? outputs.get(index) // 提示文本不包不可信标记（系统引导，非外部数据）
                            : decorateOutput(toolName, outputs.get(index), untrustedMarker, toolResultCapTokens);
                    results.add(ToolExecutionResultMessage.from(
                            request.id() == null ? "" : request.id(),
                            toolName,
                            output));
                    index++;
                }
                return Map.<String, Object>of(GraphState.MESSAGES, results);
            });
            builder.entry("agent");
            // tools_condition：最后的 AIMessage 带 tool_calls → 进工具节点。
            // #202：封闭目标集显式声明，compile 校验笔误。
            builder.addConditionalEdges("agent",
                    state -> hasPendingToolCalls(state) ? "tools" : GraphBuilder.END,
                    Set.of("tools", GraphBuilder.END));
            // 条目 5：tools 轮后路由——本批请求全部被拦截 → END（提示
            // ToolMessage 已配对，协议安全）；部分拦截 → 回 agent（拦截的
            // 提示已在 ToolMessage 里，模型据此换参数/换工具/收尾）。
            // #202：封闭目标集显式声明，compile 校验笔误。
            builder.addConditionalEdges("tools",
                    state -> ToolLoopDetector.allInterceptedAfterRound(state.messages(), loopThreshold)
                            ? GraphBuilder.END : "agent",
                    Set.of("agent", GraphBuilder.END));
        }

        // 工具池（#131/#238）：生产装配经 ModuleContext 带入共享池（超时
        // 包装 + 审计 + 指标收口）；缺省（测试直构）退回本地兜底池——
        // 保留默认超时（兜底池不搞无超时执行）。
        io.github.paopaonb666.agentbase.core.tools.ToolPool toolPool =
                ctx.toolPool() != null ? ctx.toolPool()
                        : io.github.paopaonb666.agentbase.core.tools.ToolPool.build(
                                pool, List.of(),
                                io.github.paopaonb666.agentbase.core.tools.ToolPool.DEFAULT_TOOL_TIMEOUT_SECONDS,
                                Map.of(), null);
        // 压缩终态钩子（#328）随 ModuleContext 带入（null = 未装配，零行为
        // 变化）；只动 compile 调用段——tools 节点段归并行批次。
        return builder.compile(ctx.checkpointer(), name, toolPool, ctx.compactionHook());
    }

    private static List<dev.langchain4j.agent.tool.ToolSpecification> specs(List<AgentTool> pool) {
        return pool.stream().map(AgentTool::specification).toList();
    }

    private static AiMessage lastAi(GraphState state) {
        List<ChatMessage> messages = state.messages();
        for (int i = messages.size() - 1; i >= 0; i--) {
            if (messages.get(i) instanceof AiMessage ai) {
                return ai;
            }
        }
        return null;
    }

    private static boolean hasPendingToolCalls(GraphState state) {
        AiMessage last = lastAi(state);
        return last != null && last.hasToolExecutionRequests();
    }

    /**
     * 单请求输出 → ToolMessage 文本的装饰（条目 6 外置改造预留的函数级
     * 接缝）：#84① 开关开时外部数据结果包标记进上下文；ToolCallEvent/
     * 审计日志在池内用原文（事件与上下文分离）。行为与抽方法前逐字节一致。
     *
     * <p>条目 15 TOOL_RESULTS 域：{@code toolResultCapTokens > 0} 时对装饰后
     * 的最终文本（含不可信标记行）做超限观测告警（helper 单源
     * ContextEngine.warnIfToolResultOverCap；不截断不外置）。
     */
    private static String decorateOutput(String toolName, String output, boolean untrustedMarker,
                                         int toolResultCapTokens) {
        String decorated = untrustedMarker
                ? ContextEngine.wrapUntrusted(toolName, output)
                : output;
        ContextEngine.warnIfToolResultOverCap(toolName, decorated, toolResultCapTokens);
        return decorated;
    }

    /** 供 planner 等自定义模块复用：把「本轮贴尾注入」风格的无更新空 Map 返回。 */
    public static Map<String, Object> noUpdate() {
        return new LinkedHashMap<>();
    }
}
