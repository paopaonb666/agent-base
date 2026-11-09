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
        int maxTokens = io.github.paopaonb666.agentbase.core.context.ContextEngine.resolveInputBudget(
                ctx.settings(), ctx.llm()); // #4：per-model 窗口推导

        GraphBuilder builder = GraphBuilder.begin();
        builder.addNode("agent", (GraphState state, NodeContext nodeCtx) -> {
            // 上下文工程（H3）：注入去重 + token 预算修剪，对所有模块生效。
            List<ChatMessage> modelInput =
                    ContextEngine.buildModelInput(state.messages(), maxTokens);
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
            // 工具失败在池内归一成模型可读文本反馈（坏掉的工具绝不打断对话）。
            builder.addNode("tools", (GraphState state, NodeContext nodeCtx) -> {
                AiMessage last = lastAi(state);
                if (last == null || !last.hasToolExecutionRequests()) {
                    return Map.of();
                }
                List<ChatMessage> results = new ArrayList<>();
                List<ToolExecutionRequest> requests = last.toolExecutionRequests();
                // #337：只读并发、有副作用串行；池未接（测试直构）退化为串行。
                List<String> outputs = nodeCtx.tools() == null
                        ? requests.stream().map(r -> nodeCtx.callTool(r.name(), r.arguments())).toList()
                        : nodeCtx.tools().callAll(requests, nodeCtx.config(), nodeCtx,
                                request -> nodeCtx.callTool(request.name(), request.arguments()));
                int index = 0;
                for (ToolExecutionRequest request : requests) {
                    results.add(ToolExecutionResultMessage.from(
                            request.id() == null ? "" : request.id(),
                            request.name() == null ? "" : request.name(),
                            outputs.get(index)));
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
            builder.addEdge("tools", "agent");
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
        return builder.compile(ctx.checkpointer(), name, toolPool);
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

    /** 供 planner 等自定义模块复用：把「本轮贴尾注入」风格的无更新空 Map 返回。 */
    public static Map<String, Object> noUpdate() {
        return new LinkedHashMap<>();
    }
}
