package io.github.paopaonb666.agentbase.modules.planner.graph;

import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import io.github.paopaonb666.agentbase.core.AgentModule;
import io.github.paopaonb666.agentbase.core.context.ContextEngine;
import io.github.paopaonb666.agentbase.core.graph.GraphBuilder;
import io.github.paopaonb666.agentbase.core.graph.GraphState;
import io.github.paopaonb666.agentbase.core.graph.NodeContext;
import io.github.paopaonb666.agentbase.core.graphs.ToolLoopDetector;
import io.github.paopaonb666.agentbase.core.tools.AgentTool;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * execute 节点（B3）：子任务内联 ReAct 循环——对应 Python 侧
 * {@code modules/planner/nodes_execute.py}。
 *
 * <p>决策 1：循环内联在节点里而不是子图——外层图每子任务恒定 2 个
 * super-step（execute+check），recursion_limit 不随子任务数和工具轮数膨胀。
 * 复用基座件：{@link ContextEngine#buildModelInput}（注入去重 + 预算修剪）、
 * 共享工具池（超时 + 审计包装器在直调路径上同样生效——本节点经
 * {@code NodeContext.callTool} 直接调池工具）。
 *
 * <p>子任务级失败只有两种：工具轮数耗尽、空输出（check 的空输出门改判）。
 * 循环局部累积 {@code produced}，结束时整体进 messages 通道（checkpointer /
 * 记忆形成 / 历史回放都能看到完整过程）。任何状态变更都拷贝重建 Task
 * （决策 7：不得原地修改）。
 */
public final class ExecuteNode {

    /** 子任务结果摘要截断（写回 Task.summary 的长度上限）。 */
    private static final int SUMMARY_MAX_CHARS = 500;

    private ExecuteNode() {}

    /** 构造 execute 节点（LangGraph 节点的统一签名：状态进、状态更新字典出）。 */
    public static GraphBuilder.Node make(AgentModule.ModuleContext ctx) {
        List<AgentTool> pool = ctx.tools();
        // 池非空才绑工具（对应 Python bind_tools(pool) if pool else ctx.llm）。
        List<ToolSpecification> specs = pool.isEmpty()
                ? List.of()
                : pool.stream().map(AgentTool::specification).toList();
        // #84①：不可信标记开关在节点构造期固化（与 Graphs 同款判据单源）。
        boolean untrustedMarker = ctx.settings() != null
                && ctx.settings().toolkit.toolResultUntrustedMarker;
        // 条目 15：TOOL_RESULTS 域观测上限在构造期固化（0 = 完全跳过，零开销
        // 零日志；>0 只告警不截断——外置由条目 6 承接）。SYSTEM 域扣减随
        // subtaskInput 运行期走 helper（execute 提示含 goal/子任务文本，
        // 构造期不存在——与 Graphs 的静态 prompt 形态不同）。
        int toolResultCapTokens = ctx.settings() == null
                ? 0 : ctx.settings().memory.contextBudgetToolResultTokens;
        // 条目 6：外置接线在构造期固化——fileStore 与溢出阈值（store 缺席或
        // 阈值 0 = 外置停用，executeToolCalls 原样返回，行为退回条目 15）。
        io.github.paopaonb666.agentbase.extensions.filestore.UploadedFileStore fileStore = ctx.fileStore();
        int spillThresholdChars = ctx.settings() == null
                ? 0 : ctx.settings().toolkit.toolResultSpillThreshold;

        return (GraphState state, NodeContext nodeCtx) -> {
            int subtaskToolRounds = ctx.settings().planner.subtaskToolRounds;
            List<Map<String, Object>> tasks = PlannerState.tasks(state);
            int cursor = PlannerState.cursor(state);
            if (cursor >= tasks.size()) {
                return Map.of(); // 防御：replan 可能清空剩余任务
            }
            // 状态变更拷贝重建 Task（决策 7）。
            Map<String, Object> task = PlannerState.copyTask(tasks.get(cursor));
            task.put(PlannerState.STATUS, PlannerState.RUNNING);
            task.put(PlannerState.ATTEMPTS,
                    PlannerState.intOf(tasks.get(cursor), PlannerState.ATTEMPTS) + 1);
            tasks.set(cursor, task);
            PlanStreaming.emitPlan(nodeCtx, tasks, "progress",
                    "子任务 " + (cursor + 1) + "/" + tasks.size() + " 执行中");

            List<ChatMessage> inputs = subtaskInput(state, task, tasks, cursor, ctx);
            List<ChatMessage> produced = new ArrayList<>();
            boolean exhausted = true;
            String summary = "";
            // 条目 5：工具循环检测阈值（0 = 关闭）；loopBrokenTool 非空 =
            // 本子任务因连续同参循环中断收尾（区别于"工具轮数耗尽"）。
            int loopThreshold = ctx.settings().agentToolLoopThreshold;
            String loopBrokenTool = null;
            for (int round = 0; round < subtaskToolRounds; round++) {
                List<ChatMessage> callInput = new ArrayList<>(inputs.size() + produced.size());
                callInput.addAll(inputs);
                callInput.addAll(produced);
                AiMessage finalMessage =
                        ctx.llm().stream(callInput, specs.isEmpty() ? null : specs, nodeCtx::emitDelta);
                if (finalMessage == null) {
                    exhausted = false; // 模型无产出：交给 check 的空输出门改判
                    break;
                }
                produced.add(finalMessage);
                if (!finalMessage.hasToolExecutionRequests()) {
                    String text = ContextEngine.contentText(finalMessage).strip();
                    summary = text.substring(0, Math.min(text.length(), SUMMARY_MAX_CHARS));
                    exhausted = false;
                    break;
                }
                if (pool.isEmpty()) {
                    break; // 无工具池可执行（对应 Python run_tool_calls is None）
                }
                List<ToolExecutionRequest> toolCalls = finalMessage.toolExecutionRequests();
                // 条目 5：连续同参循环检测（判据单源 ToolLoopDetector）。
                // produced 即本子任务局部的工具轮序列——循环本就 per-subtask
                // 局部，从 produced 派生免状态通道，也不会误伤跨子任务的历史。
                int[] loopRepeats = ToolLoopDetector.consecutiveRepeatCounts(produced, toolCalls);
                boolean anyIntercepted = false;
                for (int i = 0; i < toolCalls.size(); i++) {
                    anyIntercepted |= ToolLoopDetector.isIntercepted(loopRepeats[i], loopThreshold);
                }
                if (anyIntercepted) {
                    // 达阈值：被拦截的请求不执行、回喂提示 ToolMessage（配对
                    // 完整）；其余照常执行后中断 ReAct 循环，子任务按 FAILED
                    // 收尾（check 的空输出门/verdict 语义不受影响）。
                    for (int i = 0; i < toolCalls.size(); i++) {
                        if (ToolLoopDetector.isIntercepted(loopRepeats[i], loopThreshold)) {
                            String name = toolCalls.get(i).name() == null ? "" : toolCalls.get(i).name();
                            ToolLoopDetector.logIntercepted(name, loopRepeats[i], loopThreshold);
                            loopBrokenTool = name;
                            break;
                        }
                    }
                    produced.addAll(executeToolCallsWithLoopBreaks(toolCalls, loopRepeats,
                            loopThreshold, nodeCtx, untrustedMarker, toolResultCapTokens,
                            fileStore, spillThresholdChars));
                    break;
                }
                produced.addAll(executeToolCalls(toolCalls, nodeCtx, untrustedMarker,
                        toolResultCapTokens, fileStore, spillThresholdChars));
            }
            String status = exhausted || loopBrokenTool != null
                    ? PlannerState.FAILED : PlannerState.DONE;
            Map<String, Object> updated = PlannerState.copyTask(task);
            updated.put(PlannerState.STATUS, status);
            updated.put(PlannerState.SUMMARY, loopBrokenTool != null
                    ? ToolLoopDetector.loopBreakSummary(loopBrokenTool)
                    : exhausted ? "工具轮数耗尽" : summary);
            tasks.set(cursor, updated);
            PlanStreaming.emitPlan(nodeCtx, tasks, "progress", "子任务 " + (cursor + 1) + " " + status);
            return Map.of(
                    GraphState.MESSAGES, produced,
                    PlannerState.TASKS, tasks);
        };
    }

    /**
     * 一轮 tool_calls 的执行：每个结果/失败都归一为 ToolMessage（对应
     * Python {@code _execute_tool_calls}）。未知工具与执行失败都由共享池
     * 归一成模型可读文本（坏掉的工具绝不打断对话）。
     *
     * <p>条目 6 外置落点：叠加顺序与 Graphs.decorateOutput 同规则——
     * <b>先 spill 后 wrap 后 warn</b>：超阈值结果经
     * {@link ContextEngine#spillIfNeeded} 外置成摘要块（外置行存原文不带
     * 标记），#84① 开关开时外部数据结果在摘要块上包标记（标记仍是第一行）。
     * MCP 归属经 {@code nodeCtx.tools().isMcpTool(name)} 传入 2 参判据
     * （MCP 工具命名是 "{server}_{tool}"，非 mcp_ 前缀；池未接判 false）。
     *
     * <p>条目 15 TOOL_RESULTS 域：{@code toolResultCapTokens > 0} 时对最终
     * 进上下文的文本（含 #84 不可信标记行）做超限观测告警（helper 单源
     * ContextEngine.warnIfToolResultOverCap，与 Graphs.decorateOutput 同规则；
     * 外置生效时最终文本=摘要块，天然小）。
     */
    private static List<ChatMessage> executeToolCalls(List<ToolExecutionRequest> toolCalls,
                                                      NodeContext nodeCtx,
                                                      boolean untrustedMarker,
                                                      int toolResultCapTokens,
                                                      io.github.paopaonb666.agentbase.extensions.filestore.UploadedFileStore fileStore,
                                                      int spillThresholdChars) {
        List<ChatMessage> results = new ArrayList<>(toolCalls.size());
        // #337：只读工具并发、有副作用串行，结果按请求序（池内通道保归一）；
        // 池未接（测试直构）退化为串行 callTool。
        List<String> outputs = nodeCtx.tools() == null
                ? toolCalls.stream().map(c -> nodeCtx.callTool(
                        c.name() == null ? "" : c.name(),
                        c.arguments() == null ? "{}" : c.arguments())).toList()
                : nodeCtx.tools().callAll(toolCalls, nodeCtx.config(), nodeCtx,
                        call -> nodeCtx.callTool(call.name() == null ? "" : call.name(),
                                call.arguments() == null ? "{}" : call.arguments()));
        int index = 0;
        for (ToolExecutionRequest call : toolCalls) {
            String name = call.name() == null ? "" : call.name();
            String output = outputs.get(index);
            // 条目 6：先 spill（外置摘要块是上下文文本基座）后 wrap 后 warn。
            String threadId = nodeCtx.config() == null ? "" : nodeCtx.config().threadId();
            String userId = nodeCtx.config() == null ? "" : nodeCtx.config().userId();
            output = ContextEngine.spillIfNeeded(name, output, fileStore,
                    threadId, userId, spillThresholdChars);
            if (untrustedMarker) {
                // #84①：与 Graphs 同规则（判据单源在 ContextEngine）。
                boolean mcpTool = nodeCtx.tools() != null && nodeCtx.tools().isMcpTool(name);
                output = ContextEngine.wrapUntrusted(name, output, mcpTool);
            }
            ContextEngine.warnIfToolResultOverCap(name, output, toolResultCapTokens);
            results.add(ToolExecutionResultMessage.from(
                    call.id() == null ? "" : call.id(), name, output));
            index++;
        }
        return results;
    }

    /**
     * 一轮 tool_calls 的循环中断执行（条目 5）：被拦截的请求不执行、回喂
     * 提示 ToolMessage（配对完整，#68 先例形态——提示文本不包不可信标记、
     * 不做 TOOL_RESULTS 域计量、不外置，系统引导非外部数据）；其余照常执行，
     * 结果按请求序（与 Graphs tools 节点同规则，判据单源 ToolLoopDetector）。
     */
    private static List<ChatMessage> executeToolCallsWithLoopBreaks(List<ToolExecutionRequest> toolCalls,
                                                                    int[] loopRepeats,
                                                                    int loopThreshold,
                                                                    NodeContext nodeCtx,
                                                                    boolean untrustedMarker,
                                                                    int toolResultCapTokens,
                                                                    io.github.paopaonb666.agentbase.extensions.filestore.UploadedFileStore fileStore,
                                                                    int spillThresholdChars) {
        List<ChatMessage> results = new ArrayList<>(toolCalls.size());
        List<ToolExecutionRequest> executable = new ArrayList<>();
        List<Integer> executableSlots = new ArrayList<>();
        for (int i = 0; i < toolCalls.size(); i++) {
            ToolExecutionRequest call = toolCalls.get(i);
            if (ToolLoopDetector.isIntercepted(loopRepeats[i], loopThreshold)) {
                String name = call.name() == null ? "" : call.name();
                results.add(ToolExecutionResultMessage.from(
                        call.id() == null ? "" : call.id(), name,
                        ToolLoopDetector.loopHintText(name, loopRepeats[i])));
            } else {
                executable.add(call);
                executableSlots.add(i);
            }
        }
        List<ChatMessage> executed = executeToolCalls(executable, nodeCtx, untrustedMarker,
                toolResultCapTokens, fileStore, spillThresholdChars);
        for (int k = 0; k < executed.size(); k++) {
            results.add(executableSlots.get(k), executed.get(k));
        }
        return results;
    }

    /**
     * 子任务输入组装：EXECUTE_SYSTEM（总目标 + 已完成结果 + 当前子任务）
     * + 注入去重与预算修剪后的历史（对应 Python {@code _subtask_input}，
     * 提示词文案逐字见 {@link PlannerPrompts#execute}）。
     *
     * <p>预算推导（条目 15 / R2-F1）：execute 提示运行期才存在（含 goal 与
     * 已完成子任务文本），SYSTEM 域扣减随节点执行走
     * {@link ContextEngine#budgetAfterSystem}——窗口生效时提示计入预算挤占
     * 历史体，window=0 与旧口径逐字节一致。
     */
    private static List<ChatMessage> subtaskInput(GraphState state, Map<String, Object> task,
                                                  List<Map<String, Object>> tasks, int cursor,
                                                  AgentModule.ModuleContext ctx) {
        StringBuilder done = new StringBuilder();
        for (int i = 0; i < cursor && i < tasks.size(); i++) {
            Map<String, Object> t = tasks.get(i);
            if (PlannerState.DONE.equals(t.get(PlannerState.STATUS))) {
                if (done.length() > 0) {
                    done.append('\n');
                }
                String summary = PlannerState.str(t, PlannerState.SUMMARY);
                done.append("#").append(PlannerState.intOf(t, PlannerState.ID))
                        .append(" ").append(PlannerState.str(t, PlannerState.GOAL))
                        .append("：").append(summary, 0, Math.min(summary.length(), 200));
            }
        }
        List<ChatMessage> messages = PlannerState.messages(state);
        String goal = messages.isEmpty() ? "（未知）" : ContextEngine.contentText(messages.get(messages.size() - 1));
        String prompt = PlannerPrompts.execute(
                goal, done.isEmpty() ? "（无）" : done.toString(),
                PlannerState.str(task, PlannerState.GOAL));
        int maxTokens = ContextEngine.budgetAfterSystem(ctx.settings(), ctx.llm(), prompt);
        List<ChatMessage> input = new ArrayList<>();
        input.add(SystemMessage.from(prompt));
        input.addAll(ContextEngine.buildModelInput(messages, maxTokens, ctx.llm().tokenFactor()));
        return input;
    }
}
