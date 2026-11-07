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
        int maxTokens = ctx.settings().memory.contextMaxTokens;

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

            List<ChatMessage> inputs = subtaskInput(state, task, tasks, cursor, maxTokens);
            List<ChatMessage> produced = new ArrayList<>();
            boolean exhausted = true;
            String summary = "";
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
                produced.addAll(executeToolCalls(finalMessage.toolExecutionRequests(), nodeCtx));
            }
            String status = exhausted ? PlannerState.FAILED : PlannerState.DONE;
            Map<String, Object> updated = PlannerState.copyTask(task);
            updated.put(PlannerState.STATUS, status);
            updated.put(PlannerState.SUMMARY, exhausted ? "工具轮数耗尽" : summary);
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
     */
    private static List<ChatMessage> executeToolCalls(List<ToolExecutionRequest> toolCalls,
                                                      NodeContext nodeCtx) {
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
            results.add(ToolExecutionResultMessage.from(
                    call.id() == null ? "" : call.id(), name, outputs.get(index)));
            index++;
        }
        return results;
    }

    /**
     * 子任务输入组装：EXECUTE_SYSTEM（总目标 + 已完成结果 + 当前子任务）
     * + 注入去重与预算修剪后的历史（对应 Python {@code _subtask_input}，
     * 提示词文案逐字见 {@link PlannerPrompts#execute}）。
     */
    private static List<ChatMessage> subtaskInput(GraphState state, Map<String, Object> task,
                                                  List<Map<String, Object>> tasks, int cursor,
                                                  int maxTokens) {
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
        List<ChatMessage> input = new ArrayList<>();
        input.add(SystemMessage.from(PlannerPrompts.execute(
                goal, done.isEmpty() ? "（无）" : done.toString(),
                PlannerState.str(task, PlannerState.GOAL))));
        input.addAll(ContextEngine.buildModelInput(messages, maxTokens));
        return input;
    }
}
