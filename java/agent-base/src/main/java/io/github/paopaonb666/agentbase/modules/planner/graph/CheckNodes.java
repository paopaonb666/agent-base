package io.github.paopaonb666.agentbase.modules.planner.graph;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import io.github.paopaonb666.agentbase.core.AgentModule;
import io.github.paopaonb666.agentbase.core.context.ContextEngine;
import io.github.paopaonb666.agentbase.core.graph.GraphBuilder;
import io.github.paopaonb666.agentbase.core.graph.GraphState;
import io.github.paopaonb666.agentbase.core.graph.NodeContext;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * check / replan / synthesize 节点与路由（B4）——对应 Python 侧
 * {@code modules/planner/nodes_check.py}。
 *
 * <ul>
 *   <li>check：空输出一致性门（评审修正 3——模型中途放弃不得静默放行）+
 *       cursor 推进到下一个 pending；</li>
 *   <li>replan：预算内（{@code PLANNER_MAX_REPLANS}）重规划剩余工作，保留
 *       done 结果，解析失败以 failed 任务目标兜底；</li>
 *   <li>synthesize：无工具的 best-effort 综合——replan 预算之外绝不再开
 *       第二条执行路径（决策 2），如实标注未完成项；</li>
 *   <li>路由：有 failed 且预算未耗尽 → replan；有 pending → execute；
 *       否则 → synthesize。</li>
 * </ul>
 *
 * <p>所有节点与路由禁止对非 messages 键直读（决策 6，走
 * {@link PlannerState} 访问助手）。
 */
public final class CheckNodes {

    /** replan 提示里 done/failed 摘要的截断长度（summary[:150]）。 */
    private static final int REPLAN_SUMMARY_MAX_CHARS = 150;
    /** synthesize 计划清单里每个子任务摘要的截断长度（summary[:300]）。 */
    private static final int SYNTH_SUMMARY_MAX_CHARS = 300;

    private CheckNodes() {}

    /** check 节点：空输出门 + cursor 推进。 */
    public static GraphBuilder.Node makeCheck(AgentModule.ModuleContext ctx) {
        int minChars = ctx.settings().planner.minSummaryChars;
        return (GraphState state, NodeContext nodeCtx) -> {
            List<Map<String, Object>> tasks = PlannerState.tasks(state);
            int cur = PlannerState.cursor(state);
            if (cur < tasks.size()) {
                Map<String, Object> t = tasks.get(cur);
                if (PlannerState.DONE.equals(t.get(PlannerState.STATUS))
                        && PlannerState.str(t, PlannerState.SUMMARY).strip().isEmpty()) {
                    // 评审修正 3：模型中途放弃（无产出就停止调工具）不得静默放行。
                    Map<String, Object> failed = PlannerState.copyTask(t);
                    failed.put(PlannerState.STATUS, PlannerState.FAILED);
                    failed.put(PlannerState.SUMMARY, "输出为空");
                    tasks.set(cur, failed);
                } else if (minChars > 0 && PlannerState.DONE.equals(t.get(PlannerState.STATUS))
                        && PlannerState.str(t, PlannerState.SUMMARY).length() < minChars) {
                    Map<String, Object> failed = PlannerState.copyTask(t);
                    failed.put(PlannerState.STATUS, PlannerState.FAILED);
                    failed.put(PlannerState.SUMMARY, "输出不足 " + minChars + " 字");
                    tasks.set(cur, failed);
                }
            }
            int nextIdx = nextPending(tasks);
            PlanPersist.persistPlan(ctx, nodeCtx, tasks, nextIdx, PlannerState.replans(state));
            return Map.of(
                    PlannerState.TASKS, tasks,
                    PlannerState.CURSOR, nextIdx);
        };
    }

    /** replan 节点：LLM 重新拆解未完成项（保留已完成结果）。 */
    public static GraphBuilder.Node makeReplan(AgentModule.ModuleContext ctx) {
        int maxSubtasks = ctx.settings().planner.maxSubtasks;
        return (GraphState state, NodeContext nodeCtx) -> {
            List<Map<String, Object>> tasks = PlannerState.tasks(state);
            List<Map<String, Object>> done = new ArrayList<>();
            List<Map<String, Object>> failed = new ArrayList<>();
            for (Map<String, Object> t : tasks) {
                if (PlannerState.DONE.equals(t.get(PlannerState.STATUS))) {
                    done.add(t);
                } else if (PlannerState.FAILED.equals(t.get(PlannerState.STATUS))) {
                    failed.add(t);
                }
            }
            List<ChatMessage> messages = PlannerState.messages(state);
            String goal = PlannerState.lastHumanText(messages);
            if (goal.isEmpty()) {
                goal = "（未提供目标）";
            }
            String raw = invokeText(ctx, PlannerPrompts.replan(
                    goal,
                    summaryLines(done, REPLAN_SUMMARY_MAX_CHARS),
                    summaryLines(failed, REPLAN_SUMMARY_MAX_CHARS),
                    maxSubtasks));
            String fallbackGoal = failed.isEmpty()
                    ? goal
                    : PlannerState.str(failed.get(failed.size() - 1), PlannerState.GOAL);
            List<Map<String, Object>> newTasks = PlanNode.parseTasks(raw, maxSubtasks, fallbackGoal);
            // 保留已完成结果，追加新任务，总量仍受 max_subtasks 约束
            // （done 任务保留原 id；与 Python merged = [*done, *new_tasks][:max] 一致）。
            List<Map<String, Object>> merged = new ArrayList<>(done.size() + newTasks.size());
            merged.addAll(done);
            merged.addAll(newTasks);
            merged = new ArrayList<>(merged.subList(0, Math.min(merged.size(), maxSubtasks)));
            PlanStreaming.emitPlan(nodeCtx, merged, "replanned", "重规划为 " + merged.size() + " 个子任务");
            PlanPersist.persistPlan(ctx, nodeCtx, merged, done.size(), PlannerState.replans(state) + 1);
            return Map.of(
                    PlannerState.TASKS, merged,
                    PlannerState.CURSOR, done.size(),
                    PlannerState.REPLANS, PlannerState.replans(state) + 1);
        };
    }

    /** synthesize 节点：无工具的 best-effort 综合，emit PlanEvent(done)。 */
    public static GraphBuilder.Node makeSynthesize(AgentModule.ModuleContext ctx) {
        int maxTokens = ctx.settings().memory.contextMaxTokens;
        return (GraphState state, NodeContext nodeCtx) -> {
            // 保持无工具（决策 2）：不在 replan 预算之外开第二条执行路径。
            StringBuilder plan = new StringBuilder();
            for (Map<String, Object> t : PlannerState.tasks(state)) {
                if (plan.length() > 0) {
                    plan.append('\n');
                }
                String summary = PlannerState.str(t, PlannerState.SUMMARY);
                plan.append("#").append(PlannerState.intOf(t, PlannerState.ID))
                        .append(" [").append(PlannerState.str(t, PlannerState.STATUS)).append("] ")
                        .append(PlannerState.str(t, PlannerState.GOAL)).append("：")
                        .append(summary, 0, Math.min(summary.length(), SYNTH_SUMMARY_MAX_CHARS));
            }
            List<ChatMessage> input = new ArrayList<>();
            input.add(SystemMessage.from(PlannerPrompts.synth(plan.toString())));
            input.addAll(ContextEngine.buildModelInput(PlannerState.messages(state), maxTokens));
            AiMessage finalMessage = ctx.llm().stream(input, null, nodeCtx::emitDelta);
            PlanStreaming.emitPlan(nodeCtx, PlannerState.tasks(state), "done", null);
            PlanPersist.persistPlan(ctx, nodeCtx, PlannerState.tasks(state),
                    PlannerState.cursor(state), PlannerState.replans(state));
            if (finalMessage == null) {
                return Map.of();
            }
            return Map.<String, Object>of(GraphState.MESSAGES, List.of(finalMessage));
        };
    }

    /**
     * check 之后的路由：有 failed 且预算未耗尽 → replan；有 pending →
     * execute；否则 → synthesize。
     */
    public static GraphBuilder.Router routeAfterCheck(AgentModule.ModuleContext ctx) {
        int maxReplans = ctx.settings().planner.maxReplans;
        return (GraphState state) -> {
            List<Map<String, Object>> tasks = PlannerState.tasks(state);
            boolean hasFailed = false;
            boolean hasPending = false;
            for (Map<String, Object> t : tasks) {
                String status = PlannerState.str(t, PlannerState.STATUS);
                hasFailed |= PlannerState.FAILED.equals(status);
                hasPending |= PlannerState.PENDING.equals(status);
            }
            if (hasFailed && PlannerState.replans(state) < maxReplans) {
                return "replan";
            }
            if (hasPending) {
                return "execute";
            }
            return "synthesize";
        };
    }

    /** 第一个 pending 子任务的下标；无则列表长度（对应 Python next(...)）。 */
    private static int nextPending(List<Map<String, Object>> tasks) {
        for (int i = 0; i < tasks.size(); i++) {
            if (PlannerState.PENDING.equals(PlannerState.str(tasks.get(i), PlannerState.STATUS))) {
                return i;
            }
        }
        return tasks.size();
    }

    /** done/failed 的清单行：`#{id} {goal}：{summary[:150]}`，空清单为（无）。 */
    private static String summaryLines(List<Map<String, Object>> tasks, int maxChars) {
        StringBuilder sb = new StringBuilder();
        for (Map<String, Object> t : tasks) {
            if (sb.length() > 0) {
                sb.append('\n');
            }
            String summary = PlannerState.str(t, PlannerState.SUMMARY);
            sb.append("#").append(PlannerState.intOf(t, PlannerState.ID))
                    .append(" ").append(PlannerState.str(t, PlannerState.GOAL))
                    .append("：").append(summary, 0, Math.min(summary.length(), maxChars));
        }
        return sb.isEmpty() ? "（无）" : sb.toString();
    }

    /** 非流式单次调用（对应 Python replan 的 ainvoke）。 */
    private static String invokeText(AgentModule.ModuleContext ctx, String prompt) {
        AiMessage reply = ctx.llm().invoke(List.<ChatMessage>of(UserMessage.from(prompt)));
        String text = reply == null ? null : reply.text();
        return text == null ? "" : text;
    }
}
