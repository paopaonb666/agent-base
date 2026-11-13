package io.github.paopaonb666.agentbase.modules.planner.graph;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.UserMessage;
import io.github.paopaonb666.agentbase.core.AgentModule;
import io.github.paopaonb666.agentbase.core.graph.GraphBuilder;
import io.github.paopaonb666.agentbase.core.graph.GraphState;
import io.github.paopaonb666.agentbase.core.graph.NodeContext;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * plan 节点（B2）：把用户目标拆解为有界 JSON 子任务清单——对应 Python 侧
 * {@code modules/planner/nodes_plan.py}。
 *
 * <p>解析（{@link #parseTasks}）容忍模型输出的常见噪声（```json 围栏、前后
 * 散文），规范化为 Task 列表（重排 id 1..n、截断到 max_subtasks、全部
 * pending）；任何解析失败都降级为单任务——规划失败不能让整轮对话失败。
 *
 * <p>对齐说明：Python 侧 {@code _parse_tasks} 从不返回空列表（失败即降级
 * 单任务），因此 {@code if not tasks} 的重试分支实际不可达；Java 侧按同一
 * 结构移植（保留 {@code if (tasks.isEmpty())} 重试分支），语义以源为准。
 *
 * <p>与 Python 侧一致：本节点用<b>非流式</b>调用（{@code ainvoke} →
 * {@code ModelRuntime.invoke}），不向 UI 发 token 增量。
 */
public final class PlanNode {

    /** 解析失败反馈在重试 prompt 里引用的上次输出截断长度（raw[:200]）。 */
    private static final int RETRY_ECHO_MAX_CHARS = 200;

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private PlanNode() {}

    /** 构造 plan 节点（LangGraph 节点的统一签名：状态进、状态更新字典出）。 */
    public static GraphBuilder.Node make(AgentModule.ModuleContext ctx) {
        int maxSubtasks = ctx.settings().planner.maxSubtasks;
        // A-#62：计划审批开关在构造期固化（off=默认，运行期零开销）。
        boolean planApproval = ctx.settings().planner.planApproval;
        return (GraphState state, NodeContext nodeCtx) -> {
            // #66 跨轮延续：快照已注入 tasks（continue_plan=true）→ 跳过
            // 重拆，从注入的 cursor 续跑；plan 事件 status=progress。
            List<Map<String, Object>> seeded = PlannerState.tasks(state);
            if (!seeded.isEmpty()) {
                PlanStreaming.emitPlan(nodeCtx, seeded, "progress",
                        "延续既有计划（cursor=" + PlannerState.cursor(state) + "）");
                return Map.of(
                        PlannerState.TASKS, seeded,
                        PlannerState.CURSOR, PlannerState.cursor(state),
                        PlannerState.REPLANS, PlannerState.replans(state));
            }
            List<ChatMessage> messages = PlannerState.messages(state);
            String goal = PlannerState.lastHumanText(messages);
            if (goal.isEmpty()) {
                goal = "（未提供目标）";
            }
            String raw = invokeText(ctx, PlannerPrompts.plan(maxSubtasks, goal));
            List<Map<String, Object>> tasks = parseTasks(raw, maxSubtasks, goal);
            if (tasks.isEmpty()) {
                // 第一次解析失败：带错误反馈重试一次（见类注释：parseTasks
                // 失败即降级单任务，本分支与 Python 源同构、实际不可达）。
                String retryPrompt = PlannerPrompts.plan(maxSubtasks, goal)
                        + "\n\n你上次的输出无法解析为 JSON 数组："
                        + raw.substring(0, Math.min(raw.length(), RETRY_ECHO_MAX_CHARS));
                tasks = parseTasks(invokeText(ctx, retryPrompt), maxSubtasks, goal);
            }
            PlanStreaming.emitPlan(nodeCtx, tasks, "created", "拆解出 " + tasks.size() + " 个子任务");
            PlanPersist.persistPlan(ctx, nodeCtx, tasks, 0, 0);
            Map<String, Object> update = new java.util.LinkedHashMap<>();
            update.put(PlannerState.TASKS, tasks);
            update.put(PlannerState.CURSOR, 0);
            update.put(PlannerState.REPLANS, 0);
            if (planApproval) {
                // A-#62 marker 型暂停：tasks 已在 update 里（图层 apply 后
                // 落盘 PAUSED）——审批端点 POST .../approvals/plan 决定，
                // resume 以 entryOverride=execute 续跑。已有批准决定 → 消费
                // 清账本照常执行（拒绝决定在 execute 节点消费）。
                if (io.github.paopaonb666.agentbase.core.tools.ApprovalLedger
                        .status(nodeCtx.config().threadId(), PLAN_CALL_ID, 0)
                        == io.github.paopaonb666.agentbase.core.tools.ApprovalLedger.Status.APPROVED) {
                    io.github.paopaonb666.agentbase.core.tools.ApprovalLedger
                            .clear(nodeCtx.config().threadId(), PLAN_CALL_ID);
                } else {
                    update.put("pending_approval",
                            Map.of("resume_node", "execute", "detail", "plan"));
                }
            }
            return update;
        };
    }

    /** 非流式单次调用（对应 Python ainvoke）：prompt 作为一条 human 消息。 */
    private static String invokeText(AgentModule.ModuleContext ctx, String prompt) {
        AiMessage reply = ctx.llm().invoke(List.<ChatMessage>of(UserMessage.from(prompt)));
        String text = reply == null ? null : reply.text();
        return text == null ? "" : text;
    }

    /** A-#62：计划审批决定在账本里的键（与工具 call_id 命名空间隔离）。 */
    public static final String PLAN_CALL_ID = "plan";

    /**
     * 模型输出 → 规范化 Task 列表；任何解析失败都降级为单任务。
     *
     * <p>抽取用平衡括号扫描（#258，{@code JsonBlocks.firstBalancedArray}）：
     * 取<b>首个</b>完整 JSON 数组、字符串字面量内的括号不计数——贪婪
     * DOTALL 正则（{@code \[.*\]}）会把「多 JSON 块 + 尾注」的输出误取到
     * 最后一段括号。非数组 / 空数组 / 无可用 goal 都算失败；goals 截断到
     * {@code max_subtasks}；重排 id 1..n、全部 pending、空摘要、0 次尝试。
     */
    static List<Map<String, Object>> parseTasks(String raw, int maxSubtasks, String fallbackGoal) {
        List<String> goals = new ArrayList<>();
        try {
            String block = io.github.paopaonb666.agentbase.core.text.JsonBlocks
                    .firstBalancedArray(raw == null ? "" : raw);
            if (block == null) {
                throw new IllegalStateException("no JSON array found");
            }
            JsonNode parsed = MAPPER.readTree(block);
            if (!parsed.isArray() || parsed.isEmpty()) {
                throw new IllegalStateException("JSON array empty or not a list");
            }
            for (JsonNode item : parsed) {
                if (item.isObject()) {
                    String goal = item.path("goal").asText("").strip();
                    if (!goal.isEmpty()) {
                        goals.add(goal);
                    }
                }
            }
            if (goals.isEmpty()) {
                throw new IllegalStateException("no usable goals");
            }
        } catch (Exception e) {
            // ValueError / TypeError / JSONDecodeError / AttributeError 同款
            // 降级：以目标兜底单任务，规划失败不拖垮整轮对话。
            List<Map<String, Object>> fallback = new ArrayList<>();
            fallback.add(PlannerState.newTask(1, fallbackGoal));
            return fallback;
        }
        List<String> bounded = goals.subList(0, Math.min(goals.size(), maxSubtasks));
        List<Map<String, Object>> tasks = new ArrayList<>(bounded.size());
        for (int i = 0; i < bounded.size(); i++) {
            tasks.add(PlannerState.newTask(i + 1, bounded.get(i)));
        }
        return tasks;
    }
}
