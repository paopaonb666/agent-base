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
import java.util.regex.Matcher;
import java.util.regex.Pattern;

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

    /** 模型输出中的 JSON 数组（贪婪匹配首 [ 到末 ]，对应 Python re DOTALL）。 */
    private static final Pattern JSON_ARRAY = Pattern.compile("\\[.*\\]", Pattern.DOTALL);

    /** 解析失败反馈在重试 prompt 里引用的上次输出截断长度（raw[:200]）。 */
    private static final int RETRY_ECHO_MAX_CHARS = 200;

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private PlanNode() {}

    /** 构造 plan 节点（LangGraph 节点的统一签名：状态进、状态更新字典出）。 */
    public static GraphBuilder.Node make(AgentModule.ModuleContext ctx) {
        int maxSubtasks = ctx.settings().planner.maxSubtasks;
        return (GraphState state, NodeContext nodeCtx) -> {
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
            return Map.of(
                    PlannerState.TASKS, tasks,
                    PlannerState.CURSOR, 0,
                    PlannerState.REPLANS, 0);
        };
    }

    /** 非流式单次调用（对应 Python ainvoke）：prompt 作为一条 human 消息。 */
    private static String invokeText(AgentModule.ModuleContext ctx, String prompt) {
        AiMessage reply = ctx.llm().invoke(List.<ChatMessage>of(UserMessage.from(prompt)));
        String text = reply == null ? null : reply.text();
        return text == null ? "" : text;
    }

    /**
     * 模型输出 → 规范化 Task 列表；任何解析失败都降级为单任务。
     *
     * <p>对齐 Python {@code _parse_tasks}：正则抽 {@code [.*]}（贪婪到末
     * ]，DOTALL）；非数组 / 空数组 / 无可用 goal 都算失败；goals 截断到
     * {@code max_subtasks}；重排 id 1..n、全部 pending、空摘要、0 次尝试。
     */
    static List<Map<String, Object>> parseTasks(String raw, int maxSubtasks, String fallbackGoal) {
        List<String> goals = new ArrayList<>();
        try {
            Matcher matcher = JSON_ARRAY.matcher(raw == null ? "" : raw);
            if (!matcher.find()) {
                throw new IllegalStateException("no JSON array found");
            }
            JsonNode parsed = MAPPER.readTree(matcher.group(0));
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
