package io.github.paopaonb666.agentbase.modules.planner.graph;

import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.UserMessage;
import io.github.paopaonb666.agentbase.core.context.ContextEngine;
import io.github.paopaonb666.agentbase.core.graph.GraphState;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * planner 状态模型与跨图访问助手（决策 6）——对应 Python 侧
 * {@code modules/planner/state.py}。
 *
 * <p>planner 图可能恢复在 chat 写过的线程上（决策 5：图选择与线程命名空间
 * 解耦），checkpoint 里没有 tasks/cursor/replans 键。本模块内的强制约定：
 * <b>任何节点与路由函数都不得对非 messages 键做直读</b>，一律走这里的
 * 访问助手（{@code getOrDefault} 兜底默认值）。
 *
 * <p>状态通道（{@link GraphState} extras 的替换通道）：{@code tasks}
 * （子任务清单）、{@code cursor}（下一个待执行子任务的下标，check 负责推进）、
 * {@code replans}（已发生的重规划次数，预算 = planner.max_replans）。
 *
 * <p>决策 7：<b>任何状态变更都拷贝重建</b>（{@link #copyTasks} /
 * {@link #copyTask}），不得原地修改——与 Python 侧"不得原地修改 Task"
 * 的约定一致。
 */
public final class PlannerState {

    /** GraphState extras 通道名：子任务清单（List&lt;Map&gt;）。 */
    public static final String TASKS = "tasks";
    /** GraphState extras 通道名：下一个待执行子任务的下标（int）。 */
    public static final String CURSOR = "cursor";
    /** GraphState extras 通道名：已发生的重规划次数（int）。 */
    public static final String REPLANS = "replans";

    // Task dict 的键（对应 Python Task TypedDict）。
    public static final String ID = "id";
    public static final String GOAL = "goal";
    public static final String STATUS = "status";
    /** 子任务执行结果摘要（execute 结束时写入；check 的空输出门据此改判）。 */
    public static final String SUMMARY = "summary";
    public static final String ATTEMPTS = "attempts";

    // TaskStatus（Python Literal["pending", "running", "done", "failed"]）。
    public static final String PENDING = "pending";
    public static final String RUNNING = "running";
    public static final String DONE = "done";
    public static final String FAILED = "failed";

    private PlannerState() {}

    /** messages 通道（GraphState 的一等通道，无兜底问题）。 */
    public static List<ChatMessage> messages(GraphState state) {
        return state.messages();
    }

    /** 子任务清单（拷贝外层列表；元素在修改点再拷贝——决策 7）。 */
    public static List<Map<String, Object>> tasks(GraphState state) {
        List<Map<String, Object>> raw = state.getOrDefault(TASKS, List.<Map<String, Object>>of());
        return new ArrayList<>(raw == null ? List.<Map<String, Object>>of() : raw);
    }

    /** 下一个待执行子任务的下标（checkpoint 缺键时兜底 0）。 */
    public static int cursor(GraphState state) {
        Integer cursor = state.getOrDefault(CURSOR, 0);
        return cursor == null ? 0 : cursor;
    }

    /** 已发生的重规划次数（checkpoint 缺键时兜底 0）。 */
    public static int replans(GraphState state) {
        Integer replans = state.getOrDefault(REPLANS, 0);
        return replans == null ? 0 : replans;
    }

    /** Task 读取字符串键（null 兜底空串，对应 Python str(...) 的宽松读法）。 */
    public static String str(Map<String, Object> task, String key) {
        Object value = task.get(key);
        return value == null ? "" : String.valueOf(value);
    }

    /** Task 读取整数键（null / 非数兜底 0，对应 Python int(... or 0)）。 */
    public static int intOf(Map<String, Object> task, String key) {
        Object value = task.get(key);
        return value instanceof Number number ? number.intValue() : 0;
    }

    /** 规范化的新 Task：全部 pending、空摘要、0 次尝试（对应 Python _parse_tasks 的构造）。 */
    public static Map<String, Object> newTask(int id, String goal) {
        Map<String, Object> task = new LinkedHashMap<>();
        task.put(ID, id);
        task.put(GOAL, goal);
        task.put(STATUS, PENDING);
        task.put(SUMMARY, "");
        task.put(ATTEMPTS, 0);
        return task;
    }

    /** 拷贝单个 Task（决策 7：修改前先拷贝）。 */
    public static Map<String, Object> copyTask(Map<String, Object> task) {
        return new LinkedHashMap<>(task);
    }

    /** 深拷贝 Task 列表（节点返回前保证不与上层共享可变结构）。 */
    public static List<Map<String, Object>> copyTasks(List<Map<String, Object>> tasks) {
        List<Map<String, Object>> copy = new ArrayList<>(tasks.size());
        for (Map<String, Object> task : tasks) {
            copy.add(copyTask(task));
        }
        return copy;
    }

    /** 最后一条 human 消息的文本（兼容多模态 content blocks）；无则空串。 */
    public static String lastHumanText(List<ChatMessage> messages) {
        for (int i = messages.size() - 1; i >= 0; i--) {
            if (messages.get(i) instanceof UserMessage user) {
                return ContextEngine.contentText(user).strip();
            }
        }
        return "";
    }
}
