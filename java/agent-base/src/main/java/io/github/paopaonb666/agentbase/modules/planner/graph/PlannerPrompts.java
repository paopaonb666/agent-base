package io.github.paopaonb666.agentbase.modules.planner.graph;

/**
 * planner 的中文提示词（B2）——对应 Python 侧
 * {@code modules/planner/prompts.py}，文案逐字复制（给模型看的 prompt
 * 是设计依据的一部分）。占位符按 Python {@code str.format} 的语义换成
 * {@code %s/%d}；PLAN_PROMPT 首行在 Python 源里是反斜杠续行（无换行），
 * 此处保持同一字面形态。
 */
public final class PlannerPrompts {

    private PlannerPrompts() {}

    /** 拆解子任务（plan / 首次与重试共用；重试时在末尾追加解析失败反馈）。 */
    public static String plan(int maxSubtasks, String goal) {
        return """
                你是任务规划器。把用户请求拆解为有序、可独立验证的子任务（最多 %d 条）。
                只输出 JSON 数组，不要输出任何其他文字，格式：
                [{"id": 1, "goal": "一句话目标"}, ...]
                简单请求输出单条。请求：%s""".formatted(maxSubtasks, goal);
    }

    /** 预算内重规划：只输出剩余需要执行的子任务，保留已完成结果不再规划。 */
    public static String replan(String goal, String done, String failed, int maxSubtasks) {
        return """
                你是任务规划器。一次多步执行中有子任务失败了，请重新规划剩余工作。
                总目标：%s

                已完成的子任务（不要重复规划）：
                %s

                失败的子任务及原因：
                %s

                请输出新的 JSON 数组，只包含**剩余需要执行**的子任务（最多 %d 条），格式：
                [{"id": 1, "goal": "一句话目标"}, ...]
                只输出 JSON 数组，不要输出任何其他文字。""".formatted(goal, done, failed, maxSubtasks);
    }

    /** 子任务执行（execute 节点的 SystemMessage）。 */
    public static String execute(String goal, String done, String current) {
        return """
                你正在执行一个多步计划中的一个子任务。
                总目标：%s
                已完成子任务及其结果：
                %s
                当前子任务：%s
                围绕当前子任务工作（可用工具）；完成后用一段话总结结果，不要再调用工具。"""
                .formatted(goal, done, current);
    }

    /** 最终综合（synthesize 节点的 SystemMessage；保持无工具）。 */
    public static String synth(String plan) {
        return """
                你是执行总结者。下面是一次多步执行的子任务清单与各自结果。
                综合它们给用户一个连贯的最终答复；如实标注未完成/失败的子任务，不要虚构。
                子任务清单：
                %s""".formatted(plan);
    }
}
