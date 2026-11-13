package io.github.paopaonb666.agentbase.modules.planner.graph;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * #96 场景 eval · planner 线（smoke 集，按计划裁决缩水为任务分解 golden）：
 * 真实场景 goal → 规划 LLM 输出（脚本）→ {@link PlanNode#parseTasks} 分解
 * 契约——多任务有序编号、上界截断、降级单任务三态。端到端图执行路径由
 * {@code PlannerGraphTest} 四条执行路径覆盖，本类只钉「任务分解」这一层
 * 的场景级 golden（3-5 条 smoke，为后续 planner 量化扩线提供地基）。
 */
class PlanDecompositionSmokeTest {

    private static final int MAX_SUBTASKS = 4;

    @Test
    void scenarioPlanDecomposesIntoOrderedTasks() {
        // 真实场景：迁移项目的规划输出 → 三个有序子任务。
        String raw = """
                ```json
                [
                  {"goal": "梳理现有库表结构并输出依赖清单"},
                  {"goal": "在新库搭建同构 schema 并演练迁移脚本"},
                  {"goal": "灰度切换读写流量并观察核心指标"}
                ]
                ```""";
        List<Map<String, Object>> tasks = PlanNode.parseTasks(raw, MAX_SUBTASKS, "兜底目标");
        assertEquals(3, tasks.size());
        for (int i = 0; i < 3; i++) {
            assertEquals(i + 1, tasks.get(i).get(PlannerState.ID));
            assertEquals("pending", tasks.get(i).get(PlannerState.STATUS));
        }
        assertEquals("梳理现有库表结构并输出依赖清单", tasks.get(0).get(PlannerState.GOAL));
        assertEquals("在新库搭建同构 schema 并演练迁移脚本", tasks.get(1).get(PlannerState.GOAL));
        assertEquals("灰度切换读写流量并观察核心指标", tasks.get(2).get(PlannerState.GOAL));
    }

    @Test
    void scenarioPlanRespectsMaxSubtasksCap() {
        // 模型超发 6 条 → 截到上界 4（#259 截断审计在节点层，这里钉分解层）。
        StringBuilder raw = new StringBuilder("[");
        for (int i = 1; i <= 6; i++) {
            if (i > 1) {
                raw.append(',');
            }
            raw.append("{\"goal\": \"步骤").append(i).append("\"}");
        }
        raw.append(']');
        List<Map<String, Object>> tasks = PlanNode.parseTasks(raw.toString(), MAX_SUBTASKS, "兜底");
        assertEquals(4, tasks.size(), "超发截断到 maxSubtasks");
        assertEquals("步骤4", tasks.get(3).get(PlannerState.GOAL));
    }

    @Test
    void scenarioGarbagePlanFallsBackToSingleGoalTask() {
        // 规划 LLM 输出不可解析（纯文本/空数组/无 goal 键）→ 以原 goal 兜底
        // 单任务（规划失败不拖垮整轮对话）。
        for (String garbage : new String[]{"我不会输出 JSON", "[]", "[{\"title\": \"x\"}]", ""}) {
            List<Map<String, Object>> tasks = PlanNode.parseTasks(garbage, MAX_SUBTASKS, "迁移数据库");
            assertEquals(1, tasks.size(), "输入=" + garbage);
            assertEquals(1, tasks.get(0).get(PlannerState.ID));
            assertEquals("迁移数据库", tasks.get(0).get(PlannerState.GOAL));
            assertEquals("pending", tasks.get(0).get(PlannerState.STATUS));
        }
    }

    @Test
    void scenarioSingleTaskPlanPassesThrough() {
        // 简单诉求：单条规划 → 单任务原样（不强行拆分）。
        List<Map<String, Object>> tasks = PlanNode.parseTasks(
                "[{\"goal\": \"查询当前服务版本号\"}]", MAX_SUBTASKS, "兜底");
        assertEquals(1, tasks.size());
        assertEquals("查询当前服务版本号", tasks.get(0).get(PlannerState.GOAL));
    }
}
