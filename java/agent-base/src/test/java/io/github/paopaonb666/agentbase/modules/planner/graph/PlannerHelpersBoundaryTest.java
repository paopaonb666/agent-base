package io.github.paopaonb666.agentbase.modules.planner.graph;

import dev.langchain4j.data.message.UserMessage;
import io.github.paopaonb666.agentbase.core.AgentModule;
import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.core.graph.GraphState;
import io.github.paopaonb666.agentbase.core.graph.NodeContext;
import io.github.paopaonb666.agentbase.core.graph.RunConfig;
import io.github.paopaonb666.agentbase.extensions.events.Events;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A-#304 planner 支撑件边界测试三件：{@link PlannerPrompts}（占位替换/
 * 括号内容不被注入文本破坏）、{@link PlanPersist}（planStore 缺席 no-op、
 * 记录形状）、{@link PlanStreaming}（detail 空→null 的 wire 约定、空清单）。
 * 图级行为由 PlannerGraphTest 覆盖，本类只钉支撑件的边界。
 */
class PlannerHelpersBoundaryTest {

    // ── PlannerPrompts ───────────────────────────────────────────────────

    @Test
    void promptsSubstituteGoalAndBudgetWithoutTemplateBreakage() {
        // 目标含大括号/JSON 形态/换行——占位替换后模板结构不被破坏。
        String goal = "处理 {\"a\": 1} 与\n多行目标";
        String plan = PlannerPrompts.plan(4, goal);
        assertTrue(plan.contains(goal), "plan 含原 goal");
        assertTrue(plan.contains("4"), "plan 含子任务上限");
        // replan 三段摘要 + 预算；execute 三段；synth/verdict 同构。
        assertTrue(PlannerPrompts.replan(goal, "已完成A", "失败B", 3).contains("失败B"));
        assertTrue(PlannerPrompts.execute(goal, "done-A", "当前B").contains("当前B"));
        assertTrue(PlannerPrompts.synth("计划文本").contains("计划文本"));
        assertTrue(PlannerPrompts.verdict(goal, "输出摘要", 200).contains("输出摘要"));
        // 空串 goal 不炸（退化占位）。
        assertTrue(PlannerPrompts.plan(2, "").contains("2"));
    }

    // ── PlanPersist ──────────────────────────────────────────────────────

    /** 记录 emit 事件的 NodeContext 桩（emit/config/cancelled/tools；其余默认）。 */
    private static NodeContext recordingCtx(List<Object> events, RunConfig config) {
        return new NodeContext() {
            @Override public void emit(Object event) {
                events.add(event);
            }

            @Override public RunConfig config() {
                return config;
            }

            @Override public io.github.paopaonb666.agentbase.core.tools.ToolPool tools() {
                return null;
            }

            @Override public String callTool(String name, String argsJson) {
                return "";
            }

            @Override public boolean cancelled() {
                return false;
            }
        };
    }

    @Test
    void persistPlanRecordsViaPlanStoreAndSurvivesAbsentStore() {
        // planStore 在场：savePlan 记录 (tasks/cursor/replans)。
        List<Object> events = new ArrayList<>();
        var planStore = new AgentModule.PlanStorePort() {
            final List<List<Map<String, Object>>> saves = new ArrayList<>();

            @Override public void savePlan(String threadId, String module,
                    List<Map<String, Object>> tasks, int cursor, int replans) {
                saves.add(tasks);
            }

            @Override public Map<String, Object> getPlan(String threadId) {
                return null;
            }

            @Override public void deletePlan(String threadId) {
            }
        };
        var moduleCtx = new AgentModule.ModuleContext(new Settings(), null, null,
                List.of(), null, planStore);
        List<Map<String, Object>> tasks = PlannerState.copyTasks(List.of(
                PlannerState.newTask(1, "任务一")));
        PlanPersist.persistPlan(moduleCtx, recordingCtx(events,
                RunConfig.of("planner:t1", "u1")), tasks, 0, 0);
        assertEquals(1, planStore.saves.size());
        assertEquals("任务一", planStore.saves.get(0).get(0).get(PlannerState.GOAL));

        // planStore 缺席 → no-op 不炸（M10 读表面缺席 = 计划延续停用）。
        var storeless = new AgentModule.ModuleContext(new Settings(), null, null,
                List.of(), null, null);
        PlanPersist.persistPlan(storeless, recordingCtx(events,
                RunConfig.of("planner:t1", "u1")), tasks, 1, 0);
        assertEquals(1, planStore.saves.size(), "缺席不追加");
    }

    // ── PlanStreaming ────────────────────────────────────────────────────

    @Test
    void emitPlanRendersFullTaskListAndNullEmptyDetail() {
        List<Object> events = new ArrayList<>();
        NodeContext nodeCtx = recordingCtx(events, RunConfig.of("planner:t1", "u1"));
        List<Map<String, Object>> tasks = PlannerState.copyTasks(List.of(
                PlannerState.newTask(1, "A"),
                PlannerState.newTask(2, "B")));
        tasks.get(1).put(PlannerState.STATUS, PlannerState.DONE);
        PlanStreaming.emitPlan(nodeCtx, tasks, "progress", "子任务 2 done");
        assertEquals(1, events.size());
        Events.PlanEvent event = (Events.PlanEvent) events.get(0);
        assertEquals("progress", event.status());
        assertEquals("子任务 2 done", event.detail());
        assertEquals(2, event.plan().size());
        assertEquals("pending", event.plan().get(0).status());
        assertEquals("done", event.plan().get(1).status());

        // 空 detail → null（wire 与 Python model_dump_json 一致）。
        PlanStreaming.emitPlan(nodeCtx, tasks, "created", "");
        Events.PlanEvent nullDetail = (Events.PlanEvent) events.get(1);
        assertNull(nullDetail.detail());
        // 空清单 → 空计划事件（不炸）。
        PlanStreaming.emitPlan(nodeCtx, List.of(), "done", null);
        assertEquals(0, ((Events.PlanEvent) events.get(2)).plan().size());
    }
}
