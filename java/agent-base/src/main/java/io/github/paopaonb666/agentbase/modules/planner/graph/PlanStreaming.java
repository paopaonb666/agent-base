package io.github.paopaonb666.agentbase.modules.planner.graph;

import io.github.paopaonb666.agentbase.core.graph.NodeContext;
import io.github.paopaonb666.agentbase.extensions.events.Events;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * planner 向 UI 发计划事件（B5/M9）——对应 Python 侧
 * {@code modules/planner/streaming.py}。载荷与
 * {@link Events.PlanEvent} 一一对应，服务器侧按该模型校验后透传。
 *
 * <p>尽力而为的旁路信号：上下文缺失 / 写入失败都静默放弃（Python 侧
 * {@code contextlib.suppress(Exception)} + {@code get_stream_writer} 的
 * 容错语义；Java 侧 {@code NodeContext.emit} 自带同样的容错——队列满 /
 * 取消时静默丢弃，不中断节点执行）。
 */
public final class PlanStreaming {

    private PlanStreaming() {}

    /**
     * 发送全量任务清单（前端整表替换）。status 见 PlanEvent 契约：
     * created（拆解完成）/ progress（子任务状态变化）/ replanned（重规划）/
     * done（执行结束）。detail 为空时按 null 处理（线上格式保留 null 字段，
     * 与 Python Pydantic {@code model_dump_json} 一致）。
     */
    public static void emitPlan(NodeContext nodeCtx, List<Map<String, Object>> tasks,
                                String status, String detail) {
        List<Events.PlanTask> plan = new ArrayList<>(tasks.size());
        for (Map<String, Object> task : tasks) {
            plan.add(new Events.PlanTask(
                    PlannerState.intOf(task, PlannerState.ID),
                    PlannerState.str(task, PlannerState.GOAL),
                    PlannerState.str(task, PlannerState.STATUS)));
        }
        nodeCtx.emit(new Events.PlanEvent(status, plan,
                detail == null || detail.isEmpty() ? null : detail));
    }
}
