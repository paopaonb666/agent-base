package io.github.paopaonb666.agentbase.modules.planner.graph;

import io.github.paopaonb666.agentbase.core.AgentModule;
import io.github.paopaonb666.agentbase.core.graph.CompiledGraph;
import io.github.paopaonb666.agentbase.core.graph.GraphBuilder;
import io.github.paopaonb666.agentbase.modules.planner.PlannerModule;

import java.util.Set;

/**
 * planner 模块的图构建（M9）：plan → execute → check →(replan)* →
 * synthesize——对应 Python 侧 {@code modules/planner/graph.py}。
 *
 * <p>五节点拓扑（决策 1）：execute 内联 ReAct 循环，外层图每子任务恒定
 * 2 个 super-step；完成判定用规则（轮数耗尽 / 空输出门），不用额外 LLM
 * verdict；replan 有预算上限（{@code PLANNER_MAX_REPLANS}），耗尽后走
 * best-effort synthesize。以 {@code name="planner"} 编译（supervisor 把
 * 图作为 sub-agent 编排时按模块名识别）。
 */
public final class PlannerGraphs {

    private PlannerGraphs() {}

    /** 构建并编译 planner 图（以 name="planner" 编译）。 */
    public static CompiledGraph buildPlannerGraph(AgentModule.ModuleContext ctx) {
        return GraphBuilder.begin()
                .addNode("plan", PlanNode.make(ctx))
                .addNode("execute", ExecuteNode.make(ctx))
                .addNode("check", CheckNodes.makeCheck(ctx))
                .addNode("replan", CheckNodes.makeReplan(ctx))
                .addNode("synthesize", CheckNodes.makeSynthesize(ctx))
                .entry("plan")
                .addEdge("plan", "execute")
                .addEdge("execute", "check")
                // 路由：有 failed 且预算未耗尽 → replan；有 pending → execute；
                // 否则 → synthesize。#202：封闭目标集显式声明，compile 校验笔误。
                .addConditionalEdges("check", CheckNodes.routeAfterCheck(ctx),
                        Set.of("replan", "execute", "synthesize"))
                .addEdge("replan", "execute")
                .addEdge("synthesize", GraphBuilder.END)
                .compile(ctx.checkpointer(), PlannerModule.MODULE_NAME);
    }
}
