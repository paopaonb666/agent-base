package io.github.paopaonb666.agentbase.modules.planner;

import io.github.paopaonb666.agentbase.core.AgentModule;
import io.github.paopaonb666.agentbase.core.graph.CompiledGraph;
import io.github.paopaonb666.agentbase.core.tools.AgentTool;
import io.github.paopaonb666.agentbase.modules.planner.graph.PlannerGraphs;

import java.util.List;

/**
 * {@code PlannerModule} 的 AgentModule 实现——plan-execute-replan 任务规划
 * 模块（M9）：拆解、逐条执行、预算内重规划。
 *
 * <p>Python 侧唯一自定义拓扑模块的 Java 移植：不走共享图工厂
 * （core/graphs），保留 escape hatch——直接使用 core.context 与图运行时
 * 原生 API 自行组装五节点拓扑（plan → execute → check →(replan)* →
 * synthesize），见 {@link PlannerGraphs}。
 */
public final class PlannerModule implements AgentModule {

    /** 稳定标识符；必须与 AGENT_MODULES 条目一致（registry 强制执行）。 */
    public static final String MODULE_NAME = "planner";

    @Override
    public String name() {
        return MODULE_NAME;
    }

    /** 给模型看的摘要（supervisor 路由依据）；与 Python module.py 逐字一致。 */
    @Override
    public String description() {
        return "把复杂请求拆解为子任务并逐条执行（plan-execute-replan）";
    }

    @Override
    public CompiledGraph buildGraph(AgentModule.ModuleContext ctx) {
        return PlannerGraphs.buildPlannerGraph(ctx);
    }

    /**
     * 复用共享工具池；本模块不贡献私有工具。
     */
    @Override
    public List<AgentTool> getTools() {
        return List.of();
    }
}
