package io.github.paopaonb666.agentbase.modules.planner.graph;

import io.github.paopaonb666.agentbase.core.AgentModule;
import io.github.paopaonb666.agentbase.core.ThreadId;
import io.github.paopaonb666.agentbase.core.graph.NodeContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Map;

/**
 * 计划快照的旁路持久化（M10 跨轮延续）——对应 Python 侧
 * {@code modules/planner/persist.py}。
 *
 * <p>planner 图在每个状态变更点（拆解 / check 推进 / 重规划 / 完成）把
 * tasks/cursor/replans upsert 进 plan_snapshots；plan 检查端点读表
 * 而非图状态——chat 轮穿插（chat 图的 checkpoint 只含 messages 通道）
 * 不再重置计划。
 *
 * <p>旁路语义：写失败只记日志，绝不影响图执行（与记忆审计同款约定）。
 * {@code planStore} 缺席（存储后端不可用）时退化为只写图状态，跨轮延续
 * 不可用但图行为不变。
 */
public final class PlanPersist {

    private static final Logger LOG = LoggerFactory.getLogger(PlanPersist.class);

    private PlanPersist() {}

    /** 把当前计划快照落表（旁路；失败只记日志）。 */
    public static void persistPlan(AgentModule.ModuleContext ctx, NodeContext nodeCtx,
                                   List<Map<String, Object>> tasks, int cursor, int replans) {
        AgentModule.PlanStorePort store = ctx.planStore();
        if (store == null) {
            return;
        }
        String threadId = nodeCtx.config().threadId();
        ThreadId parsed = ThreadId.tryParse(threadId);
        if (parsed == null) {
            return;
        }
        try {
            store.savePlan(threadId, parsed.module(), tasks, cursor, replans);
        } catch (Exception e) {
            LOG.warn("planner: 计划快照写入失败（不影响图执行）", e);
        }
    }
}
