package io.github.paopaonb666.agentbase.core.graph;

/**
 * 图执行配置（对应 Python 侧 RunnableConfig.configurable 的受控子集）。
 *
 * <p>threadId 是带模块前缀的命名空间形态（{@code module:raw}），与
 * checkpointer 键、thread_index、记忆 agent_id 的约定共用 ThreadId 协议。
 *
 * @param entryOverride 入口覆盖（A-#26/#8）：审批暂停续跑时指向快照
 *                      pending_approval.resume_node（如 tools 节点）——
 *                      非空时运行从该节点开始而非图入口；null = 既有口径。
 */
public record RunConfig(String threadId, String userId, int recursionLimit, String entryOverride) {

    public RunConfig {
        if (threadId == null || threadId.isEmpty()) {
            throw new IllegalArgumentException("RunConfig.threadId 不能为空");
        }
        if (recursionLimit < 1) {
            throw new IllegalArgumentException("RunConfig.recursionLimit 必须 >= 1");
        }
    }

    /** 三参便捷构造（entryOverride=null——既有调用方/测试的旧口径）。 */
    public RunConfig(String threadId, String userId, int recursionLimit) {
        this(threadId, userId, recursionLimit, null);
    }

    /** 默认 recursion_limit 由调用方传入；本工厂用于测试快捷构造。 */
    public static RunConfig of(String threadId, String userId) {
        return new RunConfig(threadId, userId, 25);
    }
}
