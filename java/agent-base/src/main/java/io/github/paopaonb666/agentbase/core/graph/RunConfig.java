package io.github.paopaonb666.agentbase.core.graph;

/**
 * 图执行配置（对应 Python 侧 RunnableConfig.configurable 的受控子集）。
 *
 * <p>threadId 是带模块前缀的命名空间形态（{@code module:raw}），与
 * checkpointer 键、thread_index、记忆 agent_id 的约定共用 ThreadId 协议。
 */
public record RunConfig(String threadId, String userId, int recursionLimit) {

    public RunConfig {
        if (threadId == null || threadId.isEmpty()) {
            throw new IllegalArgumentException("RunConfig.threadId 不能为空");
        }
        if (recursionLimit < 1) {
            throw new IllegalArgumentException("RunConfig.recursionLimit 必须 >= 1");
        }
    }

    /** 默认 recursion_limit 由调用方传入；本工厂用于测试快捷构造。 */
    public static RunConfig of(String threadId, String userId) {
        return new RunConfig(threadId, userId, 25);
    }
}
