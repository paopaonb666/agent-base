package io.github.paopaonb666.agentbase.core.tools;

import java.util.concurrent.ConcurrentHashMap;

/**
 * 工具审批决定账本（A-#26）：threadId+provider call_id → 批准/拒绝。
 * 静态进程级（与 THREAD_LOCKS 同口径——单进程部署语义）；键用 provider
 * 侧 tool_call id（快照悬挂 AiMessage 的 id 在续跑时稳定，审计 callId
 * 不是）。决定带时间戳，消费方按 TTL 判过期（过期=视同拒绝，提示文本
 * 说明超时）；决定被消费（工具执行或回喂拒绝文本）后由消费方清除。
 */
public final class ApprovalLedger {

    /** 决定状态（消费方视角）。 */
    public enum Status {
        /** 无决定（调用方不该执行——pause 分支的前提）。 */
        ABSENT,
        /** 已批准：正常执行。 */
        APPROVED,
        /** 已拒绝：回喂拒绝文本，不执行。 */
        REJECTED,
        /** 决定超时未被消费：视同拒绝（提示文本说明超时）。 */
        EXPIRED
    }

    public record Decision(boolean approved, long decidedAtMillis) {
    }

    private static final ConcurrentHashMap<String, Decision> LEDGER = new ConcurrentHashMap<>();

    private ApprovalLedger() {
    }

    private static String key(String threadId, String callId) {
        return threadId + "|" + callId;
    }

    /** 记录审批决定（审批端点调用；重复决定覆盖——最后一次为准）。 */
    public static void record(String threadId, String callId, boolean approved) {
        LEDGER.put(key(threadId, callId), new Decision(approved, System.currentTimeMillis()));
    }

    /** 测试注记形态：显式决定时刻（超时路径的确定性验证）。 */
    public static void recordAt(String threadId, String callId, boolean approved,
                                long decidedAtMillis) {
        LEDGER.put(key(threadId, callId), new Decision(approved, decidedAtMillis));
    }

    /**
     * 查决定状态：无记录=ABSENT；记录过期（{@code ttlSeconds > 0} 且超过
     * 有效期）返回 EXPIRED 并清除记录；否则 APPROVED/REJECTED。
     */
    public static Status status(String threadId, String callId, long ttlSeconds) {
        return status(threadId, callId, ttlSeconds, true);
    }

    /**
     * 查决定状态，{@code consumeExpired} 决定 EXPIRED 时是否同时清除记录。
     * 消费方（hintText / ExecuteNode 的回喂与执行点）用 true——过期即消费；
     * 判定方用 false：同一门判定里 pauseReason 先问、hintText 后问，判定
     * 阶段若把过期记录提前清掉，回喂文本只能落到 ABSENT 分支（GUI 验收
     * 二轮 D6：「缺少审批决定」冒充「审批已超时」，模型收到错误归因）。
     */
    public static Status status(String threadId, String callId, long ttlSeconds,
                                boolean consumeExpired) {
        Decision decision = LEDGER.get(key(threadId, callId));
        if (decision == null) {
            return Status.ABSENT;
        }
        if (ttlSeconds > 0
                && System.currentTimeMillis() - decision.decidedAtMillis() > ttlSeconds * 1000L) {
            if (consumeExpired) {
                LEDGER.remove(key(threadId, callId));
            }
            return Status.EXPIRED;
        }
        return decision.approved() ? Status.APPROVED : Status.REJECTED;
    }

    /** 决定已被消费（执行或回喂拒绝文本）后清除。 */
    public static void clear(String threadId, String callId) {
        LEDGER.remove(key(threadId, callId));
    }
}
