package io.github.paopaonb666.agentbase.core.graph;

/**
 * 审批暂停信号（A-#26）：工具节点在执行前发现命中危险名单且尚无决定的
 * 调用时抛出——图执行层接住后把状态带 pending_approval 标记落盘并以
 * PAUSED 终态收口，等审批端点决策后续跑（entryOverride 指回本节点）。
 * 携带暂停发生时的节点名（resume_node）与待审批调用的摘要（事件 detail）。
 */
public final class PauseForApprovalException extends RuntimeException {

    private final String resumeNode;
    private final String detail;

    public PauseForApprovalException(String resumeNode, String detail) {
        super("approval required: " + detail);
        this.resumeNode = resumeNode;
        this.detail = detail;
    }

    /** 续跑入口节点（抛出暂停的节点——工具执行在批准后从它继续）。 */
    public String resumeNode() {
        return resumeNode;
    }

    /** 事件 detail（pending_calls=N;tools=name1|name2 形态）。 */
    public String detail() {
        return detail;
    }
}
