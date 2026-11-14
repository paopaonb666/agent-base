package io.github.paopaonb666.agentbase.core.graphs;

import dev.langchain4j.agent.tool.ToolExecutionRequest;
import io.github.paopaonb666.agentbase.core.tools.ApprovalLedger;

import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * 危险工具审批门（A-#26）判定单源：名单解析、暂停判定与拒绝/超时回喂
 * 文本都在这里（Graphs 的 tools 节点与 planner ExecuteNode 两接线点共用，
 * 与 ToolLoopDetector 的节点层单源先例同型）。
 *
 * <p>流程：工具节点执行前先 {@link #pauseReason}——任一请求命中名单且
 * 无有效决定 → 抛 {@code PauseForApprovalException}（run 暂停落盘，等审批）
 * ；有决定的请求按 {@link #hintText} 分桶：APPROVED → 正常执行（null）、
 * REJECTED/EXPIRED → 回喂提示文本不执行（循环拦截槽位同型）。决定被消费
 * （执行或回喂）后清账本。
 */
public final class ToolApprovalGate {

    private ToolApprovalGate() {
    }

    /** 名单解析：{@code "off"} 或逗号分隔工具名 → 小写集合（off=空集）。 */
    public static Set<String> parseDangerous(String mode) {
        if (mode == null || mode.isBlank() || "off".equalsIgnoreCase(mode.trim())) {
            return Set.of();
        }
        return Set.of(mode.trim().toLowerCase(Locale.ROOT).split("\\s*,\\s*"));
    }

    /** 请求是否命中危险名单。 */
    public static boolean isDangerous(ToolExecutionRequest request, Set<String> dangerous) {
        String name = request.name() == null ? "" : request.name().toLowerCase(Locale.ROOT);
        return dangerous.contains(name);
    }

    /**
     * 暂停判定：任一请求命中名单且无有效决定（ABSENT——EXPIRED 视同已
     * 拒绝，不暂停）→ 返回暂停 detail（pending_calls=N;tools=a|b），否则
     * null。判定在<b>任何执行之前</b>：一个未决调用即整批暂停（无副作用的
     * 调用也不跑——避免「批一半」的不可回退状态）。
     */
    public static String pauseReason(List<ToolExecutionRequest> requests, Set<String> dangerous,
                                     String threadId, long ttlSeconds) {
        if (dangerous.isEmpty()) {
            return null;
        }
        int pending = 0;
        StringBuilder names = new StringBuilder();
        for (ToolExecutionRequest request : requests) {
            if (!isDangerous(request, dangerous)) {
                continue;
            }
            // 判定不消费（D6）：EXPIRED 记录留待同一次门判定里 hintText
            // 消费——否则回喂文本落到 ABSENT 分支（「缺少审批决定」冒充
            // 「审批已超时」，模型收到错误归因）。
            ApprovalLedger.Status status =
                    ApprovalLedger.status(threadId, request.id(), ttlSeconds, false);
            if (status == ApprovalLedger.Status.ABSENT) {
                pending++;
                if (names.length() > 0) {
                    names.append('|');
                }
                names.append(request.name() == null ? "" : request.name());
            }
        }
        return pending == 0 ? null
                : "pending_calls=" + pending + ";tools=" + names;
    }

    /**
     * 决定分桶（在 {@link #pauseReason} 通过后逐请求调用）：APPROVED →
     * null（照常执行）；REJECTED → 拒绝提示；EXPIRED → 超时提示——两种
     * 提示都清账本并回喂（循环拦截槽位同型：配对完整、协议安全）。
     */
    public static String hintText(ToolExecutionRequest request, Set<String> dangerous,
                                  String threadId, long ttlSeconds) {
        if (!isDangerous(request, dangerous)) {
            return null;
        }
        String callId = request.id();
        ApprovalLedger.Status status = ApprovalLedger.status(threadId, callId, ttlSeconds);
        switch (status) {
            case APPROVED -> {
                return null;
            }
            case REJECTED -> {
                ApprovalLedger.clear(threadId, callId);
                return "该工具调用（%s）已被用户拒绝，未执行。请调整方案后继续，"
                        .formatted(request.name()) + "不要重复发起相同调用。";
            }
            case EXPIRED -> {
                ApprovalLedger.clear(threadId, callId);
                return "该工具调用（%s）的审批已超时（未在有效期内收到决定），"
                        .formatted(request.name()) + "未执行。请调整方案后继续。";
            }
            default -> {
                // ABSENT 走不到这里（pauseReason 已拦截整批）——防御性视同拒绝。
                return "该工具调用（%s）缺少审批决定，未执行。".formatted(request.name());
            }
        }
    }
}
