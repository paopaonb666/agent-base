package io.github.paopaonb666.agentbase.core.graphs;

import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.output.TokenUsage;
import io.github.paopaonb666.agentbase.core.AgentModule;
import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.core.graph.CompiledGraph;
import io.github.paopaonb666.agentbase.core.graph.GraphState;
import io.github.paopaonb666.agentbase.core.graph.InMemoryCheckpointer;
import io.github.paopaonb666.agentbase.core.graph.RunConfig;
import io.github.paopaonb666.agentbase.core.llm.ModelRuntime;
import io.github.paopaonb666.agentbase.core.tools.AgentTool;
import io.github.paopaonb666.agentbase.core.tools.ApprovalLedger;
import io.github.paopaonb666.agentbase.core.tools.ToolPool;
import org.junit.jupiter.api.Test;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A-#26 危险工具审批门：名单解析/暂停判定/决定分桶的单元语义 + 图级
 * e2e（暂停落盘→批准续跑执行 / 拒绝回喂提示 / 超时视同拒绝）。e2e 走
 * {@code Graphs.buildSingleAgentGraph} 的真实 tools 节点（脚本化 LLM +
 * 计数桩工具）。
 */
class ToolApprovalGateTest {

    private static ToolExecutionRequest call(String id, String tool) {
        return ToolExecutionRequest.builder().id(id).name(tool).arguments("{}").build();
    }

    // ── 单元语义 ─────────────────────────────────────────────────────────

    @Test
    void parseDangerousOffIsEmptyAndListIsLowercased() {
        assertTrue(ToolApprovalGate.parseDangerous("off").isEmpty());
        assertTrue(ToolApprovalGate.parseDangerous(null).isEmpty());
        assertTrue(ToolApprovalGate.parseDangerous("").isEmpty());
        assertEquals(java.util.Set.of("deploy_prod", "send_mail"),
                ToolApprovalGate.parseDangerous("Deploy_Prod, send_mail"));
    }

    @Test
    void pauseReasonTriggersOnlyOnUndecidedDangerousCalls() {
        var dangerous = ToolApprovalGate.parseDangerous("deploy_prod");
        // 未命中名单 → 不暂停。
        assertNull(ToolApprovalGate.pauseReason(List.of(call("p1", "search_web")),
                dangerous, "pt", 600));
        // 命中且无决定 → 暂停 detail（线程 id 独立——静态账本跨方法共享）。
        String reason = ToolApprovalGate.pauseReason(List.of(call("p2", "deploy_prod")),
                dangerous, "pt", 600);
        assertEquals("pending_calls=1;tools=deploy_prod", reason);
        // 已批准/已拒绝/已过期 → 不暂停（决定存在即不阻塞）。
        ApprovalLedger.record("pt", "p2", true);
        assertNull(ToolApprovalGate.pauseReason(List.of(call("p2", "deploy_prod")),
                dangerous, "pt", 600));
        ApprovalLedger.clear("pt", "p2");
    }

    @Test
    void hintTextBucketsApproveRejectExpire() {
        var dangerous = ToolApprovalGate.parseDangerous("deploy_prod");
        // 批准 → null（照常执行）。
        ApprovalLedger.record("ht", "h1", true);
        assertNull(ToolApprovalGate.hintText(call("h1", "deploy_prod"), dangerous, "ht", 600));
        // 拒绝 → 提示文本 + 账本清除（消费即清）。
        ApprovalLedger.record("ht", "h2", false);
        String rejected = ToolApprovalGate.hintText(call("h2", "deploy_prod"), dangerous, "ht", 600);
        assertTrue(rejected.contains("已被用户拒绝"), rejected);
        assertEquals(ApprovalLedger.Status.ABSENT,
                ApprovalLedger.status("ht", "h2", 600));
        // 超时（显式决定时刻回拨）→ 超时提示 + 清除。
        ApprovalLedger.recordAt("ht", "h3", true, System.currentTimeMillis() - 60_000);
        String expired = ToolApprovalGate.hintText(call("h3", "deploy_prod"), dangerous, "ht", 10);
        assertTrue(expired.contains("审批已超时"), expired);
        assertEquals(ApprovalLedger.Status.ABSENT, ApprovalLedger.status("ht", "h3", 10));
    }

    // ── 图级 e2e（真实 tools 节点）───────────────────────────────────────

    /** 脚本化主力：按序返回预置 AiMessage（工具调用轮→收尾轮）。 */
    private static final class ScriptedMain implements ModelRuntime {
        final Deque<AiMessage> responses = new ArrayDeque<>();

        @Override
        public dev.langchain4j.model.chat.ChatModel chat() {
            throw new UnsupportedOperationException();
        }

        @Override
        public StreamingChatModel streaming() {
            return null;
        }

        @Override
        public AiMessage stream(List<ChatMessage> messages,
                                List<dev.langchain4j.agent.tool.ToolSpecification> tools,
                                java.util.function.Consumer<String> onDelta) {
            AiMessage next = responses.poll();
            if (next == null) {
                throw new IllegalStateException("scripted main exhausted");
            }
            return next;
        }

        @Override
        public AiMessage invoke(List<ChatMessage> messages) {
            throw new UnsupportedOperationException();
        }
    }

    /** e2e 图：脚本化 LLM 发起危险工具调用 → 暂停 → 批准/拒绝后续跑。 */
    private record Harness(CompiledGraph graph, ScriptedMain main, AtomicInteger executions) {
    }

    private static Harness harness(String approvalMode) {
        Settings settings = Settings.load(java.util.Map.of("ENV", "development",
                "TOOL_APPROVAL_MODE", approvalMode));
        AtomicInteger executions = new AtomicInteger();
        AgentTool deploy = new AgentTool() {
            @Override public String name() { return "deploy_prod"; }
            @Override public String description() { return "部署到生产"; }
            @Override public dev.langchain4j.agent.tool.ToolSpecification specification() {
                return dev.langchain4j.agent.tool.ToolSpecification.builder()
                        .name("deploy_prod").description("部署到生产").build();
            }
            @Override public String call(String argsJson, ToolContext ctx) {
                executions.incrementAndGet();
                return "已部署";
            }
        };
        ToolPool pool = ToolPool.build(List.of(), List.of(deploy),
                ToolPool.DEFAULT_TOOL_TIMEOUT_SECONDS, Map.of(), null);
        ScriptedMain main = new ScriptedMain();
        // 第一轮：chat 发起危险工具调用；续跑后第二轮：收尾文本（END）。
        main.responses.add(AiMessage.builder().toolExecutionRequests(List.of(call("call-1", "deploy_prod"))).build());
        main.responses.add(AiMessage.from("部署完成，收尾。"));
        AgentModule.ModuleContext ctx = new AgentModule.ModuleContext(
                settings, main, new InMemoryCheckpointer(), List.of(), null, null, pool);
        // 工具经第 4 参进 specs——ctx.tools() 为空会让 buildSingleAgentGraph
        // 走单节点分支（根本没有 tools 节点）。
        return new Harness(io.github.paopaonb666.agentbase.core.graphs.Graphs
                .buildSingleAgentGraph(ctx, "chat", null, List.of(deploy)), main, executions);
    }

    @Test
    void dangerousCallPausesRunThenApprovedResumeExecutes() {
        Harness harness = harness("deploy_prod");
        String threadId = "chat:t1";
        CompiledGraph.RunOutcome first = harness.graph().streamWithTermination(
                new GraphState(), RunConfig.of(threadId, "u1"), null);
        assertEquals(io.github.paopaonb666.agentbase.extensions.events.Events.RunTermination.PAUSED,
                first.termination(), "命中名单且无决定 → 暂停");
        assertEquals(0, harness.executions().get(), "暂停前工具绝不执行");
        // 快照完整性：悬挂 tool_calls 有意保留 + pending_approval 标记。
        GraphState snapshot = harness.graph().getState(RunConfig.of(threadId, "u1"));
        assertTrue(snapshot.extra().containsKey("pending_approval"));
        assertTrue(snapshot.messages().get(snapshot.messages().size() - 1)
                instanceof AiMessage kept && kept.hasToolExecutionRequests());
        // 批准 → entryOverride=tools 续跑：工具执行、图走到 END。
        ApprovalLedger.record(threadId, "call-1", true);
        CompiledGraph.RunOutcome resumed = harness.graph().streamWithTermination(
                new GraphState(), new RunConfig(threadId, "u1", 25, "tools"), null);
        assertEquals(io.github.paopaonb666.agentbase.extensions.events.Events.RunTermination.DONE,
                resumed.termination());
        assertEquals(1, harness.executions().get(), "批准后续跑执行恰好一次");
        // pending_approval 退场；结果 ToolMessage 与收尾文本都在。
        assertFalse(harness.graph().getState(RunConfig.of(threadId, "u1"))
                .extra().containsKey("pending_approval"));
        assertTrue(resumed.state().messages().stream()
                .anyMatch(m -> m instanceof ToolExecutionResultMessage tool
                        && "已部署".equals(tool.text())));
        assertEquals("done", io.github.paopaonb666.agentbase.core.graph.RunRegistry
                .lastRun(threadId).termination());
    }

    @Test
    void rejectedDecisionFeedsHintTextWithoutExecution() {
        Harness harness = harness("deploy_prod");
        String threadId = "chat:t2";
        harness.graph().streamWithTermination(new GraphState(), RunConfig.of(threadId, "u1"), null);
        ApprovalLedger.record(threadId, "call-1", false);
        CompiledGraph.RunOutcome resumed = harness.graph().streamWithTermination(
                new GraphState(), new RunConfig(threadId, "u1", 25, "tools"), null);
        assertEquals(io.github.paopaonb666.agentbase.extensions.events.Events.RunTermination.DONE,
                resumed.termination());
        assertEquals(0, harness.executions().get(), "拒绝 → 绝不执行");
        assertTrue(resumed.state().messages().stream()
                .anyMatch(m -> m instanceof ToolExecutionResultMessage tool
                        && tool.text().contains("已被用户拒绝")));
    }

    @Test
    void offModeKeepsExecutionPathByteIdentical() {
        // 名单 off（默认）：危险工具照常执行——零行为变化回归锁。
        Harness harness = harness("off");
        CompiledGraph.RunOutcome outcome = harness.graph().streamWithTermination(
                new GraphState(), RunConfig.of("chat:t3", "u1"), null);
        assertEquals(io.github.paopaonb666.agentbase.extensions.events.Events.RunTermination.DONE,
                outcome.termination());
        assertEquals(1, harness.executions().get());
    }

    @Test
    void nonDangerousToolsNeverPause() {
        // 名单 deploy_prod 但模型调用的是普通工具 → 不暂停照常执行。
        Settings settings = Settings.load(java.util.Map.of("ENV", "development",
                "TOOL_APPROVAL_MODE", "deploy_prod"));
        AtomicInteger executions = new AtomicInteger();
        AgentTool search = new AgentTool() {
            @Override public String name() { return "search_web"; }
            @Override public String description() { return "搜索"; }
            @Override public dev.langchain4j.agent.tool.ToolSpecification specification() {
                return dev.langchain4j.agent.tool.ToolSpecification.builder()
                        .name("search_web").description("搜索").build();
            }
            @Override public String call(String argsJson, ToolContext ctx) {
                executions.incrementAndGet();
                return "结果";
            }
        };
        ToolPool pool = ToolPool.build(List.of(), List.of(search),
                ToolPool.DEFAULT_TOOL_TIMEOUT_SECONDS, Map.of(), null);
        ScriptedMain main = new ScriptedMain();
        main.responses.add(AiMessage.builder().toolExecutionRequests(List.of(call("call-1", "search_web"))).build());
        main.responses.add(AiMessage.from("完成。"));
        AgentModule.ModuleContext ctx = new AgentModule.ModuleContext(
                settings, main, new InMemoryCheckpointer(), List.of(), null, null, pool);
        CompiledGraph graph = io.github.paopaonb666.agentbase.core.graphs.Graphs
                .buildSingleAgentGraph(ctx, "chat", null, List.of(search));
        CompiledGraph.RunOutcome outcome = graph.streamWithTermination(
                new GraphState(), RunConfig.of("chat:t4", "u1"), null);
        assertEquals(io.github.paopaonb666.agentbase.extensions.events.Events.RunTermination.DONE,
                outcome.termination());
        assertEquals(1, executions.get());
    }
}
