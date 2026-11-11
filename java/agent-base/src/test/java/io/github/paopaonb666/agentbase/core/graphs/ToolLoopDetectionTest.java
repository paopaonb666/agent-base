package io.github.paopaonb666.agentbase.core.graphs;

import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import io.github.paopaonb666.agentbase.core.AgentModule;
import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.core.graph.CompiledGraph;
import io.github.paopaonb666.agentbase.core.graph.GraphState;
import io.github.paopaonb666.agentbase.core.graph.InMemoryCheckpointer;
import io.github.paopaonb666.agentbase.core.graph.RunConfig;
import io.github.paopaonb666.agentbase.core.llm.ModelRuntime;
import io.github.paopaonb666.agentbase.core.tools.AgentTool;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;
import java.util.function.Supplier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 工具循环检测 harness（roadmap 大件 5 / 条目 5）：连续同参调用达阈值后
 * 第 threshold+1 次请求被拦截——不执行、回喂结构化提示 ToolMessage（配对
 * 完整、协议安全），图经条件边正常 DONE 而不是烧到 recursionLimit 裸错。
 *
 * <p>脚本桩先例：同包 {@code ToolPoolBindingTest.untrustedMarkerWrapsExternalToolResultsInContextOnly}
 * 的两段式 stub LLM（工厂形态 + 轮次计数）+ fake AgentTool。
 */
class ToolLoopDetectionTest {

    /** 计数调用的 fake 工具：记录被执行次数（拦截发生在 callAll 之前，不触达本工具体）。 */
    private static final class CountingTool implements AgentTool {
        private final String name;
        final AtomicInteger calls = new AtomicInteger();

        CountingTool(String name) {
            this.name = name;
        }

        @Override public String name() { return name; }

        @Override public String description() {
            return "循环检测测试桩：返回自增序号，用于断言实际执行次数。";
        }

        @Override public ToolSpecification specification() {
            return ToolSpecification.builder().name(name()).description(description())
                    .parameters(JsonObjectSchema.builder().build()).build();
        }

        @Override public String call(String argsJson, ToolContext ctx) {
            return name + " 结果-" + calls.incrementAndGet();
        }
    }

    /**
     * 脚本化 stub LLM（工厂形态：每次图运行取独立实例，轮次计数互不串扰）。
     * 按轮次弹出脚本消息；脚本耗尽回纯文本兜底（拦截失效时测试仍干净失败）。
     */
    private static Supplier<ModelRuntime> scriptedLlm(List<AiMessage> script) {
        return () -> {
            AtomicInteger round = new AtomicInteger();
            return new ModelRuntime() {
                private AiMessage next() {
                    int r = round.getAndIncrement();
                    return r < script.size() ? script.get(r) : AiMessage.from("脚本已耗尽");
                }

                @Override public ChatModel chat() {
                    throw new UnsupportedOperationException("测试桩不走向线");
                }

                @Override public StreamingChatModel streaming() {
                    throw new UnsupportedOperationException("测试桩不走向线");
                }

                @Override public AiMessage stream(List<ChatMessage> messages,
                                                  List<ToolSpecification> tools, Consumer<String> onDelta) {
                    return next();
                }

                @Override public AiMessage invoke(List<ChatMessage> messages) {
                    return next();
                }
            };
        };
    }

    private static ToolExecutionRequest call(String id, String tool, String args) {
        return ToolExecutionRequest.builder().id(id).name(tool).arguments(args).build();
    }

    private static AiMessage aiWithCalls(ToolExecutionRequest... requests) {
        return AiMessage.from("", List.of(requests));
    }

    private static AgentModule.ModuleContext ctx(Settings settings, ModelRuntime llm,
                                                 List<AgentTool> tools) {
        return new AgentModule.ModuleContext(settings, llm, new InMemoryCheckpointer(),
                tools, null, null);
    }

    private static Settings settings(String threshold) {
        Map<String, String> overrides = threshold == null
                ? Map.of("ENV", "development")
                : Map.of("ENV", "development", "AGENT_TOOL_LOOP_THRESHOLD", threshold);
        return Settings.load(overrides, null);
    }

    private static List<ToolExecutionResultMessage> toolMessages(GraphState state) {
        return state.messages().stream()
                .filter(m -> m instanceof ToolExecutionResultMessage)
                .map(m -> (ToolExecutionResultMessage) m)
                .toList();
    }

    /** tool_calls 全配对：每个 AiMessage 的请求 id 都有同 id 的 ToolMessage。 */
    private static void assertAllToolCallsPaired(GraphState state) {
        Set<String> requested = new HashSet<>();
        Set<String> answered = new HashSet<>();
        for (ChatMessage message : state.messages()) {
            if (message instanceof AiMessage ai && ai.hasToolExecutionRequests()) {
                for (ToolExecutionRequest request : ai.toolExecutionRequests()) {
                    requested.add(request.id());
                }
            } else if (message instanceof ToolExecutionResultMessage tm) {
                answered.add(tm.id());
            }
        }
        assertEquals(requested, answered, "tool_calls 必须与 ToolMessage 逐条配对");
    }

    /** 1. 连续 3 轮同参 → 工具只执行 2 次，第 3 条 ToolMessage 是提示，图正常结束。 */
    @Test
    void loopBreaksAfterConsecutiveRepeats() {
        CountingTool probe = new CountingTool("probe");
        List<AiMessage> script = List.of(
                aiWithCalls(call("c1", "probe", "{}")),
                aiWithCalls(call("c2", "probe", "{}")),
                aiWithCalls(call("c3", "probe", "{}")));
        CompiledGraph graph = Graphs.buildSingleAgentGraph(
                ctx(settings(null), scriptedLlm(script).get(), List.of(probe)),
                "chat", null, null);
        CompiledGraph.RunOutcome outcome = graph.streamWithTermination(
                new GraphState(List.of(UserMessage.from("开始")), Map.of()),
                RunConfig.of("chat:t50", "u1"), null);
        assertEquals(io.github.paopaonb666.agentbase.extensions.events.Events.RunTermination.DONE,
                outcome.termination(), "全拦截 → 条件边 END：run 正常完结，不烧 recursionLimit");
        GraphState out = outcome.state();

        assertEquals(2, probe.calls.get(), "阈值 2：前 2 次执行，第 3 次被拦截");
        List<ToolExecutionResultMessage> toolMessages = toolMessages(out);
        assertEquals(3, toolMessages.size(), "3 条请求都有配对 ToolMessage");
        assertEquals("probe 结果-1", toolMessages.get(0).text());
        assertEquals("probe 结果-2", toolMessages.get(1).text());
        assertEquals(ToolLoopDetector.loopHintText("probe", 2), toolMessages.get(2).text(),
                "第 3 条 ToolMessage 是结构化收敛提示");
        assertAllToolCallsPaired(out);
    }

    /** 2. 参数变化重置连续计数：不同参数不拦截。 */
    @Test
    void differentArgsResetsStreak() {
        CountingTool probe = new CountingTool("probe");
        List<AiMessage> script = List.of(
                aiWithCalls(call("c1", "probe", "{\"q\":\"a\"}")),
                aiWithCalls(call("c2", "probe", "{\"q\":\"b\"}")),
                aiWithCalls(call("c3", "probe", "{\"q\":\"a\"}")),
                AiMessage.from("done"));
        CompiledGraph graph = Graphs.buildSingleAgentGraph(
                ctx(settings(null), scriptedLlm(script).get(), List.of(probe)),
                "chat", null, null);
        GraphState out = graph.invoke(new GraphState(List.of(UserMessage.from("开始")), Map.of()),
                RunConfig.of("chat:t52", "u1"));

        assertEquals(3, probe.calls.get(), "参数逐轮变化 → 全部照常执行");
        for (ToolExecutionResultMessage tm : toolMessages(out)) {
            assertNotEquals(ToolLoopDetector.loopHintText("probe", 2), tm.text(),
                    "任何一轮都不应出现拦截提示");
        }
        assertAllToolCallsPaired(out);
    }

    /** 3. AGENT_TOOL_LOOP_THRESHOLD=0 = 关闭：逐字节旧行为（同参也不拦）。 */
    @Test
    void thresholdZeroDisablesDetection() {
        CountingTool probe = new CountingTool("probe");
        List<AiMessage> script = List.of(
                aiWithCalls(call("c1", "probe", "{}")),
                aiWithCalls(call("c2", "probe", "{}")),
                aiWithCalls(call("c3", "probe", "{}")),
                AiMessage.from("done"));
        CompiledGraph graph = Graphs.buildSingleAgentGraph(
                ctx(settings("0"), scriptedLlm(script).get(), List.of(probe)),
                "chat", null, null);
        GraphState out = graph.invoke(new GraphState(List.of(UserMessage.from("开始")), Map.of()),
                RunConfig.of("chat:t53", "u1"));

        assertEquals(3, probe.calls.get(), "0 = 关闭：连续同参也全部执行");
        for (ToolExecutionResultMessage tm : toolMessages(out)) {
            assertNotEquals(ToolLoopDetector.loopHintText("probe", 2), tm.text());
        }
        assertAllToolCallsPaired(out);
    }

    /** 4. 同批 1 重复 1 正常：正常的照常执行、结果按请求序、路由回 agent。 */
    @Test
    void batchWithPartialRepeatsStillExecutesOthers() {
        CountingTool probe = new CountingTool("probe");
        CountingTool other = new CountingTool("other");
        List<AiMessage> script = List.of(
                aiWithCalls(call("c1", "probe", "{\"q\":\"a\"}")),
                aiWithCalls(call("c2", "probe", "{\"q\":\"a\"}"),
                        call("c3", "other", "{\"x\":1}")),
                aiWithCalls(call("c4", "probe", "{\"q\":\"a\"}"),
                        call("c5", "other", "{\"x\":2}")),
                AiMessage.from("done"));
        CompiledGraph graph = Graphs.buildSingleAgentGraph(
                ctx(settings(null), scriptedLlm(script).get(), List.of(probe, other)),
                "chat", null, null);
        GraphState out = graph.invoke(new GraphState(List.of(UserMessage.from("开始")), Map.of()),
                RunConfig.of("chat:t54", "u1"));

        assertEquals(2, probe.calls.get(), "第 3 轮的 probe 同参请求被拦截不执行");
        assertEquals(2, other.calls.get(), "同批的正常请求照常执行");
        // 第 3 轮的两条 ToolMessage 按请求序：probe 槽位是提示、other 槽位是结果。
        List<ToolExecutionResultMessage> toolMessages = toolMessages(out);
        assertEquals(5, toolMessages.size());
        assertEquals(ToolLoopDetector.loopHintText("probe", 2), toolMessages.get(3).text());
        assertEquals("other 结果-2", toolMessages.get(4).text());
        assertAllToolCallsPaired(out);
        // 部分拦截 → 路由回 agent：模型看到提示后收尾出 done。
        assertTrue(out.messages().get(out.messages().size() - 1) instanceof AiMessage last
                        && "done".equals(last.text()),
                "部分拦截 → 路由回 agent，模型收尾：" + out.messages().get(out.messages().size() - 1));
    }

    /** 5a. key 归一：键重排 / 空白差异视为相同；工具名参与 key。 */
    @Test
    void canonicalKeyTreatsKeyReorderAndWhitespaceAsSame() {
        String a = ToolLoopDetector.canonicalKey("probe", "{\"b\": 1, \"a\": \"x\"}");
        String b = ToolLoopDetector.canonicalKey("probe", "{ \"a\":\"x\", \"b\":1 }");
        assertEquals(a, b, "键重排 + 空白差异应归一为同 key");
        assertNotEquals(ToolLoopDetector.canonicalKey("probe", "{}"),
                ToolLoopDetector.canonicalKey("other", "{}"),
                "工具名参与 key：同名同参才算重复");
    }

    /** 5b. 非法 JSON 退回原串比较；null 参数与 "{}" 同键。 */
    @Test
    void canonicalKeyFallsBackToRawStringOnInvalidJson() {
        assertEquals("probe not-json", ToolLoopDetector.canonicalKey("probe", "not-json"));
        assertEquals(ToolLoopDetector.canonicalKey("probe", null),
                ToolLoopDetector.canonicalKey("probe", "{}"),
                "null 参数按执行通道同款归一视作 {}");
    }

    /** 5c. 跨轮历史重复不算连续：上一轮的工具轮隔着 UserMessage 即断开。 */
    @Test
    void historyAcrossUserTurnDoesNotCountAsConsecutive() {
        // 上一轮（UserMessage 之前）的同参工具轮 + 本轮 pending 请求：不连续。
        List<ChatMessage> acrossTurns = List.of(
                UserMessage.from("第一轮"),
                aiWithCalls(call("c1", "probe", "{}")),
                ToolExecutionResultMessage.from("c1", "probe", "结果"),
                UserMessage.from("第二轮"),
                aiWithCalls(call("c2", "probe", "{}")));
        assertEquals(0, ToolLoopDetector.consecutiveRepeatCounts(acrossTurns,
                        List.of(call("c2", "probe", "{}")))[0],
                "UserMessage 断开连续性：上一轮的历史重复不累计");
        // 对照：中间无 UserMessage 的连续两轮才计 2。
        List<ChatMessage> consecutive = new ArrayList<>(List.of(
                UserMessage.from("第一轮"),
                aiWithCalls(call("c1", "probe", "{}")),
                ToolExecutionResultMessage.from("c1", "probe", "结果"),
                aiWithCalls(call("c2", "probe", "{}")),
                ToolExecutionResultMessage.from("c2", "probe", "结果"),
                aiWithCalls(call("c3", "probe", "{}"))));
        assertEquals(2, ToolLoopDetector.consecutiveRepeatCounts(consecutive,
                List.of(call("c3", "probe", "{}")))[0]);
    }

    /** 5d. 路由判定：tools 轮补齐后复算——全拦截 → END；部分拦截 → 回 agent。 */
    @Test
    void routerDetectsFullyInterceptedRoundAfterItCompletes() {
        int threshold = 2;
        List<ChatMessage> fully = List.of(
                aiWithCalls(call("c1", "probe", "{}")),
                ToolExecutionResultMessage.from("c1", "probe", "结果-1"),
                aiWithCalls(call("c2", "probe", "{}")),
                ToolExecutionResultMessage.from("c2", "probe", "结果-2"),
                aiWithCalls(call("c3", "probe", "{}")),
                ToolExecutionResultMessage.from("c3", "probe",
                        ToolLoopDetector.loopHintText("probe", 2)));
        assertTrue(ToolLoopDetector.allInterceptedAfterRound(fully, threshold),
                "本批全部被拦截 → 条件边路由 END");
        List<ChatMessage> partial = List.of(
                aiWithCalls(call("c1", "probe", "{}")),
                ToolExecutionResultMessage.from("c1", "probe", "结果-1"),
                aiWithCalls(call("c2", "probe", "{}")),
                ToolExecutionResultMessage.from("c2", "probe", "结果-2"),
                aiWithCalls(call("c3", "probe", "{}"), call("c4", "probe", "{\"q\":1}")),
                ToolExecutionResultMessage.from("c3", "probe",
                        ToolLoopDetector.loopHintText("probe", 2)),
                ToolExecutionResultMessage.from("c4", "probe", "结果-新"));
        assertTrue(!ToolLoopDetector.allInterceptedAfterRound(partial, threshold),
                "同批混入新请求（未被拦截）→ 路由回 agent");
    }

    /** 5e. L93/R4-b 归一收口：嵌套键序与数字形态归一同 key；值不同仍不同。 */
    @Test
    void canonicalKeyTreatsNestedKeyOrderAndNumericFormAsSame() {
        // 嵌套对象键序：旧实现只排顶层、嵌套保持原序——现全深度归一。
        assertEquals(ToolLoopDetector.canonicalKey("probe", "{\"a\":{\"x\":1,\"y\":2}}"),
                ToolLoopDetector.canonicalKey("probe", "{\"a\":{\"y\":2,\"x\":1}}"),
                "嵌套对象键序不同应归一为同 key");
        // 数字形态：1 / 1.0 / 1.00 同 key（1e0 亦然）。
        assertEquals(ToolLoopDetector.canonicalKey("probe", "{\"n\":1}"),
                ToolLoopDetector.canonicalKey("probe", "{\"n\":1.0}"),
                "数字 1 vs 1.0 应归一为同 key");
        assertEquals(ToolLoopDetector.canonicalKey("probe", "{\"n\":1.0}"),
                ToolLoopDetector.canonicalKey("probe", "{\"n\":1.00}"),
                "数字 1.0 vs 1.00 应归一为同 key");
        assertEquals(ToolLoopDetector.canonicalKey("probe", "{\"n\":1}"),
                ToolLoopDetector.canonicalKey("probe", "{\"n\":1e0}"),
                "数字 1 vs 1e0 应归一为同 key");
        // 归一递归穿 List：数组内对象的键序同样归一（元素序本身保留）。
        assertEquals(ToolLoopDetector.canonicalKey("probe", "{\"l\":[{\"b\":1,\"a\":2}]}"),
                ToolLoopDetector.canonicalKey("probe", "{\"l\":[{\"a\":2,\"b\":1}]}"),
                "数组内嵌套对象键序不同应归一为同 key");
        // 方向安全：值确实不同 → 归一后仍不同（不误拦）。
        assertNotEquals(ToolLoopDetector.canonicalKey("probe", "{\"a\":{\"x\":1}}"),
                ToolLoopDetector.canonicalKey("probe", "{\"a\":{\"x\":2}}"),
                "嵌套值不同必须仍是不同 key");
        assertNotEquals(ToolLoopDetector.canonicalKey("probe", "{\"n\":1}"),
                ToolLoopDetector.canonicalKey("probe", "{\"n\":2}"),
                "数值不同必须仍是不同 key");
        assertNotEquals(ToolLoopDetector.canonicalKey("probe", "{\"l\":[1,2]}"),
                ToolLoopDetector.canonicalKey("probe", "{\"l\":[2,1]}"),
                "数组元素序有语义：换序是不同参数");
    }

    /** 5f. L93 端到端：跨轮换数字写法（1 → 1.0 → 1.00）的同参请求仍被拦截。 */
    @Test
    void numericFormVariationStillInterceptedEndToEnd() {
        CountingTool probe = new CountingTool("probe");
        List<AiMessage> script = List.of(
                aiWithCalls(call("c1", "probe", "{\"q\":{\"a\":1}}")),
                aiWithCalls(call("c2", "probe", "{\"q\":{\"a\":1.0}}")),
                aiWithCalls(call("c3", "probe", "{\"q\":{\"a\":1.00}}")));
        CompiledGraph graph = Graphs.buildSingleAgentGraph(
                ctx(settings(null), scriptedLlm(script).get(), List.of(probe)),
                "chat", null, null);
        CompiledGraph.RunOutcome outcome = graph.streamWithTermination(
                new GraphState(List.of(UserMessage.from("开始")), Map.of()),
                RunConfig.of("chat:t55", "u1"), null);
        assertEquals(io.github.paopaonb666.agentbase.extensions.events.Events.RunTermination.DONE,
                outcome.termination(), "全拦截 → 条件边 END");
        GraphState out = outcome.state();

        assertEquals(2, probe.calls.get(), "数字写法换形态不重置连续计数：第 3 次被拦截");
        List<ToolExecutionResultMessage> toolMessages = toolMessages(out);
        assertEquals(3, toolMessages.size());
        assertEquals(ToolLoopDetector.loopHintText("probe", 2), toolMessages.get(2).text(),
                "第 3 条 ToolMessage 是结构化收敛提示");
        assertAllToolCallsPaired(out);
    }
}
