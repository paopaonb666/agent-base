package io.github.paopaonb666.agentbase.modules.planner;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import io.github.paopaonb666.agentbase.core.AgentModule;
import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.core.context.ContextEngine;
import io.github.paopaonb666.agentbase.core.graph.CompiledGraph;
import io.github.paopaonb666.agentbase.core.graph.GraphState;
import io.github.paopaonb666.agentbase.core.graph.InMemoryCheckpointer;
import io.github.paopaonb666.agentbase.core.graph.RunConfig;
import io.github.paopaonb666.agentbase.core.llm.ModelRuntime;
import io.github.paopaonb666.agentbase.modules.planner.graph.PlannerGraphs;
import io.github.paopaonb666.agentbase.modules.planner.graph.PlannerState;
import org.junit.jupiter.api.Test;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 语义判分（#63，PLANNER_VERDICT=llm）：partial 定向补做一轮、failed 进
 * replan、判分失败 best-effort 保持 done、三态解析容错。
 */
class PlannerVerdictTest {

    /** 脚本化模型替身（与 PlannerGraphTest 同构，局部自包含）。 */
    private static final class ScriptedModel implements ModelRuntime {
        private final ArrayDeque<AiMessage> script = new ArrayDeque<>();

        ScriptedModel(String... responses) {
            for (String response : responses) {
                script.add(AiMessage.from(response));
            }
        }

        @Override
        public ChatModel chat() {
            return new ChatModel() {
                @Override
                public ChatResponse doChat(ChatRequest request) {
                    return ChatResponse.builder().aiMessage(script.poll()).build();
                }
            };
        }

        @Override
        public dev.langchain4j.model.chat.StreamingChatModel streaming() {
            return null;
        }

        @Override
        public AiMessage stream(List<ChatMessage> messages, List<dev.langchain4j.agent.tool.ToolSpecification> tools,
                                Consumer<String> onDelta) {
            return invoke(messages);
        }

        @Override
        public AiMessage invoke(List<ChatMessage> messages) {
            AiMessage next = script.poll();
            if (next == null) {
                throw new IllegalStateException("scripted responses exhausted");
            }
            return next;
        }
    }

    private static AgentModule.ModuleContext ctx(ModelRuntime model) {
        Settings settings = Settings.load(Map.of("PLANNER_VERDICT", "llm"), null);
        return new AgentModule.ModuleContext(settings, model, new InMemoryCheckpointer(),
                List.of(), null, null);
    }

    /** 跑完一张单任务计划，返回最终状态（事件丢弃）。 */
    private GraphState runPlan(ModelRuntime model) {
        CompiledGraph graph = PlannerGraphs.buildPlannerGraph(ctx(model));
        return graph.stream(
                new GraphState(List.of(UserMessage.from("做一个目标")), Map.of()),
                new RunConfig("planner:v1", "u1", 25),
                event -> {
                });
    }

    @Test
    void partialVerdictTriggersOneDirectedRetry() {
        ScriptedModel model = new ScriptedModel(
                "[{\"id\": 1, \"goal\": \"写报告\"}]",
                "只写了开头",
                "partial 缺少结论部分",
                "完整报告含结论",
                "done 现在完整",
                "综合：报告完成");
        GraphState result = runPlan(model);
        List<Map<String, Object>> tasks = result.getOrDefault(PlannerState.TASKS, List.of());
        assertEquals(1, tasks.size());
        assertEquals("done", tasks.get(0).get(PlannerState.STATUS));
        assertEquals("完整报告含结论", tasks.get(0).get(PlannerState.SUMMARY));
        assertEquals(1, tasks.get(0).get("verdict_retries"));
        // 补做轮把 goal 指引传给了 executor：messages 里应出现带指引的输入
        // （第二条 human 消息的 execute 输入）——用 attempts 间接证明重跑。
        assertEquals(2, tasks.get(0).get(PlannerState.ATTEMPTS));
    }

    @Test
    void failedVerdictRoutesToReplan() {
        ScriptedModel model = new ScriptedModel(
                "[{\"id\": 1, \"goal\": \"写报告\"}]",
                "完全跑题的内容",
                "failed 与目标无关",
                "[{\"id\": 1, \"goal\": \"重拆：认真写报告\"}]",
                "重拆后的合格报告",
                "done 报告合格",
                "综合：完成");
        GraphState result = runPlan(model);
        List<Map<String, Object>> tasks = result.getOrDefault(PlannerState.TASKS, List.of());
        assertEquals(1, tasks.size());
        assertEquals("done", tasks.get(0).get(PlannerState.STATUS));
        assertEquals("重拆后的合格报告", tasks.get(0).get(PlannerState.SUMMARY));
        assertEquals(1, result.getOrDefault(PlannerState.REPLANS, -1));
    }

    @Test
    void unparseableVerdictKeepsDone() {
        ScriptedModel model = new ScriptedModel(
                "[{\"id\": 1, \"goal\": \"写报告\"}]",
                "正常输出",
                "（判分器答非所问的一段话）",
                "综合：完成");
        GraphState result = runPlan(model);
        List<Map<String, Object>> tasks = result.getOrDefault(PlannerState.TASKS, List.of());
        assertEquals("done", tasks.get(0).get(PlannerState.STATUS));
        assertEquals("正常输出", tasks.get(0).get(PlannerState.SUMMARY));
        assertTrue(!String.valueOf(tasks.get(0).get(PlannerState.SUMMARY)).contains("判分"));
    }

    @Test
    void parseVerdictToleratesFormatting() {
        var verdict = io.github.paopaonb666.agentbase.modules.planner.graph.CheckNodes.class;
        assertEquals("done", io.github.paopaonb666.agentbase.modules.planner.graph.CheckNodes
                .parseVerdict("done 目标达成"));
        assertEquals("failed", io.github.paopaonb666.agentbase.modules.planner.graph.CheckNodes
                .parseVerdict("failed：跑题"));
        assertEquals("partial", io.github.paopaonb666.agentbase.modules.planner.graph.CheckNodes
                .parseVerdict("PARTIAL: 缺一半"));
        assertEquals("partial", io.github.paopaonb666.agentbase.modules.planner.graph.CheckNodes
                .parseVerdict("**partial** 缺少结论"));
        assertEquals("", io.github.paopaonb666.agentbase.modules.planner.graph.CheckNodes
                .parseVerdict(""));
        assertEquals("", io.github.paopaonb666.agentbase.modules.planner.graph.CheckNodes
                .parseVerdict(null));
        assertEquals("", io.github.paopaonb666.agentbase.modules.planner.graph.CheckNodes
                .parseVerdict("我无法判断"));
    }
}
