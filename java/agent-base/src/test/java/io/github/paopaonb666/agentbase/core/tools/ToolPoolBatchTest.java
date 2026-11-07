package io.github.paopaonb666.agentbase.core.tools;

import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import io.github.paopaonb666.agentbase.core.graph.RunConfig;
import io.github.paopaonb666.agentbase.toolkit.search.WebSearchTool;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 只读标记（#336）与批量执行（#337）语义锁定：readOnly 默认 true、
 * 外部副作用面显式 false；callAll 只读并发（并发度可观测）、有副作用
 * 串行在调用线程、结果严格按请求序、串行槽异常原样传播（handoff 语义）。
 */
class ToolPoolBatchTest {

    private static final RunConfig CONFIG = RunConfig.of("chat:batch", "u");

    /** 可配置只读性的计数工具。 */
    private static AgentTool probeTool(String name, boolean readOnly,
                                       AtomicReference<String> executedOn) {
        return new AgentTool() {
            @Override
            public String name() {
                return name;
            }

            @Override
            public String description() {
                return "probe";
            }

            @Override
            public ToolSpecification specification() {
                return ToolSpecification.builder().name(name).description("probe")
                        .parameters(JsonObjectSchema.builder().build()).build();
            }

            @Override
            public boolean readOnly() {
                return readOnly;
            }

            @Override
            public String call(String argsJson, ToolContext ctx) {
                executedOn.set(Thread.currentThread().getName());
                return "done:" + name;
            }
        };
    }

    private static ToolExecutionRequest req(String name) {
        return ToolExecutionRequest.builder().name(name).arguments("{}").build();
    }

    @Test
    void readOnlyDefaultsToTrueAndExternalToolsDeclareFalse() {
        // 默认 true：模块/记忆工具等匿名实现全部吃到
        AgentTool anonymous = probeTool("probe", true, new AtomicReference<>());
        assertTrue(anonymous.readOnly());
        // #336：web_search 显式声明非只读（查询词出网）
        assertFalse(new WebSearchTool(null, null, 10.0).readOnly());
    }

    @Test
    void readOnlyToolsRunConcurrently() throws Exception {
        // 两把互等的闩：串行实现必死锁到超时（工具自报 TIMEOUT），并发实现
        // 两个都顺利到齐。
        CountDownLatch bothStarted = new CountDownLatch(2);
        AgentTool a = new AgentTool() {
            @Override
            public String name() { return "slow_a"; }

            @Override
            public String description() { return "probe"; }

            @Override
            public ToolSpecification specification() {
                return ToolSpecification.builder().name("slow_a").build();
            }

            @Override
            public String call(String argsJson, ToolContext ctx) throws Exception {
                bothStarted.countDown();
                if (!bothStarted.await(3, TimeUnit.SECONDS)) {
                    return "TIMEOUT";
                }
                return "ok-a";
            }
        };
        AgentTool b = new AgentTool() {
            @Override
            public String name() { return "slow_b"; }

            @Override
            public String description() { return "probe"; }

            @Override
            public ToolSpecification specification() {
                return ToolSpecification.builder().name("slow_b").build();
            }

            @Override
            public String call(String argsJson, ToolContext ctx) throws Exception {
                bothStarted.countDown();
                if (!bothStarted.await(3, TimeUnit.SECONDS)) {
                    return "TIMEOUT";
                }
                return "ok-b";
            }
        };
        ToolPool pool = ToolPool.build(List.of(a, b), List.of(), 30.0, Map.of(), null);
        List<String> outputs = pool.callAll(List.of(req("slow_a"), req("slow_b")),
                CONFIG, null, r -> pool.call(r.name(), r.arguments(), CONFIG, null));
        assertEquals(List.of("ok-a", "ok-b"), outputs);
    }

    @Test
    void sideEffectToolsNeverOverlapAndOrderIsPreserved() throws Exception {
        // 「串行」语义 = 不重叠、按提交序（pool.call 本身把工具体派到 worker
        // 线程，线程身份不可观测）。第二个有副作用工具启动时第一个必须已结束。
        AtomicBoolean effectInFlight = new AtomicBoolean(false);
        AtomicBoolean overlapDetected = new AtomicBoolean(false);
        AgentTool effect = new AgentTool() {
            @Override
            public String name() { return "write_thing"; }

            @Override
            public String description() { return "probe"; }

            @Override
            public ToolSpecification specification() {
                return ToolSpecification.builder().name("write_thing").build();
            }

            @Override
            public boolean readOnly() {
                return false;
            }

            @Override
            public String call(String argsJson, ToolContext ctx) throws Exception {
                if (!effectInFlight.compareAndSet(false, true)) {
                    overlapDetected.set(true);
                }
                Thread.sleep(30);
                effectInFlight.set(false);
                return "done:write_thing";
            }
        };
        AgentTool pure = probeTool("read_thing", true, new AtomicReference<>());
        ToolPool pool = ToolPool.build(List.of(effect, pure), List.of(), 30.0, Map.of(), null);

        List<String> outputs = pool.callAll(
                List.of(req("read_thing"), req("write_thing"), req("write_thing"),
                        req("read_thing")),
                CONFIG, null, r -> pool.call(r.name(), r.arguments(), CONFIG, null));

        assertEquals(List.of("done:read_thing", "done:write_thing", "done:write_thing",
                "done:read_thing"), outputs);
        assertFalse(overlapDetected.get(), "有副作用工具不得并发重叠");
    }

    @Test
    void singleRequestTakesFastPath() {
        AtomicReference<String> thread = new AtomicReference<>();
        ToolPool pool = ToolPool.build(
                List.of(probeTool("only", false, thread)), List.of(), 30.0, Map.of(), null);
        List<String> outputs = pool.callAll(List.of(req("only")), CONFIG, null,
                r -> pool.call(r.name(), r.arguments(), CONFIG, null));
        assertEquals(List.of("done:only"), outputs);
    }

    @Test
    void serialRunnerExceptionsPropagateLikeTheOldLoop() {
        AgentTool poolTool = probeTool("pure_tool", true, new AtomicReference<>());
        ToolPool pool = ToolPool.build(List.of(poolTool), List.of(), 30.0, Map.of(), null);
        // 串行槽（池外工具 → 非只读）：runner 异常必须原样抛（handoff 语义）
        assertThrows(IllegalStateException.class, () -> pool.callAll(
                List.of(req("external_thing"), req("pure_tool")), CONFIG, null,
                r -> {
                    if (r.name().equals("external_thing")) {
                        throw new IllegalStateException("handoff blew up");
                    }
                    return pool.call(r.name(), r.arguments(), CONFIG, null);
                }));
    }

    @Test
    void emptyRequestListReturnsEmpty() {
        ToolPool pool = ToolPool.build(List.of(), List.of(), 30.0, Map.of(), null);
        assertTrue(pool.callAll(List.of(), CONFIG, null,
                r -> pool.call(r.name(), r.arguments(), CONFIG, null)).isEmpty());
    }
}
