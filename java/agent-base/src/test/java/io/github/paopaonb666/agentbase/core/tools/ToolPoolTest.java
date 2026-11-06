package io.github.paopaonb666.agentbase.core.tools;

import dev.langchain4j.agent.tool.ToolSpecification;
import io.github.paopaonb666.agentbase.core.graph.NodeContext;
import io.github.paopaonb666.agentbase.core.graph.RunConfig;
import io.github.paopaonb666.agentbase.extensions.events.Events;
import io.github.paopaonb666.agentbase.extensions.toollog.ToolCallRecord;
import io.github.paopaonb666.agentbase.extensions.toollog.ToolCallRecorder;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * ToolPool 超时/审计/归一测试（roadmap #288）。
 *
 * <p>池是全部模块的工具执行唯一收口：超时、错误归一、审计截断、事件
 * 配对与重名 fail-fast 在这里一次锁定。事件与审计里参数键序为字母序
 * （TreeMap）、`_raw` 为原始 JSON 串（#126/#127 的裁决语义，改动时先改
 * 这里再改 Python 侧）。
 */
class ToolPoolTest {

    /** 录制事件的 NodeContext 替身。 */
    private static final class RecordingContext implements NodeContext {
        final List<Object> events = new ArrayList<>();
        private final RunConfig config =
                new RunConfig("chat:pool-test", "tester", 25);

        @Override
        public RunConfig config() {
            return config;
        }

        @Override
        public void emit(Object event) {
            events.add(event);
        }

        @Override
        public String callTool(String name, String argsJson) {
            return "";
        }

        @Override
        public ToolPool tools() {
            return null;
        }

        @Override
        public boolean cancelled() {
            return false;
        }
    }

    /** 固定返回文本的工具。 */
    private static AgentTool echoTool(String name, String reply) {
        return new AgentTool() {
            @Override
            public String name() {
                return name;
            }

            @Override
            public String description() {
                return "echo fixture";
            }

            @Override
            public ToolSpecification specification() {
                return ToolSpecification.builder().name(name).description("echo fixture").build();
            }

            @Override
            public String call(String argsJson, ToolContext ctx) {
                return reply;
            }
        };
    }

    private static final class RecordingRecorder implements ToolCallRecorder {
        final List<ToolCallRecord> records = new ArrayList<>();

        @Override
        public void record(ToolCallRecord record) {
            records.add(record);
        }

        @Override
        public List<Map<String, Object>> listForThread(String threadId, int limit) {
            return List.of();
        }
    }

    @Test
    void callRoutesToToolAndReturnsResult() {
        ToolPool pool = ToolPool.build(
                List.of(echoTool("echo", "pong")), List.of(), 30.0, Map.of(), null);
        RecordingContext ctx = new RecordingContext();

        assertEquals("pong", pool.call("echo", "{\"k\":1}", ctx.config(), ctx));
    }

    @Test
    void unknownToolReturnsNormalizedText() {
        ToolPool pool = ToolPool.build(List.of(), List.of(), 30.0, Map.of(), null);
        // 与 Python ToolNode handle_tool_error 同构的归一文案（一致性佐证域）。
        assertEquals("tool execution failed: unknown tool 'nope'",
                pool.call("nope", "{}", null, null));
    }

    @Test
    void timeoutReturnsNormalizedTextWhileWorkerFinishesInBackground() throws Exception {
        // 慢工具：忙等 400ms（模拟不可中断的阻塞 IO——interrupt 不打断），
        // 池超时 0.05s。调用方立刻拿到归一超时文本；工具线程留在后台跑完
        // （"有界泄漏"——记录在案的权衡）。
        AtomicBoolean finished = new AtomicBoolean(false);
        AgentTool slow = new AgentTool() {
            @Override
            public String name() {
                return "slow";
            }

            @Override
            public String description() {
                return "slow fixture";
            }

            @Override
            public ToolSpecification specification() {
                return ToolSpecification.builder().name("slow").build();
            }

            @Override
            public String call(String argsJson, ToolContext ctx) throws Exception {
                // 忙等：Thread.interrupt 只置位不打断——对齐不可中断 IO。
                long end = System.currentTimeMillis() + 400;
                while (System.currentTimeMillis() < end) {
                    Thread.onSpinWait();
                }
                finished.set(true);
                return "late";
            }
        };
        ToolPool pool = ToolPool.build(List.of(slow), List.of(), 0.05, Map.of(), null);
        RecordingContext ctx = new RecordingContext();

        long started = System.nanoTime();
        String result = pool.call("slow", "{}", ctx.config(), ctx);
        long elapsedMs = (System.nanoTime() - started) / 1_000_000;

        assertEquals("tool execution failed: tool 'slow' exceeded 0.05s timeout", result);
        assertTrue(elapsedMs < 200, "caller should return at timeout, took " + elapsedMs + "ms");
        // 有界泄漏：工作线程不受 future.cancel(true) 影响，跑完为止。
        long deadline = System.nanoTime() + 2_000_000_000L;
        while (!finished.get() && System.nanoTime() < deadline) {
            Thread.sleep(10);
        }
        assertTrue(finished.get(), "background worker should finish naturally");
        // end 事件带 outcome=timeout。
        Events.ToolCallEvent end = (Events.ToolCallEvent) ctx.events.get(ctx.events.size() - 1);
        assertEquals("end", end.phase());
        assertEquals("timeout", end.status());
    }

    @Test
    void toolExceptionIsNormalizedNotPropagated() {
        // 坏掉的工具绝不打断对话：异常 → 模型可读文本。
        AgentTool broken = new AgentTool() {
            @Override
            public String name() {
                return "broken";
            }

            @Override
            public String description() {
                return "broken fixture";
            }

            @Override
            public ToolSpecification specification() {
                return ToolSpecification.builder().name("broken").build();
            }

            @Override
            public String call(String argsJson, ToolContext ctx) {
                throw new IllegalStateException("内部端点 http://10.0.0.1:9200/_cat 爆了");
            }
        };
        ToolPool pool = ToolPool.build(List.of(broken), List.of(), 30.0, Map.of(), null);
        RecordingContext ctx = new RecordingContext();

        String result = pool.call("broken", "{}", ctx.config(), ctx);
        // URL 脱敏（\S+ 吞掉路径整体）+ 归一前缀。
        assertEquals("tool execution failed: 内部端点 <redacted-url> 爆了", result);
        // outcome=error。
        Events.ToolCallEvent end = (Events.ToolCallEvent) ctx.events.get(ctx.events.size() - 1);
        assertEquals("error", end.status());
        assertTrue(end.error().contains("redacted-url"));
    }

    @Test
    void handleToolErrorCapsAt300CharsAndHandlesNullMessage() {
        // 300 字符上限含前缀整体；无消息异常是空串（不是 "null"）。
        Exception noMessage = new RuntimeException() {
            @Override
            public String getMessage() {
                return null;
            }
        };
        assertEquals("tool execution failed: ", ToolPool.handleToolError(noMessage));

        Exception longMessage = new RuntimeException("x".repeat(500));
        String normalized = ToolPool.handleToolError(longMessage);
        assertEquals(300, normalized.length());
        assertTrue(normalized.startsWith("tool execution failed: "));
    }

    @Test
    void auditRecordTruncatesArgsResultError() {
        RecordingRecorder recorder = new RecordingRecorder();
        AgentTool big = echoTool("big", "r".repeat(5000));
        ToolPool pool = ToolPool.build(List.of(big), List.of(), 30.0, Map.of(), recorder);
        RecordingContext ctx = new RecordingContext();

        pool.call("big", "{\"a\":\"" + "q".repeat(3000) + "\"}", ctx.config(), ctx);

        assertEquals(1, recorder.records.size());
        ToolCallRecord record = recorder.records.get(0);
        assertEquals("big", record.tool());
        assertEquals("ok", record.status());
        assertEquals(ToolCallRecord.ARGS_MAX_CHARS, record.argsJson().length());
        assertEquals(ToolCallRecord.RESULT_MAX_CHARS, record.resultText().length());
        // 线程与模块来自 ThreadId 解析（解析失败留空）。
        assertEquals("chat:pool-test", record.threadId());
        assertEquals("chat", record.module());
    }

    @Test
    void startAndEndEventsPairOnCallId() {
        RecordingRecorder recorder = new RecordingRecorder();
        ToolPool pool = ToolPool.build(
                List.of(echoTool("echo", "pong")), List.of(), 30.0, Map.of(), recorder);
        RecordingContext ctx = new RecordingContext();

        pool.call("echo", "{\"k\":1}", ctx.config(), ctx);

        assertEquals(2, ctx.events.size());
        Events.ToolCallEvent start = (Events.ToolCallEvent) ctx.events.get(0);
        Events.ToolCallEvent end = (Events.ToolCallEvent) ctx.events.get(1);
        assertEquals("start", start.phase());
        assertEquals("end", end.phase());
        assertEquals(start.callId(), end.callId());
        assertEquals("ok", end.status());
        assertTrue(end.durationMs() >= 0);
        // 参数键序稳定（TreeMap 字母序）与 _raw 降级：#126/#127 裁决语义。
        assertEquals(Map.of("k", 1), start.args());
    }

    @Test
    void malformedArgsDegradeToRawField() {
        RecordingContext ctx = new RecordingContext();
        ToolPool pool = ToolPool.build(
                List.of(echoTool("echo", "pong")), List.of(), 30.0, Map.of(), null);

        pool.call("echo", "not json at all", ctx.config(), ctx);

        Events.ToolCallEvent start = (Events.ToolCallEvent) ctx.events.get(0);
        assertEquals(Map.of("_raw", "not json at all"), start.args());
    }

    @Test
    void duplicateToolNameFailsFast() {
        assertThrows(ToolPool.ToolPoolError.class, () -> ToolPool.build(
                List.of(echoTool("dup", "a")), List.of(echoTool("dup", "b")), 30.0, Map.of(), null));
    }

    @Test
    void perToolTimeoutOverridesDefault() throws Exception {
        // per-tool 超时覆盖默认值：默认 30s 太久，用 Map 指定 0.05s。
        AgentTool slow = echoTool("slow", "late");
        // slow 工具实际挂 300ms。
        AgentTool reallySlow = new AgentTool() {
            @Override
            public String name() {
                return "slow";
            }

            @Override
            public String description() {
                return "slow";
            }

            @Override
            public ToolSpecification specification() {
                return ToolSpecification.builder().name("slow").build();
            }

            @Override
            public String call(String argsJson, ToolContext ctx) throws Exception {
                Thread.sleep(300);
                return "late";
            }
        };
        ToolPool pool = ToolPool.build(List.of(reallySlow), List.of(), 30.0,
                Map.of("slow", 0.05), null);
        RecordingContext ctx = new RecordingContext();
        long started = System.nanoTime();
        String result = pool.call("slow", "{}", ctx.config(), ctx);
        long elapsedMs = (System.nanoTime() - started) / 1_000_000;
        assertTrue(elapsedMs < 200);
        assertTrue(result.contains("exceeded 0.05s timeout"));
    }

    @Test
    void specificationsAndRegisteredToolsFollowAssemblyOrder() {
        ToolPool pool = ToolPool.build(
                List.of(echoTool("m1", "a")),
                List.of(echoTool("t1", "b"), echoTool("t2", "c")),
                30.0, Map.of(), null);
        assertFalse(pool.isEmpty());
        // Map.copyOf 不保序：三个名字必须全部在列（顺序契约属 #16 改造范围）。
        assertEquals(List.of("m1", "t1", "t2"),
                pool.specifications().stream().map(ToolSpecification::name).sorted().toList());
        assertEquals(List.of("m1", "t1", "t2"),
                pool.registeredTools().stream().map(AgentTool::name).sorted().toList());
        assertEquals(3, pool.specifications().size());
        assertEquals(3, pool.registeredTools().size());
    }
}
