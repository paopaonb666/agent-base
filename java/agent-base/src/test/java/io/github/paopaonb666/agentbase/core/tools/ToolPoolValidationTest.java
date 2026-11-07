package io.github.paopaonb666.agentbase.core.tools;

import com.fasterxml.jackson.databind.ObjectMapper;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import io.github.paopaonb666.agentbase.core.graph.RunConfig;
import io.github.paopaonb666.agentbase.extensions.toollog.ToolCallRecord;
import io.github.paopaonb666.agentbase.toolkit.ToolArgsValidator;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 参数 Schema 校验与回喂契约（#334/#335）：执行前轻校验、失败短路
 * （不进工具线程）、outcome=invalid 进审计、契约文案锁定。
 */
class ToolPoolValidationTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    /** 带声明 schema 的工具：记录调用次数（校验短路时必须为 0）。 */
    private static AgentTool schemaTool(ToolSpecification spec, AtomicInteger invocations) {
        return new AgentTool() {
            @Override
            public String name() {
                return spec.name();
            }

            @Override
            public String description() {
                return "schema fixture";
            }

            @Override
            public ToolSpecification specification() {
                return spec;
            }

            @Override
            public String call(String argsJson, ToolContext ctx) {
                invocations.incrementAndGet();
                return "called";
            }
        };
    }

    private static ToolSpecification queryToolSpec() {
        return ToolSpecification.builder()
                .name("query")
                .description("fixture")
                .parameters(JsonObjectSchema.builder()
                        .addStringProperty("query", "查询词")
                        .addIntegerProperty("limit", "条数")
                        .addEnumProperty("mode", List.of("fast", "deep"))
                        .required("query")
                        .build())
                .build();
    }

    // ── 校验器单测 ────────────────────────────────────────────────

    @Test
    void missingRequiredIsReported() throws Exception {
        List<String> errors = ToolArgsValidator.validate(queryToolSpec().parameters(),
                MAPPER.readTree("{\"limit\": 5}"));
        assertEquals(List.of("缺少必需参数 'query'"), errors);
    }

    @Test
    void typeMismatchIsReportedPerDeclaredSchema() throws Exception {
        List<String> errors = ToolArgsValidator.validate(queryToolSpec().parameters(),
                MAPPER.readTree("{\"query\": 42, \"limit\": \"many\"}"));
        assertEquals(2, errors.size());
        assertTrue(errors.get(0).contains("参数 'query' 校验失败: 类型应为 string，实为 integer"));
        assertTrue(errors.get(1).contains("参数 'limit' 校验失败: 类型应为 integer，实为 string"));
    }

    @Test
    void enumValueOutsideSetIsReported() throws Exception {
        List<String> errors = ToolArgsValidator.validate(queryToolSpec().parameters(),
                MAPPER.readTree("{\"query\": \"x\", \"mode\": \"turbo\"}"));
        assertEquals(1, errors.size());
        assertTrue(errors.get(0).contains("取值应为 [fast, deep] 之一"));
    }

    @Test
    void arrayItemTypeIsCheckedRecursively() throws Exception {
        ToolSpecification spec = ToolSpecification.builder()
                .name("tags")
                .parameters(JsonObjectSchema.builder()
                        .addProperty("tags", dev.langchain4j.model.chat.request.json.JsonArraySchema.builder()
                                .items(new dev.langchain4j.model.chat.request.json.JsonStringSchema())
                                .build())
                        .required("tags")
                        .build())
                .build();
        List<String> errors = ToolArgsValidator.validate(spec.parameters(),
                MAPPER.readTree("{\"tags\": [\"a\", 1]}"));
        assertEquals(1, errors.size());
        assertTrue(errors.get(0).contains("参数 'tags[1]' 校验失败"));
    }

    @Test
    void untypedObjectSchemaAcceptsAnything() throws Exception {
        // 零参工具（handoff 类）：模型多给的键不失败
        List<String> errors = ToolArgsValidator.validate(
                JsonObjectSchema.builder().build(),
                MAPPER.readTree("{\"anything\": [1, 2, 3]}"));
        assertTrue(errors.isEmpty());
    }

    @Test
    void unknownPropertiesPassDeclaredOnesStillChecked() throws Exception {
        List<String> errors = ToolArgsValidator.validate(queryToolSpec().parameters(),
                MAPPER.readTree("{\"query\": \"x\", \"extra\": true}"));
        assertTrue(errors.isEmpty());
    }

    // ── 池级契约：短路、审计、事件 ─────────────────────────────────

    @Test
    void invalidArgsShortCircuitWithoutInvokingToolAndFeedContractText() {
        AtomicInteger invocations = new AtomicInteger();
        ToolPool pool = ToolPool.build(
                List.of(schemaTool(queryToolSpec(), invocations)), List.of(), 30.0, Map.of(), null);

        String result = pool.call("query", "{\"limit\": 5}", RunConfig.of("chat:t", "u"), null);

        // #335 契约文本：结构化定位 + 修正指引，不执行工具
        assertEquals("缺少必需参数 'query'。请修正参数后重试", result);
        assertEquals(0, invocations.get());
    }

    @Test
    void invalidArgsAreAuditedWithOutcomeInvalid() {
        AtomicInteger invocations = new AtomicInteger();
        RecordingRecorder recorder = new RecordingRecorder();
        ToolPool pool = ToolPool.build(
                List.of(schemaTool(queryToolSpec(), invocations)), List.of(), 30.0, Map.of(),
                recorder);

        pool.call("query", "not-json-at-all", RunConfig.of("chat:t", "u"), null);

        assertEquals(0, invocations.get());
        assertEquals(1, recorder.records.size());
        assertEquals("invalid", recorder.records.get(0).status());
        assertTrue(recorder.records.get(0).errorText().endsWith("请修正参数后重试"));
    }

    @Test
    void validArgsStillExecuteNormally() {
        AtomicInteger invocations = new AtomicInteger();
        ToolPool pool = ToolPool.build(
                List.of(schemaTool(queryToolSpec(), invocations)), List.of(), 30.0, Map.of(), null);

        assertEquals("called", pool.call("query",
                "{\"query\": \"x\", \"limit\": 3, \"mode\": \"fast\"}",
                RunConfig.of("chat:t", "u"), null));
        assertEquals(1, invocations.get());
    }

    /** 与 ToolPoolTest 同形的审计替身（局部重复以保持测试自包含）。 */
    private static final class RecordingRecorder implements io.github.paopaonb666.agentbase.extensions.toollog.ToolCallRecorder {
        final List<ToolCallRecord> records = new java.util.ArrayList<>();

        @Override
        public void record(ToolCallRecord record) {
            records.add(record);
        }

        @Override
        public List<Map<String, Object>> listForThread(String threadId, int limit) {
            return List.of();
        }
    }
}
