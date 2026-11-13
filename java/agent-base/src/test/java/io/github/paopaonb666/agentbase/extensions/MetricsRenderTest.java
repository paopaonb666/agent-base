package io.github.paopaonb666.agentbase.extensions;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import io.github.paopaonb666.agentbase.entrypoints.server.routes.SystemRoutes;
import org.junit.jupiter.api.Test;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.http.ResponseEntity;

import java.util.List;
import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * /metrics 渲染契约测试（roadmap #294）：Prometheus 文本展示格式的行形状
 * （计数器/直方图桶/sum/count 的 %.6f 口径）、累计计数与滑动窗口分母的
 * 有意分歧、工具与记忆 outcome 封闭枚举（防基数爆炸）、空段的历史换行
 * 行为，以及四段拼接的端点响应。行形状用新建实例隔离断言；端点拼接用
 * 进程级单例（所有序列以 nanoTime 命名，避免跨测试串扰）。
 */
class MetricsRenderTest {

    private static String uniq(String prefix) {
        return prefix + "-" + System.nanoTime();
    }

    @Test
    void httpHistogramRendersCounterBucketsSumCount() {
        String route = "/v1/render/" + System.nanoTime();
        Metrics.HttpMetrics http = new Metrics.HttpMetrics();
        http.observe("GET", route, 200, 0.02);
        http.observe("GET", route, 200, 0.6);
        http.observe("GET", route, 500, 0.2);
        String body = http.render();

        assertTrue(body.contains(
                "http_requests_total{method=\"GET\",route=\"" + route + "\",status=\"200\"} 2"));
        assertTrue(body.contains(
                "http_requests_total{method=\"GET\",route=\"" + route + "\",status=\"500\"} 1"));
        // 直方图按路由聚合（不分状态）：桶为累计口径（<= bound）。
        // #265：逐 le 值 golden 断言覆盖全桶序列（含新增 30/60/120/300）。
        assertTrue(body.contains(
                "http_request_duration_seconds_bucket{route=\"" + route + "\",le=\"0.01\"} 0"));
        assertTrue(body.contains(
                "http_request_duration_seconds_bucket{route=\"" + route + "\",le=\"0.05\"} 1"));
        assertTrue(body.contains(
                "http_request_duration_seconds_bucket{route=\"" + route + "\",le=\"0.1\"} 1"));
        assertTrue(body.contains(
                "http_request_duration_seconds_bucket{route=\"" + route + "\",le=\"0.5\"} 2"));
        assertTrue(body.contains(
                "http_request_duration_seconds_bucket{route=\"" + route + "\",le=\"1.0\"} 3"));
        assertTrue(body.contains(
                "http_request_duration_seconds_bucket{route=\"" + route + "\",le=\"5.0\"} 3"));
        assertTrue(body.contains(
                "http_request_duration_seconds_bucket{route=\"" + route + "\",le=\"10.0\"} 3"));
        assertTrue(body.contains(
                "http_request_duration_seconds_bucket{route=\"" + route + "\",le=\"30.0\"} 3"));
        assertTrue(body.contains(
                "http_request_duration_seconds_bucket{route=\"" + route + "\",le=\"60.0\"} 3"));
        assertTrue(body.contains(
                "http_request_duration_seconds_bucket{route=\"" + route + "\",le=\"120.0\"} 3"));
        assertTrue(body.contains(
                "http_request_duration_seconds_bucket{route=\"" + route + "\",le=\"300.0\"} 3"));
        assertTrue(body.contains(
                "http_request_duration_seconds_bucket{route=\"" + route + "\",le=\"+Inf\"} 3"));
        assertTrue(body.contains(
                "http_request_duration_seconds_sum{route=\"" + route + "\"} 0.820000"));
        assertTrue(body.contains(
                "http_request_duration_seconds_count{route=\"" + route + "\"} 3"));
        // 以换行结尾（Prometheus 文本格式约定）。
        assertTrue(body.endsWith("\n"));
    }

    @Test
    void emptyHttpSectionIsBareNewline() {
        // 历史行为锁定：HTTP 段空时多一个换行（与 Python 渲染逐字节对齐）；
        // 工具/记忆段空时为空串（无尾换行）。
        assertEquals("\n", new Metrics.HttpMetrics().render());
        assertEquals("", new Metrics.ToolMetrics().renderToolMetrics());
        assertEquals("", new Metrics.MemoryMetricsImpl().renderMemoryMetrics());
    }

    @Test
    void slidingWindowCapsHistogramWhileCounterStaysCumulative() {
        String route = "/v1/flood/" + System.nanoTime();
        Metrics.HttpMetrics http = new Metrics.HttpMetrics();
        for (int i = 0; i < Metrics.MAX_SAMPLES_PER_ROUTE + 10; i++) {
            http.observe("POST", route, 200, 0.001);
        }
        // 有意分歧（类注释声明）：*_total 累计，*_count/bucket 只反映窗口。
        assertTrue(http.render().contains(
                "http_requests_total{method=\"POST\",route=\"" + route + "\",status=\"200\"} "
                        + (Metrics.MAX_SAMPLES_PER_ROUTE + 10)));
        assertTrue(http.render().contains(
                "http_request_duration_seconds_count{route=\"" + route + "\"} "
                        + Metrics.MAX_SAMPLES_PER_ROUTE));
    }

    @Test
    void extendedBucketsResolveLongToolDurations() {
        // #265 分辨力用例：默认工具超时 30s、SSE 分钟级——旧上限 10.0 下
        // 45s 样本只能落 +Inf。扩桶后 45s 落 60.0 桶（120/300 亦覆盖之），
        // 不落 30.0 桶；+Inf 仍是样本总数口径。
        String tool = uniq("slow");
        Metrics.ToolMetrics toolMetrics = new Metrics.ToolMetrics();
        toolMetrics.observeTool(tool, "ok", 45.0);
        String body = toolMetrics.renderToolMetrics();
        assertTrue(body.contains(
                "tool_duration_seconds_bucket{tool=\"" + tool + "\",le=\"30.0\"} 0"));
        assertTrue(body.contains(
                "tool_duration_seconds_bucket{tool=\"" + tool + "\",le=\"60.0\"} 1"));
        assertTrue(body.contains(
                "tool_duration_seconds_bucket{tool=\"" + tool + "\",le=\"120.0\"} 1"));
        assertTrue(body.contains(
                "tool_duration_seconds_bucket{tool=\"" + tool + "\",le=\"300.0\"} 1"));
        assertTrue(body.contains(
                "tool_duration_seconds_bucket{tool=\"" + tool + "\",le=\"+Inf\"} 1"));
    }

    @Test
    void toolOutcomeEnumIsClosed() {
        String tool = "calc-" + System.nanoTime();
        Metrics.ToolMetrics toolMetrics = new Metrics.ToolMetrics();
        IllegalArgumentException unknown = assertThrows(IllegalArgumentException.class,
                () -> toolMetrics.observeTool(tool, "banana", 0.1));
        assertTrue(unknown.getMessage().contains("ok, timeout, error, invalid"));
        toolMetrics.observeTool(tool, "ok", 0.1);
        toolMetrics.observeTool(tool, "invalid", 0.0);
        String body = toolMetrics.renderToolMetrics();
        assertTrue(body.contains("tool_executions_total{tool=\"" + tool + "\",outcome=\"ok\"} 1"));
        assertTrue(body.contains("tool_executions_total{tool=\"" + tool + "\",outcome=\"invalid\"} 1"));
        assertTrue(body.contains("tool_duration_seconds_count{tool=\"" + tool + "\"} 2"));
    }

    @Test
    void registerNamesEnablesUnknownToolDriftWarning() {
        // #33：装配期闭集（registerNames，幂等合并）是工具指标基数的主
        // 防线；未注册名照常按原名计数、不抛错（指标层不能成为故障点），
        // 每名仅 WARN 一次（漂移探测，WARN 断言模式抄 CostMeterTest）。
        String known = uniq("known");
        String unknown = uniq("unknown");
        Metrics.ToolMetrics toolMetrics = new Metrics.ToolMetrics();
        toolMetrics.registerNames(List.of(known, "spare-" + System.nanoTime()));
        toolMetrics.registerNames(List.of(known)); // 幂等合并：重复注入无副作用

        Logger metricsLogger = (Logger) LoggerFactory.getLogger(Metrics.ToolMetrics.class);
        ListAppender<ILoggingEvent> appender = new ListAppender<>();
        appender.start();
        metricsLogger.addAppender(appender);
        try {
            toolMetrics.observeTool(known, "ok", 0.1);      // 注册名：不告警
            toolMetrics.observeTool(unknown, "ok", 0.2);    // 未注册：WARN 一次
            toolMetrics.observeTool(unknown, "error", 0.3); // 同名再观测：不再 WARN
        } finally {
            metricsLogger.detachAppender(appender);
            appender.stop();
        }

        assertEquals(1, warnCount(appender, unknown));
        String body = toolMetrics.renderToolMetrics();
        assertTrue(body.contains(
                "tool_executions_total{tool=\"" + known + "\",outcome=\"ok\"} 1"));
        assertTrue(body.contains(
                "tool_executions_total{tool=\"" + unknown + "\",outcome=\"ok\"} 1"));
        assertTrue(body.contains(
                "tool_executions_total{tool=\"" + unknown + "\",outcome=\"error\"} 1"));
        assertTrue(body.contains("tool_duration_seconds_count{tool=\"" + unknown + "\"} 2"));
    }

    @Test
    void noRegistryProvidedCountsNormallyWithoutWarning() {
        // #33：registerNames 后置可选——直构路径（如 ToolPoolTest 直建池、
        // 本类既有用例的裸实例）没有注册调用：未注册名照常计数、不抛错、
        // 不告警（没有闭集就无从谈起「未注册」，行为与从前完全一致）。
        String tool = uniq("direct");
        Metrics.ToolMetrics toolMetrics = new Metrics.ToolMetrics();

        Logger metricsLogger = (Logger) LoggerFactory.getLogger(Metrics.ToolMetrics.class);
        ListAppender<ILoggingEvent> appender = new ListAppender<>();
        appender.start();
        metricsLogger.addAppender(appender);
        try {
            toolMetrics.observeTool(tool, "ok", 0.1);
        } finally {
            metricsLogger.detachAppender(appender);
            appender.stop();
        }

        assertEquals(0, warnCount(appender, tool));
        assertTrue(toolMetrics.renderToolMetrics().contains(
                "tool_executions_total{tool=\"" + tool + "\",outcome=\"ok\"} 1"));
    }

    /** WARN 计数（CostMeterTest 模式）：按级别与消息片段过滤。 */
    private static long warnCount(ListAppender<ILoggingEvent> appender, String contains) {
        return appender.list.stream()
                .filter(e -> e.getLevel() == Level.WARN)
                .filter(e -> e.getFormattedMessage().contains(contains))
                .count();
    }

    @Test
    void memoryOutcomeEnumIsClosed() {
        String op = "add-" + System.nanoTime();
        Metrics.MemoryMetricsImpl memory = new Metrics.MemoryMetricsImpl();
        IllegalArgumentException unknown = assertThrows(IllegalArgumentException.class,
                () -> memory.observe(op, "banana", 0.1));
        assertTrue(unknown.getMessage().contains("ok, error, degraded"));
        memory.observe(op, "degraded", 0.5);
        String body = memory.renderMemoryMetrics();
        assertTrue(body.contains("memory_operations_total{op=\"" + op + "\",outcome=\"degraded\"} 1"));
        assertTrue(body.contains("memory_duration_seconds_count{op=\"" + op + "\"} 1"));
    }

    @Test
    void costRendersTokenTotalsAndCostLine() {
        String model = "gpt-x-" + System.nanoTime();
        String profile = "main";
        Metrics.CostMetrics cost = new Metrics.CostMetrics();
        cost.observeTokens(model, profile, "input", 120);
        cost.observeTokens(model, profile, "output", 80);
        cost.observeCost(1.25);
        String body = cost.renderCostMetrics();
        assertTrue(body.contains(
                "llm_tokens_total{model=\"" + model + "\",profile=\"" + profile
                        + "\",kind=\"input\"} 120"));
        assertTrue(body.contains(
                "llm_tokens_total{model=\"" + model + "\",profile=\"" + profile
                        + "\",kind=\"output\"} 80"));
        assertTrue(body.contains("llm_cost_cny_total 1.250000"));
        // llm_cost_cny_total 是进程级累计（不按 model 分解）：锁定行形状。
        assertTrue(Pattern.compile("llm_cost_cny_total \\d+\\.\\d{6}").matcher(body).find());
    }

    @Test
    void estimatedInputKindRendersAsTokenTotalLine() {
        // #340 对账位：kind 集合开放（KEY_ORDER 是逐元素字典序排序，非封闭
        // 枚举）——estimated_input 按既有行形状渲染（与 input/output 同
        // 模板），是 LangChain4jRuntime 发射的运维对账指标（与 kind="input"
        // 按 {model, profile} 对照校准 LLM_TOKEN_FACTOR）。
        String model = "gpt-est-" + System.nanoTime();
        Metrics.CostMetrics cost = new Metrics.CostMetrics();
        cost.observeTokens(model, "fast", "estimated_input", 42);
        cost.observeTokens(model, "fast", "input", 100);
        String body = cost.renderCostMetrics();
        assertTrue(body.contains(
                "llm_tokens_total{model=\"" + model + "\",profile=\"fast\",kind=\"estimated_input\"} 42"));
        assertTrue(body.contains(
                "llm_tokens_total{model=\"" + model + "\",profile=\"fast\",kind=\"input\"} 100"));
        // 字典序：estimated_input 排在 input 之前（e < i）。
        assertTrue(body.indexOf("kind=\"estimated_input\"") < body.indexOf("kind=\"input\"}"),
                "KEY_ORDER 逐元素字典序：estimated_input 行在前");
    }

    // ── 缓做池 #19：上下文窗口压力仪表（llm_context_pressure） ─────────────

    @Test
    void contextPressureRendersLastAndHighPerKeyInKeyOrder() {
        // 行形状：每键两行（stat="last"/"high"），%.4f（Locale.ROOT），
        // KEY_ORDER 字典序；空 map 渲染 ""（与 tool/memory 空段同形）。
        // 新建实例隔离断言（类注释约定）。
        assertEquals("", new Metrics.ContextMetrics().renderContextMetrics());
        Metrics.ContextMetrics context = new Metrics.ContextMetrics();
        context.observePressure("gpt-ctx-b", "main", 0.5);
        context.observePressure("gpt-ctx-a", "fast", 0.25);
        context.observePressure("gpt-ctx-a", "fast", 0.75);
        String body = context.renderContextMetrics();
        assertTrue(body.contains(
                "llm_context_pressure{model=\"gpt-ctx-a\",profile=\"fast\",stat=\"last\"} 0.7500"));
        assertTrue(body.contains(
                "llm_context_pressure{model=\"gpt-ctx-a\",profile=\"fast\",stat=\"high\"} 0.7500"));
        assertTrue(body.contains(
                "llm_context_pressure{model=\"gpt-ctx-b\",profile=\"main\",stat=\"last\"} 0.5000"));
        assertTrue(body.contains(
                "llm_context_pressure{model=\"gpt-ctx-b\",profile=\"main\",stat=\"high\"} 0.5000"));
        // 字典序：gpt-ctx-a 键整体排在 gpt-ctx-b 前。
        assertTrue(body.indexOf("model=\"gpt-ctx-a\"") < body.indexOf("model=\"gpt-ctx-b\""),
                "KEY_ORDER 逐元素字典序：a 键在前");
        // 4 位小数口径锁定（任意行都形如 x.xxxx）。
        assertTrue(Pattern.compile(
                        "llm_context_pressure\\{model=\"[^\"]+\",profile=\"[^\"]+\",stat=\"(last|high)\"} \\d+\\.\\d{4}")
                        .matcher(body).find());
        assertTrue(body.endsWith("\n"));
    }

    @Test
    void contextPressureHighWatermarkNeverFalls() {
        // high 水位只升不降：observe 0.5 再 0.3 → last=0.3、high=0.5。
        Metrics.ContextMetrics context = new Metrics.ContextMetrics();
        context.observePressure("gpt-wm", "main", 0.5);
        context.observePressure("gpt-wm", "main", 0.3);
        String body = context.renderContextMetrics();
        assertTrue(body.contains(
                "llm_context_pressure{model=\"gpt-wm\",profile=\"main\",stat=\"last\"} 0.3000"));
        assertTrue(body.contains(
                "llm_context_pressure{model=\"gpt-wm\",profile=\"main\",stat=\"high\"} 0.5000"),
                "high 水位不随后续低观测回落");
    }

    @Test
    void contextPressureIgnoresNegativeAndNonFiniteRatios() {
        // 无意义观测不建键：ratio<0 / NaN / ±Inf 全忽略。
        Metrics.ContextMetrics context = new Metrics.ContextMetrics();
        context.observePressure("gpt-bad", "main", -0.1);
        context.observePressure("gpt-bad", "main", Double.NaN);
        context.observePressure("gpt-bad", "main", Double.POSITIVE_INFINITY);
        context.observePressure("gpt-bad", "main", Double.NEGATIVE_INFINITY);
        assertEquals("", context.renderContextMetrics(), "负值/非有限值不应建键");
    }

    @Test
    void metricsEndpointConcatenatesAllFourSections() {
        String route = uniq("/v1/concat/");
        Metrics.http().observe("GET", route, 200, 0.01);
        String tool = uniq("t");
        Metrics.tool().observeTool(tool, "ok", 0.01);
        String op = uniq("o");
        Metrics.memory().observe(op, "ok", 0.01);
        String model = uniq("m");
        Metrics.cost().observeTokens(model, "main", "input", 7);

        SystemRoutes routes = new SystemRoutes(null, providerOf(null));
        ResponseEntity<String> response = routes.metrics();
        String body = response.getBody();
        assertTrue(body.contains("http_requests_total{method=\"GET\",route=\"" + route + "\""));
        assertTrue(body.contains("tool_executions_total{tool=\"" + tool + "\""));
        assertTrue(body.contains("memory_operations_total{op=\"" + op + "\""));
        assertTrue(body.contains("llm_tokens_total{model=\"" + model + "\""));
        assertTrue(body.contains("llm_cost_cny_total"));
        assertEquals("text/plain;version=0.0.4;charset=utf-8",
                response.getHeaders().getContentType().toString());
    }

    private static ObjectProvider<io.github.paopaonb666.agentbase.extensions.costmeter.CostGovernor>
            providerOf(io.github.paopaonb666.agentbase.extensions.costmeter.CostGovernor governor) {
        return new ObjectProvider<>() {
            @Override
            public io.github.paopaonb666.agentbase.extensions.costmeter.CostGovernor getObject() {
                return governor;
            }

            @Override
            public io.github.paopaonb666.agentbase.extensions.costmeter.CostGovernor getObject(Object... args) {
                return governor;
            }

            @Override
            public io.github.paopaonb666.agentbase.extensions.costmeter.CostGovernor getIfAvailable() {
                return governor;
            }

            @Override
            public io.github.paopaonb666.agentbase.extensions.costmeter.CostGovernor getIfUnique() {
                return governor;
            }
        };
    }

    @Test
    void embeddingCooldownGaugeRendersRemainingSecondsAndDecays() throws Exception {
        // A-#166：冷却截止进 gauge（秒向上取整）；时间基自然衰减到 0。生产
        // 只在失败时打未来截止、更晚截止覆盖更早——用例按时间升序打点。
        Metrics.observeEmbeddingCooldownUntil(System.currentTimeMillis() + 400);
        assertTrue(Metrics.renderEmbeddingCooldownMetrics()
                        .contains("agent_embedding_cooldown_seconds_remaining 1"),
                "400ms 余量向上取整为 1s");
        Thread.sleep(700);
        assertTrue(Metrics.renderEmbeddingCooldownMetrics()
                        .contains("agent_embedding_cooldown_seconds_remaining 0"),
                "截止过后自然衰减到 0");
        Metrics.observeEmbeddingCooldownUntil(System.currentTimeMillis() + 4_500);
        assertTrue(Metrics.renderEmbeddingCooldownMetrics()
                        .contains("agent_embedding_cooldown_seconds_remaining 5"),
                "4500ms 余量向上取整为 5s");
    }
}
