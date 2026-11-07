package io.github.paopaonb666.agentbase.extensions;

import io.github.paopaonb666.agentbase.entrypoints.server.routes.SystemRoutes;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.http.ResponseEntity;

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
}
