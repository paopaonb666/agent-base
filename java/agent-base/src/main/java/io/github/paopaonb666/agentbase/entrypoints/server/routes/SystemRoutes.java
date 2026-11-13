package io.github.paopaonb666.agentbase.entrypoints.server.routes;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import io.github.paopaonb666.agentbase.core.AgentRuntime;
import io.github.paopaonb666.agentbase.core.llm.LlmFactory;
import io.github.paopaonb666.agentbase.entrypoints.server.Background;
import io.github.paopaonb666.agentbase.extensions.costmeter.CostMeters;
import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.extensions.Metrics;
import io.github.paopaonb666.agentbase.extensions.costmeter.CostGovernor;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * 系统端点：健康探针、Prometheus 指标、模块清单。
 * 对应 Python 侧 {@code routes/system.py}。
 */
@RestController
public class SystemRoutes {

    private static final Logger LOG = LoggerFactory.getLogger(SystemRoutes.class);

    // 模型端点探活（/health）：成功结果 TTL 缓存 30s，避免每个探活请求
    // 都打真实网络；失败结果不缓存（#206「或穿透」口径——provider 刚挂
    // 时 /health 没有说谎窗口，每次 health 都复检）。
    private static final double MODEL_PROBE_TTL_SECONDS = 30.0;
    private static final double MODEL_PROBE_TIMEOUT_SECONDS = 3.0;

    /** 探活结果缓存：{base_url: (monotonic 时间, 结果)}（进程级单例口径）。 */
    private static final Map<String, Map.Entry<Double, String>> MODEL_PROBE_CACHE =
            new LinkedHashMap<>();

    /** /models 拉取器（可测性分离：测试注入假应答）。 */
    interface ModelsFetcher {
        java.net.http.HttpResponse<String> fetch(String url, String apiKey) throws Exception;
    }

    private static final ModelsFetcher DEFAULT_FETCHER = (url, apiKey) -> {
        java.net.http.HttpClient client = java.net.http.HttpClient.newBuilder()
                .connectTimeout(java.time.Duration.ofSeconds((long) MODEL_PROBE_TIMEOUT_SECONDS))
                .build();
        java.net.http.HttpRequest request = java.net.http.HttpRequest.newBuilder()
                .uri(java.net.URI.create(url))
                .header("Authorization", "Bearer " + apiKey)
                .timeout(java.time.Duration.ofSeconds((long) MODEL_PROBE_TIMEOUT_SECONDS))
                .GET().build();
        return client.send(request, java.net.http.HttpResponse.BodyHandlers.ofString());
    };

    private final AgentRuntime runtime;
    private final CostGovernor governor;

    public SystemRoutes(AgentRuntime runtime,
                        org.springframework.beans.factory.ObjectProvider<CostGovernor> governorProvider) {
        this.runtime = runtime;
        // COST_ENABLED=false 时 governor bean 缺席（null）——可选注入。
        this.governor = governorProvider.getIfAvailable();
    }

    /**
     * 探测 openai 兼容端点（成功 TTL 缓存，失败穿透）。ok 需要同时满足：
     * 2xx 且响应体是含 {@code data} 数组的 JSON（#205——200 错误页/网关
     * 502/404 都不再冒充健康）；失败结果带原因（错误页/网关故障/网络
     * 异常三类可区分，进 health 明细）且不缓存——故障恢复立即可见。
     */
    static String probeModel(Settings settings) {
        return probeModel(settings, DEFAULT_FETCHER);
    }

    static String probeModel(Settings settings, ModelsFetcher fetcher) {
        String baseUrl = settings.llm.baseUrl;
        double now = System.nanoTime() / 1_000_000_000.0;
        synchronized (MODEL_PROBE_CACHE) {
            var hit = MODEL_PROBE_CACHE.get(baseUrl);
            if (hit != null && "ok".equals(hit.getValue())
                    && now - hit.getKey() < MODEL_PROBE_TTL_SECONDS) {
                return hit.getValue();
            }
        }
        String url = baseUrl.replaceAll("/+$", "") + "/models";
        String status;
        try {
            java.net.http.HttpResponse<String> response = fetcher.fetch(url, settings.llm.apiKey.value());
            if (response.statusCode() < 200 || response.statusCode() >= 300) {
                status = "error:http:" + response.statusCode();
            } else if (!modelsShapeValid(response.body())) {
                status = "error:shape:not-openai-models";
            } else {
                status = "ok";
            }
        } catch (Exception e) {
            status = "error:net:" + e.getClass().getSimpleName();
        }
        if (!"ok".equals(status)) {
            LOG.warn("health: model endpoint probe failed for {}: {}", baseUrl, status);
        } else {
            // 只缓存成功结果（#206 失败穿透）。
            synchronized (MODEL_PROBE_CACHE) {
                MODEL_PROBE_CACHE.put(baseUrl, Map.entry(now, status));
            }
        }
        return status;
    }

    /** OpenAI 兼容 /models 契约：JSON 对象且含 data 数组。 */
    private static boolean modelsShapeValid(String body) {
        try {
            com.fasterxml.jackson.databind.JsonNode root =
                    new com.fasterxml.jackson.databind.ObjectMapper().readTree(body);
            return root.isObject() && root.has("data") && root.get("data").isArray();
        } catch (Exception e) {
            return false;
        }
    }

    // ── GET /v1/modules：已注册模块清单 ─────────────────────────────────

    @GetMapping("/v1/modules")
    public Map<String, Object> listModules() {
        List<Map<String, Object>> modules = new ArrayList<>();
        runtime.modules().forEach((name, module) -> {
            Map<String, Object> view = new LinkedHashMap<>();
            view.put("name", name);
            view.put("description", module.description());
            modules.add(view);
        });
        Map<String, Object> supervisor = new LinkedHashMap<>();
        supervisor.put("name", AgentRuntime.SUPERVISOR_MODULE);
        supervisor.put("description", "多 Agent 协作：把所有已注册模块编排为 sub-agent");
        modules.add(supervisor);
        // invoke 契约发现面（#220）：mode/profile 支持值与路由校验单源同批——
        // 新增 mode/profile 只改一处，前端按此渲染选择器而不是写死字面。
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("modules", modules);
        body.put("modes", AgentsRoutes.MODES);
        body.put("profiles", LlmFactory.PROFILES);
        return body;
    }

    // ── GET /v1/tools：工具自描述清单（A-#34）────────────────────────────

    /**
     * 池内工具的自描述面（排障便利）：名+描述+参数 schema（langchain4j
     * JsonObjectSchema 的渲染形态）——与 /v1/modules 同风格，按名字稳定
     * 排序（#16 specifications() 契约）。池缺席（无工具装配）返回空表。
     */
    @GetMapping("/v1/tools")
    public Map<String, Object> listTools() {
        List<Map<String, Object>> tools = new ArrayList<>();
        var pool = runtime.tools();
        if (pool != null) {
            for (dev.langchain4j.agent.tool.ToolSpecification spec : pool.specifications()) {
                Map<String, Object> view = new LinkedHashMap<>();
                view.put("name", spec.name());
                view.put("description", spec.description() == null ? "" : spec.description());
                view.put("parameters", String.valueOf(spec.parameters()));
                tools.add(view);
            }
        }
        return Map.of("tools", tools);
    }

    // ── GET /health：组件健康；degraded 表示部分失败 ────────────────────

    @GetMapping("/health")
    public Map<String, Object> health() {
        Map<String, String> components = new LinkedHashMap<>();
        if (runtime.checkpointer() == null) {
            components.put("checkpointer", "unconfigured");
        } else {
            components.put("checkpointer", runtime.checkpointer().healthy() ? "ok" : "error");
        }
        Settings settings = runtime.settings();
        String key = settings.llm.apiKey.value().strip();
        if (key.isEmpty()) {
            components.put("model", "unconfigured");
        } else if (!settings.llm.baseUrl.startsWith("http://")
                && !settings.llm.baseUrl.startsWith("https://")) {
            components.put("model", "misconfigured");
        } else if (settings.observability.healthProbeModel) {
            // 真实探活由开关控制：探活会打真实网络，默认关闭。
            components.put("model", probeModel(settings));
        } else {
            components.put("model", "ok");
        }
        // 记忆系统：未启用不是降级（组件缺席即可），启用时探存储。
        var memory = runtime.memory();
        if (memory != null) {
            components.put("memory", memory.healthProbe());
        }
        // 快照欠账（#198）：put 失败重试后仍失败的步在内存续跑——这里让
        // 欠账可见（不参与 ok/degraded 判定：存储健康已由 checkpointer 组件
        // 报告，欠账是存储故障的下游痕迹）。
        if (io.github.paopaonb666.agentbase.core.graph.CompiledGraph.persistInDebt()) {
            components.put("snapshot",
                    "debt:" + io.github.paopaonb666.agentbase.core.graph.CompiledGraph.persistDebtSteps());
        }
        // 成本治理：仅在 COST_ENABLED=true 时出现；warn/blocked → degraded。
        if (governor != null) {
            components.put("cost", governor.status());
        }
        // MCP（#29）：启用即承诺——server 断开即 degraded（模型面已缺工具，
        // 惰性重连在下次调用自愈）。未启用组件缺席（与 memory 可选先例同构）。
        var mcpHealth = runtime.mcpServerHealth();
        if (mcpHealth != null) {
            boolean allUp = mcpHealth.values().stream().allMatch(Boolean::booleanValue);
            if (allUp) {
                components.put("mcp", "ok");
            } else {
                String down = mcpHealth.entrySet().stream()
                        .filter(e -> !e.getValue())
                        .map(Map.Entry::getKey)
                        .sorted()
                        .collect(java.util.stream.Collectors.joining(","));
                components.put("mcp", "down:" + down);
            }
        }
        // 池水位（#81）：有界池家族的运行计数——工具池（在飞/排队/泄漏累计）、
        // Background 队列、成本账本丢弃。观测数值段，不参与 ok/degraded 判定
        // （与 snapshot debt 同哲学：数值是排障线索，不是健康语义）。
        var pools = new StringBuilder("bg_pending:").append(Background.pending())
                .append(",cost_dropped:").append(CostMeters.cost().droppedRecordCount());
        if (runtime.tools() != null) {
            pools.insert(0, "tool:" + runtime.tools().activeThreadCount()
                    + "/" + runtime.tools().queuedCount()
                    + ",leaked:" + runtime.tools().leakedTaskCount() + ",");
        }
        components.put("pools", pools.toString());
        // A-#214：sse 在途水位（在飞/闸门上限）与 threads 活跃数——观测数值
        // 段，不参与 ok/degraded 判定（与 pools 同哲学）。闸门缺席（测试直
        // 构 SystemRoutes）按 0/0 报告。
        var gate = io.github.paopaonb666.agentbase.entrypoints.server.RunGate.active();
        components.put("sse", "inflight:" + (gate == null ? 0 : gate.inFlight())
                + "/" + (gate == null ? 0 : gate.maxConcurrent()));
        components.put("threads", "active:" + io.github.paopaonb666.agentbase.core.graph.RunRegistry.activeCount());
        boolean allOk = components.entrySet().stream()
                .filter(e -> !"pools".equals(e.getKey()) && !"sse".equals(e.getKey())
                        && !"threads".equals(e.getKey()))
                .allMatch(e -> "ok".equals(e.getValue()));
        return Map.of("status", allOk ? "ok" : "degraded", "components", components);
    }

    // ── GET /metrics：Prometheus 文本（HTTP + tool + memory + cost 四段） ─

    @GetMapping(value = "/metrics", produces = "text/plain; version=0.0.4; charset=utf-8")
    public ResponseEntity<String> metrics() {
        String body = Metrics.http().render()
                + Metrics.tool().renderToolMetrics()
                + Metrics.memory().renderMemoryMetrics()
                + Metrics.cost().renderCostMetrics()
                + Metrics.events().renderEventDropMetrics()
                + Metrics.context().renderContextMetrics()
                + Metrics.renderEmbeddingCooldownMetrics();
        return ResponseEntity.ok().contentType(MediaType.parseMediaType(
                "text/plain; version=0.0.4; charset=utf-8")).body(body);
    }
}
