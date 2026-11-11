package io.github.paopaonb666.agentbase.extensions;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Deque;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentHashMap;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 带路由模板 label 的请求指标。
 *
 * <p>继承自 chat-agent 评审中的 B4/B5：计数器和直方图的 label 使用路由
 * 模板（{@code /v1/agents/{module}/invoke}），绝不用原始请求路径——按
 * 对话划分的路径会为每个 id 创建一条时间序列，让 Prometheus 基数爆炸。
 *
 * <p>刻意不引入依赖：渲染 Prometheus 文本展示格式只要几十行。
 *
 * <p><b>语义与口径声明</b>：{@code *_total} 是进程启动以来的累计计数；
 * {@code *_count} / bucket 反映滑动窗口（每序列最近
 * {@link #MAX_SAMPLES_PER_ROUTE} 个样本）——两者分母口径不同是有意为之
 * 的内存权衡。所有指标都是<b>单进程</b>口径，多实例部署下各自计数。
 *
 * <p>对应 Python 侧 {@code extensions/metrics.py}；渲染结果逐字节对齐
 * （含空指标时 HTTP 段多一个换行的历史行为）。
 */
public final class Metrics {

    private Metrics() {}

    // 延迟直方图共用桶（http/tool/memory 三段）：#265 扩展——旧上限 10.0 下，
    // 默认工具超时 30s（ToolPool.DEFAULT_TOOL_TIMEOUT_SECONDS）与 SSE 分钟级
    // 调用几乎全落 +Inf，直方图失去分辨力；30/60/120/300 覆盖长工具与流式
    // 调用（桶数固定，序列基数不随数据漂移）。Python metrics.py BUCKETS 同值
    // 同步改，保渲染逐字节对齐。
    public static final double[] BUCKETS =
            {0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0};

    /** 每条序列保留的延迟样本上限：直方图只反映最近的窗口。 */
    public static final int MAX_SAMPLES_PER_ROUTE = 1000;

    /** 工具执行结果的封闭枚举（label 不接纳任意字符串，防基数爆炸）。
     * invalid = 参数校验失败短路（#334/#335，工具未执行）。 */
    public static final List<String> TOOL_OUTCOMES =
            List.of("ok", "timeout", "error", "invalid");

    /** 记忆系统操作的封闭枚举（M6）。 */
    public static final List<String> MEMORY_OUTCOMES = List.of("ok", "error", "degraded");

    /** 事件丢弃原因的封闭枚举（#210：面板缺条目可诊断）。 */
    public static final List<String> EVENT_DROP_REASONS = List.of("cancelled", "ping_full");

    /** 事件丢弃计数器（#210）：只计"面板看不到"的丢失，按原因分桶。 */
    public static final class EventDropMetrics {
        private final Map<String, Long> drops = new ConcurrentHashMap<>();

        /** 记录一次事件丢弃；reason 必须属于 EVENT_DROP_REASONS。 */
        public void observeDrop(String reason) {
            if (!EVENT_DROP_REASONS.contains(reason)) {
                throw new IllegalArgumentException(
                        "unknown event drop reason '%s'; expected %s".formatted(reason, EVENT_DROP_REASONS));
            }
            drops.merge(reason, 1L, Long::sum);
        }

        public long dropped(String reason) {
            return drops.getOrDefault(reason, 0L);
        }

        public String renderEventDropMetrics() {
            List<String> lines = new ArrayList<>();
            for (Map.Entry<String, Long> e : drops.entrySet().stream()
                    .sorted(Map.Entry.comparingByKey())
                    .collect(java.util.stream.Collectors.toList())) {
                lines.add("agent_events_dropped_total{reason=\"%s\"} %d"
                        .formatted(e.getKey(), e.getValue()));
            }
            return String.join("\n", lines) + (lines.isEmpty() ? "" : "\n");
        }
    }

    /** 计数器 + 滑动窗口延迟直方图的公共基类。 */
    abstract static class LabeledHistogram {
        private final Map<List<String>, Long> counts = new ConcurrentHashMap<>();
        private final Map<String, Deque<Double>> durations = new ConcurrentHashMap<>();
        private final Object lock = new Object();

        abstract String totalName();

        abstract String bucketPrefix();

        void record(List<String> key, String series, double duration) {
            counts.merge(key, 1L, Long::sum);
            synchronized (lock) {
                Deque<Double> window = durations.computeIfAbsent(series, k -> new ArrayDeque<>());
                window.addLast(duration);
                while (window.size() > MAX_SAMPLES_PER_ROUTE) {
                    window.removeFirst();
                }
            }
        }

        String render(
                java.util.function.Function<List<String>, String> counterLabels,
                java.util.function.Function<String, String> seriesLabels,
                boolean trailingNewlineWhenEmpty) {
            List<String> lines = new ArrayList<>();
            // sorted(...)：Python 侧按键排序渲染——Java 侧按键列表逐元素字典序
            // 对齐（List 不是 Comparable，TreeMap 自然序会 CCE，#294 实测）。
            Map<List<String>, Long> sortedCounts = new LinkedHashMap<>();
            counts.entrySet().stream().sorted(Map.Entry.comparingByKey(KEY_ORDER))
                    .forEach(e -> sortedCounts.put(e.getKey(), e.getValue()));
            for (Map.Entry<List<String>, Long> e : sortedCounts.entrySet()) {
                lines.add("%s{%s} %d".formatted(totalName(), counterLabels.apply(e.getKey()), e.getValue()));
            }
            Map<String, Deque<Double>> sortedDurations = new TreeMap<>(durations);
            for (Map.Entry<String, Deque<Double>> e : sortedDurations.entrySet()) {
                String lab = seriesLabels.apply(e.getKey());
                List<Double> samples;
                synchronized (lock) {
                    samples = new ArrayList<>(e.getValue());
                }
                for (double bound : BUCKETS) {
                    long cum = samples.stream().filter(d -> d <= bound).count();
                    lines.add("%s_bucket{%s,le=\"%s\"} %d".formatted(bucketPrefix(), lab, bound, cum));
                }
                lines.add("%s_bucket{%s,le=\"+Inf\"} %d".formatted(bucketPrefix(), lab, samples.size()));
                double sum = samples.stream().mapToDouble(Double::doubleValue).sum();
                lines.add(String.format(java.util.Locale.ROOT, "%s_sum{%s} %.6f",
                        bucketPrefix(), lab, sum));
                lines.add("%s_count{%s} %d".formatted(bucketPrefix(), lab, samples.size()));
            }
            if (lines.isEmpty()) {
                return trailingNewlineWhenEmpty ? "\n" : "";
            }
            return String.join("\n", lines) + "\n";
        }
    }

    /** 计数器键（label 元组）的确定性排序：逐元素字典序，短键在前。 */
    private static final java.util.Comparator<List<String>> KEY_ORDER = (a, b) -> {
        int n = Math.min(a.size(), b.size());
        for (int i = 0; i < n; i++) {
            int c = a.get(i).compareTo(b.get(i));
            if (c != 0) {
                return c;
            }
        }
        return Integer.compare(a.size(), b.size());
    };

    /** 进程内的请求计数器 + 延迟直方图。 */
    public static final class HttpMetrics extends LabeledHistogram {
        @Override
        String totalName() { return "http_requests_total"; }

        @Override
        String bucketPrefix() { return "http_request_duration_seconds"; }

        /** 在其路由模板下记录一个已完成的请求。 */
        public void observe(String method, String route, int status, double duration) {
            record(List.of(method, route, String.valueOf(status)), route, duration);
        }

        /** 渲染 Prometheus 文本展示格式。 */
        public String render() {
            return render(
                    key -> "method=\"%s\",route=\"%s\",status=\"%s\"".formatted(key.get(0), key.get(1), key.get(2)),
                    route -> "route=\"%s\"".formatted(route),
                    true);
        }
    }

    /** 进程内的工具执行计数器 + 延迟直方图。 */
    public static final class ToolMetrics extends LabeledHistogram {

        private static final Logger LOG = LoggerFactory.getLogger(ToolMetrics.class);

        /** 装配期注入的工具名闭集（#33）：tool label 基数的主防线。 */
        private final Set<String> registeredNames = ConcurrentHashMap.newKeySet();

        /** 已告警过的未注册工具名：每名一次 WARN，防日志刷屏。 */
        private final Set<String> warnedUnknownNames = ConcurrentHashMap.newKeySet();

        @Override
        String totalName() { return "tool_executions_total"; }

        @Override
        String bucketPrefix() { return "tool_duration_seconds"; }

        /**
         * 注入装配期工具名闭集（#33 基数显式化）：幂等合并，可在不同装配
         * 批次重复调用。基数防漂移设计——工具名集合本就是装配期闭集
         * （ToolPool.build 对重名 fail-fast），该闭集是 tool label 基数的
         * <b>主防线</b>；本方法把闭集显式交给指标层，让绕过装配路径的
         * 观测可被探测。注册是可选的后置步骤：未注入注册集（直构/测试
         * 路径）时 {@link #observeTool} 行为与从前完全一致。
         */
        public void registerNames(Collection<String> names) {
            registeredNames.addAll(names);
        }

        /**
         * 记录一次已完成的工具执行；outcome 必须属于 TOOL_OUTCOMES。
         *
         * <p>未注册名<b>照常按原名计数，不抛错</b>（抛错会把指标层变成
         * 故障点），仅对每个未注册名 WARN 一次（漂移探测）：真实新工具
         * 漏注册或绕过装配路径的观测由此可诊断；计数按原名保留，不做
         * other 兜底——兜底会掩盖漏注册本身。未注入注册集时不告警
         * （没有闭集就无从谈起「未注册」）。
         */
        public void observeTool(String name, String outcome, double duration) {
            if (!TOOL_OUTCOMES.contains(outcome)) {
                throw new IllegalArgumentException(
                        "unknown tool outcome '%s'; expected %s".formatted(outcome, TOOL_OUTCOMES));
            }
            if (!registeredNames.isEmpty() && !registeredNames.contains(name)
                    && warnedUnknownNames.add(name)) {
                LOG.warn("tool '{}': observed but not in the registered name set "
                        + "(assembly-time closed set, #33); if this is a new tool it must be "
                        + "added at assembly or metric label cardinality will drift", name);
            }
            record(List.of(name, outcome), name, duration);
        }

        /** 渲染工具指标的 Prometheus 文本展示格式（与 HttpMetrics.render 拼接）。 */
        public String renderToolMetrics() {
            return render(
                    key -> "tool=\"%s\",outcome=\"%s\"".formatted(key.get(0), key.get(1)),
                    name -> "tool=\"%s\"".formatted(name),
                    false);
        }
    }

    /** 进程内的记忆操作计数器 + 延迟直方图。 */
    public static final class MemoryMetricsImpl extends LabeledHistogram {
        @Override
        String totalName() { return "memory_operations_total"; }

        @Override
        String bucketPrefix() { return "memory_duration_seconds"; }

        /** 记录一次已完成的记忆操作；outcome 必须属于 MEMORY_OUTCOMES。 */
        public void observe(String op, String outcome, double duration) {
            if (!MEMORY_OUTCOMES.contains(outcome)) {
                throw new IllegalArgumentException(
                        "unknown memory outcome '%s'; expected %s".formatted(outcome, MEMORY_OUTCOMES));
            }
            record(List.of(op, outcome), op, duration);
        }

        /** 渲染记忆指标的 Prometheus 文本展示格式。 */
        public String renderMemoryMetrics() {
            return render(
                    key -> "op=\"%s\",outcome=\"%s\"".formatted(key.get(0), key.get(1)),
                    op -> "op=\"%s\"".formatted(op),
                    false);
        }
    }

    /**
     * 进程内的 LLM 用量计数器（T4.1 成本治理）。
     *
     * <p>kind 集合开放（KEY_ORDER 是逐元素字典序排序，非封闭枚举——任意
     * kind 均按既有行形状渲染）：{@code "input"} / {@code "output"} 来自
     * CostMeter.observe（真实 usage 记账）；{@code "estimated_input"} 是
     * #340 的运维<b>对账</b>位——LangChain4jRuntime 在每次真实调用的 usage
     * 记账时刻发射（ContextEngine.estimateInputTokens 口径，乘 tokenFactor
     * 对齐 provider 分词），与 kind="input" 按 {model, profile} 对照校准
     * LLM_TOKEN_FACTOR。
     */
    public static final class CostMetrics {
        private final Map<List<String>, Long> tokens = new ConcurrentHashMap<>();
        private volatile double costCnyTotal = 0.0;

        public void observeTokens(String model, String profile, String kind, long amount) {
            tokens.merge(List.of(model, profile, kind), amount, Long::sum);
        }

        public synchronized void observeCost(double costCny) {
            costCnyTotal += costCny;
        }

        public String renderCostMetrics() {
            List<String> lines = new ArrayList<>();
            for (Map.Entry<List<String>, Long> e : tokens.entrySet().stream()
                    .sorted(Map.Entry.comparingByKey(KEY_ORDER))
                    .collect(java.util.stream.Collectors.toList())) {
                lines.add("llm_tokens_total{model=\"%s\",profile=\"%s\",kind=\"%s\"} %d"
                        .formatted(e.getKey().get(0), e.getKey().get(1), e.getKey().get(2), e.getValue()));
            }
            lines.add(String.format(java.util.Locale.ROOT, "llm_cost_cny_total %.6f", costCnyTotal));
            return String.join("\n", lines) + "\n";
        }
    }

    /**
     * 进程内的上下文窗口压力仪表（缓做池 #19）。
     *
     * <p>观测<b>观测输入</b> token 用量占上下文窗口的比例（窗口压力），
     * 供容量排障——纯观测面，不碰预算逻辑。发射点在 LangChain4jRuntime
     * 三条真实调用路径的 estimated_input 记账处（est / contextWindowTokens）；
     * 分面 {model, profile} 与 llm_tokens_total 同构，值 = [last, high]：
     * last 是最近一次占用比，high 是进程内峰值水位（只升不降）。
     * 占用比 &gt; 1 合法（估算输入超窗口——排障正是要看这个），不做钳制。
     */
    public static final class ContextMetrics {
        /** {model, profile} → [last, high]（更新走 synchronized，观测面无锁读）。 */
        private final Map<List<String>, double[]> pressure = new ConcurrentHashMap<>();

        /**
         * 记录一次窗口占用比。ratio &lt; 0 或非有限值（NaN/Inf）忽略——无意义
         * 的观测不建键；high 水位只升不降（max）。
         */
        public synchronized void observePressure(String model, String profile, double ratio) {
            if (ratio < 0 || !Double.isFinite(ratio)) {
                return;
            }
            double[] cell = pressure.computeIfAbsent(List.of(model, profile), k -> new double[2]);
            cell[0] = ratio;
            cell[1] = Math.max(cell[1], ratio);
        }

        /**
         * 渲染窗口压力的 Prometheus 文本展示格式：每键两行（stat="last" 与
         * stat="high"，%.4f，KEY_ORDER 字典序）。空 map 返回 ""（与
         * tool/memory 空段同形）。窗口未配置（=0）的运行时不发射——分母
         * 缺失，NaN 无意义。
         */
        public String renderContextMetrics() {
            if (pressure.isEmpty()) {
                return "";
            }
            List<String> lines = new ArrayList<>();
            for (Map.Entry<List<String>, double[]> e : pressure.entrySet().stream()
                    .sorted(Map.Entry.comparingByKey(KEY_ORDER))
                    .collect(java.util.stream.Collectors.toList())) {
                lines.add(String.format(java.util.Locale.ROOT,
                        "llm_context_pressure{model=\"%s\",profile=\"%s\",stat=\"last\"} %.4f",
                        e.getKey().get(0), e.getKey().get(1), e.getValue()[0]));
                lines.add(String.format(java.util.Locale.ROOT,
                        "llm_context_pressure{model=\"%s\",profile=\"%s\",stat=\"high\"} %.4f",
                        e.getKey().get(0), e.getKey().get(1), e.getValue()[1]));
            }
            return String.join("\n", lines) + "\n";
        }
    }

    // ── 进程级单例（对应 Python 的 TOOL_METRICS / MEMORY_METRICS / COST_METRICS） ──

    private static final HttpMetrics HTTP = new HttpMetrics();
    private static final ToolMetrics TOOL = new ToolMetrics();
    private static final MemoryMetricsImpl MEMORY = new MemoryMetricsImpl();
    private static final CostMetrics COST = new CostMetrics();
    private static final EventDropMetrics EVENTS = new EventDropMetrics();
    private static final ContextMetrics CONTEXT = new ContextMetrics();

    public static HttpMetrics http() { return HTTP; }

    public static ToolMetrics tool() { return TOOL; }

    public static MemoryMetricsImpl memory() { return MEMORY; }

    public static CostMetrics cost() { return COST; }

    public static EventDropMetrics events() { return EVENTS; }

    public static ContextMetrics context() { return CONTEXT; }
}
