package io.github.paopaonb666.agentbase.memory.embeddings;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import io.github.paopaonb666.agentbase.core.config.Settings;

/**
 * 语义向量客户端（M6b）：openai 兼容 /embeddings + 关键词降级。
 *
 * <p>对话 LLM（如 DeepSeek）与 embedding 服务是两套独立配置：默认指向
 * 硅基流动（{@code https://api.siliconflow.cn/v1}，BAAI/bge-m3，1024 维，
 * 中英双语）。降级链（enterprise 语义）：
 *
 * <ul>
 * <li>{@code MEMORY_EMBEDDING_API_KEY} 未配置 → {@link NullEmbedding}，
 *     检索退化为 BM25 关键词 + 时间衰减 + 显著度，其余功能不受影响；</li>
 * <li>调用失败（网络/HTTP/响应异常）→ 本次返回 {@code null} 并进入短暂
 *     冷却（默认 60s 内不再打真实网络），之后自动恢复——一个挂掉的
 *     embedding 端点绝不能拖垮对话主流程，也不能被每个请求反复重试打爆。</li>
 * </ul>
 *
 * <p>对应 Python 侧 {@code agent_base_memory/embeddings.py}，逐算法移植。
 */
public final class Embeddings {

    private Embeddings() {}

    private static final Logger LOG = LoggerFactory.getLogger(Embeddings.class);

    private static final ObjectMapper JSON = new ObjectMapper();

    /**
     * 单条输入的字符封顶：bge-m3 的窗口是 8192 token，超长文本截断即可
     * （记忆与文档分块的正常长度远小于此；截断是防御而非常规路径）。
     */
    public static final int MAX_INPUT_CHARS = 6000;

    /** 请求体/响应体的日志截断长度：绝不让 embedding 原文或整包响应进日志。 */
    private static final int LOG_BODY_LIMIT = 200;

    /** probe() 结果缓存时长（/health 会被负载均衡轮询，不能每次都打网络）。 */
    private static final long PROBE_CACHE_NANOS = 60_000_000_000L;

    /** embedding 调用失败（网络/HTTP/响应异常）；由客户端内部捕获并降级。 */
    public static final class EmbeddingError extends RuntimeException {
        public EmbeddingError(String message) {
            super(message);
        }

        public EmbeddingError(String message, Throwable cause) {
            super(message, cause);
        }
    }

    // ── 降级实现 ─────────────────────────────────────────────────────

    /** 不向量化：embedding 未配置时的常驻实现，检索走关键词路径。 */
    public static final class NullEmbedding implements EmbeddingClient {

        @Override
        public Integer dims() {
            return null;
        }

        @Override
        public List<List<Float>> embed(List<String> texts) {
            return null;
        }

        @Override
        public boolean probe() {
            // 未配置 embedding 不是降级（Python 侧 NullEmbedding 无 probe
            // 属性，health_probe 跳过探测——语义对齐：恒 ok）。
            return true;
        }
    }

    // ── openai 兼容实现 ──────────────────────────────────────────────

    /**
     * openai 兼容 /embeddings 端点客户端（硅基流动/智谱/自建均可）。
     *
     * <p>失败冷却：一次失败后 {@code cooldownSeconds} 内的调用直接返回
     * null，不打真实网络——避免半死不活的端点把每轮对话都拖慢一个
     * 超时周期。
     */
    public static final class OpenAiCompatibleEmbedding implements EmbeddingClient {

        private final String baseUrl;
        private final String apiKey;
        private final String model;
        private final Integer dims;
        private final int batchSize;
        private final long timeoutMillis;
        private final long cooldownNanos;
        private final HttpClient injectedClient;

        private HttpClient client;
        private long blockedUntilNanos = 0L;
        private boolean dimsWarned = false;
        private boolean probeDone = false;
        private boolean probeOk = false;
        private long probeAtNanos = 0L;

        public OpenAiCompatibleEmbedding(String baseUrl, String apiKey, String model, int dims,
                                         int batchSize, double timeoutSeconds) {
            this(baseUrl, apiKey, model, dims, batchSize, timeoutSeconds, 60.0, null);
        }

        public OpenAiCompatibleEmbedding(String baseUrl, String apiKey, String model, Integer dims,
                                         int batchSize, double timeoutSeconds,
                                         double cooldownSeconds, HttpClient client) {
            this.baseUrl = stripTrailingSlashes(baseUrl);
            this.apiKey = apiKey;
            this.model = model;
            this.dims = dims;
            this.batchSize = Math.max(1, batchSize);
            this.timeoutMillis = (long) (timeoutSeconds * 1000.0);
            this.cooldownNanos = (long) (cooldownSeconds * 1_000_000_000.0);
            this.injectedClient = client;
            this.client = client;
        }

        @Override
        public Integer dims() {
            return dims;
        }

        @Override
        public List<List<Float>> embed(List<String> texts) {
            if (texts == null || texts.isEmpty()) {
                return List.of();
            }
            if (System.nanoTime() < blockedUntilNanos) {
                return null;
            }
            List<List<Float>> vectors = new ArrayList<>();
            try {
                for (int start = 0; start < texts.size(); start += batchSize) {
                    List<String> window = texts.subList(start, Math.min(start + batchSize, texts.size()));
                    List<String> batch = new ArrayList<>(window.size());
                    for (String text : window) {
                        batch.add(truncateCodePoints(text, MAX_INPUT_CHARS));
                    }
                    vectors.addAll(embedBatch(batch));
                }
            } catch (EmbeddingError exc) {
                blockedUntilNanos = System.nanoTime() + cooldownNanos;
                LOG.warn("memory: embedding 不可用，{}s 内走关键词降级：{}",
                        cooldownNanos / 1_000_000_000.0, exc.getMessage());
                return null;
            }
            return vectors;
        }

        private List<List<Float>> embedBatch(List<String> batch) {
            // 请求体键序与 Python 一致：model → input → encoding_format。
            Map<String, Object> body = new LinkedHashMap<>();
            body.put("model", model);
            body.put("input", batch);
            body.put("encoding_format", "float");
            String json;
            try {
                json = JSON.writeValueAsString(body);
            } catch (IOException e) {
                throw new EmbeddingError("请求序列化失败：" + e.getClass().getSimpleName(), e);
            }
            HttpRequest request = HttpRequest.newBuilder(URI.create(baseUrl + "/embeddings"))
                    .timeout(Duration.ofMillis(timeoutMillis))
                    .header("Authorization", "Bearer " + apiKey)
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(json, StandardCharsets.UTF_8))
                    .build();
            HttpResponse<String> response = send(request);
            if (response.statusCode() != 200) {
                throw new EmbeddingError("HTTP " + response.statusCode() + "："
                        + truncateCodePoints(response.body(), LOG_BODY_LIMIT));
            }
            JsonNode payload;
            try {
                payload = JSON.readTree(response.body());
            } catch (IOException e) {
                throw new EmbeddingError("响应格式异常："
                        + truncateCodePoints(response.body(), LOG_BODY_LIMIT));
            }
            JsonNode data = payload.get("data");
            if (data == null) {
                throw new EmbeddingError("响应格式异常："
                        + truncateCodePoints(response.body(), LOG_BODY_LIMIT));
            }
            if (!data.isArray() || data.size() != batch.size()) {
                throw new EmbeddingError("响应条数不符：期望 " + batch.size() + "，实际 "
                        + (data.isArray() ? data.size() : "非列表"));
            }
            List<JsonNode> ordered = new ArrayList<>();
            data.forEach(ordered::add);
            ordered.sort(Comparator.comparingInt(item -> item.path("index").asInt(0)));
            List<List<Float>> vectors = new ArrayList<>(ordered.size());
            for (JsonNode item : ordered) {
                JsonNode vector = item.get("embedding");
                if (vector == null || !vector.isArray()) {
                    throw new EmbeddingError("响应缺少 embedding 数组");
                }
                checkDims(vector.size());
                List<Float> values = new ArrayList<>(vector.size());
                vector.forEach(value -> values.add((float) value.asDouble()));
                vectors.add(values);
            }
            return vectors;
        }

        private HttpResponse<String> send(HttpRequest request) {
            try {
                return http().send(request, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
            } catch (IOException e) {
                throw new EmbeddingError("网络错误：" + e.getClass().getSimpleName(), e);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new EmbeddingError("网络错误：InterruptedException", e);
            }
        }

        private void checkDims(int actual) {
            // 配置维度与实际不符只告警一次：以实际维度为准（逐行记录）。
            if (dims != null && actual != dims && !dimsWarned) {
                dimsWarned = true;
                LOG.warn("memory: embedding 维度与配置不符（配置 {}，实际 {}）——"
                        + "历史向量如与新维度混存将无法做余弦相似，建议清库重建", dims, actual);
            }
        }

        @Override
        public boolean probe() {
            // 与 LLM 探活同一语义：任何 HTTP 应答都算 ok（网络可达即证明），
            // 配额/鉴权不属于健康判定。结果缓存 60s。
            long now = System.nanoTime();
            if (probeDone && now - probeAtNanos < PROBE_CACHE_NANOS) {
                return probeOk;
            }
            boolean ok = false;
            try {
                HttpRequest request = HttpRequest.newBuilder(URI.create(baseUrl + "/models"))
                        .timeout(Duration.ofMillis(timeoutMillis))
                        .header("Authorization", "Bearer " + apiKey)
                        .GET()
                        .build();
                http().send(request, HttpResponse.BodyHandlers.discarding());
                ok = true;
            } catch (Exception e) {
                LOG.warn("memory: embedding 端点探活失败 {}", baseUrl);
            }
            probeAtNanos = now;
            probeOk = ok;
            probeDone = true;
            return ok;
        }

        private HttpClient http() {
            if (client == null) {
                client = HttpClient.newBuilder()
                        .connectTimeout(Duration.ofMillis(timeoutMillis))
                        .build();
            }
            return client;
        }
    }

    // ── 装配 ─────────────────────────────────────────────────────────

    /** 按 settings 装配 embedding 客户端：未启用/未配 key 一律降级。 */
    public static EmbeddingClient build(Settings settings) {
        if (!settings.memory.embeddingEnabled) {
            LOG.info("memory: 语义检索已禁用（MEMORY_EMBEDDING_ENABLED=false）");
            return new NullEmbedding();
        }
        String apiKey = settings.memory.embeddingApiKey.strip();
        if (apiKey.isEmpty()) {
            LOG.info("memory: 未配置 MEMORY_EMBEDDING_API_KEY，检索走关键词降级");
            return new NullEmbedding();
        }
        return new OpenAiCompatibleEmbedding(
                settings.memory.embeddingBaseUrl,
                apiKey,
                settings.memory.embeddingModel,
                settings.memory.embeddingDims,
                settings.memory.embeddingBatchSize,
                settings.memory.embeddingTimeoutSeconds);
    }

    // ── 底层工具 ─────────────────────────────────────────────────────

    /** Python {@code base_url.rstrip("/")}：去掉全部结尾斜杠。 */
    private static String stripTrailingSlashes(String url) {
        String out = url == null ? "" : url;
        while (out.endsWith("/")) {
            out = out.substring(0, out.length() - 1);
        }
        return out;
    }

    /** 码点级截断（对应 Python {@code s[:max]}）。 */
    static String truncateCodePoints(String s, int max) {
        if (s == null || s.codePointCount(0, s.length()) <= max) {
            return s;
        }
        return s.substring(0, s.offsetByCodePoints(0, max));
    }
}
