package io.github.paopaonb666.agentbase.memory.embeddings;

import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.ArrayDeque;
import java.util.List;
import java.util.Queue;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Embeddings 维度契约测试（roadmap #165/旧 #60）：返回向量维度与配置
 * 不符时<b>拒写</b>——DimensionMismatchError 直接穿透冷却降级（失配是
 * 配置错误不是端点故障，降级成存纯文本等于让换模型静默发生），混维度
 * 向量绝不入库；维度相符或未配置 dims 时正常返回。
 */
class EmbeddingsTest {

    private HttpServer server;

    private Embeddings.OpenAiCompatibleEmbedding client(Integer dims) throws IOException {
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.start();
        String base = "http://127.0.0.1:" + server.getAddress().getPort() + "/v1";
        return new Embeddings.OpenAiCompatibleEmbedding(base, "test-key", "test-model",
                dims, 4, 5.0, 60.0, null);
    }

    /** 逐请求弹出响应向量（冷却穿透测试需要两次不同应答）。 */
    private void serveEmbeddings(Queue<String> vectorJsons) {
        server.createContext("/v1/embeddings", exchange -> {
            String vector = vectorJsons.isEmpty() ? "[]" : vectorJsons.poll();
            byte[] body = ("{\"data\": [{\"index\": 0, \"embedding\": " + vector + "}]}")
                    .getBytes(StandardCharsets.UTF_8);
            exchange.getResponseHeaders().add("Content-Type", "application/json");
            exchange.sendResponseHeaders(200, body.length);
            try (var out = exchange.getResponseBody()) {
                out.write(body);
            }
        });
    }

    @AfterEach
    void stopServer() {
        if (server != null) {
            server.stop(0);
        }
    }

    @Test
    void dimensionMismatchRejectsTheWriteInsteadOfWarning() throws IOException {
        Embeddings.OpenAiCompatibleEmbedding client = client(3);
        serveEmbeddings(new ArrayDeque<>(List.of("[0.1, 0.2]")));   // 配置 3 维，实际 2 维
        Embeddings.DimensionMismatchError error = assertThrows(
                Embeddings.DimensionMismatchError.class,
                () -> client.embed(List.of("一段文本")));
        assertTrue(error.getMessage().contains("3"), error.getMessage());
        assertTrue(error.getMessage().contains("2"), error.getMessage());
        assertTrue(error.getMessage().contains("拒绝写入"), error.getMessage());
    }

    @Test
    void matchingDimensionsPassThrough() throws IOException {
        Embeddings.OpenAiCompatibleEmbedding client = client(3);
        serveEmbeddings(new ArrayDeque<>(List.of("[0.1, 0.2, 0.3]")));
        List<List<Float>> vectors = client.embed(List.of("一段文本"));
        assertEquals(List.of(0.1f, 0.2f, 0.3f), vectors.get(0));
    }

    @Test
    void unconfiguredDimsSkipTheCheck() throws IOException {
        Embeddings.OpenAiCompatibleEmbedding client = client(null);
        serveEmbeddings(new ArrayDeque<>(List.of("[0.1, 0.2]")));
        List<List<Float>> vectors = client.embed(List.of("一段文本"));
        assertNotNull(vectors);
        assertEquals(List.of(0.1f, 0.2f), vectors.get(0));
    }

    @Test
    void mismatchDoesNotTriggerCooldown() throws IOException {
        // 失配是配置错误：不进入 60s 冷却——第一次拒后模型维度修正的
        // 下一发请求照常打网络（若误入冷却会直接返回 null）。
        Embeddings.OpenAiCompatibleEmbedding client = client(3);
        serveEmbeddings(new ArrayDeque<>(List.of("[0.1, 0.2]", "[0.1, 0.2, 0.3]")));
        assertThrows(Embeddings.DimensionMismatchError.class,
                () -> client.embed(List.of("触发失配")));
        List<List<Float>> vectors = client.embed(List.of("修正后"));
        assertEquals(List.of(0.1f, 0.2f, 0.3f), vectors.get(0));
    }

    // ── #297：批量分片 / 冷却 / probe 缓存 / NullEmbedding ──────────────

    @Test
    void largeInputIsSlicedIntoBatchWindows() throws IOException {
        // batchSize=2：6 条输入切 3 窗，响应按 index 重排还原顺序。
        java.util.concurrent.atomic.AtomicInteger requests = new java.util.concurrent.atomic.AtomicInteger();
        com.fasterxml.jackson.databind.ObjectMapper mapper = new com.fasterxml.jackson.databind.ObjectMapper();
        server = com.sun.net.httpserver.HttpServer.create(
                new java.net.InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/v1/embeddings", exchange -> {
            requests.incrementAndGet();
            String bodyText = new String(exchange.getRequestBody().readAllBytes(),
                    StandardCharsets.UTF_8);
            int count = mapper.readTree(bodyText).get("input").size();
            StringBuilder data = new StringBuilder("[");
            for (int i = count - 1; i >= 0; i--) {
                if (i < count - 1) {
                    data.append(',');
                }
                data.append("{\"index\": ").append(i)
                        .append(", \"embedding\": [").append(i).append(".0]}");
            }
            data.append("]");
            byte[] body = ("{\"data\": " + data + "}").getBytes(StandardCharsets.UTF_8);
            exchange.getResponseHeaders().add("Content-Type", "application/json");
            exchange.sendResponseHeaders(200, body.length);
            try (var out = exchange.getResponseBody()) {
                out.write(body);
            }
        });
        server.start();
        String base = "http://127.0.0.1:" + server.getAddress().getPort() + "/v1";
        Embeddings.OpenAiCompatibleEmbedding client = new Embeddings.OpenAiCompatibleEmbedding(
                base, "k", "m", 1, 2, 5.0, 60.0, null);
        List<List<Float>> vectors = client.embed(List.of("a", "b", "c", "d", "e", "f"));
        assertEquals(3, requests.get(), "6 条输入按 batchSize=2 切 3 窗");
        // index 为窗口内下标（OpenAI 契约）——客户端按窗口重排还原输入顺序：
        // 窗口1 [a,b] → [0,1]，窗口2 [c,d] → [0,1]，窗口3 [e,f] → [0,1]。
        assertEquals(List.of(0.0f), vectors.get(0));
        assertEquals(List.of(1.0f), vectors.get(1));
        assertEquals(List.of(0.0f), vectors.get(2));
        assertEquals(List.of(1.0f), vectors.get(5));
    }

    @Test
    void serverErrorEntersCooldownAndReturnsNull() throws IOException {
        Embeddings.OpenAiCompatibleEmbedding client = client(3);
        server.createContext("/v1/embeddings", exchange -> {
            exchange.sendResponseHeaders(500, 0);
            exchange.close();
        });
        assertEquals(null, client.embed(List.of("一次失败")));
        // 冷却期内不发网络直接 null（冷却 60s 远超测试时长）。
        assertEquals(null, client.embed(List.of("冷却期内")));
    }

    @Test
    void probeCachesResultWithinTtl() throws IOException {
        Embeddings.OpenAiCompatibleEmbedding client = client(3);
        java.util.concurrent.atomic.AtomicInteger probes = new java.util.concurrent.atomic.AtomicInteger();
        server.createContext("/v1/models", exchange -> {
            probes.incrementAndGet();
            exchange.sendResponseHeaders(200, 0);
            exchange.close();
        });
        assertTrue(client.probe());
        assertTrue(client.probe());
        assertEquals(1, probes.get(), "probe 结果 60s 内复用缓存");
    }

    @Test
    void nullEmbeddingNeverVectorizesAndProbesOk() {
        Embeddings.NullEmbedding nullClient = new Embeddings.NullEmbedding();
        assertEquals(null, nullClient.dims());
        assertEquals(null, nullClient.embed(List.of("任意")));
        assertTrue(nullClient.probe(), "未配置 embedding 不算降级（health 探针恒绿）");
    }
}
