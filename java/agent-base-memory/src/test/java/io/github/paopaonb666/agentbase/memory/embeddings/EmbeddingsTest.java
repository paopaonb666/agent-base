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
}
