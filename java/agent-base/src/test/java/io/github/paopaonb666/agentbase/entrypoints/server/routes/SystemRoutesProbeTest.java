package io.github.paopaonb666.agentbase.entrypoints.server.routes;

import io.github.paopaonb666.agentbase.core.config.Settings;
import org.junit.jupiter.api.Test;

import java.net.http.HttpResponse;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 模型探活契约测试（roadmap #205/#206）：2xx + OpenAI /models 形状
 * （data 数组）才算 ok——200 错误页/网关 502/网络异常三类失败可区分
 * 且进 health 明细；失败短 TTL（5s）复检，成功长 TTL（30s）。
 */
class SystemRoutesProbeTest {

    private static Settings settingsWith(String baseUrl) {
        return Settings.load(java.util.Map.of(
                "ENV", "development",
                "LLM_BASE_URL", baseUrl), null);
    }

    @SuppressWarnings("unchecked")
    private static HttpResponse<String> response(int status, String body) {
        HttpResponse<String> r = org.mockito.Mockito.mock(HttpResponse.class);
        org.mockito.Mockito.when(r.statusCode()).thenReturn(status);
        org.mockito.Mockito.when(r.body()).thenReturn(body);
        return r;
    }

    private static SystemRoutes.ModelsFetcher fetcherReturning(HttpResponse<String> r,
                                                               AtomicInteger calls) {
        return (url, apiKey) -> {
            calls.incrementAndGet();
            return r;
        };
    }

    @Test
    void openAiShapeWith2xxIsOk() {
        AtomicInteger calls = new AtomicInteger();
        var fetcher = fetcherReturning(
                response(200, "{\"object\": \"list\", \"data\": [{\"id\": \"m\"}]}"), calls);
        assertEquals("ok", SystemRoutes.probeModel(
                settingsWith("http://probe-ok-" + System.nanoTime()), fetcher));
        assertEquals(1, calls.get());
    }

    @Test
    void httpErrorPagesAndGatewayFailuresAreNotOk() {
        long uniq = System.nanoTime();
        // 200 但 HTML 错误页（反代错误页冒充健康）→ 形状不符。
        AtomicInteger html = new AtomicInteger();
        assertEquals("error:shape:not-openai-models",
                SystemRoutes.probeModel(settingsWith("http://probe-html-" + uniq),
                        fetcherReturning(response(200, "<html>oops</html>"), html)));
        // 网关 502 → http 明细。
        AtomicInteger gateway = new AtomicInteger();
        assertEquals("error:http:502",
                SystemRoutes.probeModel(settingsWith("http://probe-502-" + uniq),
                        fetcherReturning(response(502, "bad gateway"), gateway)));
        // 网络异常 → net 明细。
        assertEquals("error:net:ConnectException",
                SystemRoutes.probeModel(settingsWith("http://probe-net-" + uniq),
                        (url, key) -> {
                            throw new java.net.ConnectException("refused");
                        }));
    }

    @Test
    void failuresPassThroughWhileSuccessStaysCached() throws Exception {
        String url = "http://probe-ttl-" + System.nanoTime();
        Settings settings = settingsWith(url);
        AtomicInteger calls = new AtomicInteger();

        // 失败穿透：连续两次探测各打一次真实网络（故障恢复立即可见）。
        SystemRoutes.ModelsFetcher failing = fetcherReturning(response(500, "x"), calls);
        assertEquals("error:http:500", SystemRoutes.probeModel(settings, failing));
        assertEquals("error:http:500", SystemRoutes.probeModel(settings, failing));
        assertEquals(2, calls.get(), "失败结果不缓存——每次 health 都复检");

        // 恢复成 2xx+形状：ok；成功结果进入 30s TTL 缓存。
        SystemRoutes.ModelsFetcher counting = (u, k) -> {
            calls.incrementAndGet();
            return response(200, "{\"data\": []}");
        };
        assertEquals("ok", SystemRoutes.probeModel(settings, counting));
        assertEquals(3, calls.get());
        assertEquals("ok", SystemRoutes.probeModel(settings, counting));
        assertEquals(3, calls.get(), "成功结果 30s 内复用缓存，不打网络");
    }
}
