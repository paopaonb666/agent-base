package io.github.paopaonb666.agentbase.entrypoints.server;

import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.extensions.Metrics;
import jakarta.servlet.DispatcherType;
import jakarta.servlet.FilterChain;
import org.junit.jupiter.api.Test;
import org.springframework.mock.web.MockHttpServletRequest;
import org.springframework.mock.web.MockHttpServletResponse;

import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * MetricsFilter 观测语义（review C-P3）：非异步请求装配点收口一次；
 * 异步请求（SSE——controller 返回 SseEmitter 后 filter 链即返回）在
 * REQUEST 趟不观测，等容器在流终态重入 ASYNC dispatch 才按全生命周期
 * 时长收口（对齐 Python Starlette 中间件量完整流）。
 */
class WebFiltersMetricsFilterTest {

    @Test
    void nonAsyncRequestObservedOnceAtRequestDispatch() throws Exception {
        String route = uniq("/v1/mf/plain/");
        MockHttpServletRequest req = new MockHttpServletRequest("GET", route);
        req.setAttribute(org.springframework.web.servlet.HandlerMapping.BEST_MATCHING_PATTERN_ATTRIBUTE, route);
        MockHttpServletResponse resp = new MockHttpServletResponse();
        FilterChain chain = (request, response) -> { };
        new WebFilters.MetricsFilter().doFilter(req, resp, chain);
        assertTrue(Metrics.http().render().contains(
                        "http_requests_total{method=\"GET\",route=\"" + route + "\",status=\"200\"} 1"),
                "非异步请求应在 REQUEST 趟恰观测一次");
    }

    @Test
    void asyncRequestObservedOnlyAtAsyncDispatch() throws Exception {
        // 修复前 REQUEST 趟立即观测（装配时长 + 状态恒 200），ASYNC 收口
        // 缺失——本测试在该实现下必红（第一断言：装配点不应有观测）。
        String route = uniq("/v1/mf/async/");
        MockHttpServletRequest req = new MockHttpServletRequest("GET", route);
        req.setAttribute(org.springframework.web.servlet.HandlerMapping.BEST_MATCHING_PATTERN_ATTRIBUTE, route);
        req.setAsyncStarted(true);
        MockHttpServletResponse resp = new MockHttpServletResponse();
        FilterChain chain = (request, response) -> { };
        new WebFilters.MetricsFilter().doFilter(req, resp, chain);
        assertFalse(Metrics.http().render().contains("route=\"" + route + "\""),
                "异步请求的 REQUEST 趟（装配点）不得观测，等 ASYNC dispatch 收口");
        // 流终态：容器 ASYNC dispatch 重入过滤器链（同一请求对象，attribute 保留）。
        req.setDispatcherType(DispatcherType.ASYNC);
        req.setAsyncStarted(false);
        Thread.sleep(20); // 与 REQUEST 趟拉开可观测的时间差
        new WebFilters.MetricsFilter().doFilter(req, resp, chain);
        assertTrue(Metrics.http().render().contains(
                        "http_requests_total{method=\"GET\",route=\"" + route + "\",status=\"200\"} 1"),
                "ASYNC 趟（流终态）恰观测一次");
    }

    @Test
    void metricsFilterRegistrationCoversAsyncDispatch() {
        // 收口观测依赖 ASYNC dispatch 重入：注册面必须显式声明
        // REQUEST + ASYNC（FilterRegistrationBean 缺省只挂 REQUEST）。
        var registration = new ServerConfiguration()
                .metricsFilter(Settings.load(java.util.Map.of("ENV", "development")));
        assertTrue(registration.determineDispatcherTypes().contains(DispatcherType.REQUEST));
        assertTrue(registration.determineDispatcherTypes().contains(DispatcherType.ASYNC),
                "MetricsFilter 必须注册 ASYNC dispatch");
    }

    private static String uniq(String prefix) {
        return prefix + UUID.randomUUID();
    }
}
