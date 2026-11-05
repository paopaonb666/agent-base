package io.github.paopaonb666.agentbase.entrypoints.server;

import java.io.IOException;

import io.github.paopaonb666.agentbase.extensions.Metrics;
import io.github.paopaonb666.agentbase.extensions.Observability;

import jakarta.servlet.Filter;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.ServletRequest;
import jakarta.servlet.ServletResponse;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;

/**
 * HTTP 中间件（对应 Python 侧 create_app 的两个中间件）：
 *
 * <ul>
 *   <li>{@link RequestIdFilter}：接受/校验客户端 X-Request-ID（白名单，
 *       非法则另发新 id），MDC 贯穿日志，响应回写 X-Request-ID；</li>
 *   <li>{@link MetricsFilter}：按<b>路由模板</b>（非原始路径）记录
 *       http_requests_total / 时长直方图——按对话划分的路径会让
 *       Prometheus 基数爆炸。</li>
 * </ul>
 */
public final class WebFilters {

    private WebFilters() {}

    public static final class RequestIdFilter implements Filter {

        @Override
        public void doFilter(ServletRequest request, ServletResponse response, FilterChain chain)
                throws IOException, ServletException {
            HttpServletRequest req = (HttpServletRequest) request;
            HttpServletResponse resp = (HttpServletResponse) response;
            String incoming = req.getHeader("X-Request-ID");
            String requestId = incoming != null
                    && AuthBackends.REQUEST_ID_RE.matcher(incoming).matches()
                    ? incoming
                    : Observability.newRequestId();
            // MDC 直接装/卸（withRequestId 的 Runnable 形态无法透传受检异常）。
            String previous = org.slf4j.MDC.get(Observability.REQUEST_ID_KEY);
            org.slf4j.MDC.put(Observability.REQUEST_ID_KEY, requestId);
            try {
                resp.setHeader("X-Request-ID", requestId);
                chain.doFilter(request, response);
            } finally {
                if (previous == null) {
                    org.slf4j.MDC.remove(Observability.REQUEST_ID_KEY);
                } else {
                    org.slf4j.MDC.put(Observability.REQUEST_ID_KEY, previous);
                }
            }
        }
    }

    public static final class MetricsFilter implements Filter {

        @Override
        public void doFilter(ServletRequest request, ServletResponse response, FilterChain chain)
                throws IOException, ServletException {
            HttpServletRequest req = (HttpServletRequest) request;
            long started = System.nanoTime();
            try {
                chain.doFilter(request, response);
            } finally {
                int status = response instanceof HttpServletResponse http
                        ? http.getStatus()
                        : 200;
                Metrics.http().observe(req.getMethod(), routeOf(req), status,
                        (System.nanoTime() - started) / 1_000_000_000.0);
            }
        }

        /** 路由模板（Spring 匹配后注入 BEST_MATCHING_PATTERN 属性）。 */
        private static String routeOf(HttpServletRequest req) {
            Object pattern = req.getAttribute(
                    org.springframework.web.servlet.HandlerMapping.BEST_MATCHING_PATTERN_ATTRIBUTE);
            return pattern instanceof String template ? template : req.getRequestURI();
        }
    }
}
