package io.github.paopaonb666.agentbase.entrypoints.server;

import java.io.IOException;

import io.github.paopaonb666.agentbase.extensions.Metrics;
import io.github.paopaonb666.agentbase.extensions.Observability;

import jakarta.servlet.DispatcherType;
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

        /** REQUEST 趟记录的起始时刻（ASYNC dispatch 读回计全生命周期时长）。 */
        private static final String START_NANO_ATTR = MetricsFilter.class.getName() + ".startNano";

        @Override
        public void doFilter(ServletRequest request, ServletResponse response, FilterChain chain)
                throws IOException, ServletException {
            HttpServletRequest req = (HttpServletRequest) request;
            // 异步请求（SSE）量全生命周期（review C-P3，对齐 Python 中间件
            // 量完整流）：REQUEST 趟只记录起始时刻——controller 返回
            // SseEmitter 后 filter 链即返回，此处观测到的只有数毫秒的装配
            // 时长；等容器在流终态重入 ASYNC dispatch 时再收口（起始时刻从
            // request attribute 读回）。流内 error 事件不改变 HTTP status，
            // metrics status 恒为响应状态码（与 Python 同口径；终态事件到
            // status 的映射不做——需要 SseStreams 内部终态分类传出，超 S）。
            boolean initial = req.getDispatcherType() != DispatcherType.ASYNC;
            long started;
            if (initial) {
                started = System.nanoTime();
                req.setAttribute(START_NANO_ATTR, started);
            } else {
                started = req.getAttribute(START_NANO_ATTR) instanceof Long recorded
                        ? recorded
                        : System.nanoTime();
            }
            try {
                chain.doFilter(request, response);
            } finally {
                boolean deferToAsyncDispatch = initial && req.isAsyncStarted();
                if (!deferToAsyncDispatch) {
                    int status = response instanceof HttpServletResponse http
                            ? http.getStatus()
                            : 200;
                    Metrics.http().observe(req.getMethod(), routeOf(req), status,
                            (System.nanoTime() - started) / 1_000_000_000.0);
                }
            }
        }

        /** 路由模板（Spring 匹配后注入 BEST_MATCHING_PATTERN 属性）。 */
        private static String routeOf(HttpServletRequest req) {
            Object pattern = req.getAttribute(
                    org.springframework.web.servlet.HandlerMapping.BEST_MATCHING_PATTERN_ATTRIBUTE);
            // 未匹配路由（404/405）恒用 "unmatched"（对应 Python server/__init__
            // 的 route template 回退）——原始 URI 会让扫描器打出的每个路径都
            // 成为独立 Prometheus 序列，基数爆炸。
            return pattern instanceof String template ? template : "unmatched";
        }
    }
}
