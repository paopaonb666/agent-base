"""请求指标（阶段 4）的测试：路由模板 label、滑动窗口上限。"""

from __future__ import annotations

from agent_base.extensions.metrics import MAX_SAMPLES_PER_ROUTE, Metrics


def test_observe_and_render_counters() -> None:
    metrics = Metrics()
    metrics.observe("GET", "/health", 200, 0.001)
    metrics.observe("GET", "/health", 200, 0.002)
    metrics.observe("POST", "/v1/agents/{module}/invoke", 200, 0.5)
    rendered = metrics.render()
    assert 'http_requests_total{method="GET",route="/health",status="200"} 2' in rendered
    assert (
        'http_requests_total{method="POST",route="/v1/agents/{module}/invoke",status="200"} 1'
        in rendered
    )
    assert 'http_request_duration_seconds_count{route="/health"} 2' in rendered


def test_duration_samples_capped_at_window() -> None:
    """样本数超过窗口上限时只保留最近的样本（内存有界）。"""
    metrics = Metrics()
    for i in range(MAX_SAMPLES_PER_ROUTE + 50):
        metrics.observe("GET", "/health", 200, float(i))
    metrics.observe("GET", "/health", 200, 0.0)
    rendered = metrics.render()
    # *_count 反映窗口内样本数，而不是累计请求数。
    assert f'http_request_duration_seconds_count{{route="/health"}} {MAX_SAMPLES_PER_ROUTE}' in (
        rendered
    )
    # 窗口滑动后，最早的样本已被挤出：最近的 0.0 在，最早的 0.0（第 1 个）不在。
    assert 'http_request_duration_seconds_bucket{route="/health",le="0.01"} 1' in rendered
    assert 'http_request_duration_seconds_bucket{route="/health",le="+Inf"}' in rendered
