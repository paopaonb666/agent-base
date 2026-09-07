"""带路由模板 label 的请求指标（阶段 4）。

继承自 chat-agent 评审中的 B4/B5：计数器和直方图的 label 使用路由模板
（``/v1/agents/{module}/invoke``），绝不用原始请求路径——按对话划分的
路径会为每个 id 创建一条时间序列，让 Prometheus 基数爆炸。

刻意不引入依赖：渲染 Prometheus 文本展示格式只要几十行，而把
prometheus-client 挡在基座之外能让依赖树保持轻薄。token / 业务指标
属于模块。
"""

from __future__ import annotations

from collections import deque

BUCKETS: tuple[float, ...] = (0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0)

# 每条路由保留的延迟样本上限：长驻进程的直方图只反映最近的窗口，
# 否则样本列表随请求数单调增长（内存 + 渲染成本）。
MAX_SAMPLES_PER_ROUTE = 1000


class Metrics:
    """进程内的请求计数器 + 延迟直方图，事件循环安全。

    服务器在单个 asyncio 循环中记录观测值，所以普通 dict 不需要加锁。
    延迟样本是滑动窗口（每条路由最近 MAX_SAMPLES_PER_ROUTE 个），
    ``*_count`` 因此反映的是窗口内请求数，而 ``http_requests_total``
    是进程启动以来的累计值。
    """

    def __init__(self) -> None:
        self._requests: dict[tuple[str, str, int], int] = {}
        self._durations: dict[str, deque[float]] = {}

    def observe(self, method: str, route: str, status: int, duration: float) -> None:
        """在其路由模板下记录一个已完成的请求。"""
        key = (method, route, status)
        self._requests[key] = self._requests.get(key, 0) + 1
        samples = self._durations.setdefault(route, deque(maxlen=MAX_SAMPLES_PER_ROUTE))
        samples.append(duration)

    def render(self) -> str:
        """渲染 Prometheus 文本展示格式。"""
        lines: list[str] = []
        for (method, route, status), count in sorted(self._requests.items()):
            lines.append(
                f'http_requests_total{{method="{method}",route="{route}",'
                f'status="{status}"}} {count}'
            )
        for route, durations in sorted(self._durations.items()):
            for bound in BUCKETS:
                cum = sum(1 for d in durations if d <= bound)
                lines.append(
                    f'http_request_duration_seconds_bucket{{route="{route}",le="{bound}"}} {cum}'
                )
            lines.append(
                f'http_request_duration_seconds_bucket{{route="{route}",'
                f'le="+Inf"}} {len(durations)}'
            )
            lines.append(
                f'http_request_duration_seconds_sum{{route="{route}"}} {sum(durations):.6f}'
            )
            lines.append(f'http_request_duration_seconds_count{{route="{route}"}} {len(durations)}')
        return "\n".join(lines) + "\n"
