"""Request metrics with route-template labels (Stage 4).

Inherits B4/B5 from the chat-agent review: the counter and histogram labels
use the ROUTE TEMPLATE (``/v1/agents/{module}/invoke``), never the raw
request path — a per-conversation path would create one time series per
id and explode Prometheus cardinality.

Deliberately dependency-free: rendering the Prometheus text exposition
format is a few dozen lines, and keeping prometheus-client out of the base
keeps the dependency tree thin. Token / business metrics belong to modules.
"""

from __future__ import annotations

BUCKETS: tuple[float, ...] = (0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0)


class Metrics:
    """In-process request counters + latency histograms, event-loop safe.

    The server records observations from a single asyncio loop, so plain
    dicts need no locking.
    """

    def __init__(self) -> None:
        self._requests: dict[tuple[str, str, int], int] = {}
        self._durations: dict[str, list[float]] = {}

    def observe(self, method: str, route: str, status: int, duration: float) -> None:
        """Record one completed request under its route template."""
        key = (method, route, status)
        self._requests[key] = self._requests.get(key, 0) + 1
        self._durations.setdefault(route, []).append(duration)

    def render(self) -> str:
        """Render the Prometheus text exposition format."""
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
