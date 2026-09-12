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

# 工具执行结果的封闭枚举：label 不接纳任意字符串，与路由模板同源地
# 防基数爆炸。
TOOL_OUTCOMES: tuple[str, ...] = ("ok", "timeout", "error")


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


class ToolMetrics:
    """进程内的工具执行计数器 + 延迟直方图，事件循环安全。

    工具名来自代码（注册表条目与模块 ``get_tools`` 的静态定义），不是
    用户输入，因此基数天然有界——这与 HTTP 路由必须用模板 label 的
    道理相同，但无需再设白名单。

    池的 ``_TimeoutTool`` 是所有工具执行的唯一收口（模块工具与工具库
    工具 alike），在它身上挂钩即可覆盖全部工具。
    """

    def __init__(self) -> None:
        self._executions: dict[tuple[str, str], int] = {}
        self._durations: dict[str, deque[float]] = {}

    def observe_tool(self, name: str, outcome: str, duration: float) -> None:
        """记录一次已完成的工具执行；``outcome`` 必须属于 TOOL_OUTCOMES。"""
        if outcome not in TOOL_OUTCOMES:
            raise ValueError(f"unknown tool outcome {outcome!r}; expected {TOOL_OUTCOMES}")
        key = (name, outcome)
        self._executions[key] = self._executions.get(key, 0) + 1
        samples = self._durations.setdefault(name, deque(maxlen=MAX_SAMPLES_PER_ROUTE))
        samples.append(duration)

    def render_tool_metrics(self) -> str:
        """渲染工具指标的 Prometheus 文本展示格式（与 Metrics.render 拼接）。"""
        lines: list[str] = []
        for (name, outcome), count in sorted(self._executions.items()):
            lines.append(f'tool_executions_total{{tool="{name}",outcome="{outcome}"}} {count}')
        for name, durations in sorted(self._durations.items()):
            for bound in BUCKETS:
                cum = sum(1 for d in durations if d <= bound)
                lines.append(f'tool_duration_seconds_bucket{{tool="{name}",le="{bound}"}} {cum}')
            lines.append(
                f'tool_duration_seconds_bucket{{tool="{name}",le="+Inf"}} {len(durations)}'
            )
            lines.append(f'tool_duration_seconds_sum{{tool="{name}"}} {sum(durations):.6f}')
            lines.append(f'tool_duration_seconds_count{{tool="{name}"}} {len(durations)}')
        return "\n".join(lines) + "\n" if lines else ""


# 进程级单例：工具池在 bootstrap 装配一次，全局挂钩是它的自然对应物；
# 与按应用实例化的 Metrics 不同，它没有"每个应用一套"的需求。
TOOL_METRICS = ToolMetrics()


# 记忆系统操作的封闭枚举（M6）：与 TOOL_OUTCOMES 同源，label 不接纳
# 任意字符串，防基数爆炸。M6c 的管线操作（extract/consolidate/…）沿用
# 同一个封闭集。
MEMORY_OUTCOMES: tuple[str, ...] = ("ok", "error", "degraded")


class MemoryMetrics:
    """进程内的记忆操作计数器 + 延迟直方图，事件循环安全。

    操作名来自代码（service/pipeline 的静态调用点），不是用户输入，
    基数天然有界。
    """

    def __init__(self) -> None:
        self._ops: dict[tuple[str, str], int] = {}
        self._durations: dict[str, deque[float]] = {}

    def observe(self, op: str, outcome: str, duration: float) -> None:
        """记录一次已完成的记忆操作；outcome 必须属于 MEMORY_OUTCOMES。"""
        if outcome not in MEMORY_OUTCOMES:
            raise ValueError(f"unknown memory outcome {outcome!r}; expected {MEMORY_OUTCOMES}")
        key = (op, outcome)
        self._ops[key] = self._ops.get(key, 0) + 1
        self._durations.setdefault(op, deque(maxlen=MAX_SAMPLES_PER_ROUTE)).append(duration)

    def render_memory_metrics(self) -> str:
        """渲染记忆指标的 Prometheus 文本展示格式。"""
        lines: list[str] = []
        for (op, outcome), count in sorted(self._ops.items()):
            lines.append(f'memory_operations_total{{op="{op}",outcome="{outcome}"}} {count}')
        for op, durations in sorted(self._durations.items()):
            for bound in BUCKETS:
                cum = sum(1 for d in durations if d <= bound)
                lines.append(f'memory_duration_seconds_bucket{{op="{op}",le="{bound}"}} {cum}')
            lines.append(
                f'memory_duration_seconds_bucket{{op="{op}",le="+Inf"}} {len(durations)}'
            )
            lines.append(f'memory_duration_seconds_sum{{op="{op}"}} {sum(durations):.6f}')
            lines.append(f'memory_duration_seconds_count{{op="{op}"}} {len(durations)}')
        return "\n".join(lines) + "\n" if lines else ""


MEMORY_METRICS = MemoryMetrics()
