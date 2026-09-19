"""带路由模板 label 的请求指标（阶段 4）。

继承自 chat-agent 评审中的 B4/B5：计数器和直方图的 label 使用路由模板
（``/v1/agents/{module}/invoke``），绝不用原始请求路径——按对话划分的
路径会为每个 id 创建一条时间序列，让 Prometheus 基数爆炸。

刻意不引入依赖：渲染 Prometheus 文本展示格式只要几十行，而把
prometheus-client 挡在基座之外能让依赖树保持轻薄。token / 业务指标
属于模块。

结构（M1 治理）：``_LabeledHistogram`` 公共基类承载"计数器 + 滑动窗口
直方图 + bucket 渲染"，三套近乎复制的实现归一为三个薄子类，新指标只
需声明名字模板与 label 集合。

**语义与口径声明**：``*_total`` 是进程启动以来的累计计数；
``*_count`` / bucket 反映滑动窗口（每序列最近 MAX_SAMPLES_PER_ROUTE
个样本）——两者分母口径不同是有意为之的内存权衡，因此本子系统不是
Prometheus 规范意义上的累计直方图；``rate()`` 应基于 ``*_total`` 计算。
所有指标都是**单进程**口径，多 worker 部署下各进程各自计数（聚合需在
采集侧完成，见 README）。
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable

BUCKETS: tuple[float, ...] = (0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0)

# 每条序列保留的延迟样本上限：长驻进程的直方图只反映最近的窗口，
# 否则样本列表随请求数单调增长（内存 + 渲染成本）。
MAX_SAMPLES_PER_ROUTE = 1000

# 工具执行结果的封闭枚举：label 不接纳任意字符串，与路由模板同源地
# 防基数爆炸。
TOOL_OUTCOMES: tuple[str, ...] = ("ok", "timeout", "error")

# 记忆系统操作的封闭枚举（M6）：与 TOOL_OUTCOMES 同源，label 不接纳
# 任意字符串，防基数爆炸。M6c 的管线操作（extract/consolidate/…）沿用
# 同一个封闭集。
MEMORY_OUTCOMES: tuple[str, ...] = ("ok", "error", "degraded")


class _LabeledHistogram:
    """计数器 + 滑动窗口延迟直方图的公共基类（M1）。

    子类提供 total/bucket 指标名前缀与两个 label 渲染函数（计数键的
    label 集与延迟序列的 label 集），观测与渲染全部复用基类。
    """

    total_name: str
    bucket_prefix: str

    def __init__(self) -> None:
        self._counts: dict[tuple[str, ...], int] = {}
        self._durations: dict[str, deque[float]] = {}

    def _record(self, key: tuple[str, ...], series: str, duration: float) -> None:
        """记录一次观测：计数累计 + 样本入滑动窗口。"""
        self._counts[key] = self._counts.get(key, 0) + 1
        self._durations.setdefault(series, deque(maxlen=MAX_SAMPLES_PER_ROUTE)).append(duration)

    def _render(
        self,
        counter_labels: Callable[[tuple[str, ...]], str],
        series_labels: Callable[[str], str],
        *,
        # 历史行为：HTTP 指标即使为空也输出一个换行；tool/memory 为空
        # 时输出空串。保持渲染结果逐字节不变。
        trailing_newline_when_empty: bool,
    ) -> str:
        lines: list[str] = []
        for key, count in sorted(self._counts.items()):
            lines.append(f"{self.total_name}{{{counter_labels(key)}}} {count}")
        for series, durations in sorted(self._durations.items()):
            lab = series_labels(series)
            for bound in BUCKETS:
                cum = sum(1 for d in durations if d <= bound)
                lines.append(f'{self.bucket_prefix}_bucket{{{lab},le="{bound}"}} {cum}')
            lines.append(f'{self.bucket_prefix}_bucket{{{lab},le="+Inf"}} {len(durations)}')
            lines.append(f"{self.bucket_prefix}_sum{{{lab}}} {sum(durations):.6f}")
            lines.append(f"{self.bucket_prefix}_count{{{lab}}} {len(durations)}")
        if not lines:
            return "\n" if trailing_newline_when_empty else ""
        return "\n".join(lines) + "\n"


class Metrics(_LabeledHistogram):
    """进程内的请求计数器 + 延迟直方图，事件循环安全。

    服务器在单个 asyncio 循环中记录观测值，所以普通 dict 不需要加锁。
    口径声明见模块 docstring（``*_total`` 累计 vs 窗口 ``*_count``）。
    """

    def __init__(self) -> None:
        super().__init__()
        self.total_name = "http_requests_total"
        self.bucket_prefix = "http_request_duration_seconds"

    def observe(self, method: str, route: str, status: int, duration: float) -> None:
        """在其路由模板下记录一个已完成的请求。"""
        self._record((method, route, str(status)), route, duration)

    def render(self) -> str:
        """渲染 Prometheus 文本展示格式。"""

        def counter_labels(key: tuple[str, ...]) -> str:
            method, route, status = key
            return f'method="{method}",route="{route}",status="{status}"'

        return self._render(
            counter_labels, lambda route: f'route="{route}"', trailing_newline_when_empty=True
        )


class ToolMetrics(_LabeledHistogram):
    """进程内的工具执行计数器 + 延迟直方图，事件循环安全。

    工具名来自代码（注册表条目与模块 ``get_tools`` 的静态定义），不是
    用户输入，因此基数天然有界——这与 HTTP 路由必须用模板 label 的
    道理相同，但无需再设白名单。

    池的 ``_TimeoutTool`` 是所有工具执行的唯一收口（模块工具与工具库
    工具 alike），在它身上挂钩即可覆盖全部工具。
    """

    def __init__(self) -> None:
        super().__init__()
        self.total_name = "tool_executions_total"
        self.bucket_prefix = "tool_duration_seconds"

    def observe_tool(self, name: str, outcome: str, duration: float) -> None:
        """记录一次已完成的工具执行；``outcome`` 必须属于 TOOL_OUTCOMES。"""
        if outcome not in TOOL_OUTCOMES:
            raise ValueError(f"unknown tool outcome {outcome!r}; expected {TOOL_OUTCOMES}")
        self._record((name, outcome), name, duration)

    def render_tool_metrics(self) -> str:
        """渲染工具指标的 Prometheus 文本展示格式（与 Metrics.render 拼接）。"""

        def counter_labels(key: tuple[str, ...]) -> str:
            name, outcome = key
            return f'tool="{name}",outcome="{outcome}"'

        return self._render(
            counter_labels, lambda name: f'tool="{name}"', trailing_newline_when_empty=False
        )


# 进程级单例：工具池在 bootstrap 装配一次，全局挂钩是它的自然对应物；
# 与按应用实例化的 Metrics 不同，它没有"每个应用一套"的需求。
TOOL_METRICS = ToolMetrics()


class MemoryMetrics(_LabeledHistogram):
    """进程内的记忆操作计数器 + 延迟直方图，事件循环安全。

    操作名来自代码（service/pipeline 的静态调用点），不是用户输入，
    基数天然有界。
    """

    def __init__(self) -> None:
        super().__init__()
        self.total_name = "memory_operations_total"
        self.bucket_prefix = "memory_duration_seconds"

    def observe(self, op: str, outcome: str, duration: float) -> None:
        """记录一次已完成的记忆操作；outcome 必须属于 MEMORY_OUTCOMES。"""
        if outcome not in MEMORY_OUTCOMES:
            raise ValueError(f"unknown memory outcome {outcome!r}; expected {MEMORY_OUTCOMES}")
        self._record((op, outcome), op, duration)

    def render_memory_metrics(self) -> str:
        """渲染记忆指标的 Prometheus 文本展示格式。"""

        def counter_labels(key: tuple[str, ...]) -> str:
            op, outcome = key
            return f'op="{op}",outcome="{outcome}"'

        return self._render(
            counter_labels, lambda op: f'op="{op}"', trailing_newline_when_empty=False
        )


MEMORY_METRICS = MemoryMetrics()
