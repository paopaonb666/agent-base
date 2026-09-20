"""成本计量（T4.1）："给 Agent 装电表"。

三件套：

- **采集**（``CostMeterHandler``）：模型级 LangChain callback，从
  ``on_llm_end`` 的 usage（``usage_metadata`` 或 OpenAI 风格
  ``llm_output.token_usage``）提取 token 数——``build_llm`` 在
  ``COST_ENABLED=true`` 时把它挂到每个客户端上，是全部 LLM 调用的
  唯一收口（图执行、形成管线、planner alike）；
- **累计**（``CostMeter``）：进程内按日/月聚合 + 价目表计价
  （``COST_PRICES_JSON``，单位 元/百万 tokens；未知模型计 0 并告警
  一次）。同步更新，预算门（T4.2）读它不需要 await；
- **账本**（``MemoryCostLedger`` / ``SqliteCostLedger``）：pending 行的
  落库恢复面——进程重启后预算累计从账本恢复。MySQL 后端暂回退内存
  账本（重启清零，方向安全：只会少记不会误熔断），表结构随 sqlite
  DDL 预留。

已知局限：单进程口径（与 /metrics 同款）；价目手工维护。
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Protocol

from langchain_core.callbacks import BaseCallbackHandler

from agent_base.extensions.metrics import COST_METRICS
from agent_base.extensions.observability import get_request_id

logger = logging.getLogger(__name__)

# sqlite 账本 DDL（幂等自举）。MySQL 的同名表随 sqlite 形态预留，
# mysql 后端的账本接入是后续任务（当前回退内存账本并告警）。
_COST_USAGE_DDL = """
CREATE TABLE IF NOT EXISTS cost_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    request_id TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL,
    profile TEXT NOT NULL DEFAULT 'main',
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens INTEGER NOT NULL DEFAULT 0,
    cost_cny REAL NOT NULL DEFAULT 0
);
"""
_COST_USAGE_INDEX_DDL = "CREATE INDEX IF NOT EXISTS idx_cost_usage_ts ON cost_usage (ts);"


@dataclass(frozen=True)
class UsageRecord:
    """一次模型调用的用量行。"""

    ts: float
    request_id: str
    model: str
    profile: str
    prompt_tokens: int
    completion_tokens: int
    cache_read_tokens: int
    cost_cny: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _usage_of(message: Any) -> tuple[int, int, int] | None:
    usage = getattr(message, "usage_metadata", None)
    if not usage:
        return None
    details = usage.get("input_token_details") or {}
    return (
        int(usage.get("input_tokens", 0) or 0),
        int(usage.get("output_tokens", 0) or 0),
        int(details.get("cache_read", 0) or 0),
    )


def _iter_generations(response: Any) -> Any:
    """兼容两种形态：真实管线的 ``LLMResult``（嵌套 list）与直接构造的
    ``ChatResult``（扁平 list）。"""
    for group in getattr(response, "generations", None) or []:
        if isinstance(group, (list, tuple)):
            yield from group
        else:
            yield group


def _extract_usage(response: Any) -> tuple[int, int, int]:
    """从 LLMResult / ChatResult 提取 (prompt, completion, cache_read)。

    优先 standard 的 ``usage_metadata``（挂在 generation 消息上，流式
    聚合与非流式一致；astream 路径的生成项本身就是分块对象，两种形态
    都探测）；回退 OpenAI 风格 ``llm_output.token_usage``；都没有就是
    全零（模型不回报 usage，不是错误）。
    """
    for generation in _iter_generations(response):
        try:
            message = getattr(generation, "message", None)
        except AttributeError:
            message = None
        for candidate in (message, generation):
            if candidate is None:
                continue
            usage = _usage_of(candidate)
            if usage is not None:
                return usage
    token_usage = (getattr(response, "llm_output", None) or {}).get("token_usage") or {}
    if token_usage:
        details = token_usage.get("prompt_tokens_details") or {}
        return (
            int(token_usage.get("prompt_tokens", 0) or 0),
            int(token_usage.get("completion_tokens", 0) or 0),
            int(details.get("cached_tokens", 0) or 0),
        )
    return 0, 0, 0


def _price_of(
    prices: dict[str, dict[str, float]], model: str, prompt: int, completion: int, cache_read: int
) -> float:
    """按价目表计价（元/百万 tokens）；未知模型 0 元（调用方告警）。"""
    table = prices.get(model)
    if not table:
        return 0.0
    cache_read = min(cache_read, prompt)
    miss = prompt - cache_read
    return (
        miss / 1e6 * float(table.get("input_miss", 0.0))
        + cache_read / 1e6 * float(table.get("input_hit", 0.0))
        + completion / 1e6 * float(table.get("output", 0.0))
    )


class CostMeter:
    """进程内用量累计器：按日/月聚合 + 待落库行队列。

    同步更新（callback 在事件循环内触发），预算门直接读聚合值。
    """

    def __init__(self, prices_json: str = "") -> None:
        self._prices: dict[str, dict[str, float]] = {}
        self._configure(prices_json)
        self._pending: deque[UsageRecord] = deque()
        # day-iso → 当日累计（元）；month-iso → 当月累计。
        self._day_totals: dict[str, float] = {}
        self._month_totals: dict[str, float] = {}
        self._warned_models: set[str] = set()

    def _configure(self, prices_json: str) -> None:
        try:
            data = json.loads(prices_json) if prices_json.strip() else {}
            self._prices = data if isinstance(data, dict) else {}
        except ValueError:
            logger.warning("cost: COST_PRICES_JSON 不是合法 JSON，按空价目运行")
            self._prices = {}

    def configure(self, prices_json: str) -> None:
        """服务装配时注入价目表（覆盖默认单例的空价目）。"""
        self._configure(prices_json)
        self._warned_models.clear()

    def observe(
        self, *, model: str, profile: str, prompt: int, completion: int, cache_read: int
    ) -> None:
        if model not in self._prices and model not in self._warned_models:
            self._warned_models.add(model)
            logger.warning(
                "cost: 模型 %r 不在 COST_PRICES_JSON 价目表中，按 0 元计量（补齐价目后账单才完整）",
                model,
            )
        cost = _price_of(self._prices, model, prompt, completion, cache_read)
        now = time.time()
        day = datetime.fromtimestamp(now).date().isoformat()
        month = day[:7]
        self._day_totals[day] = self._day_totals.get(day, 0.0) + cost
        self._month_totals[month] = self._month_totals.get(month, 0.0) + cost
        self._pending.append(
            UsageRecord(
                ts=now,
                request_id=get_request_id(),
                model=model,
                profile=profile,
                prompt_tokens=prompt,
                completion_tokens=completion,
                cache_read_tokens=cache_read,
                cost_cny=cost,
            )
        )
        COST_METRICS.observe_tokens(model, profile, "input", prompt)
        COST_METRICS.observe_tokens(model, profile, "output", completion)
        COST_METRICS.observe_cost(cost)

    def take_pending(self) -> list[UsageRecord]:
        """取走待落库行（账本 flush 时调用）。"""
        rows = list(self._pending)
        self._pending.clear()
        return rows

    def day_total_cny(self, day: str | None = None) -> float:
        key = day or datetime.fromtimestamp(time.time()).date().isoformat()
        return self._day_totals.get(key, 0.0)

    def month_total_cny(self, month: str | None = None) -> float:
        return self._month_totals.get(
            month or datetime.fromtimestamp(time.time()).date().isoformat()[:7], 0.0
        )

    def restore_day(self, day: str, cost: float) -> None:
        """进程重启后从账本恢复当日累计（叠加，不清零内存值）。"""
        self._day_totals[day] = self._day_totals.get(day, 0.0) + cost


class CostMeterHandler(BaseCallbackHandler):
    """模型级 usage 采集 callback（``build_llm`` 在 cost enabled 时挂载）。"""

    def __init__(self, *, model: str, profile: str, meter: CostMeter) -> None:
        super().__init__()
        self.model = model
        self.profile = profile
        self._meter = meter

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        prompt, completion, cache_read = _extract_usage(response)
        try:
            self._meter.observe(
                model=self.model,
                profile=self.profile,
                prompt=prompt,
                completion=completion,
                cache_read=cache_read,
            )
        except Exception:  # 计量绝不影响调用方
            logger.warning("cost: 用量记录失败", exc_info=True)


class CostLedger(Protocol):
    """账本协议：批量落库 + 按时间点累计（重启恢复用）。"""

    async def insert(self, rows: list[UsageRecord]) -> None: ...

    async def totals_since(self, ts: float) -> float: ...


class MemoryCostLedger:
    """内存账本（测试、memory 后端与 mysql 后端的回退形态）。"""

    def __init__(self) -> None:
        self._rows: list[UsageRecord] = []

    async def insert(self, rows: list[UsageRecord]) -> None:
        self._rows.extend(rows)

    async def totals_since(self, ts: float) -> float:
        return sum(row.cost_cny for row in self._rows if row.ts >= ts)


class SqliteCostLedger:
    """sqlite 账本：跟随 checkpointer 的 sqlite 库文件，独立连接。"""

    _db: Any  # aiosqlite.Connection（惰性导入，注解用 Any）

    @classmethod
    async def create(cls, db_path: str) -> SqliteCostLedger:
        import aiosqlite

        self = cls()
        self._db = await aiosqlite.connect(db_path)
        await self._db.execute(_COST_USAGE_DDL)
        await self._db.execute(_COST_USAGE_INDEX_DDL)
        await self._db.commit()
        return self

    async def insert(self, rows: list[UsageRecord]) -> None:
        if not rows:
            return
        await self._db.executemany(
            "INSERT INTO cost_usage (ts, request_id, model, profile, prompt_tokens,"
            " completion_tokens, cache_read_tokens, cost_cny)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    r.ts,
                    r.request_id,
                    r.model,
                    r.profile,
                    r.prompt_tokens,
                    r.completion_tokens,
                    r.cache_read_tokens,
                    r.cost_cny,
                )
                for r in rows
            ],
        )
        await self._db.commit()

    async def totals_since(self, ts: float) -> float:
        async with self._db.execute(
            "SELECT COALESCE(SUM(cost_cny), 0) FROM cost_usage WHERE ts >= ?", (ts,)
        ) as cursor:
            row = await cursor.fetchone()
            return float(row[0]) if row else 0.0

    async def aclose(self) -> None:
        await self._db.close()


# 进程级单例：与 TOOL_METRICS 同理，全部模型客户端共享一份累计。
COST_METER = CostMeter()
