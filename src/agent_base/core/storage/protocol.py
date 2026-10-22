"""基座存储协议：线程索引与计划快照（与记忆子系统无关）。"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from agent_base.core.storage.models import ThreadIndex


@runtime_checkable
class ThreadIndexRepo(Protocol):
    """会话索引（thread_index 表，S1 线程作用域的数据面）。

    线程元数据自管：invoke 时 upsert（属主/模块/标题/更新时间），线程
    端点的列表与属主校验改查该表——不再全表扫描 checkpointer，也不再
    向任意用户返回全库线程列表。
    """

    async def upsert_thread_index(self, entry: ThreadIndex) -> None: ...

    async def get_thread_index(self, thread_id: str) -> ThreadIndex | None: ...

    async def list_thread_index(
        self, user_id: str, *, module: str | None = None, limit: int = 50
    ) -> list[ThreadIndex]: ...

    async def delete_thread_index(self, thread_id: str) -> None: ...


@runtime_checkable
class PlanRepo(Protocol):
    """计划快照（plan_snapshots 表，M10 跨轮延续的数据面）。

    planner 图在每个状态变更点（拆解/推进/重规划/完成）upsert 快照；
    plan 检查端点读表而非图状态——chat 轮穿插不再重置计划（chat 图的
    checkpoint 只含 messages 通道）。``tasks`` 是序列化好的任务清单
    （``list[dict]``，后端自行 JSON 化）。
    """

    async def save_plan(
        self,
        *,
        thread_id: str,
        module: str,
        tasks: list[dict[str, Any]],
        cursor: int,
        replans: int,
    ) -> None: ...

    async def get_plan(self, thread_id: str) -> dict[str, Any] | None: ...

    async def delete_plan(self, thread_id: str) -> None: ...


class CoreStore(ThreadIndexRepo, PlanRepo, Protocol):
    """基座存储：线程索引 + 计划快照，外加连接释放。"""

    async def aclose(self) -> None: ...
