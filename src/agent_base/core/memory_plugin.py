"""记忆子系统的装配点：entry point 发现（0.3.0 拆分）。

基座源码**不** import ``agent_base_memory``。记忆实现通过发行包的
entry point 注册自己：

```toml
[project.entry-points."agent_base.memory"]
service = "agent_base_memory.plugin:build"
routes = "agent_base_memory.api:routers"
```

- ``service``：``async build(*, settings, llm, fast_llm, redis=None) ->
  MemoryPlugin | None``——``redis``（共享状态服务，ADR-012）自 0.2 契约
  起为**可选**形参：接受的实现可用它把捕获锁/注入快照外置到 Redis；
  0.1 老实现按旧三参签名调用，不受影响；
- ``routes``：``routers() -> list[APIRouter]``（管理 API，由 ``create_app`` 挂载）。

语义：``MEMORY_ENABLED=true`` 即"启用即承诺"——找不到实现就报错并提示
安装，而不是静默降级；``false`` 时零发现零导入，与没有记忆包时等价。
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool

from agent_base.core.contracts import ChatInvokerPort, MemoryServicePort

if TYPE_CHECKING:  # pragma: no cover - import avoided at runtime
    from agent_base.core.config import Settings
    from agent_base.extensions.redissvc import RedisService

ENTRY_POINT_GROUP = "agent_base.memory"
_SERVICE = "service"
_ROUTES = "routes"

logger = logging.getLogger(__name__)


class MemoryPluginError(RuntimeError):
    """记忆子系统已启用，但实现包缺失或装配失败。"""


@dataclass
class MemoryPlugin:
    """记忆实现返回给组合根的装配结果。"""

    service: MemoryServicePort
    tools: list[BaseTool]


def _select(name: str) -> Any | None:
    matches = [ep for ep in entry_points(group=ENTRY_POINT_GROUP) if ep.name == name]
    if not matches:
        return None
    if len(matches) > 1:
        logger.warning(
            "memory plugin: %d entry points named %r in %r; using %s",
            len(matches),
            name,
            ENTRY_POINT_GROUP,
            matches[0].value,
        )
    return matches[0]


def _missing() -> MemoryPluginError:
    return MemoryPluginError(
        "MEMORY_ENABLED=true but no memory implementation is installed. "
        "Install the optional package (pip install 'agent-base[memory]') "
        "or set MEMORY_ENABLED=false."
    )


async def build_memory_plugin(
    *,
    settings: Settings,
    llm: BaseChatModel | None,
    fast_llm: ChatInvokerPort | None,
    redis: RedisService | None = None,
) -> MemoryPlugin | None:
    """装配记忆服务；未启用返回 None，启用但实现缺失抛错。

    ``redis``（ADR-012）：按实现的签名选择性透传——0.2+ 的记忆包接受
    它并外置捕获锁/注入快照；0.1 老实现（三参签名）按原样调用。
    """
    if not settings.memory.enabled:
        return None
    entry = _select(_SERVICE)
    if entry is None:
        raise _missing()
    builder = entry.load()
    kwargs: dict[str, Any] = {"settings": settings, "llm": llm, "fast_llm": fast_llm}
    try:
        accepts_redis = "redis" in inspect.signature(builder).parameters
    except (TypeError, ValueError):  # pragma: no cover - 非函数式 builder 防御
        accepts_redis = False
    if accepts_redis:
        kwargs["redis"] = redis
    plugin = await builder(**kwargs)
    if plugin is None or not isinstance(plugin, MemoryPlugin):
        raise MemoryPluginError(
            f"memory plugin entry point {entry.value!r} returned no MemoryPlugin"
        )
    return plugin


def load_memory_routers() -> list[APIRouter]:
    """记忆管理 API 的路由（未启用时应由调用方跳过）。"""
    entry = _select(_ROUTES)
    if entry is None:
        raise _missing()
    return list(entry.load()())


__all__ = [
    "ENTRY_POINT_GROUP",
    "MemoryPlugin",
    "MemoryPluginError",
    "build_memory_plugin",
    "load_memory_routers",
]
