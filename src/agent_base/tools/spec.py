"""``ToolSpec`` 条目类型：注册表的声明式元数据。

独立成模块是为了打断导入环——``registry`` 需要聚合各工具子包声明的
条目，而工具子包需要 ``ToolSpec`` 类型来声明它们；类型定义本身不依赖
任何一方。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool

if TYPE_CHECKING:  # pragma: no cover - import avoided at runtime
    from agent_base.core.config import Settings


class ToolkitError(ValueError):
    """工具库装配失败（启用了不可用的工具、未知工具名等）。"""


# 工具分层（ADR-010）：basic 永远绑定给模型；advanced 参与向量路由
# （top-k 召回）。两档是封闭枚举——分层语义只被 ToolRouter 消费。
KNOWN_TOOL_TIERS: tuple[str, ...] = ("basic", "advanced")


@dataclass(frozen=True)
class ToolSpec:
    """一个基座内置工具的注册条目。

    ``name``        —— 稳定标识；= ``TOOLKIT_ENABLED`` 条目 = 池中的工具名
    ``category``    —— 类别（basic / network / code），仅供观测与文档
    ``safety``      —— 安全级别（read / network / exec），为将来的审批门
                        预留的元数据；本阶段只登记不强制
    ``tier``        —— 分层（basic / advanced，ADR-010）：basic 永远绑定，
                        advanced 参与向量路由的 top-k 召回；默认 basic
    ``factory``     —— 惰性工厂：仅在工具被启用且可用时才调用，产出加入
                        共享池的 ``BaseTool`` 实例（可携带 settings 装配的
                        依赖，如搜索管理器）
    ``available``   —— 可用性谓词：依赖是否安装、必需配置是否齐全
    ``unavailable_reason`` —— 不可用时的可读修复指引（进入 ToolkitError）
    ``timeout``     —— 独立超时覆盖；None 沿用全局 ``TOOL_TIMEOUT_SECONDS``
    """

    name: str
    category: str
    safety: str
    factory: Callable[[Settings], list[BaseTool]]
    available: Callable[[Settings], bool]
    unavailable_reason: str
    timeout: float | None = None
    tier: str = "basic"

    def __post_init__(self) -> None:
        if self.tier not in KNOWN_TOOL_TIERS:
            raise ToolkitError(
                f"tool {self.name!r} declares unknown tier {self.tier!r}; "
                f"expected one of {list(KNOWN_TOOL_TIERS)}"
            )


__all__ = ["KNOWN_TOOL_TIERS", "ToolkitError", "ToolSpec"]
