"""工具库注册表（M1 治理设施）。

把基座内置工具以 **声明式元数据** 的形式登记在一张表里：名字、类别、
安全级别、按需工厂、可用性谓词与独立超时。``build_toolkit_tools`` 在
bootstrap 时按 ``TOOLKIT_ENABLED`` 装配启用的工具并追加进共享池。

治理语义：

- **默认最小暴露**——``TOOLKIT_ENABLED`` 默认只含零依赖工具；高风险
  （``python_repl``）与依赖外部服务（``web_search``）的工具必须显式
  开启才存在。
- **启用即承诺**——显式启用的工具若依赖缺失或配置不全，是启动时的
  配置错误（``ToolkitError``），不是运行时的静默缺席；错误信息带上
  可执行的修复指引。
- **名字即契约**——``TOOLKIT_ENABLED`` 的合法取值就是注册表的键，
  拼错在启动时快速失败（与 ``AGENT_MODULES`` 的语义一致）。
- **per-tool 超时**——注册表可以声明独立超时（如网络搜索显著短于
  全局预算），经 ``build_tool_pool(timeouts=...)`` 下发到池。

本模块顶层导入各工具子包，后者只依赖 ``tools.spec``（类型）与各自的
第三方库——不存在反向依赖，因此无循环导入。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool

from agent_base.tools.basic import BASIC_TOOL_SPECS
from agent_base.tools.repl import PYTHON_REPL_SPEC
from agent_base.tools.search.tool import WEB_SEARCH_SPEC
from agent_base.tools.spec import ToolkitError, ToolSpec

if TYPE_CHECKING:  # pragma: no cover - import avoided at runtime
    from agent_base.core.config import Settings

# 注册表本体：键与 ToolSpec.name 一致，由各工具子包声明条目。
TOOLKIT: dict[str, ToolSpec] = {
    spec.name: spec for spec in (*BASIC_TOOL_SPECS, WEB_SEARCH_SPEC, PYTHON_REPL_SPEC)
}


def build_toolkit_tools(settings: Settings) -> list[BaseTool]:
    """按 ``settings.toolkit_enabled`` 装配启用的内置工具。

    启用但不可用 → ``ToolkitError``（快速失败 + 修复指引）；未启用 →
    工厂根本不会被调用（依赖缺失完全无感）。
    """
    tools: list[BaseTool] = []
    for name in settings.toolkit_enabled:
        spec = TOOLKIT.get(name)
        if spec is None:
            raise ToolkitError(
                f"unknown TOOLKIT_ENABLED entry {name!r}; known toolkit tools: {sorted(TOOLKIT)}"
            )
        if not spec.available(settings):
            raise ToolkitError(
                f"toolkit tool {name!r} is enabled but unavailable: {spec.unavailable_reason}"
            )
        tools.extend(spec.factory(settings))
    return tools


def toolkit_timeouts(settings: Settings) -> dict[str, float]:
    """收集已启用工具声明的独立超时（供 ``build_tool_pool(timeouts=...)``）。"""
    return {
        name: spec.timeout
        for name, spec in TOOLKIT.items()
        if name in settings.toolkit_enabled and spec.timeout is not None
    }


def known_toolkit_names() -> list[str]:
    """注册表里的全部工具名（错误信息与文档用）。"""
    return sorted(TOOLKIT)


__all__ = [
    "ToolSpec",
    "ToolkitError",
    "build_toolkit_tools",
    "known_toolkit_names",
    "toolkit_timeouts",
]
