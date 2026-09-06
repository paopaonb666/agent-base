"""显式清单的模块注册表（ADR-002）。

``AGENT_MODULES`` 是“哪些模块处于激活状态、以及以什么顺序”的唯一事实
来源。对于每个名字，registry 会：

1. 用严格的标识符模式校验名字（反路径遍历 / 反导入任意代码）；
2. 导入 ``{prefix}.{name}``（默认 ``agent_base.modules.{name}``）；
3. 读取 ``module`` 属性——即 ``AgentModule`` 实例；
4. 校验它符合契约、且其声明的 ``name`` 与请求的名字一致。

任何失败都会中止启动——有问题的模块永远不会被静默跳过。
"""

from __future__ import annotations

import importlib
import re
from typing import Any, cast

from agent_base.core.contracts import AgentModule

# 模块名：小写标识符，无路径分隔符，不能以数字开头。
# importlib 会防御关键字，但这里使用严格白名单能让失败信息更可操作，
# 并排除 ``..`` 之类的逃逸写法。
_MODULE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class RegistryError(ValueError):
    """模块加载失败（未知名字 / 重复 / 契约问题）时抛出。"""


def load_modules(
    names: list[str],
    *,
    prefix: str = "agent_base.modules",
) -> dict[str, AgentModule]:
    """按顺序加载并校验指定的模块。

    返回有序映射 ``name -> AgentModule``（插入顺序 == 装配顺序）。
    遇到第一个问题即抛出 ``RegistryError``。

    ``prefix`` 可注入，以便测试把 registry 指向一个 fixture 包，
    从而证明该机制与基座本身解耦。
    """
    modules: dict[str, AgentModule] = {}
    for name in names:
        if name in modules:
            raise RegistryError(f"duplicate module {name!r} in AGENT_MODULES")
        modules[name] = _load_one(name, prefix=prefix)
    return modules


def _load_one(name: str, *, prefix: str) -> AgentModule:
    _validate_name(name)
    full_name = f"{prefix}.{name}"
    try:
        package = importlib.import_module(full_name)
    except ModuleNotFoundError as exc:
        if exc.name == full_name:
            raise RegistryError(f"unknown module {name!r} (no such package {full_name!r})") from exc
        raise RegistryError(f"module {name!r} failed to import: {exc}") from exc

    obj = getattr(package, "module", None)
    if obj is None:
        raise RegistryError(
            f"module {name!r} does not expose a 'module' object in {full_name}.__init__"
        )
    _validate_contract(obj, name)
    return cast(AgentModule, obj)


def _validate_name(name: str) -> None:
    if not isinstance(name, str) or not _MODULE_NAME_RE.match(name):
        raise RegistryError(f"invalid module name {name!r}; must match {_MODULE_NAME_RE.pattern}")


def _validate_contract(obj: Any, expected_name: str) -> None:
    declared = getattr(obj, "name", None)
    if declared != expected_name:
        raise RegistryError(f"module loaded as {expected_name!r} but declares name {declared!r}")
    description = getattr(obj, "description", None)
    if not isinstance(description, str) or not description:
        raise RegistryError(f"module {expected_name!r} must declare a non-empty 'description'")
    if not callable(getattr(obj, "build_graph", None)):
        raise RegistryError(f"module {expected_name!r} must implement 'build_graph(ctx)'")
    if not callable(getattr(obj, "get_tools", None)):
        raise RegistryError(f"module {expected_name!r} must implement 'get_tools()'")
