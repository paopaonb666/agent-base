"""显式清单的模块注册表（ADR-002）。

``AGENT_MODULES`` 是“哪些模块处于激活状态、以及以什么顺序”的唯一事实
来源。对于每个名字，registry 会：

1. 用严格的标识符模式校验名字（反路径遍历 / 反导入任意代码）；
2. 依次尝试各个模块包，导入 ``{package}.{name}``（默认只有
   ``agent_base.modules``，可用 ``AGENT_MODULE_PACKAGES`` 追加独立分发的
   模块包，同名时以靠前的包为准）；
3. 读取 ``module`` 属性——即 ``AgentModule`` 实例；
4. 校验它符合契约、且其声明的 ``name`` 与请求的名字一致。

任何失败都会中止启动——有问题的模块永远不会被静默跳过。
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Sequence
from typing import Any, cast

from agent_base.core.contracts import AgentModule

# 默认模块包：基座自带的样板模块（chat/writer/hello/planner）。
DEFAULT_MODULE_PACKAGES: tuple[str, ...] = ("agent_base.modules",)

# 模块名：小写标识符，无路径分隔符，不能以数字开头。
# importlib 会防御关键字，但这里使用严格白名单能让失败信息更可操作，
# 并排除 ``..`` 之类的逃逸写法。
_MODULE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")

# 模块包（导入前缀）：小写标识符，点分隔；同样只允许白名单字符，
# 避免配置里的笔误变成奇怪的导入错误。
_PACKAGE_RE = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$")


class RegistryError(ValueError):
    """模块加载失败（未知名字 / 重复 / 契约问题）时抛出。"""


def load_modules(
    names: list[str],
    *,
    prefix: str | None = None,
    packages: Sequence[str] | None = None,
) -> dict[str, AgentModule]:
    """按顺序加载并校验指定的模块。

    返回有序映射 ``name -> AgentModule``（插入顺序 == 装配顺序）。
    遇到第一个问题即抛出 ``RegistryError``。

    ``packages`` 是模块包的导入前缀，按顺序尝试；``AGENT_MODULE_PACKAGES``
    由此注入，独立分发的模块包（``acme_modules`` 之类）只要加进去即可，
    基座不需要改导入图。``prefix`` 是单包快捷方式（测试把 registry 指向
    fixture 包时用它），两者同时给出时 ``prefix`` 优先。
    """
    search = _search_packages(prefix=prefix, packages=packages)
    modules: dict[str, AgentModule] = {}
    for name in names:
        if name in modules:
            raise RegistryError(f"duplicate module {name!r} in AGENT_MODULES")
        modules[name] = _load_one(name, packages=search)
    return modules


def _search_packages(*, prefix: str | None, packages: Sequence[str] | None) -> list[str]:
    if prefix is not None:
        candidates = [prefix]
    elif packages is not None:
        candidates = [str(p) for p in packages]
    else:
        candidates = list(DEFAULT_MODULE_PACKAGES)
    if not candidates:
        raise RegistryError("no module packages configured (AGENT_MODULE_PACKAGES is empty)")
    for candidate in candidates:
        if not _PACKAGE_RE.match(candidate):
            raise RegistryError(
                f"invalid module package {candidate!r}; must match {_PACKAGE_RE.pattern}"
            )
    return candidates


def _load_one(name: str, *, packages: Sequence[str]) -> AgentModule:
    _validate_name(name)
    for package_name in packages:
        full_name = f"{package_name}.{name}"
        try:
            package = importlib.import_module(full_name)
        except ModuleNotFoundError as exc:
            # 包本身不存在时 exc.name 是缺失的前缀（如 "acme_modules"），而
            # 不是完整模块名；模块内部 import 失败则是别的名字。两种情形分开：
            # 前者换下一个包，后者按"导入失败"上报。
            missing = exc.name or ""
            if missing and (full_name == missing or full_name.startswith(missing + ".")):
                continue
            raise RegistryError(
                f"module {name!r} failed to import from {package_name!r}: {exc}"
            ) from exc
        except Exception as exc:
            # SyntaxError、NameError 等 import 期错误同样以 RegistryError 归一：
            # 本文件的契约是"任何失败都以 RegistryError 中止"，入口层据此
            # 给出统一的报错形态，而不是裸堆栈。
            raise RegistryError(
                f"module {name!r} failed to import from {package_name!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        obj = getattr(package, "module", None)
        if obj is None:
            raise RegistryError(
                f"module {name!r} does not expose a 'module' object in {full_name}.__init__"
            )
        _validate_contract(obj, name)
        return cast(AgentModule, obj)
    tried = ", ".join(repr(p) for p in packages)
    raise RegistryError(f"unknown module {name!r} (not found in packages: {tried})")


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
