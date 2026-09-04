"""Explicit-manifest module registry (ADR-002).

``AGENT_MODULES`` is the single source of truth for which modules are active
and in what order. For each name the registry:

1. validates the name against a strict identifier pattern (anti path-traversal
   / import-arbitrary-code);
2. imports ``{prefix}.{name}`` (default ``agent_base.modules.{name}``);
3. reads the ``module`` attribute — the ``AgentModule`` instance;
4. validates it conforms to the contract and that its declared ``name``
   matches the requested one.

Any failure aborts startup — a bad module is never silently skipped.
"""

from __future__ import annotations

import importlib
import re
from typing import Any, cast

from agent_base.core.contracts import AgentModule

# Module names: lowercase identifier, no path separators, no leading digit.
# importlib guards keywords, but a strict allowlist here keeps the failure
# message actionable and rules out ``..``-style escapes.
_MODULE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class RegistryError(ValueError):
    """Raised when module loading fails (unknown name / dup / contract)."""


def load_modules(
    names: list[str],
    *,
    prefix: str = "agent_base.modules",
) -> dict[str, AgentModule]:
    """Load and validate the named modules in order.

    Returns an ordered mapping ``name -> AgentModule`` (insertion order ==
    assembly order). Raises ``RegistryError`` on the first problem.

    ``prefix`` is injectable so tests can point the registry at a fixture
    package and prove the mechanism is decoupled from the base itself.
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
