"""Tests for core.registry (explicit-manifest loading, ADR-002)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_base.core.contracts import AgentModule
from agent_base.core.registry import RegistryError, load_modules


def test_load_chat_module() -> None:
    modules = load_modules(["chat"])
    assert list(modules) == ["chat"]
    assert modules["chat"].name == "chat"
    assert isinstance(modules["chat"], AgentModule)


def test_duplicate_name_rejected() -> None:
    with pytest.raises(RegistryError, match="duplicate"):
        load_modules(["chat", "chat"])


def test_unknown_module_rejected() -> None:
    with pytest.raises(RegistryError, match="unknown module"):
        load_modules(["definitely_not_a_module_xyz"])


@pytest.mark.parametrize("bad", ["../evil", "has space", "1chat", "Chat", "a-b", ""])
def test_invalid_name_rejected(bad: str) -> None:
    with pytest.raises(RegistryError, match="invalid module name"):
        load_modules([bad])


# ── fixture-module tests: prove the mechanism is decoupled from the base ──


def _write_module(tmp_path: Path, package: str, name: str, body: str = "") -> None:
    mod_dir = tmp_path / package / name
    mod_dir.mkdir(parents=True)
    (tmp_path / package / "__init__.py").write_text("", encoding="utf-8")
    (mod_dir / "__init__.py").write_text(body, encoding="utf-8")


def test_module_without_object_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_module(tmp_path, "fm_empty", "empty")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(RegistryError, match="does not expose a 'module' object"):
        load_modules(["empty"], prefix="fm_empty")


def test_module_name_mismatch_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    body = (
        "class Fake:\n"
        "    name = 'other'\n"
        "    description = 'fake'\n"
        "    def build_graph(self, ctx):\n"
        "        raise NotImplementedError\n"
        "    def get_tools(self):\n"
        "        return []\n"
        "module = Fake()\n"
    )
    _write_module(tmp_path, "fm_mismatch", "mismatched", body)
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(RegistryError, match="declares name"):
        load_modules(["mismatched"], prefix="fm_mismatch")


def test_hello_module_loads_without_touching_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The module-guide's five-step recipe, exercised against a throwaway package.

    Proof that adding a module requires no base-code change: the registry
    resolves it purely from the injected prefix + name, then validates the
    contract.
    """
    body = (
        "class Hello:\n"
        "    def __init__(self):\n"
        "        self.name = 'hello'\n"
        "        self.description = 'hello sample'\n"
        "    def build_graph(self, ctx):\n"
        "        raise NotImplementedError\n"
        "    def get_tools(self):\n"
        "        return []\n"
        "module = Hello()\n"
    )
    _write_module(tmp_path, "fm_hello", "hello", body)
    monkeypatch.syspath_prepend(str(tmp_path))
    modules = load_modules(["hello"], prefix="fm_hello")
    assert modules["hello"].name == "hello"
    assert modules["hello"].description == "hello sample"
