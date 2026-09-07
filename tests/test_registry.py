"""core.registry 的测试（显式清单加载，ADR-002）。"""

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


# ── fixture 模块测试：证明该机制与基座解耦 ──


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
    """module-guide 五步配方，针对一个一次性包来演练。

    证明新增模块无需任何基座代码改动：registry 纯粹从注入的 prefix + name
    解析它，然后校验契约。
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


# ── 契约校验分支 ──


def test_module_empty_description_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    body = (
        "class Fake:\n"
        "    name = 'nodesc'\n"
        "    description = ''\n"
        "    def build_graph(self, ctx):\n"
        "        raise NotImplementedError\n"
        "    def get_tools(self):\n"
        "        return []\n"
        "module = Fake()\n"
    )
    _write_module(tmp_path, "fm_nodesc", "nodesc", body)
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(RegistryError, match="non-empty 'description'"):
        load_modules(["nodesc"], prefix="fm_nodesc")


def test_module_missing_build_graph_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = (
        "class Fake:\n"
        "    name = 'nograph'\n"
        "    description = 'fake'\n"
        "    def get_tools(self):\n"
        "        return []\n"
        "module = Fake()\n"
    )
    _write_module(tmp_path, "fm_nograph", "nograph", body)
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(RegistryError, match="build_graph"):
        load_modules(["nograph"], prefix="fm_nograph")


def test_module_missing_get_tools_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    body = (
        "class Fake:\n"
        "    name = 'notools'\n"
        "    description = 'fake'\n"
        "    def build_graph(self, ctx):\n"
        "        raise NotImplementedError\n"
        "module = Fake()\n"
    )
    _write_module(tmp_path, "fm_notools", "notools", body)
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(RegistryError, match="get_tools"):
        load_modules(["notools"], prefix="fm_notools")


def test_module_import_failure_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """自身 import 就爆炸的模块，应以 import 错误失败，而非 'unknown'。"""
    body = "import module_that_does_not_exist_anywhere_123\n"
    _write_module(tmp_path, "fm_broken", "broken", body)
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(RegistryError, match="failed to import"):
        load_modules(["broken"], prefix="fm_broken")


def test_module_syntax_error_normalized(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """语法错误等非 ModuleNotFoundError 的 import 失败同样归一为 RegistryError。"""
    _write_module(tmp_path, "fm_syntax", "syntax_err", "def broken(:\n    pass\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(RegistryError, match="SyntaxError"):
        load_modules(["syntax_err"], prefix="fm_syntax")
