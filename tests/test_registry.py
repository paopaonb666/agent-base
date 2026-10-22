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


# ── 多模块包（AGENT_MODULE_PACKAGES）：独立分发的模块包 ──


def test_module_found_in_second_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """第一个包没有该模块时继续找下一个包；包本身不存在也按"跳过"处理。"""
    body = (
        "class Hello:\n"
        "    def __init__(self):\n"
        "        self.name = 'hello'\n"
        "        self.description = 'hello from second'\n"
        "    def build_graph(self, ctx):\n"
        "        raise NotImplementedError\n"
        "    def get_tools(self):\n"
        "        return []\n"
        "module = Hello()\n"
    )
    _write_module(tmp_path, "fm_second", "hello", body)
    monkeypatch.syspath_prepend(str(tmp_path))
    modules = load_modules(["hello"], packages=["fm_absent_package", "fm_second"])
    assert modules["hello"].description == "hello from second"


def test_first_package_wins_when_name_repeated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """同名模块存在于多个包时，按 packages 顺序取第一个（可预测的覆盖规则）。"""
    first = (
        "class Hello:\n"
        "    def __init__(self):\n"
        "        self.name = 'hello'\n"
        "        self.description = 'first'\n"
        "    def build_graph(self, ctx):\n"
        "        raise NotImplementedError\n"
        "    def get_tools(self):\n"
        "        return []\n"
        "module = Hello()\n"
    )
    second = first.replace("'first'", "'second'")
    _write_module(tmp_path, "fm_p1", "hello", first)
    _write_module(tmp_path, "fm_p2", "hello", second)
    monkeypatch.syspath_prepend(str(tmp_path))
    modules = load_modules(["hello"], packages=["fm_p1", "fm_p2"])
    assert modules["hello"].description == "first"


def test_unknown_module_reports_searched_packages() -> None:
    with pytest.raises(RegistryError, match="not found in packages"):
        load_modules(["definitely_not_a_module_xyz"], packages=["agent_base.modules"])


def test_empty_packages_rejected() -> None:
    with pytest.raises(RegistryError, match="no module packages configured"):
        load_modules(["chat"], packages=[])


@pytest.mark.parametrize("bad", ["../evil", "Has-Dash", "9pkg", ""])
def test_invalid_package_name_rejected(bad: str) -> None:
    with pytest.raises(RegistryError, match="invalid module package"):
        load_modules(["chat"], packages=[bad])
