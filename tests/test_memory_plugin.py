"""记忆插件装配点（0.3.0 拆分）：entry point 发现语义。

这些用例不装真实记忆包也能跑——它们替换 ``importlib.metadata`` 的入口点，
验证"启用即承诺"、禁用零发现、以及基座导入图里不出现记忆包。
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import APIRouter

from agent_base.core import memory_plugin
from agent_base.core.memory_plugin import (
    ENTRY_POINT_GROUP,
    MemoryPlugin,
    MemoryPluginError,
    build_memory_plugin,
    load_memory_routers,
)


class _FakeEntryPoint:
    """最小 entry point：只实现 ``name`` / ``value`` / ``load()``。"""

    def __init__(self, name: str, value: str, loader: Any) -> None:
        self.name = name
        self.value = value
        self._loader = loader

    def load(self) -> Any:
        return self._loader


def _entry_points(*entries: _FakeEntryPoint):  # type: ignore[no-untyped-def]
    def fake(*, group: str) -> list[_FakeEntryPoint]:
        assert group == ENTRY_POINT_GROUP
        return list(entries)

    return fake


def _settings(enabled: bool) -> Any:
    return SimpleNamespace(memory=SimpleNamespace(enabled=enabled))


def _runtime_settings(state_dir: Path) -> Any:
    """启动用例用的最小 Settings：不读 .env，也不要求真实 API key。

    CI 里没有 .env，``Settings()`` 会因为 ``LLM_API_KEY`` 缺失在组合根之前
    就报错——那样测到的就不是记忆插件的装配语义了。sqlite 状态库落到
    tmp_path，避免用例在工作区里留文件。
    """
    from agent_base.core.config import Settings

    return Settings(
        _env_file=None,
        llm={"api_key": "sk-test"},
        checkpointer={"sqlite_path": str(state_dir / "state.db")},
    )


def _service_builder(plugin: Any):  # type: ignore[no-untyped-def]
    async def build(*, settings: Any, llm: Any, fast_llm: Any) -> Any:
        assert settings.memory.enabled is True
        assert llm is None
        assert fast_llm is None
        return plugin

    return build


@pytest.mark.asyncio
async def test_disabled_returns_none_without_looking_up_entry_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(*, group: str) -> list[Any]:
        raise AssertionError(f"disabled memory must not touch {group}")

    monkeypatch.setattr(memory_plugin, "entry_points", explode)
    assert await build_memory_plugin(settings=_settings(False), llm=None, fast_llm=None) is None


@pytest.mark.asyncio
async def test_missing_service_entry_point_fails_with_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(memory_plugin, "entry_points", _entry_points())
    with pytest.raises(MemoryPluginError) as excinfo:
        await build_memory_plugin(settings=_settings(True), llm=None, fast_llm=None)
    message = str(excinfo.value)
    assert "MEMORY_ENABLED=true" in message
    assert "pip install 'agent-base[memory]'" in message


@pytest.mark.asyncio
async def test_service_entry_point_returns_plugin(monkeypatch: pytest.MonkeyPatch) -> None:
    service = object()
    plugin = MemoryPlugin(service=service, tools=[])  # type: ignore[arg-type]
    entry = _FakeEntryPoint("service", "fake:build", _service_builder(plugin))
    monkeypatch.setattr(memory_plugin, "entry_points", _entry_points(entry))

    assert await build_memory_plugin(settings=_settings(True), llm=None, fast_llm=None) is plugin


@pytest.mark.asyncio
async def test_service_entry_point_returning_wrong_type_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = _FakeEntryPoint("service", "fake:build", _service_builder(object()))
    monkeypatch.setattr(memory_plugin, "entry_points", _entry_points(entry))

    with pytest.raises(MemoryPluginError, match="returned no MemoryPlugin"):
        await build_memory_plugin(settings=_settings(True), llm=None, fast_llm=None)


@pytest.mark.asyncio
async def test_duplicate_service_entry_points_use_the_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = MemoryPlugin(service=object(), tools=[])  # type: ignore[arg-type]
    second = MemoryPlugin(service=object(), tools=[])  # type: ignore[arg-type]
    entries = (
        _FakeEntryPoint("service", "first:build", _service_builder(first)),
        _FakeEntryPoint("service", "second:build", _service_builder(second)),
    )
    monkeypatch.setattr(memory_plugin, "entry_points", _entry_points(*entries))

    assert await build_memory_plugin(settings=_settings(True), llm=None, fast_llm=None) is first


def test_routes_are_loaded_from_entry_point(monkeypatch: pytest.MonkeyPatch) -> None:
    router = APIRouter()
    entry = _FakeEntryPoint("routes", "fake:routers", lambda: [router])
    monkeypatch.setattr(memory_plugin, "entry_points", _entry_points(entry))

    assert load_memory_routers() == [router]


def test_missing_routes_entry_point_fails_with_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(memory_plugin, "entry_points", _entry_points())
    with pytest.raises(MemoryPluginError, match=r"pip install 'agent-base\[memory\]'"):
        load_memory_routers()


def test_base_runtime_import_graph_never_imports_memory_package() -> None:
    """组合根与插件契约的导入图里不应出现 ``agent_base_memory``。

    ``agent_base.entrypoints.server`` 在模块级构造 ``app``（uvicorn 目标），
    构造时才会经 entry point 装载记忆包，因此这里只检查组合根一侧；
    静态检查见下一个用例。
    """
    code = (
        "import sys\n"
        "import agent_base.core.bootstrap\n"
        "import agent_base.core.memory_plugin\n"
        "leaked = sorted(m for m in sys.modules if m.split('.')[0] == 'agent_base_memory')\n"
        "assert not leaked, leaked\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr


def test_base_sources_have_no_static_import_of_memory_package() -> None:
    """基座源码不得出现 ``import agent_base_memory``（依赖方向靠测试守住）。"""
    source_root = Path(__file__).resolve().parents[1] / "src" / "agent_base"
    offenders: list[str] = []
    for path in source_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name.split(".")[0] == "agent_base_memory" for name in names):
                offenders.append(f"{path.relative_to(source_root)}:{node.lineno}")
    assert not offenders, offenders


@pytest.mark.asyncio
async def test_startup_fails_fast_when_memory_package_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """MEMORY_ENABLED=true 且没装记忆包：组合根在启动时报可操作的错误。"""
    from agent_base.core.bootstrap import create_runtime

    monkeypatch.setattr(memory_plugin, "entry_points", _entry_points())
    monkeypatch.setenv("MEMORY_ENABLED", "true")

    with pytest.raises(MemoryPluginError, match=r"pip install 'agent-base\[memory\]'"):
        await create_runtime(_runtime_settings(tmp_path))


@pytest.mark.asyncio
async def test_startup_without_memory_package_works_when_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """MEMORY_ENABLED=false：零发现、零导入，运行时照常装配。"""
    from agent_base.core.bootstrap import create_runtime

    monkeypatch.setattr(memory_plugin, "entry_points", _entry_points())
    monkeypatch.setenv("MEMORY_ENABLED", "false")

    runtime = await create_runtime(_runtime_settings(tmp_path))
    try:
        assert runtime.memory is None
        assert runtime.thread_index is not None
    finally:
        await runtime.close()
