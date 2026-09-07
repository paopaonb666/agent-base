"""core.bootstrap 的单元测试：模块图缓存 + supervisor 保留名。"""

from __future__ import annotations

from typing import Any

import pytest

from agent_base.core.bootstrap import SUPERVISOR_MODULE, AgentRuntime, validate_module_names
from agent_base.core.config import Settings
from agent_base.core.registry import RegistryError


class _CountingModule:
    """记录 build_graph 调用次数的模块替身。"""

    def __init__(self, name: str) -> None:
        self.name = name
        self.description = "counting stand-in"
        self.build_calls = 0

    def build_graph(self, ctx: Any) -> object:
        self.build_calls += 1
        return object()

    def get_tools(self) -> list[Any]:
        return []


def test_graph_is_built_once_and_cached() -> None:
    """server 的每个请求都会调用 graph()——编译结果必须缓存。"""
    module = _CountingModule("chat")
    runtime = AgentRuntime(settings=Settings(_env_file=None), llm=None, modules={"chat": module})
    first = runtime.graph("chat")
    second = runtime.graph("chat")
    assert first is second
    assert module.build_calls == 1


def test_reserved_supervisor_name_rejected() -> None:
    """业务模块占用 supervisor 保留名必须启动即失败，而不是得到误导报错。"""
    impostor = _CountingModule(SUPERVISOR_MODULE)
    modules: dict[str, Any] = {SUPERVISOR_MODULE: impostor}
    with pytest.raises(RegistryError, match="reserved"):
        validate_module_names(modules)
