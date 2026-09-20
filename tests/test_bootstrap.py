"""core.bootstrap 的单元测试：模块图缓存 + supervisor 保留名。"""

from __future__ import annotations

from typing import Any

import pytest

from agent_base.core.bootstrap import (
    SUPERVISOR_MODULE,
    AgentRuntime,
    create_runtime,
    validate_module_names,
)
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


# -- 快档装配（T0.2：形成管线的 fast 侧在组合根接线） -----------------------


def _fast_settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "llm_api_key": "sk-test",
        "checkpointer_backend": "memory",
        "llm_fast_base_url": "https://fast.example.com/v1",
        "llm_fast_model": "glm-4.5-flash",
    }
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


async def test_create_runtime_wires_resilient_fast_side() -> None:
    """LLM_FAST_* 已配置且非 all_main：管线拿到 ResilientLLM，回退主力。"""
    from agent_base.core.llm import ResilientLLM

    rt = await create_runtime(_fast_settings())
    try:
        assert rt.memory is not None and rt.memory.pipeline is not None
        fast = rt.memory.pipeline._fast_llm
        assert isinstance(fast, ResilientLLM)
        assert fast._fallback is rt.memory.pipeline._llm
    finally:
        await rt.close()


async def test_create_runtime_all_main_has_no_fast_side() -> None:
    """all_main 档：即使 LLM_FAST_* 已配置也不装配快档（历史行为）。"""
    rt = await create_runtime(_fast_settings(memory_pipeline_profile="all_main"))
    try:
        assert rt.memory is not None and rt.memory.pipeline is not None
        assert rt.memory.pipeline._fast_llm is None
    finally:
        await rt.close()


async def test_create_runtime_unconfigured_fast_has_no_fast_side() -> None:
    """LLM_FAST_* 未配置：不装配快档。"""
    rt = await create_runtime(
        Settings(_env_file=None, llm_api_key="sk-test", checkpointer_backend="memory")
    )
    try:
        assert rt.memory is not None and rt.memory.pipeline is not None
        assert rt.memory.pipeline._fast_llm is None
    finally:
        await rt.close()
