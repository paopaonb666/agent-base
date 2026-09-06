"""共享工具池（阶段 3）。

每个模块通过 ``AgentModule.get_tools()`` 贡献工具；每个运行时装配一次
池，再通过 ``ModuleContext.tools`` 交还给模块。模块把池绑定到它们的
LLM，并通过 LangGraph 的 ``ToolNode`` 执行它，后者把工具失败归一成
``ToolMessage`` 反馈（模型看到错误并能恢复——工具异常绝不会让图崩溃）。

每个工具都用挂钟超时（``TOOL_TIMEOUT_SECONDS``）包装。在异步路径
（服务器和 CLI 使用的路径）中，超时会干净地取消 await；在同步路径中，
底层调用仍会在其工作线程中继续运行，但超过截止时间后结果会被丢弃
（有界泄漏，已记录在案的权衡——Python 线程无法被杀死）。
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any

from langchain_core.tools import BaseTool

from agent_base.core.contracts import AgentModule

DEFAULT_TOOL_TIMEOUT_SECONDS = 30.0


class ToolTimeoutError(TimeoutError):
    """某个工具超出了其挂钟预算而被截断。"""


class ToolPoolError(ValueError):
    """无法装配工具池（工具名重复）时抛出。"""


def handle_tool_error(exc: Exception) -> str:
    """把任何工具失败归一成模型可读的反馈。

    LangGraph 1.2 默认的 ``handle_tool_errors`` 只会转换参数错误，
    其余的一律重新抛出；基座的契约更强——坏掉的工具绝不能打断对话，
    因此每个异常都被转换为模型能够回应的 ``ToolMessage``（阶段 3）。
    """
    return f"tool execution failed: {exc!r}"


class _TimeoutTool(BaseTool):
    """包装 ``inner``，让同步和异步两种运行模式都遵守同一挂钟预算。"""

    inner: BaseTool
    timeout: float

    def _run(self, **kwargs: Any) -> Any:
        # 执行器 + future.result：即使工作线程本身无法被中断，
        # 截止时间也会在我们这一侧得到遵守。
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.inner.invoke, kwargs)
            try:
                return future.result(timeout=self.timeout)
            except FutureTimeoutError as exc:
                raise ToolTimeoutError(
                    f"tool {self.name!r} exceeded {self.timeout}s timeout"
                ) from exc

    async def _arun(self, **kwargs: Any) -> Any:
        try:
            return await asyncio.wait_for(self.inner.ainvoke(kwargs), timeout=self.timeout)
        except TimeoutError as exc:  # asyncio.TimeoutError 即内置的 TimeoutError（3.11+）
            raise ToolTimeoutError(f"tool {self.name!r} exceeded {self.timeout}s timeout") from exc


def build_tool_pool(
    modules: dict[str, AgentModule],
    *,
    timeout: float = DEFAULT_TOOL_TIMEOUT_SECONDS,
) -> list[BaseTool]:
    """把每个模块的工具收集进一个带超时包装、无冲突的池。

    顺序遵循模块装配顺序（``AGENT_MODULES``）。重复的工具名会中止启动——
    两个同名工具会让工具调用路由变得模糊，因此这是一个快速失败的
    配置错误。
    """
    pool: list[BaseTool] = []
    seen: set[str] = set()
    for module in modules.values():
        for tool in module.get_tools():
            if tool.name in seen:
                raise ToolPoolError(
                    f"duplicate tool name {tool.name!r} contributed by module "
                    f"{module.name!r}; tool names must be unique across modules"
                )
            seen.add(tool.name)
            pool.append(
                _TimeoutTool(
                    name=tool.name,
                    description=tool.description,
                    args_schema=tool.args_schema,
                    inner=tool,
                    timeout=timeout,
                )
            )
    return pool
