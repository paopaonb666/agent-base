"""chat 模块贡献的工具。

阶段 3：这个样板工具演示了完整的“贡献 → 池 → ``ToolNode``”往返。
每个模块都遵循此布局；这里返回的工具通过 ``ModuleContext.tools``
在模块之间共享。
"""

from __future__ import annotations

from langchain_core.tools import BaseTool, tool


@tool
def echo(text: str) -> str:
    """把给定的文本原样返回，并加上 'echo:' 前缀。

    用于证明共享工具池端到端可用的样板工具；随着模块增多，请替换为
    真正的模块工具。
    """
    return f"echo: {text}"


def get_tools() -> list[BaseTool]:
    """返回本模块贡献给共享池的工具。"""
    return [echo]
