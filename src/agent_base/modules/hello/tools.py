"""hello 模块贡献的工具（module-guide 第 4 步）。"""

from __future__ import annotations

from langchain_core.tools import BaseTool


def get_tools() -> list[BaseTool]:
    """hello 模块不贡献任何工具。"""
    return []
