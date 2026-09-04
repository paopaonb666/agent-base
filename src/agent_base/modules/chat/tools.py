"""Tools contributed by the chat module.

The shared tool pool and ``ToolNode`` execution land in Stage 3. Until then
the sample module contributes an empty tool list — this file exists to fix
the physical layout (graph / module / tools) that every module follows.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool


def get_tools() -> list[BaseTool]:
    """Return the tools this module contributes to the shared pool."""
    return []
