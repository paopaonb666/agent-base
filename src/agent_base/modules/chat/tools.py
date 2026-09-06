"""Tools contributed by the chat module.

Stage 3: the sample tool demonstrates the full contribution → pool →
``ToolNode`` round-trip. Every module follows this layout; tools returned
here are shared across modules through ``ModuleContext.tools``.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool, tool


@tool
def echo(text: str) -> str:
    """Echo the given text back, prefixed with 'echo:'.

    Sample tool proving the shared tool pool works end to end; replace with
    real module tools as modules grow.
    """
    return f"echo: {text}"


def get_tools() -> list[BaseTool]:
    """Return the tools this module contributes to the shared pool."""
    return [echo]
