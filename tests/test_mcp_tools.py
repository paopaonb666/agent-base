"""MCP 工具来源的测试：启用语义、快速失败、配置校验、池集成。"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.tools import BaseTool, tool

from agent_base.core.config import Settings
from agent_base.core.tools import ToolPoolError, build_tool_pool
from agent_base.tools.mcp import McpError, load_mcp_tools


def _settings(**mcp: Any) -> Settings:
    return Settings(_env_file=None, mcp=mcp)


@tool
def _fake_fetch(url: str) -> str:
    """适配器返回的替身工具。"""
    return f"fetched {url}"


async def test_disabled_never_touches_the_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MCP_ENABLED=false（默认）时零导入——适配器工厂被碰即失败。"""

    def _must_not_import() -> object:
        raise AssertionError("disabled path must not import the adapter")

    monkeypatch.setattr("agent_base.tools.mcp._import_client", _must_not_import)
    settings = _settings(enabled=False, servers_json='{"docs": {"transport": "stdio"}}')
    assert await load_mcp_tools(settings) == []


async def test_enabled_with_empty_json_fails_fast() -> None:
    settings = _settings(enabled=True, servers_json="")
    with pytest.raises(McpError, match="MCP_SERVERS_JSON 为空"):
        await load_mcp_tools(settings)


async def test_enabled_with_bad_json_fails_fast() -> None:
    settings = _settings(enabled=True, servers_json="{not json")
    with pytest.raises(McpError, match="不是合法 JSON"):
        await load_mcp_tools(settings)


async def test_enabled_without_extra_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    """启用但未装 [mcp] extras：启动即报错并带安装指引。"""

    def _no_adapter() -> object:
        raise ImportError("No module named 'langchain_mcp_adapters'")

    monkeypatch.setattr("agent_base.tools.mcp._import_client", _no_adapter)
    settings = _settings(
        enabled=True,
        servers_json='{"docs": {"transport": "stdio", "command": "uvx"}}',
    )
    with pytest.raises(McpError, match=r"\[mcp\] extras"):
        await load_mcp_tools(settings)


@pytest.mark.parametrize(
    ("servers_json", "message"),
    [
        ('{"Bad-Name": {"transport": "stdio", "command": "x"}}', "server 名"),
        ('{"docs": ["not", "an object"]}', "配置必须是对象"),
        ('{"docs": {"command": "uvx"}}', "transport"),
        ('{"docs": {"transport": "stdio"}}', '需要 "command"'),
        ('{"docs": {"transport": "streamable_http"}}', '需要 "url"'),
    ],
)
async def test_invalid_server_configs_rejected(servers_json: str, message: str) -> None:
    settings = _settings(enabled=True, servers_json=servers_json)
    with pytest.raises(McpError, match=message):
        await load_mcp_tools(settings)


async def test_valid_config_loads_tools_with_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """合法配置走适配器：连接配置透传、强制服务器名前缀、工具进池。"""
    captured: dict[str, Any] = {}

    class _FakeClient:
        def __init__(self, connections: Any, **kwargs: Any) -> None:
            captured["connections"] = connections
            captured["kwargs"] = kwargs

        async def get_tools(self) -> list[BaseTool]:
            return [_fake_fetch]

    monkeypatch.setattr("agent_base.tools.mcp._import_client", lambda: _FakeClient)
    settings = _settings(
        enabled=True,
        servers_json='{"docs": {"transport": "stdio", "command": "uvx", "args": ["m"]}}',
    )
    tools = await load_mcp_tools(settings)
    assert [t.name for t in tools] == ["_fake_fetch"]
    assert isinstance(captured["connections"], dict) and "docs" in captured["connections"]
    assert captured["kwargs"] == {"tool_name_prefix": True}


async def test_unreachable_server_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    """启用且配置合法但 server 连不上：启动即失败，不静默缺席。"""

    class _DeadClient:
        def __init__(self, connections: Any, **kwargs: Any) -> None:
            pass

        async def get_tools(self) -> list[BaseTool]:
            raise ConnectionRefusedError("no server")

    monkeypatch.setattr("agent_base.tools.mcp._import_client", lambda: _DeadClient)
    settings = _settings(
        enabled=True,
        servers_json='{"docs": {"transport": "stdio", "command": "uvx"}}',
    )
    with pytest.raises(McpError, match="连接失败"):
        await load_mcp_tools(settings)


def test_pool_rejects_mcp_name_collision() -> None:
    """MCP 工具并入池后同样受重名快速失败保护（与模块工具撞名即中止）。"""

    @tool
    def docs_fetch(url: str) -> str:
        """冒名的 docs_fetch——与 mcp 前缀工具重名。"""
        return url

    @tool
    def mcp_tool(url: str) -> str:
        """适配器产出的前缀化工具替身。"""
        return url

    mcp_tool.name = "docs_fetch"  # 模拟与模块工具撞名的前缀化 MCP 工具

    class _FakeModule:
        name = "m"
        description = "stand-in"

        def get_tools(self) -> list[BaseTool]:
            return [docs_fetch]

    with pytest.raises(ToolPoolError, match="duplicate tool name 'docs_fetch'"):
        build_tool_pool({"m": _FakeModule()}, extra_tools=[mcp_tool])
