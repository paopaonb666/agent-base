"""MCP 工具来源：把 MCP Server 接成共享池的外部工具（可选 extras）。

语义对齐工具库（``tools/registry.py``）的「启用即承诺」：

- ``MCP_ENABLED=false``（默认）：零解析、零导入——装配路径与本模块
  不存在时完全一致；
- ``MCP_ENABLED=true`` 是承诺：``[mcp]`` extras（langchain-mcp-adapters）
  必须已安装、``MCP_SERVERS_JSON`` 必须是合法的非空配置、server 必须
  能连上，否则启动即 ``McpError``，错误信息带可执行的修复指引；绝不
  静默缺席。

工具名由适配器统一加服务器名前缀（``{server}_{tool}``），经
``build_tool_pool`` 并入共享池——超时包装、审计收口与重名快速失败
全部自动继承，不另开旁路。
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from langchain_core.tools import BaseTool

if TYPE_CHECKING:  # pragma: no cover - import avoided at runtime
    from agent_base.core.config import Settings

# server 名会成为工具名前缀，与模块名同样严格（registry 同款白名单，
# 反路径遍历 / 反导入任意代码）。
_SERVER_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")

# langchain-mcp-adapters 0.3 的连接形态（transport 字面量）。
_KNOWN_TRANSPORTS = ("stdio", "streamable_http", "sse", "websocket")


class McpError(ValueError):
    """MCP 工具来源装配失败（缺依赖 / 配置非法 / 连不上）时抛出。"""


def _parse_servers(servers_json: str) -> dict[str, Any]:
    """解析并校验 ``MCP_SERVERS_JSON``（服务器名 → 连接配置）。

    只做结构性校验（名字白名单、transport 与 command/url 在位）；连接
    级参数由适配器在连接时校验。返回值直接作为适配器的 connections
    （按 Any 传递——真实 TypedDict 形态以适配器版本为准）。
    """
    text = servers_json.strip()
    if not text:
        raise McpError(
            "MCP_ENABLED=true 但 MCP_SERVERS_JSON 为空；示例："
            '{"docs": {"transport": "stdio", "command": "uvx",'
            ' "args": ["mcp-server-fetch"]}}'
        )
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise McpError(f"MCP_SERVERS_JSON 不是合法 JSON：{exc}") from exc
    if not isinstance(data, dict) or not data:
        raise McpError('MCP_SERVERS_JSON 必须是非空对象：{"server-name": {...}}')
    for name, conn in data.items():
        if not isinstance(name, str) or not _SERVER_NAME_RE.match(name):
            raise McpError(
                f"MCP server 名 {name!r} 非法：须匹配 {_SERVER_NAME_RE.pattern}"
                "（它会成为工具名前缀）"
            )
        if not isinstance(conn, dict):
            raise McpError(f"MCP server {name!r} 的配置必须是对象")
        transport = conn.get("transport")
        if transport not in _KNOWN_TRANSPORTS:
            raise McpError(
                f"MCP server {name!r} 的 transport {transport!r} 不支持；"
                f"expected one of {list(_KNOWN_TRANSPORTS)}"
            )
        if transport == "stdio" and not conn.get("command"):
            raise McpError(f'MCP server {name!r}：transport "stdio" 需要 "command" 字段')
        if transport in ("streamable_http", "sse", "websocket") and not conn.get("url"):
            raise McpError(f'MCP server {name!r}：transport {transport!r} 需要 "url" 字段')
    return data


def _import_client() -> Any:
    """延迟导入适配器（延迟到 ``MCP_ENABLED=true`` 的装配时刻）。

    单独成函数是为了可测性：测试对它打桩即可模拟 extras 缺失 / 适配器
    行为，不必操纵 ``sys.modules`` 的导入机制细节。
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient

    return MultiServerMCPClient


async def load_mcp_tools(settings: Settings) -> list[BaseTool]:
    """按 ``MCP_*`` 配置加载 MCP Server 的工具（供共享池并入）。

    未启用返回空列表——调用方可以无条件 ``extend``。启用但装配失败抛
    ``McpError``（快速失败 + 修复指引），与 ``ToolkitError`` 同哲学。
    """
    if not settings.mcp.enabled:
        return []
    try:
        client_cls = _import_client()
    except ImportError as exc:
        raise McpError('MCP_ENABLED=true 但未安装 [mcp] extras：pip install -e ".[mcp]"') from exc
    connections = _parse_servers(settings.mcp.servers_json)
    client = client_cls(connections, tool_name_prefix=True)
    try:
        return list(await client.get_tools())
    except Exception as exc:
        raise McpError(
            f"MCP server 连接失败（server 进程起来了吗？）：{type(exc).__name__}: {exc}"
        ) from exc
