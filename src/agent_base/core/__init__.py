"""agent 基座的核心装配：config、contract、registry、llm、bootstrap。

这里没有任何业务逻辑——只有模块插入其上的运行时装配。分阶段路线图见
``Agent基座软件开发计划.md``（置于本仓库之外）。
"""

from agent_base.core.config import Settings, SettingsError
from agent_base.core.contracts import AgentModule, ModuleContext
from agent_base.core.registry import RegistryError, load_modules

__all__ = [
    "AgentModule",
    "ModuleContext",
    "RegistryError",
    "Settings",
    "SettingsError",
    "load_modules",
]
