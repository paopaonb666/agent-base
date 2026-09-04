"""Core wiring for the agent base: config, contract, registry, llm, bootstrap.

No business logic lives here — only the runtime assembly that modules plug
into. See ``Agent基座软件开发计划.md`` (kept outside this repo) for the
stage-by-stage roadmap.
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
