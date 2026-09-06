"""The ``hello`` module (module-guide step 5: enable via AGENT_MODULES).

Not enabled by default — AGENT_MODULES is the single on-switch (ADR-002).
"""

from __future__ import annotations

from agent_base.modules.hello.module import HelloModule

module = HelloModule()
