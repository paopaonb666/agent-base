"""``hello`` 模块（module-guide 第 5 步：通过 AGENT_MODULES 启用）。

默认未启用——AGENT_MODULES 是唯一的开关（ADR-002）。
"""

from __future__ import annotations

from agent_base.modules.hello.module import HelloModule

module = HelloModule()
