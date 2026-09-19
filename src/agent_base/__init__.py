"""agent-base：构建于 LangGraph 运行时之上的 Agent 软件基座。

基座的核心是 LangGraph 运行时 + 装配层 + 扩展点；应用逻辑以模块的形式
接入，模块实现 ``AgentModule`` 契约。长期记忆（``agent_base.memory``）是
基座内置的第一个"超级模块"式子系统，业务方仍通过模块契约接入各自能力。
"""

__version__ = "0.1.0"
