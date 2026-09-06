"""agent-base：构建于 LangGraph 运行时之上的 Agent 软件基座。

基座刻意不包含任何业务能力——没有 RAG、没有长期记忆、没有应用逻辑。
业务功能之后以模块的形式接入，模块实现 ``AgentModule`` 契约（阶段 1+）。
"""

__version__ = "0.1.0"
