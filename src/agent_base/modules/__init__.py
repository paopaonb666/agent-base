"""Agent 模块。

这里的每个子包都是一个实现 ``AgentModule`` 契约的 agent 模块。物理上
存在于这里**并不**意味着模块在运行时被启用——``AGENT_MODULES`` 才是
唯一事实来源（ADR-002）。registry（``core/registry.py``）会把被启用的
名字解析到这些包。
"""
