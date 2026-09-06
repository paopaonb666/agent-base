"""对话样板模块（AgentModule 的参考实现）。

这是每个新模块都要照抄的模板。它完全位于 ``core/`` 之外——证明模块
接入时无需触碰基座代码。基座之所以能发现它，仅仅是因为 ``chat`` 被
列在 ``AGENT_MODULES`` 里，且本包暴露了一个 ``module`` 对象。
"""

from agent_base.modules.chat.module import ChatModule

module = ChatModule()

__all__ = ["module"]
