"""Sample conversational module (reference AgentModule implementation).

This is the template every new module copies. It lives entirely outside
``core/`` — proof that modules plug in without touching base code. The base
discovers it only because ``chat`` is listed in ``AGENT_MODULES`` and this
package exposes a ``module`` object.
"""

from agent_base.modules.chat.module import ChatModule

module = ChatModule()

__all__ = ["module"]
