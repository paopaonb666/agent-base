"""The ``writer`` sample module — the second built-in agent (Stage 4).

Serves two purposes:
1. a second supervisor sub-agent so the multi-agent template demonstrably
   routes between specialists (chat vs writer);
2. proof that adding a module follows the five-step module guide with zero
   base changes: a directory, the contract, and an AGENT_MODULES entry.
"""

from agent_base.modules.writer.module import WriterModule

module = WriterModule()
