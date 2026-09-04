"""Agent modules.

Each subpackage here is one agent module implementing the ``AgentModule``
contract. Being physically present does NOT enable a module at runtime —
``AGENT_MODULES`` is the single source of truth (ADR-002). The registry
(``core/registry.py``) resolves enabled names to these packages.
"""
