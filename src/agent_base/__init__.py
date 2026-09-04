"""agent-base: an Agent software foundation built on the LangGraph runtime.

The base deliberately contains no business capabilities -- no RAG, no
long-term memory, no application logic. Business functionality plugs in
later as modules implementing the AgentModule contract (Stage 1+).
"""

__version__ = "0.1.0"
