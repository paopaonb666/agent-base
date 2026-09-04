"""Assemble the openai-compatible chat model.

All supported providers (deepseek / zhipu / openai-compatible) share the
same ``ChatOpenAI`` client. Switching provider means changing ``LLM_BASE_URL``
and ``LLM_MODEL`` in configuration — never code.
"""

from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI
from openai import OpenAIError

from agent_base.core.config import Settings, SettingsError


def build_llm(settings: Settings) -> ChatOpenAI:
    """Build the chat model from validated settings.

    ``LLM_API_KEY`` takes precedence; when unset, the client defers to the
    ambient ``OPENAI_API_KEY`` (langchain-openai's standard fallback). If
    neither is available, ``ChatOpenAI`` raises at construction — we convert
    that into an actionable ``SettingsError`` instead of leaking a raw
    ``openai.OpenAIError``.
    """
    kwargs: dict[str, Any] = {
        "model": settings.llm_model,
        "base_url": settings.llm_base_url,
    }
    api_key = settings.llm_api_key.get_secret_value().strip()
    if api_key:
        kwargs["api_key"] = api_key
    try:
        return ChatOpenAI(**kwargs)
    except OpenAIError as exc:
        raise SettingsError(
            "LLM_API_KEY is not configured and no ambient OPENAI_API_KEY is set; "
            "copy .env.example to .env, set LLM_API_KEY, then retry"
        ) from exc
