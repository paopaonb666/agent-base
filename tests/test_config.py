"""Tests for core.config (pydantic-settings + fail-fast)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_base.core.config import Settings, SettingsError


def test_default_modules() -> None:
    assert Settings().agent_modules == ["chat"]


def test_comma_separated_modules() -> None:
    assert Settings(agent_modules="chat,writer").agent_modules == ["chat", "writer"]


def test_comma_separated_strips_whitespace() -> None:
    assert Settings(agent_modules=" chat , writer ").agent_modules == ["chat", "writer"]


def test_empty_modules_string_yields_empty_list() -> None:
    assert Settings(agent_modules="").agent_modules == []


def test_unknown_provider_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(llm_provider="openai")


def test_unknown_checkpointer_backend_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(checkpointer_backend="redis")


def test_unknown_env_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(env="staging")


def test_production_without_key_fails_fast() -> None:
    settings = Settings(env="production", llm_api_key="")
    with pytest.raises(SettingsError, match="LLM_API_KEY is required in production"):
        settings.ensure_production_ready()


def test_production_with_key_is_ok() -> None:
    Settings(env="production", llm_api_key="sk-1").ensure_production_ready()


def test_development_without_key_is_lenient() -> None:
    Settings(env="development", llm_api_key="").ensure_production_ready()


def test_api_key_is_secret_str() -> None:
    assert Settings(llm_api_key="sk-secret").llm_api_key.get_secret_value() == "sk-secret"
