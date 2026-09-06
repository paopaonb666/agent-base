"""Tests for core.config (pydantic-settings + fail-fast)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from agent_base.core.config import Settings, SettingsError


def test_default_modules() -> None:
    # _env_file=None keeps the test hermetic against an ambient repo .env.
    assert Settings(_env_file=None).agent_modules == ["chat"]


def test_comma_separated_modules() -> None:
    assert Settings(agent_modules="chat,writer").agent_modules == ["chat", "writer"]


def test_comma_separated_strips_whitespace() -> None:
    assert Settings(agent_modules=" chat , writer ").agent_modules == ["chat", "writer"]


def test_empty_modules_string_yields_empty_list() -> None:
    assert Settings(agent_modules="").agent_modules == []


def test_env_var_comma_separated_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_MODULES", "chat,writer")
    assert Settings(_env_file=None).agent_modules == ["chat", "writer"]


def test_env_var_json_array_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_MODULES", '["chat","writer"]')
    assert Settings(_env_file=None).agent_modules == ["chat", "writer"]


def test_invalid_json_array_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_MODULES", '["chat", 3]')
    with pytest.raises(ValidationError, match="list of strings"):
        Settings(_env_file=None)


def test_malformed_json_array_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A '['-prefixed value is JSON-array territory; broken JSON must fail loudly."""
    monkeypatch.setenv("AGENT_MODULES", "[chat")
    with pytest.raises(ValidationError, match="invalid"):
        Settings(_env_file=None)


def test_json_array_of_wrong_type_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Defensive branch: '['-prefixed JSON that is not a list."""
    monkeypatch.setenv("AGENT_MODULES", "[1,2]")
    with pytest.raises(ValidationError, match="list of strings"):
        Settings(_env_file=None)


def test_json_non_list_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """The parser's own guard: a '[…]' value whose decoded JSON is not a list."""
    import agent_base.core.config as config_module

    monkeypatch.setattr(config_module.json, "loads", lambda text: {"not": "a list"})
    with pytest.raises(ValueError, match="must be a list of strings"):
        Settings(agent_modules="[anything]")


def test_dotenv_comma_separated_modules(tmp_path: Path) -> None:
    """Regression: a .env file must accept the plain comma form.

    pydantic-settings JSON-decodes list-typed env values before validation;
    without NoDecode, ``AGENT_MODULES=chat`` in .env crashed startup with
    ``error parsing value for field "agent_modules"``.
    """
    env_file = tmp_path / ".env"
    env_file.write_text("AGENT_MODULES=chat\n", encoding="utf-8")
    assert Settings(_env_file=env_file).agent_modules == ["chat"]


def test_dotenv_comma_separated_strips_whitespace(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("AGENT_MODULES= chat , writer \n", encoding="utf-8")
    assert Settings(_env_file=env_file).agent_modules == ["chat", "writer"]


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
