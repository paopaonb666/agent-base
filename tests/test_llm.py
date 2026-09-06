"""Tests for core.llm (openai-compatible client assembly)."""

from __future__ import annotations

import pytest
from openai import OpenAIError

from agent_base.core.config import Settings, SettingsError
from agent_base.core.llm import build_llm


def test_build_llm_with_key_sets_model() -> None:
    settings = Settings(_env_file=None, llm_api_key="sk-test", llm_model="deepseek-chat")
    llm = build_llm(settings)
    assert llm.model_name == "deepseek-chat"


def test_build_llm_construction_failure_becomes_settings_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raw openai error (e.g. missing key everywhere) is actionable, not leaked."""

    def _fail(**kwargs: object) -> None:
        raise OpenAIError("no api key available")

    monkeypatch.setattr("agent_base.core.llm.ChatOpenAI", _fail)
    with pytest.raises(SettingsError, match="LLM_API_KEY"):
        build_llm(Settings(_env_file=None, llm_api_key=""))
