"""Configuration for the agent base, consumed by the runtime.

This module is the single consumer of the config CONTRACT declared in
``.env.example``. pydantic-settings maps environment variables
(case-insensitive) onto these fields; ``.env`` is loaded automatically when
present.

Design notes (inherited from chat-agent review):
- B5 "配置即校验": every field that can be wrong is validated at startup,
  not silently defaulted.
- SEC-C2 lesson: no weak default secrets; production refuses to boot
  without ``LLM_API_KEY``.
"""

from __future__ import annotations

import json
from typing import Annotated

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# openai-compatible providers the base knows how to assemble. All share the
# same ChatOpenAI client; this set only guards against config typos.
KNOWN_PROVIDERS: frozenset[str] = frozenset({"deepseek", "zhipu", "openai-compatible"})

# checkpointer backends the base recognizes. sqlite is wired in Stage 3.
KNOWN_CHECKPOINTER_BACKENDS: frozenset[str] = frozenset({"memory", "sqlite"})

KNOWN_ENVIRONMENTS: frozenset[str] = frozenset({"development", "production"})


class SettingsError(ValueError):
    """Raised when configuration fails validation (fail-fast)."""


class Settings(BaseSettings):
    """Runtime configuration.

    Fields mirror the ``.env.example`` contract. Unknown env vars are ignored
    (``extra="ignore"``) so that unrelated shell variables never leak in.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # -- modules ------------------------------------------------------------
    # Module names to load; order == assembly order.
    # Accepted forms: comma-separated ("chat,writer") or a JSON array
    # (["chat","writer"]). NoDecode stops pydantic-settings from JSON-decoding
    # list-typed env values BEFORE validation — that pre-decode rejected the
    # plain comma form in .env with "error parsing value for field
    # agent_modules"; the before-validator below does the real parsing.
    # The registry resolves each name to agent_base.modules.<name>.
    agent_modules: Annotated[list[str], NoDecode] = Field(default_factory=lambda: ["chat"])

    # -- LLM (openai-compatible) -------------------------------------------
    llm_provider: str = "deepseek"
    llm_api_key: SecretStr = SecretStr("")
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "deepseek-chat"

    # -- conversation state (checkpointer; wired in Stage 3) ---------------
    checkpointer_backend: str = "memory"
    checkpointer_sqlite_path: str = "./agent_base_state.db"

    # -- environment --------------------------------------------------------
    env: str = "development"

    @field_validator("agent_modules", mode="before")
    @classmethod
    def _parse_agent_modules(cls, value: object) -> object:
        """Parse ``AGENT_MODULES`` into a list of names.

        Accepts the friendly comma form (``"chat,writer"``) and a JSON array
        (``'["chat","writer"]'``). Empty input -> empty list. Works for env
        vars, .env files, and direct init-kwargs alike.
        """
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            if text.startswith("["):
                return cls._parse_json_array(text)
            return [part.strip() for part in text.split(",") if part.strip()]
        return value

    @classmethod
    def _parse_json_array(cls, text: str) -> list[str]:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "AGENT_MODULES starts with '[' so it is parsed as a JSON array, "
                f"but the array is invalid: {text!r}"
            ) from exc
        if not isinstance(parsed, list):
            raise ValueError(f"AGENT_MODULES JSON array must be a list of strings: {text!r}")
        names: list[str] = []
        for item in parsed:
            if not isinstance(item, str):
                raise ValueError(f"AGENT_MODULES JSON array must be a list of strings: {text!r}")
            if item.strip():
                names.append(item.strip())
        return names

    @field_validator("llm_provider")
    @classmethod
    def _validate_provider(cls, value: str) -> str:
        if value not in KNOWN_PROVIDERS:
            raise ValueError(
                f"unknown LLM_PROVIDER {value!r}; expected one of {sorted(KNOWN_PROVIDERS)}"
            )
        return value

    @field_validator("checkpointer_backend")
    @classmethod
    def _validate_checkpointer_backend(cls, value: str) -> str:
        if value not in KNOWN_CHECKPOINTER_BACKENDS:
            raise ValueError(
                f"unknown CHECKPOINTER_BACKEND {value!r}; "
                f"expected one of {sorted(KNOWN_CHECKPOINTER_BACKENDS)}"
            )
        return value

    @field_validator("env")
    @classmethod
    def _validate_env(cls, value: str) -> str:
        if value not in KNOWN_ENVIRONMENTS:
            raise ValueError(f"unknown ENV {value!r}; expected one of {sorted(KNOWN_ENVIRONMENTS)}")
        return value

    def ensure_production_ready(self) -> None:
        """Fail fast on production-misconfiguration.

        Production requires a real API key; booting without one would put the
        service into a silently-broken state (chat-agent review SEC-C2).
        """
        if self.env != "production":
            return
        if not self.llm_api_key.get_secret_value().strip():
            raise SettingsError(
                "LLM_API_KEY is required in production (ENV=production); "
                "refusing to start with an empty key"
            )
