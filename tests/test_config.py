"""core.config 的测试（pydantic-settings + 快速失败）。"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from agent_base.core.config import Settings, SettingsError


def test_default_modules() -> None:
    # _env_file=None 让测试对环境中既有的仓库 .env 保持隔离。
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
    """以 '[' 开头的值属于 JSON 数组范畴；损坏的 JSON 必须响亮地失败。"""
    monkeypatch.setenv("AGENT_MODULES", "[chat")
    with pytest.raises(ValidationError, match="invalid"):
        Settings(_env_file=None)


def test_json_array_of_wrong_type_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """防御性分支：以 '[' 开头的 JSON 却不是列表。"""
    monkeypatch.setenv("AGENT_MODULES", "[1,2]")
    with pytest.raises(ValidationError, match="list of strings"):
        Settings(_env_file=None)


def test_json_non_list_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """解析器自身的防护：一个 '[…]' 值，其解码后的 JSON 不是列表。"""
    import agent_base.core.config as config_module

    monkeypatch.setattr(config_module.json, "loads", lambda text: {"not": "a list"})
    with pytest.raises(ValueError, match="must be a list of strings"):
        Settings(agent_modules="[anything]")


def test_dotenv_comma_separated_modules(tmp_path: Path) -> None:
    """回归测试：.env 文件必须接受普通的逗号形式。

    pydantic-settings 会在校验前对列表类型的 env 值做 JSON 解码；没有
    NoDecode 时，.env 中的 ``AGENT_MODULES=chat`` 会以
    ``error parsing value for field "agent_modules"`` 导致启动崩溃。
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


def test_mysql_backend_accepted() -> None:
    settings = Settings(_env_file=None, checkpointer_backend="mysql")
    assert settings.checkpointer_backend == "mysql"
    assert settings.checkpointer_mysql_host == "127.0.0.1"
    assert settings.checkpointer_mysql_port == 3306
    assert settings.checkpointer_mysql_user == "root"
    assert settings.checkpointer_mysql_database == "agent_base"


def test_mysql_password_is_secret_str() -> None:
    settings = Settings(checkpointer_mysql_password="s3cret")
    assert settings.checkpointer_mysql_password.get_secret_value() == "s3cret"


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


def test_cors_wildcard_rejected_in_any_env() -> None:
    with pytest.raises(SettingsError, match="CORS_ORIGINS"):
        Settings(_env_file=None, cors_origins="*").ensure_production_ready()
    with pytest.raises(SettingsError, match="CORS_ORIGINS"):
        Settings(_env_file=None, env="development", cors_origins=["*"]).ensure_production_ready()


def test_non_positive_tool_timeout_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, tool_timeout_seconds=0)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, tool_timeout_seconds=-1.5)


def test_nan_tool_timeout_rejected() -> None:
    # NaN 与任何数比较都是 False，<= 0 拦不住它；wait_for(timeout=nan)
    # 行为不可预测，必须在配置层拒绝。
    with pytest.raises(ValidationError):
        Settings(_env_file=None, tool_timeout_seconds=float("nan"))


def test_production_mysql_requires_connection_fields() -> None:
    settings = Settings(
        _env_file=None,
        env="production",
        llm_api_key="sk-1",
        checkpointer_backend="mysql",
        checkpointer_mysql_password="",
    )
    with pytest.raises(SettingsError, match="CHECKPOINTER_MYSQL"):
        settings.ensure_production_ready()


def test_production_mysql_complete_is_ok() -> None:
    Settings(
        _env_file=None,
        env="production",
        llm_api_key="sk-1",
        checkpointer_backend="mysql",
        checkpointer_mysql_user="app",
        checkpointer_mysql_database="agent_base",
        checkpointer_mysql_password="pw",
    ).ensure_production_ready()


# ── 文档解析限额（tools/parsing 契约） ──────────────────────────────


def test_doc_parse_defaults() -> None:
    settings = Settings(_env_file=None)
    assert settings.doc_parse_max_input_bytes == 10 * 1024 * 1024
    assert settings.doc_parse_max_output_chars == 50_000


def test_doc_parse_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOC_PARSE_MAX_INPUT_BYTES", "1024")
    monkeypatch.setenv("DOC_PARSE_MAX_OUTPUT_CHARS", "200")
    settings = Settings(_env_file=None)
    assert settings.doc_parse_max_input_bytes == 1024
    assert settings.doc_parse_max_output_chars == 200


def test_doc_parse_nonpositive_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOC_PARSE_MAX_INPUT_BYTES", "0")
    with pytest.raises(ValidationError, match="must be > 0"):
        Settings(_env_file=None)
    monkeypatch.setenv("DOC_PARSE_MAX_INPUT_BYTES", "1024")
    monkeypatch.setenv("DOC_PARSE_MAX_OUTPUT_CHARS", "-5")
    with pytest.raises(ValidationError, match="must be > 0"):
        Settings(_env_file=None)


def test_toolkit_enabled_csv_and_json(monkeypatch: pytest.MonkeyPatch) -> None:
    # 与 AGENT_MODULES 同源：逗号与 JSON 数组两种形式都必须被接受。
    monkeypatch.setenv("TOOLKIT_ENABLED", "current_time,web_search")
    assert Settings(_env_file=None).toolkit_enabled == ["current_time", "web_search"]
    monkeypatch.setenv("TOOLKIT_ENABLED", '["calculator"]')
    assert Settings(_env_file=None).toolkit_enabled == ["calculator"]
    monkeypatch.setenv("TOOLKIT_ENABLED", "")
    assert Settings(_env_file=None).toolkit_enabled == []
