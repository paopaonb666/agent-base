"""agent 基座的配置，供运行时消费。

本模块是 ``.env.example`` 中声明的配置契约（CONTRACT）的唯一消费方。
pydantic-settings 把环境变量（大小写不敏感）映射到这些字段上；存在
``.env`` 时会被自动加载。

设计说明（继承自 chat-agent 评审）：
- B5 “配置即校验”：每个可能出错的字段都在启动时校验，而不是静默地
  使用默认值。
- SEC-C2 经验：不使用弱默认密钥；production 环境在没有 ``LLM_API_KEY``
  时拒绝启动。
"""

from __future__ import annotations

import json
from typing import Annotated

from pydantic import Field, SecretStr, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# 基座知道如何装配的 openai 兼容 provider。它们共用同一个 ChatOpenAI
# 客户端；这个集合仅用于防止配置拼写错误。
KNOWN_PROVIDERS: frozenset[str] = frozenset({"deepseek", "zhipu", "openai-compatible"})

# 基座识别的 checkpointer 后端。sqlite 在阶段 3 接线，mysql 为可选后端。
KNOWN_CHECKPOINTER_BACKENDS: frozenset[str] = frozenset({"memory", "sqlite", "mysql"})

KNOWN_ENVIRONMENTS: frozenset[str] = frozenset({"development", "production"})


class SettingsError(ValueError):
    """当配置校验失败时抛出（快速失败）。"""


class Settings(BaseSettings):
    """运行时配置。

    字段与 ``.env.example`` 契约一一对应。未知环境变量会被忽略
    （``extra="ignore"``），因此无关的 shell 变量永远不会泄漏进来。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # -- 模块 ------------------------------------------------------------
    # 要加载的模块名；顺序即装配顺序。
    # 接受的格式：逗号分隔（"chat,writer"）或 JSON 数组
    # (["chat","writer"])。NoDecode 会阻止 pydantic-settings 在**校验之前**
    # 就对列表类型的 env 值做 JSON 解码——那种预解码会以
    # "error parsing value for field agent_modules" 拒绝 .env 中的普通
    # 逗号形式；下面的 before 校验器才做真正的解析。
    # registry 会把每个名字解析为 agent_base.modules.<name>。
    agent_modules: Annotated[list[str], NoDecode] = Field(default_factory=lambda: ["chat"])

    # -- 服务 ------------------------------------------------------------
    # 允许跨域调用本服务的来源（浏览器客户端，如 agent-base-ui）。
    # 与 AGENT_MODULES 一样支持 csv / JSON 数组两种形式。
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:3000"]
    )

    # -- LLM（openai 兼容） ---------------------------------------------
    llm_provider: str = "deepseek"
    llm_api_key: SecretStr = SecretStr("")
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "deepseek-chat"

    # -- 对话状态（checkpointer；阶段 3 接线） ---------------------------
    checkpointer_backend: str = "memory"
    checkpointer_sqlite_path: str = "./agent_base_state.db"

    # MySQL 后端连接参数（仅当 CHECKPOINTER_BACKEND=mysql 时生效）。
    # 要求 MySQL >= 8.0.19（或 MariaDB >= 10.7.1）。
    checkpointer_mysql_host: str = "127.0.0.1"
    checkpointer_mysql_port: int = 3306
    checkpointer_mysql_user: str = "root"
    checkpointer_mysql_password: SecretStr = SecretStr("")
    checkpointer_mysql_database: str = "agent_base"

    # -- 工具池（阶段 3） ------------------------------------------------
    # 每次工具执行的挂钟时间预算，由池的包装器强制执行。
    tool_timeout_seconds: float = 30.0

    # -- 环境 --------------------------------------------------------
    env: str = "development"

    # -- 可观测性 ------------------------------------------------------
    # 结构化 JSON 日志行（开发环境默认人类可读）。
    # 生产部署应设置 LOG_JSON=true 以获得机器可解析的日志。
    # 追踪（LangSmith/Langfuse）仍由环境开关控制，不在这里建模——
    # 见 extensions/observability.log_tracing_config。
    log_json: bool = False

    # /health 是否真实探测 LLM 端点可达性（会打真实网络，默认关闭——
    # 关闭时只校验 key 非空与 base_url 格式；生产环境可按需开启）。
    health_probe_model: bool = False

    @field_validator("agent_modules", "cors_origins", mode="before")
    @classmethod
    def _parse_csv_or_json_list(cls, value: object, info: ValidationInfo) -> object:
        """把列表类型的字段从其友好的字符串形式解析出来。

        ``AGENT_MODULES`` 和 ``CORS_ORIGINS`` 都接受逗号形式
        （``"chat,writer"`` / ``"http://localhost:3000,https://x.example.com"``）
        和 JSON 数组（``'["chat","writer"]'``）。空输入 -> 空列表。
        对环境变量、.env 文件和直接传入的 init 关键字参数都同样适用。
        """
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            if text.startswith("["):
                return cls._parse_json_array(text, info.field_name or "value")
            return [part.strip() for part in text.split(",") if part.strip()]
        return value

    @classmethod
    def _parse_json_array(cls, text: str, field: str) -> list[str]:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{field} starts with '[' so it is parsed as a JSON array, "
                f"but the array is invalid: {text!r}"
            ) from exc
        if not isinstance(parsed, list):
            raise ValueError(f"{field} JSON array must be a list of strings: {text!r}")
        names: list[str] = []
        for item in parsed:
            if not isinstance(item, str):
                raise ValueError(f"{field} JSON array must be a list of strings: {text!r}")
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

    @field_validator("tool_timeout_seconds")
    @classmethod
    def _validate_tool_timeout(cls, value: float) -> float:
        # 非正数会让工具池立即超时，等于静默禁用所有工具；NaN 的所有
        # 比较都是 False，必须用 not (value > 0) 一并拒绝。
        if not (value > 0):
            raise ValueError(f"TOOL_TIMEOUT_SECONDS must be > 0, got {value}")
        return value

    def ensure_production_ready(self) -> None:
        """在配置错误时快速失败（安全项对所有环境生效，其余限 production）。

        production 要求真实的 API key；没有它启动会让服务陷入静默损坏的
        状态（chat-agent 评审 SEC-C2）。
        """
        if "*" in self.cors_origins:
            raise SettingsError(
                'CORS_ORIGINS="*" is not allowed (in any environment): a wildcard '
                "origin lets any site call the invoke endpoint on the user's behalf"
            )
        if self.env != "production":
            return
        if not self.llm_api_key.get_secret_value().strip():
            raise SettingsError(
                "LLM_API_KEY is required in production (ENV=production); "
                "refusing to start with an empty key"
            )
        if self.checkpointer_backend == "mysql":
            missing = [
                name
                for name, value in (
                    ("CHECKPOINTER_MYSQL_USER", self.checkpointer_mysql_user),
                    ("CHECKPOINTER_MYSQL_DATABASE", self.checkpointer_mysql_database),
                    (
                        "CHECKPOINTER_MYSQL_PASSWORD",
                        self.checkpointer_mysql_password.get_secret_value(),
                    ),
                )
                if not str(value).strip()
            ]
            if missing:
                raise SettingsError(
                    f"{', '.join(missing)} required in production with CHECKPOINTER_BACKEND=mysql"
                )
