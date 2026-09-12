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

# 工具库（tools/）的搜索引擎实现清单：SEARCH_ENGINE_PRIORITY 的合法取值。
KNOWN_SEARCH_ENGINES: frozenset[str] = frozenset({"tavily", "duckduckgo"})

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
    # 默认 sqlite：对话（含失败的工具调用消息）superstep 粒度边流边写，
    # 重启可恢复；memory 仅在显式选择时使用（什么都不持久化）。
    checkpointer_backend: str = "sqlite"
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
    # 工具调用审计（M5）：每条调用（成功/超时/异常）写入
    # tool_call_records 表，存储后端跟随 CHECKPOINTER_BACKEND。
    tool_call_log_enabled: bool = True

    # -- 工具库（tools/；M1 治理设施） ------------------------------------
    # 要装配进共享池的基座内置工具名（逗号分隔或 JSON 数组）。默认只开
    # 零依赖工具；高风险工具（python_repl）与依赖外部服务的工具
    # （web_search）必须显式开启。名字必须与注册表条目一致，拼错在
    # 启动时快速失败。
    toolkit_enabled: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["current_time", "calculator", "json_query"]
    )

    # -- 网络搜索（tools/search；M3） -------------------------------------
    # 搜索引擎优先级：按顺序依次尝试，任一成功即返回。tavily 需要
    # 下面的 API key；duckduckgo 免 key（但需要安装 [search] extras）。
    search_engine_priority: str = "duckduckgo"
    tavily_api_key: SecretStr = SecretStr("")
    # 单次引擎调用的超时与结果缓存 TTL。搜索是交互式对话中的一步，
    # 超时应显著短于全局 TOOL_TIMEOUT_SECONDS（池超时是它的兜底）。
    # 注意 ddgs 会聚合多个上游引擎，受限网络下聚合经常超过 10s——
    # 默认 15s 是"多数查询能完成、失败也不会拖垮对话"的折中。
    search_timeout_seconds: float = 15.0
    search_cache_ttl_seconds: float = 300.0
    # ddgs 的上游引擎钉选。默认 "auto" 会同时扇出十几个上游（google/
    # yahoo/brave/startpage 等），受限网络下大量超时把聚合拖死，直到
    # 触发 SEARCH_TIMEOUT_SECONDS。单钉可达引擎（duckduckgo / bing）
    # 实测快一个数量级；逗号分隔可传多个，但多引擎聚合的稳定性以
    # 实测为准。
    search_ddgs_backend: str = "duckduckgo"

    # -- 文档解析（工具库 tools/parsing；M4 前置） ------------------------
    # 单个文档的输入大小封顶（字节）：解析在内存中进行，上限挡住超大
    # 文件把工具线程拖入 OOM 的路径。超限直接报错而不是截断输入。
    doc_parse_max_input_bytes: int = 10 * 1024 * 1024
    # 解析输出的文本长度封顶（字符）：超限截断并置 truncated 标志，
    # 防止整本 PDF 的文本一次性灌进模型上下文。
    doc_parse_max_output_chars: int = 50_000

    # -- 记忆系统（memory/；M6） ------------------------------------------
    # 总开关：关闭时不装配记忆服务，invoke 链路与既有行为完全一致。
    memory_enabled: bool = True
    # 语义检索（embedding）：openai 兼容 /embeddings 端点（默认指向硅基
    # 流动，BAAI/bge-m3，1024 维，中英双语）。API key 为空时自动降级为
    # BM25 关键词 + 时间衰减检索，其余记忆功能不受影响。
    memory_embedding_enabled: bool = True
    memory_embedding_base_url: str = "https://api.siliconflow.cn/v1"
    memory_embedding_api_key: SecretStr = SecretStr("")
    memory_embedding_model: str = "BAAI/bge-m3"
    memory_embedding_dims: int = 1024
    memory_embedding_batch_size: int = 16
    memory_embedding_timeout_seconds: float = 10.0
    # 召回与注入预算：每轮最多召回的记忆条数与注入上下文的字符预算。
    memory_recall_top_k: int = 6
    memory_context_max_chars: int = 3000
    # chat 图的模型侧 token 预算（M6d 上下文工程）：超过预算的旧历史被
    # 修剪出"发给模型"的输入（checkpointer 全量历史不动）。这是近似估算
    # （CJK 0.6 token/字 + ASCII 0.25 token/字符），宁小勿大。
    memory_context_max_tokens: int = 12_000
    # 记忆形成管线（M6c）：每 N 轮对话跑一次抽取+整合（后台异步，绝不
    # 阻塞对话流）。
    memory_capture_enabled: bool = True
    memory_capture_every_turns: int = 1
    # 会话滚动摘要（M6c/M6d）：线程内消息数超过阈值后触发摘要更新。
    memory_summary_enabled: bool = True
    memory_summary_trigger_messages: int = 12
    # 用户画像（M6c）：随对话合并演化的结构化画像。
    memory_profile_enabled: bool = True
    # 遗忘策略：episodic 记忆的保留天数（0 表示不过期）；检索打分的
    # 时间半衰期（天）——越久未被访问的记忆权重越低。
    memory_episodic_ttl_days: int = 30
    memory_time_decay_half_life_days: float = 14.0
    # 抽取管线的单次输入字符上限（防止长会话把 prompt 撑爆）。
    memory_extraction_max_input_chars: int = 8000
    # 文档知识库分块（M6e）：按字符数分块 + 相邻块重叠。
    memory_doc_chunk_chars: int = 800
    memory_doc_chunk_overlap: int = 100

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

    @field_validator("agent_modules", "cors_origins", "toolkit_enabled", mode="before")
    @classmethod
    def _parse_csv_or_json_list(cls, value: object, info: ValidationInfo) -> object:
        """把列表类型的字段从其友好的字符串形式解析出来。

        ``AGENT_MODULES`` / ``CORS_ORIGINS`` / ``TOOLKIT_ENABLED`` 都接受
        逗号形式（``"chat,writer"`` / ``"http://localhost:3000,https://x.example.com"``）
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

    @field_validator("doc_parse_max_input_bytes", "doc_parse_max_output_chars")
    @classmethod
    def _validate_doc_parse_limits(cls, value: int, info: ValidationInfo) -> int:
        # 非正数的限额等于让解析层要么拒绝一切输入、要么截断一切输出，
        # 必须是启动时的配置错误而不是静默运行时行为。
        if value <= 0:
            raise ValueError(f"{info.field_name} must be > 0, got {value}")
        return value

    @field_validator("search_engine_priority")
    @classmethod
    def _validate_search_priority(cls, value: str) -> str:
        engines = [part.strip().lower() for part in value.split(",") if part.strip()]
        if not engines:
            raise ValueError("SEARCH_ENGINE_PRIORITY must name at least one engine")
        unknown = [e for e in engines if e not in KNOWN_SEARCH_ENGINES]
        if unknown:
            raise ValueError(
                f"unknown SEARCH_ENGINE_PRIORITY entries {unknown}; "
                f"expected subset of {sorted(KNOWN_SEARCH_ENGINES)}"
            )
        return value

    @field_validator("search_timeout_seconds", "search_cache_ttl_seconds")
    @classmethod
    def _validate_search_durations(cls, value: float, info: ValidationInfo) -> float:
        # 与 TOOL_TIMEOUT_SECONDS 同理：NaN 的所有比较都是 False，
        # 必须用 not (value > 0) 一并拒绝。
        if not (value > 0):
            raise ValueError(f"{info.field_name} must be > 0, got {value}")
        return value

    @field_validator("search_ddgs_backend")
    @classmethod
    def _validate_search_ddgs_backend(cls, value: str) -> str:
        backend = value.strip().lower()
        if not backend:
            raise ValueError(
                "SEARCH_DDGS_BACKEND must name at least one ddgs engine "
                "(e.g. duckduckgo, bing); leave unset for the default"
            )
        return backend

    @field_validator(
        "memory_embedding_dims",
        "memory_embedding_batch_size",
        "memory_recall_top_k",
        "memory_context_max_chars",
        "memory_context_max_tokens",
        "memory_extraction_max_input_chars",
        "memory_doc_chunk_chars",
    )
    @classmethod
    def _validate_memory_positive_ints(cls, value: int, info: ValidationInfo) -> int:
        # 非正数的上限/预算等于让记忆功能要么拒绝一切、要么注入一切，
        # 必须是启动时的配置错误而不是静默运行时行为。
        if value <= 0:
            raise ValueError(f"{info.field_name} must be > 0, got {value}")
        return value

    @field_validator(
        "memory_embedding_timeout_seconds", "memory_time_decay_half_life_days"
    )
    @classmethod
    def _validate_memory_durations(cls, value: float, info: ValidationInfo) -> float:
        # NaN 的所有比较都是 False，必须用 not (value > 0) 一并拒绝。
        if not (value > 0):
            raise ValueError(f"{info.field_name} must be > 0, got {value}")
        return value

    @field_validator("memory_capture_every_turns", "memory_summary_trigger_messages")
    @classmethod
    def _validate_memory_trigger_intervals(cls, value: int, info: ValidationInfo) -> int:
        if value < 1:
            raise ValueError(f"{info.field_name} must be >= 1, got {value}")
        return value

    @field_validator("memory_episodic_ttl_days")
    @classmethod
    def _validate_memory_ttl(cls, value: int) -> int:
        # 0 是合法值：表示 episodic 记忆不过期。
        if value < 0:
            raise ValueError(f"memory_episodic_ttl_days must be >= 0, got {value}")
        return value

    @field_validator("memory_doc_chunk_overlap")
    @classmethod
    def _validate_memory_chunk_overlap(cls, value: int, info: ValidationInfo) -> int:
        if value < 0:
            raise ValueError(f"memory_doc_chunk_overlap must be >= 0, got {value}")
        # 重叠必须小于块长（与 memory_doc_chunk_chars 同属一组配置），
        # 否则分块会陷入死循环。
        chars = info.data.get("memory_doc_chunk_chars")
        if chars is not None and value >= chars:
            raise ValueError(
                f"memory_doc_chunk_overlap must be < memory_doc_chunk_chars "
                f"({chars}), got {value}"
            )
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
