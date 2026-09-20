"""agent 基座的配置，供运行时消费（M3/P1-2：按域嵌套的配置节）。

本模块是 ``.env.example`` 中声明的配置契约（CONTRACT）的唯一消费方。
pydantic-settings 把环境变量（大小写不敏感）映射到这些字段上；存在
``.env`` 时会被自动加载。

设计说明（继承自 chat-agent 评审 + 两轮设计审查）：
- B5 "配置即校验"：每个可能出错的字段都在启动时校验，而不是静默地
  使用默认值；校验随配置节就近放置，不再集中在一个胖类。
- SEC-C2 经验：不使用弱默认密钥；production 环境在没有 ``LLM_API_KEY``
  时拒绝启动。
- **按域嵌套**（M3/P1-2）：``LlmSettings`` / ``CheckpointerSettings`` /
  ``ToolkitSettings`` / ``SearchSettings`` / ``DocParseSettings`` /
  ``MemorySettings`` / ``ObservabilitySettings`` 各一个配置节，子系统
  只消费自己那一节（接口隔离）。**环境变量契约保持平铺不变**：嵌套
  节的 env 名 = 节字段名 + 内部字段名（``MEMORY_ENABLED`` /
  ``CHECKPOINTER_MYSQL_HOST`` …），``.env.example`` 无需任何迁移。
- **兼容垫片**：历史平铺读取（``settings.memory_enabled``）经
  ``__getattr__`` 代理到对应配置节（只读），既有测试与脚本不受影响；
  基座源码已全部迁移到节访问。

未知环境变量会被忽略（``extra="ignore"``）。
"""

from __future__ import annotations

import json
from typing import Annotated, Any

from pydantic import (
    AliasChoices,
    BaseModel,
    Field,
    SecretStr,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    NoDecode,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

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


def _parse_csv_or_json_list(value: object, field_name: str) -> object:
    """把列表类型的字段从其友好的字符串形式解析出来。

    ``AGENT_MODULES`` / ``CORS_ORIGINS`` / ``TOOLKIT_ENABLED`` 都接受
    逗号形式（``"chat,writer"``）和 JSON 数组（``'["chat","writer"]'``）。
    空输入 -> 空列表。对环境变量、.env 文件和直接传入的 init 关键字
    参数都同样适用。
    """
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{field_name} starts with '[' so it is parsed as a JSON array, "
                    f"but the array is invalid: {text!r}"
                ) from exc
            if not isinstance(parsed, list):
                raise ValueError(f"{field_name} JSON array must be a list of strings: {text!r}")
            names: list[str] = []
            for item in parsed:
                if not isinstance(item, str):
                    raise ValueError(f"{field_name} JSON array must be a list of strings: {text!r}")
                if item.strip():
                    names.append(item.strip())
            return names
        return [part.strip() for part in text.split(",") if part.strip()]
    return value


class LlmSettings(BaseModel):
    """LLM（openai 兼容）配置节：env 前缀 ``LLM_``。"""

    model_config = SettingsConfigDict(extra="ignore", validate_default=True)

    provider: str = "deepseek"
    api_key: SecretStr = SecretStr("")
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"

    @field_validator("provider")
    @classmethod
    def _validate_provider(cls, value: str) -> str:
        if value not in KNOWN_PROVIDERS:
            raise ValueError(
                f"unknown LLM_PROVIDER {value!r}; expected one of {sorted(KNOWN_PROVIDERS)}"
            )
        return value


class CheckpointerSettings(BaseModel):
    """对话状态（checkpointer）配置节：env 前缀 ``CHECKPOINTER_``。"""

    model_config = SettingsConfigDict(extra="ignore", validate_default=True)

    # 默认 sqlite：对话（含失败的工具调用消息）superstep 粒度边流边写，
    # 重启可恢复；memory 仅在显式选择时使用（什么都不持久化）。
    backend: str = "sqlite"
    sqlite_path: str = "./agent_base_state.db"

    # MySQL 后端连接参数（仅当 CHECKPOINTER_BACKEND=mysql 时生效）。
    # 要求 MySQL >= 8.0.19（或 MariaDB >= 10.7.1）。
    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: SecretStr = SecretStr("")
    mysql_database: str = "agent_base"

    @field_validator("backend")
    @classmethod
    def _validate_backend(cls, value: str) -> str:
        if value not in KNOWN_CHECKPOINTER_BACKENDS:
            raise ValueError(
                f"unknown CHECKPOINTER_BACKEND {value!r}; "
                f"expected one of {sorted(KNOWN_CHECKPOINTER_BACKENDS)}"
            )
        return value


class ToolkitSettings(BaseModel):
    """工具池与工具库配置节：env 前缀 ``TOOLKIT_`` / ``TOOL_``。"""

    model_config = SettingsConfigDict(extra="ignore", validate_default=True, populate_by_name=True)

    # 要装配进共享池的基座内置工具名。默认只开零依赖工具；高风险工具
    # （python_repl）与依赖外部服务的工具（web_search）必须显式开启。
    enabled: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["current_time", "calculator", "json_query"],
        validation_alias=AliasChoices("TOOLKIT_ENABLED", "toolkit_enabled"),
    )
    # 每次工具执行的挂钟时间预算，由池的包装器强制执行。
    timeout_seconds: float = Field(
        default=30.0, validation_alias=AliasChoices("TOOL_TIMEOUT_SECONDS", "tool_timeout_seconds")
    )
    # 工具调用审计（M5）：每条调用（成功/超时/异常）写入
    # tool_call_records 表，存储后端跟随 CHECKPOINTER_BACKEND。
    call_log_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("TOOL_CALL_LOG_ENABLED", "tool_call_log_enabled"),
    )

    @field_validator("enabled", mode="before")
    @classmethod
    def _parse_enabled(cls, value: object) -> object:
        return _parse_csv_or_json_list(value, "TOOLKIT_ENABLED")

    @field_validator("timeout_seconds")
    @classmethod
    def _validate_timeout(cls, value: float) -> float:
        # 非正数会让工具池立即超时，等于静默禁用所有工具；NaN 的所有
        # 比较都是 False，必须用 not (value > 0) 一并拒绝。
        if not (value > 0):
            raise ValueError(f"TOOL_TIMEOUT_SECONDS must be > 0, got {value}")
        return value


class SearchSettings(BaseModel):
    """网络搜索（tools/search）配置节：env 前缀 ``SEARCH_`` / ``TAVILY_``。"""

    model_config = SettingsConfigDict(extra="ignore", validate_default=True)

    # 搜索引擎优先级：按顺序依次尝试，任一成功即返回。tavily 需要
    # API key；duckduckgo 免 key（但需要安装 [search] extras）。
    engine_priority: str = "duckduckgo"
    tavily_api_key: SecretStr = SecretStr("")
    # 单次引擎调用的超时与结果缓存 TTL。搜索是交互式对话中的一步，
    # 超时应显著短于 TOOL_TIMEOUT_SECONDS（池超时是它的兜底）。
    # 注意 ddgs 会聚合多个上游引擎，受限网络下聚合经常超过 10s——
    # 默认 15s 是"多数查询能完成、失败也不会拖垮对话"的折中。
    timeout_seconds: float = 15.0
    cache_ttl_seconds: float = 300.0
    # ddgs 的上游引擎钉选。默认 "auto" 会同时扇出十几个上游，受限
    # 网络下大量超时把聚合拖死；单钉可达引擎实测快一个数量级。
    ddgs_backend: str = "duckduckgo"

    @field_validator("engine_priority")
    @classmethod
    def _validate_priority(cls, value: str) -> str:
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

    @field_validator("timeout_seconds", "cache_ttl_seconds")
    @classmethod
    def _validate_durations(cls, value: float, info: ValidationInfo) -> float:
        # 与 TOOL_TIMEOUT_SECONDS 同理：NaN 的所有比较都是 False，
        # 必须用 not (value > 0) 一并拒绝。
        if not (value > 0):
            raise ValueError(f"{info.field_name} must be > 0, got {value}")
        return value

    @field_validator("ddgs_backend")
    @classmethod
    def _validate_ddgs_backend(cls, value: str) -> str:
        backend = value.strip().lower()
        if not backend:
            raise ValueError(
                "SEARCH_DDGS_BACKEND must name at least one ddgs engine (e.g. duckduckgo, bing)"
            )
        return backend


class DocParseSettings(BaseModel):
    """文档解析（tools/parsing）配置节：env 前缀 ``DOC_PARSE_``。"""

    model_config = SettingsConfigDict(extra="ignore", validate_default=True)

    # 单个文档的输入大小封顶（字节）：解析在内存中进行，上限挡住超大
    # 文件把工具线程拖入 OOM 的路径。超限直接报错而不是截断输入。
    max_input_bytes: int = 10 * 1024 * 1024
    # 解析输出的文本长度封顶（字符）：超限截断并置 truncated 标志，
    # 防止整本 PDF 的文本一次性灌进模型上下文。
    max_output_chars: int = 50_000

    @field_validator("max_input_bytes", "max_output_chars")
    @classmethod
    def _validate_limits(cls, value: int, info: ValidationInfo) -> int:
        # 非正数的限额等于让解析层要么拒绝一切输入、要么截断一切输出，
        # 必须是启动时的配置错误而不是静默运行时行为。
        if value <= 0:
            raise ValueError(f"{info.field_name} must be > 0, got {value}")
        return value


class MemorySettings(BaseModel):
    """记忆系统（memory/）配置节：env 前缀 ``MEMORY_``。"""

    model_config = SettingsConfigDict(extra="ignore", validate_default=True)

    # 总开关：关闭时不装配记忆服务，invoke 链路与既有行为完全一致。
    enabled: bool = True
    # 语义检索（embedding）：openai 兼容 /embeddings 端点（默认指向硅基
    # 流动，BAAI/bge-m3，1024 维，中英双语）。API key 为空时自动降级为
    # BM25 关键词 + 时间衰减检索，其余记忆功能不受影响。
    embedding_enabled: bool = True
    embedding_base_url: str = "https://api.siliconflow.cn/v1"
    embedding_api_key: SecretStr = SecretStr("")
    embedding_model: str = "BAAI/bge-m3"
    embedding_dims: int = 1024
    embedding_batch_size: int = 16
    embedding_timeout_seconds: float = 10.0
    # 召回与注入预算：每轮最多召回的记忆条数与注入上下文的字符预算。
    recall_top_k: int = 6
    # 召回最低分阈值（0..1）：低于该分数的候选不召回——没有它，毫无
    # 关联的提问也会按时间/显著度凑满 top_k，弱相关记忆噪音很大。
    # 0.5 的依据：混合分里纯噪音（无关键词/语义证据）的天花板只有
    # recency+salience ≈ 0.3，加向量余弦噪音实测 ≤0.47；而真实相关
    # （含改写）≥0.59（bge-m3 与测试替身双双验证）。降级关键词路径
    # 由 service.search 自动放宽一半（缺向量主力分量，硬门槛会误杀
    # 老而准的关键词命中）。
    recall_min_score: float = 0.5
    # 混合检索权重（锐评 #4）：四分量分别对应向量余弦/BM25/时间衰减/
    # 显著度，和必须为 1（误差 0.01）。
    weight_vector: float = 0.40
    weight_keyword: float = 0.30
    weight_recency: float = 0.15
    weight_salience: float = 0.15
    context_max_chars: int = 3000
    # 模型侧 token 预算（M6d 上下文工程）：超过预算的旧历史被修剪出
    # "发给模型"的输入（checkpointer 全量历史不动）。
    context_max_tokens: int = 12_000
    # 记忆形成管线（M6c）：每 N 轮对话跑一次抽取+整合（后台异步）。
    capture_enabled: bool = True
    capture_every_turns: int = 1
    # 会话滚动摘要（M6c/M6d）：线程内消息数超过阈值后触发摘要更新。
    summary_enabled: bool = True
    summary_trigger_messages: int = 12
    # 用户画像（M6c）：随对话合并演化的结构化画像。
    profile_enabled: bool = True
    # 画像注入的硬截断上限（字符）：prompt 里的字数只是软约束，保存时
    # 做结构级硬截断兜底，防止画像缓慢膨胀注入预算。
    profile_max_chars: int = 1500
    # 遗忘策略：episodic 记忆的保留天数（0 表示不过期）；检索打分的
    # 时间半衰期（天）。
    episodic_ttl_days: int = 30
    time_decay_half_life_days: float = 14.0
    # 抽取管线的单次输入字符上限与消息条数窗口（两者共同约束）。
    extraction_max_input_chars: int = 8000
    extraction_max_messages: int = 24
    # 文档知识库分块（M6e）：按字符数分块 + 相邻块重叠。
    doc_chunk_chars: int = 800
    doc_chunk_overlap: int = 100
    # 记忆表 MySQL 连接池（锐评 #2）：小池即可——记忆读写是每轮一到
    # 两次的稳定流量，不是每 token。
    mysql_pool_size: int = 5
    # 记忆身份鉴权（安全加固 P0）：配置了密钥后，所有带记忆作用域的
    # 请求必须附带 ``X-User-Sig = HMAC-SHA256(X-User-Id, secret)``，
    # 否则 401。留空时 development 接受裸 X-User-Id；production 且
    # memory_enabled 时强制要求配置（快速失败）。
    auth_secret: SecretStr = SecretStr("")

    @field_validator(
        "embedding_dims",
        "embedding_batch_size",
        "recall_top_k",
        "context_max_chars",
        "context_max_tokens",
        "extraction_max_input_chars",
        "doc_chunk_chars",
        "mysql_pool_size",
        "profile_max_chars",
        "extraction_max_messages",
    )
    @classmethod
    def _validate_positive_ints(cls, value: int, info: ValidationInfo) -> int:
        if value <= 0:
            raise ValueError(f"{info.field_name} must be > 0, got {value}")
        return value

    @field_validator("recall_min_score")
    @classmethod
    def _validate_min_score(cls, value: float) -> float:
        if not (0.0 <= value < 1.0):
            raise ValueError(f"MEMORY_RECALL_MIN_SCORE must be in [0, 1), got {value}")
        return value

    @field_validator("weight_vector", "weight_keyword", "weight_recency", "weight_salience")
    @classmethod
    def _validate_weight(cls, value: float, info: ValidationInfo) -> float:
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"{info.field_name} must be in [0, 1], got {value}")
        return value

    @model_validator(mode="after")
    def _validate_weights_sum(self) -> MemorySettings:
        weights = (
            self.weight_vector,
            self.weight_keyword,
            self.weight_recency,
            self.weight_salience,
        )
        if abs(sum(weights) - 1.0) > 0.01:
            raise ValueError(
                f"memory hybrid weights must sum to 1.0 (±0.01), got {sum(weights)} from {weights}"
            )
        return self

    @field_validator("embedding_timeout_seconds", "time_decay_half_life_days")
    @classmethod
    def _validate_durations(cls, value: float, info: ValidationInfo) -> float:
        if not (value > 0):
            raise ValueError(f"{info.field_name} must be > 0, got {value}")
        return value

    @field_validator("capture_every_turns", "summary_trigger_messages")
    @classmethod
    def _validate_trigger_intervals(cls, value: int, info: ValidationInfo) -> int:
        if value < 1:
            raise ValueError(f"{info.field_name} must be >= 1, got {value}")
        return value

    @field_validator("episodic_ttl_days")
    @classmethod
    def _validate_ttl(cls, value: int) -> int:
        # 0 是合法值：表示 episodic 记忆不过期。
        if value < 0:
            raise ValueError(f"memory_episodic_ttl_days must be >= 0, got {value}")
        return value

    @field_validator("doc_chunk_overlap")
    @classmethod
    def _validate_chunk_overlap(cls, value: int, info: ValidationInfo) -> int:
        if value < 0:
            raise ValueError(f"{info.field_name} must be >= 0, got {value}")
        # 重叠必须小于块长（与 doc_chunk_chars 同属一组配置），
        # 否则分块会陷入死循环。
        chars = info.data.get("doc_chunk_chars")
        if chars is not None and value >= chars:
            raise ValueError(f"doc_chunk_overlap must be < doc_chunk_chars ({chars}), got {value}")
        return value


class ObservabilitySettings(BaseModel):
    """可观测性配置节：env ``LOG_JSON`` / ``HEALTH_PROBE_MODEL``。"""

    model_config = SettingsConfigDict(extra="ignore", validate_default=True)

    # 结构化 JSON 日志行（开发环境默认人类可读）。生产部署应设置
    # LOG_JSON=true 以获得机器可解析的日志。
    log_json: bool = False

    # /health 是否真实探测 LLM 端点可达性（会打真实网络，默认关闭）。
    health_probe_model: bool = False


class _FlatMappingSource(PydanticBaseSettingsSource):
    """平铺键值 source：把环境变量 / .env 的键**原样**放进输入字典。

    pydantic-settings 对嵌套模型默认不做平铺 env 名映射（没有
    delimiter 时），而 ``env_nested_delimiter="_"`` 又会把
    ``CHECKPOINTER_MYSQL_HOST`` 这类多词内部字段切错位。这里改为：
    平铺键原样交给 ``Settings._lift_legacy_flat_fields``（before 校验器）
    按前缀归位到配置节——既保住平铺 env 契约，又不需要逐字段别名。
    优先级仍由 source 顺序保证：init > env > dotenv。
    """

    def __init__(self, settings_cls: type[BaseSettings], mapping: dict[str, Any]) -> None:
        super().__init__(settings_cls)
        self._mapping = mapping

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        return self._mapping.get(field_name), field_name, False

    def __call__(self) -> dict[str, Any]:
        return dict(self._mapping)


# 整键迁移映射：环境变量名（小写）→ 节内字段名（前缀本身即整键时）。
_SECTION_INNER: dict[str, str] = {
    "tool_timeout_seconds": "timeout_seconds",
    "tool_call_log_enabled": "call_log_enabled",
    "log_json": "log_json",
    "health_probe_model": "health_probe_model",
    "tavily_api_key": "tavily_api_key",
}


class Settings(BaseSettings):
    """运行时配置（聚合根）：各域配置节 + 全局杂项。

    环境变量契约平铺如旧（见各节 docstring）；历史平铺**读取**
    （``settings.memory_enabled``）经 ``__getattr__`` 代理到配置节。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        validate_default=True,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """替换默认 source：env 与 .env 都以平铺形式进入归位校验器。

        尊重调用方的 ``_env_file`` 覆盖（含 ``_env_file=None``——禁用
        .env 加载，测试用）。
        """
        import os

        # ``_env_file=None``（禁用 .env）也经由该属性传入：env_file 为
        # None 时不构建 dotenv source。
        dotenv_path = getattr(dotenv_settings, "env_file", None)
        dotenv_data: dict[str, Any] = {}
        if dotenv_path:
            paths = dotenv_path if isinstance(dotenv_path, (list, tuple)) else [dotenv_path]
            try:
                from dotenv import dotenv_values

                for path in paths:
                    dotenv_data.update(
                        {
                            k: v
                            for k, v in dotenv_values(path, encoding="utf-8").items()
                            if v is not None
                        }
                    )
            except OSError:  # pragma: no cover - .env 不可读时按缺失处理
                dotenv_data = {}
        return (
            init_settings,
            _FlatMappingSource(settings_cls, dict(os.environ)),
            _FlatMappingSource(settings_cls, dotenv_data),
            file_secret_settings,
        )

    # -- 模块 ------------------------------------------------------------
    # 要加载的模块名；顺序即装配顺序。registry 会把每个名字解析为
    # agent_base.modules.<name>。
    agent_modules: Annotated[list[str], NoDecode] = Field(default_factory=lambda: ["chat"])

    # -- 服务 ------------------------------------------------------------
    # 允许跨域调用本服务的来源（浏览器客户端，如 agent-base-ui）。
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:3000"]
    )

    # -- 环境 --------------------------------------------------------
    env: str = "development"

    # -- 按域配置节 --------------------------------------------------------
    llm: LlmSettings = Field(default_factory=LlmSettings)
    checkpointer: CheckpointerSettings = Field(default_factory=CheckpointerSettings)
    toolkit: ToolkitSettings = Field(default_factory=ToolkitSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    doc_parse: DocParseSettings = Field(default_factory=DocParseSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)

    @field_validator("agent_modules", "cors_origins", mode="before")
    @classmethod
    def _parse_root_lists(cls, value: object, info: ValidationInfo) -> object:
        return _parse_csv_or_json_list(value, info.field_name or "value")

    @field_validator("env")
    @classmethod
    def _validate_env(cls, value: str) -> str:
        if value not in KNOWN_ENVIRONMENTS:
            raise ValueError(f"unknown ENV {value!r}; expected one of {sorted(KNOWN_ENVIRONMENTS)}")
        return value

    @model_validator(mode="before")
    @classmethod
    def _lift_legacy_flat_fields(cls, data: Any) -> Any:
        """兼容垫片（写入侧）：把历史平铺的关键字/环境值提升到配置节。

        ``Settings(memory_enabled=False)`` 与
        ``Settings(**{"memory.enabled": ...})`` 之外，测试与旧调用方大量
        使用平铺 kwargs；这里按字段名前缀归位到对应配置节，平铺契约
        保持可写。
        """
        if not isinstance(data, dict):
            return data
        lifted: dict[str, dict[str, Any]] = {}
        out: dict[str, Any] = {}
        sections = {
            "llm_": "llm",
            "checkpointer_": "checkpointer",
            "toolkit_": "toolkit",
            "tool_timeout_seconds": "toolkit",
            "tool_call_log_enabled": "toolkit",
            "search_": "search",
            "tavily_api_key": "search",
            "doc_parse_": "doc_parse",
            "memory_": "memory",
            "log_json": "observability",
            "health_probe_model": "observability",
        }
        for key, value in data.items():
            lower = str(key).lower()
            matched_prefix = ""
            for prefix, section in sections.items():
                if lower.startswith(prefix):
                    matched_prefix = prefix
                    target = section
                    break
            if matched_prefix:
                inner = lower[len(matched_prefix) :]
                # TOOL_*/观测整键迁移到节内的新名字。
                inner = _SECTION_INNER.get(lower, inner)
                lifted.setdefault(str(target), {})[inner] = value
            else:
                out[key] = value
        # 根级字段的 env/dotenv 键可能全大写（AGENT_MODULES）：归一小写。
        for key in [
            k
            for k in out
            if k != k.lower() and k.lower() in ("agent_modules", "cors_origins", "env")
        ]:
            # init kwargs（已是小写）优先：只补缺，不覆盖。
            out.setdefault(key.lower(), out.pop(key))
        for section, values in lifted.items():
            if section in out and isinstance(out[section], dict):
                out[section] = {**values, **out[section]}
            else:
                out[section] = values
        return out

    def __getattr__(self, name: str) -> Any:
        """兼容垫片（读取侧）：历史平铺属性代理到对应配置节（只读）。

        仅在正常属性查找失败时触发——配置节字段名不受影响。新代码应
        使用节访问（``settings.memory.enabled``）。
        """
        if name in _SECTION_INNER:
            # 整键迁移的历史字段（TOOL_TIMEOUT_SECONDS 等在节内改了名）。
            inner = _SECTION_INNER[name]
            if name.startswith("tool_"):
                section_name = "toolkit"
            elif name in ("log_json", "health_probe_model"):
                section_name = "observability"
            else:
                section_name = "search"
            return getattr(getattr(self, section_name), inner)
        if name.startswith(("checkpointer_", "doc_parse_")):
            section_name, inner = name.split("_", 1)[0], name.split("_", 1)[1]
            if name.startswith("checkpointer_"):
                section_name, inner = "checkpointer", name[len("checkpointer_") :]
            else:
                section_name, inner = "doc_parse", name[len("doc_parse_") :]
        elif name.startswith(("memory_", "llm_", "search_", "toolkit_")):
            section_name, inner = name.split("_", 1)
            if section_name == "toolkit" and name in (
                "tool_timeout_seconds",
                "tool_call_log_enabled",
            ):
                # TOOL_* 前缀的工具池字段在 toolkit 节里改了名。
                inner = {
                    "tool_timeout_seconds": "timeout_seconds",
                    "tool_call_log_enabled": "call_log_enabled",
                }[name]
                section_name = "toolkit"
        elif name in ("log_json", "health_probe_model"):
            section_name, inner = "observability", name
        elif name == "tavily_api_key":
            section_name, inner = "search", "tavily_api_key"
        else:
            raise AttributeError(name)
        section = getattr(self, section_name)
        if hasattr(section, inner):
            return getattr(section, inner)
        raise AttributeError(name)

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
        if not self.llm.api_key.get_secret_value().strip():
            raise SettingsError(
                "LLM_API_KEY is required in production (ENV=production); "
                "refusing to start with an empty key"
            )
        if self.memory.enabled and not self.memory.auth_secret.get_secret_value().strip():
            raise SettingsError(
                "MEMORY_AUTH_SECRET is required in production when MEMORY_ENABLED=true: "
                "memory isolation is enforced via signed X-User-Id "
                "(clients must send X-User-Sig = HMAC-SHA256(X-User-Id, secret))"
            )
        if self.checkpointer.backend == "mysql":
            missing = [
                name
                for name, value in (
                    ("CHECKPOINTER_MYSQL_USER", self.checkpointer.mysql_user),
                    ("CHECKPOINTER_MYSQL_DATABASE", self.checkpointer.mysql_database),
                    (
                        "CHECKPOINTER_MYSQL_PASSWORD",
                        self.checkpointer.mysql_password.get_secret_value(),
                    ),
                )
                if not str(value).strip()
            ]
            if missing:
                raise SettingsError(
                    f"{', '.join(missing)} required in production with CHECKPOINTER_BACKEND=mysql"
                )
