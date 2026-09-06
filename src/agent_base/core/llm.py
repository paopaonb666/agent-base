"""装配 openai 兼容的对话模型。

所有支持的 provider（deepseek / zhipu / openai-compatible）共用同一个
``ChatOpenAI`` 客户端。切换 provider 意味着在配置中修改 ``LLM_BASE_URL``
和 ``LLM_MODEL``——永远不改代码。
"""

from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI
from openai import OpenAIError

from agent_base.core.config import Settings, SettingsError


def build_llm(settings: Settings) -> ChatOpenAI:
    """根据已校验的 settings 构建对话模型。

    ``LLM_API_KEY`` 优先；未设置时，客户端回退到环境中的
    ``OPENAI_API_KEY``（langchain-openai 的标准回退）。如果两者都没有，
    ``ChatOpenAI`` 会在构造时抛错——我们把它转换成可操作的
    ``SettingsError``，而不是泄漏原始的 ``openai.OpenAIError``。
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
