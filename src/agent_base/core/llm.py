"""装配 openai 兼容的对话模型。

所有支持的 provider（deepseek / zhipu / openai-compatible）共用同一个
``ChatOpenAI`` 客户端。切换 provider 意味着在配置中修改 ``LLM_BASE_URL``
和 ``LLM_MODEL``——永远不改代码。

双档位（成本治理）：``build_llm(settings, profile="fast")`` 在 ``LLM_FAST_*``
已配置时返回独立客户端（免费/低价模型吃批量任务），未配置时返回与 main
相同的客户端——快档是优化不是依赖。``ResilientLLM`` 为免费档包装并发
限流（免费档普遍按并发数限速）与主力模型回退。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from langchain_openai import ChatOpenAI
from openai import OpenAIError

from agent_base.core.config import Settings, SettingsError

logger = logging.getLogger(__name__)


class _ToolSafeStreamChatOpenAI(ChatOpenAI):
    """快档专用客户端：绑定工具时退回非流式，纯聊天仍流式。

    背景（2026-09-20 实测）：SiliconFlow 的 GLM-4-9B 在流式模式下把
    工具名拆进 ``arguments`` delta（``function.name`` 为空、id 为 null），
    违反 OpenAI 流式协议——langchain 聚合不出合法的 tool_calls，图执行
    静默拿到空工具调用与空回答（同请求非流式则完全正常）。因此该模型
    一旦作为快档进入图执行路径（Tier 1 / planner fast 都会 bind_tools），
    工具轮必须走非流式让服务端聚合出正确格式；不绑工具的对话仍流式，
    SSE delta 体验不变。
    """

    async def _astream(
        self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> Any:
        if kwargs.get("tools"):
            result = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            for generation in result.generations:
                message = generation.message
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content=message.content,
                        tool_calls=getattr(message, "tool_calls", []) or [],
                        additional_kwargs=message.additional_kwargs,
                        response_metadata=message.response_metadata,
                    )
                )
            return
        async for chunk in super()._astream(messages, stop=stop, run_manager=run_manager, **kwargs):
            yield chunk


def build_llm(settings: Settings, profile: str = "main") -> ChatOpenAI:
    """根据已校验的 settings 构建对话模型。

    ``profile="fast"`` 时使用 ``LLM_FAST_*`` 配置；未配置（base_url 与
    model 任一为空）则回落主力配置，不报错。``LLM_API_KEY`` 优先；
    快档未单独配 key 时共用主力 key（同一 provider 双档的常见形态）。

    ``LLM_API_KEY`` 未设置时，客户端回退到环境中的 ``OPENAI_API_KEY``
    （langchain-openai 的标准回退）。如果两者都没有，``ChatOpenAI`` 会在
    构造时抛错——我们把它转换成可操作的 ``SettingsError``，而不是泄漏
    原始的 ``openai.OpenAIError``。
    """
    if profile == "fast" and settings.llm_fast.is_configured:
        fast = settings.llm_fast
        model = fast.model
        base_url = fast.base_url
        fast_key = fast.api_key.get_secret_value().strip()
        main_key = settings.llm.api_key.get_secret_value().strip()
        api_key = fast_key or main_key
    else:
        if profile not in ("main", "fast"):
            raise ValueError(f"unknown llm profile {profile!r}; expected 'main' or 'fast'")
        model = settings.llm.model
        base_url = settings.llm.base_url
        api_key = settings.llm.api_key.get_secret_value().strip()
    kwargs: dict[str, Any] = {"model": model, "base_url": base_url}
    if api_key:
        kwargs["api_key"] = api_key
    # 成本计量（T4.1）：enabled 时挂模型级 usage 采集 callback，并请求
    # 流式路径回报 usage（OpenAI 兼容 provider 的 stream_options；
    # 关闭时完全不发该参数，兼容不支持它的 provider）。
    if settings.cost.enabled:
        from agent_base.extensions.costmeter import COST_METER, CostMeterHandler

        kwargs["stream_usage"] = True
        kwargs["callbacks"] = [CostMeterHandler(model=model, profile=profile, meter=COST_METER)]
    else:
        kwargs["stream_usage"] = False
    # 快档走工具安全子类（见类 docstring：SiliconFlow 流式工具调用的
    # 协议违规）；主档保持原生客户端，行为零变化。
    client_cls: type[ChatOpenAI] = _ToolSafeStreamChatOpenAI if profile == "fast" else ChatOpenAI
    try:
        return client_cls(**kwargs)
    except OpenAIError as exc:
        raise SettingsError(
            "LLM_API_KEY is not configured and no ambient OPENAI_API_KEY is set; "
            "copy .env.example to .env, set LLM_API_KEY, then retry"
        ) from exc


class ResilientLLM:
    """鸭子类型包装：并发限流 + 瞬时错误重试 + 可选主力回退。

    面向后台管线（``ainvoke`` 单消息列表直答），不进 langchain
    ``BaseChatModel`` 继承树——图执行路径不使用它。

    - ``max_concurrency``：信号量限流，尊重免费档并发配额；
    - ``max_retries`` / ``base_delay``：瞬时错误（429/超时）指数退避重试；
    - ``fallback``：重试耗尽后的主力模型；为 None 时原样抛出最后一次错误。
    """

    def __init__(
        self,
        primary: Any,
        fallback: Any | None = None,
        max_retries: int = 2,
        base_delay: float = 1.0,
        max_concurrency: int = 2,
    ) -> None:
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries}")
        if base_delay < 0:
            raise ValueError(f"base_delay must be >= 0, got {base_delay}")
        if max_concurrency < 1:
            raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}")
        self._primary = primary
        self._fallback = fallback
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def ainvoke(self, messages: Any) -> Any:
        attempts = self._max_retries + 1
        last_error: BaseException | None = None
        for attempt in range(attempts):
            try:
                async with self._semaphore:
                    return await self._primary.ainvoke(messages)
            except Exception as exc:
                last_error = exc
                if attempt < attempts - 1:
                    delay = self._base_delay * (2**attempt)
                    logger.warning(
                        "llm: 快速档调用失败（第 %d/%d 次），%.1fs 后重试: %s",
                        attempt + 1,
                        attempts,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)
        if self._fallback is not None:
            logger.warning("llm: 快速档重试耗尽，回退主力模型")
            return await self._fallback.ainvoke(messages)
        assert last_error is not None
        raise last_error
