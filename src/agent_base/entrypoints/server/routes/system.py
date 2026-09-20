"""系统端点（H1 拆分）：健康探针、Prometheus 指标、模块清单。"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse

from agent_base.core.bootstrap import SUPERVISOR_MODULE
from agent_base.entrypoints.server.deps import get_runtime, memory_or_none

if TYPE_CHECKING:  # pragma: no cover - import avoided at runtime
    from agent_base.core.config import Settings

logger = logging.getLogger(__name__)

# 模型端点探活（/health）：结果缓存，避免每个探活请求都打真实网络。
MODEL_PROBE_TTL_SECONDS = 30.0
MODEL_PROBE_TIMEOUT_SECONDS = 3.0

# 探活结果缓存：{base_url: (monotonic 时间, 结果)}。
_model_probe_cache: dict[str, tuple[float, str]] = {}

router = APIRouter(tags=["system"])


async def _probe_model(settings: Settings) -> str:
    """探测 openai 兼容端点的网络可达性（带 TTL 缓存）。

    只证明"端点在网络层可达"——任何 HTTP 应答（含 401/404）都算 ok；
    配额、鉴权属于业务语义，不由 /health 判定。网络错误/超时 = error。
    （测试经包根 ``server_module._probe_model`` 打桩——路由在调用时
    经模块属性解析，补丁生效。）
    """
    cache_key = settings.llm.base_url
    now = time.monotonic()
    hit = _model_probe_cache.get(cache_key)
    if hit is not None and now - hit[0] < MODEL_PROBE_TTL_SECONDS:
        return hit[1]
    status = "error"
    try:
        async with httpx.AsyncClient(timeout=MODEL_PROBE_TIMEOUT_SECONDS) as client:
            await client.get(
                f"{cache_key.rstrip('/')}/models",
                headers={"Authorization": f"Bearer {settings.llm.api_key.get_secret_value()}"},
            )
        status = "ok"  # 任何 HTTP 应答都证明可达
    except Exception:
        logger.warning("health: model endpoint probe failed for %s", cache_key)
    _model_probe_cache[cache_key] = (now, status)
    return status


@router.get("/v1/modules")
async def list_modules(request: Request) -> dict[str, Any]:
    """列出已注册的模块（供前端配置面板选择，免去试错模块名）。"""
    rt = get_runtime(request)
    modules = [
        {"name": m.name, "description": str(getattr(m, "description", ""))}
        for m in rt.modules.values()
    ]
    modules.append(
        {
            "name": SUPERVISOR_MODULE,
            "description": "多 Agent 协作：把所有已注册模块编排为 sub-agent",
        }
    )
    return {"modules": modules}


@router.get("/health")
async def health(request: Request) -> dict[str, Any]:
    """组件健康；``degraded`` 表示部分失败（A4）。"""
    rt = get_runtime(request)
    components: dict[str, str] = {}
    if rt.checkpointer is None:
        components["checkpointer"] = "unconfigured"
    else:
        try:
            await rt.checkpointer.aget_tuple({"configurable": {"thread_id": "__health__"}})
            components["checkpointer"] = "ok"
        except Exception:
            components["checkpointer"] = "error"
    key = rt.settings.llm.api_key.get_secret_value().strip()
    if not key:
        components["model"] = "unconfigured"
    elif not rt.settings.llm.base_url.startswith(("http://", "https://")):
        components["model"] = "misconfigured"
    elif rt.settings.observability.health_probe_model:
        # 真实探活由开关控制（LLM_HEALTH_PROBE_MODEL=true）：探活会打
        # 真实网络，默认关闭——负载均衡器的主动检查通常已覆盖此需求。
        # 经包根属性解析（测试在包根打桩 _probe_model），见其 docstring。
        from agent_base.entrypoints import server as server_pkg

        probe = getattr(server_pkg, "_probe_model", None)
        components["model"] = await (probe or _probe_model)(rt.settings)
    else:
        components["model"] = "ok"
    # 记忆系统（M6）：未启用不是降级（组件缺席即可），启用时探存储。
    memory = memory_or_none(rt)
    if memory is not None:
        components["memory"] = await memory.health_probe()
    # 成本治理（T4.2）：仅在 COST_ENABLED=true 时出现；warn/blocked 都
    # 会把整体状态推到 degraded（预算见底与依赖损坏同级可见）。
    governor = getattr(request.app.state, "cost_governor", None)
    if governor is not None:
        components["cost"] = governor.status()
    status = "ok" if all(v == "ok" for v in components.values()) else "degraded"
    return {"status": status, "components": components}


@router.get("/metrics")
async def metrics_endpoint(request: Request) -> PlainTextResponse:
    from agent_base.extensions.metrics import (
        COST_METRICS,
        MEMORY_METRICS,
        TOOL_METRICS,
    )
    from agent_base.extensions.metrics import Metrics as MetricsCls

    metrics: MetricsCls = request.app.state.metrics
    body = (
        metrics.render()
        + TOOL_METRICS.render_tool_metrics()
        + MEMORY_METRICS.render_memory_metrics()
        + COST_METRICS.render_cost_metrics()
    )
    return PlainTextResponse(body, media_type="text/plain; version=0.0.4; charset=utf-8")
