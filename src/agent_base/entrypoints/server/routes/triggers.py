"""事件触发端点（ADR-007，Loop 3）：手动/webhook 触发已声明的调度任务。

外部系统（cron、CI、业务回调）经 ``POST /v1/triggers/{job_name}`` 触发
服务端立即执行一次任务——与进程内调度器共用同一份任务声明（``SCHEDULER_
JOBS_JSON``）。执行经 ``spawn_background`` 异步进行，端点立即 202；
结果看日志与 ``/health`` 的 scheduler 组件。调度器未启用时端点 503
（任务声明不存在），未知名 404。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Path, Request

from agent_base.entrypoints.server.deps import get_current_user

router = APIRouter(prefix="/v1/triggers", tags=["triggers"])

_JobPathParam = Path(max_length=64, pattern=r"^[a-z][a-z0-9_]*$")


@router.post("/{job_name}")
async def trigger_job(
    request: Request,
    job_name: str = _JobPathParam,
    user_id: str = Depends(get_current_user),
) -> dict[str, Any]:
    """触发一次任务执行（异步；执行结果看日志与 /health）。"""
    scheduler = getattr(request.app.state, "scheduler", None)
    if scheduler is None:
        raise HTTPException(
            status_code=503,
            detail="调度器未启用（SCHEDULER_ENABLED=false 或无任务声明）",
        )
    if not scheduler.trigger(job_name):
        raise HTTPException(
            status_code=404,
            detail=f"未声明的任务 {job_name!r}；已声明：{[j.name for j in scheduler.jobs]}",
        )
    return {"triggered": job_name, "status": "scheduled"}
