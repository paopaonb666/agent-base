"""夜间批记忆形成（成本治理 T3.2）：每轮管线的低成本替代路径。

用法::

    python scripts/memory_nightly_capture.py [--user-id default] [--date today] [--dry-run]

定位：把 ``MEMORY_CAPTURE_ENABLED`` 设为 false（配套建议
``MEMORY_SUMMARY_ENABLED=false``）后，服务端每轮 invoke 不再消耗形成
管线的 LLM 调用；本脚本（配合 cron/计划任务每天跑一次）用一次批量
调用完成全天各活跃线程的抽取+整合（``force=True`` 旁路门控），全程
走形成管线的快档分派（``MEMORY_PIPELINE_PROFILE=split`` + ``LLM_FAST_*``，
见 .env.example）。

核心逻辑在 ``agent_base.extensions.scheduler.run_memory_capture``（进程内
调度器与 webhook 触发共用同一实现）；本脚本是 CLI 薄壳：
``SCHEDULER_ENABLED=true`` 的部署可以直接用调度器的 ``memory_nightly_capture``
任务类型替代外部 cron。

- 线程来源：``thread_index`` 中在 ``--date``（默认今天，本地时区）之后
  活跃（updated_at）的线程，与线程列表端点同源；
- 失败安全：单线程失败不中断批次，仅记 stderr；
- ``--dry-run``：只列出将处理的线程与消息数，不做任何 LLM 调用。

退出码：处理成功（含 0 个线程）0；配置/装配问题 1。
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import date, datetime

from agent_base.core.bootstrap import AgentRuntime, create_runtime
from agent_base.extensions.scheduler import run_memory_capture


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="夜间批记忆形成（force 旁路门控）")
    parser.add_argument(
        "--user-id",
        default="default",
        help="要处理的用户（逗号分隔多个；thread_index 按用户索引，默认 default）",
    )
    parser.add_argument(
        "--date",
        default="today",
        help="处理该日期（本地时区）之后活跃的线程：today 或 YYYY-MM-DD",
    )
    parser.add_argument("--dry-run", action="store_true", help="只列出线程，不做 LLM 调用")
    return parser.parse_args()


def _since_ts(date_arg: str) -> float:
    day = date.today() if date_arg == "today" else datetime.strptime(date_arg, "%Y-%m-%d").date()
    start = datetime.combine(day, datetime.min.time())
    return start.timestamp()


async def capture_since(
    rt: AgentRuntime, *, user_ids: list[str], since_ts: float, dry_run: bool = False
) -> tuple[int, int]:
    """批量处理 ``since_ts`` 之后活跃的线程；返回 (处理数, 跳过数)。

    兼容薄壳：委托包内实现（dry-run 时把处理明细打到 stdout）。
    """
    if dry_run:
        assert rt.thread_index is not None
        processed = 0
        skipped = 0
        for user_id in user_ids:
            entries = await rt.thread_index.list_thread_index(user_id, limit=500)
            for entry in entries:
                if entry.updated_at < since_ts:
                    skipped += 1
                    continue
                print(f"[dry-run] 将处理：{entry.thread_id}（user={user_id}，{entry.title}）")
                processed += 1
        return processed, skipped
    summary = await run_memory_capture(rt, user_ids=user_ids, since_ts=since_ts, dry_run=False)
    return int(summary["processed"]), int(summary["skipped"]) + int(summary["failed"])


async def _run(args: argparse.Namespace) -> int:
    since_ts = _since_ts(args.date)
    rt = await create_runtime()
    try:
        if args.dry_run:
            print(
                f"dry-run：处理 {args.date}（>={datetime.fromtimestamp(since_ts)}）之后活跃的线程"
            )
        processed, skipped = await capture_since(
            rt,
            user_ids=[u.strip() for u in args.user_id.split(",") if u.strip()],
            since_ts=since_ts,
            dry_run=args.dry_run,
        )
        print(f"完成：处理 {processed}，跳过 {skipped}")
        return 0
    finally:
        await rt.close()


def main() -> int:
    args = _parse_args()
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("\n已停止")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
