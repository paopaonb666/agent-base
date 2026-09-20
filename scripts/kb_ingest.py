"""把本地目录的文档批量摄取进知识库（成本治理 T3.1：自用全盘问答入口）。

用法::

    python scripts/kb_ingest.py --dir D:/my/docs --module chat \
        --base-url http://localhost:8000 --user-id alice [--watch] [--interval 30]

- 遍历目录（递归）下的 pdf/docx/txt/md，逐个 POST 到既有上传端点
  ``/v1/agents/{module}/files``——解析、切块、向量化、属主记录全部
  复用服务端公开路径，脚本不碰存储细节；
- 幂等：状态文件（``--state``，默认目标目录下 ``.kb_ingest_state.json``）
  记录 sha256 -> file_id，内容未变的文件跳过；重复运行输出 skipped 计数；
- ``--watch`` 轮询模式：按 ``--interval`` 秒增量发现新文件与内容变更，
  Ctrl-C 退出；并发受 ``--concurrency`` 限制（默认 2，友好对待服务端）；
- 退出码：全部成功 0；部分/全部失败 1（stderr 列出失败清单）。

注意：服务端的知识库切块/向量化是上传完成后的后台任务，脚本返回后
``GET /v1/knowledge/search`` 可能需要一两秒才能看到新文件的分块。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

import httpx

# 允许的文档后缀（与 tools/parsing 的文档格式一致；图片是附件不是知识）。
ALLOWED_SUFFIXES: frozenset[str] = frozenset({".pdf", ".docx", ".txt", ".md"})

DEFAULT_CONCURRENCY = 2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量摄取本地文档进知识库")
    parser.add_argument("--dir", required=True, help="要摄取的文档目录（递归）")
    parser.add_argument("--module", default="chat", help="目标模块（默认 chat）")
    parser.add_argument("--base-url", default="http://localhost:8000", help="服务地址")
    parser.add_argument("--user-id", default="default", help="X-User-Id（知识库作用域）")
    parser.add_argument("--user-sig", default="", help="可选 X-User-Sig（production 鉴权）")
    parser.add_argument(
        "--state", default="", help="状态文件路径（默认 <dir>/.kb_ingest_state.json）"
    )
    parser.add_argument("--watch", action="store_true", help="轮询模式：持续增量摄取")
    parser.add_argument("--interval", type=int, default=30, help="watch 轮询间隔秒（默认 30）")
    parser.add_argument(
        "--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="并行上传数（默认 2）"
    )
    parser.add_argument("--dry-run", action="store_true", help="只列出将摄取的文件，不上传")
    return parser.parse_args()


def _load_state(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        print(f"警告：状态文件 {path} 损坏，视为空状态重新摄取", file=sys.stderr)
        return {}


def _save_state(path: Path, state: dict[str, dict[str, str]]) -> None:
    path.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")


def _discover(root: Path) -> list[Path]:
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in ALLOWED_SUFFIXES
        and p.name != ".kb_ingest_state.json"
    )


async def _upload_one(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    path: Path,
    semaphore: asyncio.Semaphore,
) -> tuple[Path, str | None, str | None]:
    """上传单个文件；返回 (path, file_id 或 None, 失败原因或 None)。"""
    try:
        data = path.read_bytes()
    except OSError as exc:
        return path, None, f"读取失败：{exc}"
    sha = hashlib.sha256(data).hexdigest()
    async with semaphore:
        try:
            response = await client.post(
                url,
                files={"file": (path.name, data)},
                headers=headers,
            )
        except httpx.HTTPError as exc:
            return path, None, f"请求失败：{exc}"
    if response.status_code not in (200, 201):
        return path, None, f"HTTP {response.status_code}：{response.text[:200]}"
    file_id = str(response.json().get("file_id", ""))
    if not file_id:
        return path, None, "响应缺少 file_id"
    return path, f"{sha}:{file_id}", None


async def _ingest_round(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    root: Path,
    state_path: Path,
    state: dict[str, dict[str, str]],
) -> int:
    """一轮摄取；返回本轮失败的文件数。"""
    headers = {"X-User-Id": args.user_id}
    if args.user_sig:
        headers["X-User-Sig"] = args.user_sig
    url = f"{args.base_url.rstrip('/')}/v1/agents/{args.module}/files"

    paths = _discover(root)
    pending: list[tuple[Path, str]] = []  # (path, sha)
    for path in paths:
        try:
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            print(f"跳过（无法读取）：{path}：{exc}", file=sys.stderr)
            continue
        if sha in state:
            continue
        pending.append((path, sha))
    # 内容相同但换过位置的文件也跳过（sha 已在状态里）。
    if args.dry_run:
        for path, _sha in pending:
            print(f"[dry-run] 将摄取：{path}")
        print(f"dry-run：共 {len(pending)} 个待摄取，{len(paths) - len(pending)} 个已跳过")
        return 0

    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    results = await asyncio.gather(
        *(_upload_one(client, url, headers, path, semaphore) for path, _sha in pending)
    )
    failures = 0
    for (path, file_id, error), (_pending_path, _pending_sha) in zip(results, pending, strict=True):
        if error is not None or file_id is None:
            failures += 1
            print(f"失败：{path}：{error}", file=sys.stderr)
            continue
        sha_part, real_id = file_id.split(":", 1)
        state[sha_part] = {"file_id": real_id, "path": str(path)}
        print(f"已摄取：{path} -> {real_id}")
    skipped = len(paths) - len(pending)
    if pending or skipped:
        print(f"本轮：上传 {len(pending) - failures}，跳过 {skipped}，失败 {failures}")
    _save_state(state_path, state)
    return failures


async def _run(args: argparse.Namespace) -> int:
    root = Path(args.dir)
    if not root.is_dir():
        print(f"目录不存在：{root}", file=sys.stderr)
        return 1
    state_path = Path(args.state) if args.state else root / ".kb_ingest_state.json"
    state = _load_state(state_path)
    async with httpx.AsyncClient(timeout=120) as client:
        failures = await _ingest_round(client, args, root, state_path, state)
        while args.watch:
            await asyncio.sleep(max(1, args.interval))
            failures = await _ingest_round(client, args, root, state_path, state)
    return 1 if failures else 0


def main() -> int:
    args = _parse_args()
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("\n已停止（watch 模式退出）")
        return 0


if __name__ == "__main__":
    sys.exit(main())
