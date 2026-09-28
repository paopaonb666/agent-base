"""身份作用域（S1 / P0-5）：可插拔的 ``AuthBackend`` + 默认 HMAC 头实现。

所有用户作用域端点（记忆 / 文件 / 会话线程）都通过 FastAPI 依赖
（``deps.get_current_user`` → ``AuthBackend.authenticate``）取得身份，
不再逐个端点手写校验——这是 S1 的结构性修复之一：身份解析只有一条
路径，遗漏只会发生在"忘记声明依赖"，而不是"校验逻辑抄漏了一半"。

默认实现 ``HmacHeaderAuth``：
- ``X-User-Id`` 头缺省为 ``default``；白名单正则与 X-Request-ID 相同
  （该值进记忆表与日志，非法值以 400 拒绝而不是静默替换）。
- 配置了 ``MEMORY_AUTH_SECRET`` 后必须附带
  ``X-User-Sig = HMAC-SHA256(message, secret)``（hex），否则 401。
  防重放（C2）默认开启（``MEMORY_AUTH_REPLAY_WINDOW_SECONDS=300``）：
  ``message = user_id + "\\n" + X-User-Timestamp + "\\n" + X-User-Nonce``，
  时间戳必须是 unix 秒且落在 ``now ± 窗口`` 内；带 nonce 的请求在
  进程内记录指纹，窗口期内重复出现即 401（重放拒绝）。窗口设 0 退回
  仅覆盖 user_id 的旧方案（截获的请求头可无限期重放，不推荐）。
- 未配置密钥仅限 development/单机自用（production 且 memory_enabled
  在启动时快速失败）。

已知局限：nonce 重放缓存是**单进程**口径（与 /metrics 同款），多
worker 部署需网关侧防重放或实现自定义 ``AuthBackend``；公网多用户
部署应在网关层替换为真实身份体系（实现 ``AuthBackend`` 即可接入，
无需改路由）。
"""

from __future__ import annotations

import hashlib
import hmac
import re
import time
from collections import OrderedDict
from typing import Protocol, runtime_checkable

from fastapi import HTTPException, Request

from agent_base.core.config import Settings

# 客户端提供的 X-Request-ID / X-User-Id 必须通过此白名单，否则拒绝并
# 另发新 id：该值会被回显到响应头并写进每条日志，放行任意字符串等于
# 允许伪造日志行 / 触发非法响应头。
REQUEST_ID_RE = re.compile(r"[A-Za-z0-9._-]{1,64}")


@runtime_checkable
class AuthBackend(Protocol):
    """从请求解析用户作用域身份；认证失败抛 HTTPException（401/400）。"""

    def authenticate(self, request: Request, settings: Settings) -> str: ...


class HmacHeaderAuth:
    """默认实现：X-User-Id + X-User-Sig HMAC 校验（含时间戳/nonce 防重放）。

    nonce 重放缓存按实例持有（``create_app`` 每应用构造一个后端实例，
    生命周期与进程一致）；LRU 上限防止长期运行下的无界增长。
    """

    def __init__(self, *, replay_cache_limit: int = 4096) -> None:
        self._seen_nonces: OrderedDict[str, float] = OrderedDict()
        self._replay_cache_limit = replay_cache_limit

    def authenticate(self, request: Request, settings: Settings) -> str:
        raw = request.headers.get("X-User-Id")
        if not raw:
            user_id = "default"
        elif not REQUEST_ID_RE.fullmatch(raw):
            raise HTTPException(
                status_code=400,
                detail="X-User-Id 非法：仅允许字母/数字/点/下划线/连字符，1-64 字符",
            )
        else:
            user_id = raw
        secret = settings.memory.auth_secret.get_secret_value().strip()
        if secret:
            window = settings.memory.auth_replay_window_seconds
            timestamp = ""
            nonce = ""
            if window > 0:
                timestamp = request.headers.get("X-User-Timestamp") or ""
                if not timestamp:
                    raise HTTPException(
                        status_code=401,
                        detail="缺少 X-User-Timestamp：该部署已启用防重放时间窗",
                    )
                try:
                    ts = int(timestamp)
                except ValueError:
                    raise HTTPException(
                        status_code=401, detail="X-User-Timestamp 必须是 unix 秒（整数）"
                    ) from None
                if abs(time.time() - ts) > window:
                    raise HTTPException(
                        status_code=401,
                        detail="X-User-Timestamp 超出容忍窗口：请求已过期或时钟漂移过大",
                    )
                raw_nonce = request.headers.get("X-User-Nonce") or ""
                if raw_nonce:
                    if not REQUEST_ID_RE.fullmatch(raw_nonce):
                        raise HTTPException(
                            status_code=400,
                            detail="X-User-Nonce 非法：仅允许字母/数字/点/下划线/连字符，1-64 字符",
                        )
                    nonce = raw_nonce
                message = f"{user_id}\n{timestamp}\n{nonce}"
            else:
                message = user_id
            sig = request.headers.get("X-User-Sig") or ""
            expected = hmac.new(
                secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256
            ).hexdigest()
            if not hmac.compare_digest(sig, expected):
                raise HTTPException(
                    status_code=401,
                    detail="X-User-Sig 缺失或不正确：该部署已启用记忆身份鉴权",
                )
            if nonce:
                # 通过了签名校验才记账/查账：错签名的 nonce 不占用缓存。
                self._reject_replay(nonce, expires_at=ts + 2 * window)
        return user_id

    def _reject_replay(self, nonce: str, *, expires_at: float) -> None:
        """窗口期内的 nonce 只允许出现一次；过期指纹视作新请求。"""
        seen_until = self._seen_nonces.get(nonce)
        if seen_until is not None and seen_until > time.time():
            raise HTTPException(status_code=401, detail="X-User-Nonce 已使用：重放被拒绝")
        self._seen_nonces[nonce] = expires_at
        self._seen_nonces.move_to_end(nonce)
        while len(self._seen_nonces) > self._replay_cache_limit:
            self._seen_nonces.popitem(last=False)
