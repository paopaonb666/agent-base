"""身份作用域（S1 / P0-5）：可插拔的 ``AuthBackend`` + 默认 HMAC 头实现。

所有用户作用域端点（记忆 / 文件 / 会话线程）都通过 FastAPI 依赖
（``deps.get_current_user`` → ``AuthBackend.authenticate``）取得身份，
不再逐个端点手写校验——这是 S1 的结构性修复之一：身份解析只有一条
路径，遗漏只会发生在"忘记声明依赖"，而不是"校验逻辑抄漏了一半"。

默认实现 ``HmacHeaderAuth``：
- ``X-User-Id`` 头缺省为 ``default``；白名单正则与 X-Request-ID 相同
  （该值进记忆表与日志，非法值以 400 拒绝而不是静默替换）。
- 配置了 ``MEMORY_AUTH_SECRET`` 后必须附带
  ``X-User-Sig = HMAC-SHA256(user_id, secret)``（hex），否则 401。
- 未配置密钥仅限 development/单机自用（production 且 memory_enabled
  在启动时快速失败）。

已知局限（README 已声明）：签名只覆盖 user_id 字符串本身，无时间戳/
nonce，截获的请求头可重放——公网多用户部署应在网关层替换为真实身份
体系（实现 ``AuthBackend`` 即可接入，无需改路由）。
"""

from __future__ import annotations

import hashlib
import hmac
import re
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
    """默认实现：X-User-Id + 可选的 X-User-Sig HMAC 校验。"""

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
            sig = request.headers.get("X-User-Sig") or ""
            expected = hmac.new(
                secret.encode("utf-8"), user_id.encode("utf-8"), hashlib.sha256
            ).hexdigest()
            if not hmac.compare_digest(sig, expected):
                raise HTTPException(
                    status_code=401,
                    detail="X-User-Sig 缺失或不正确：该部署已启用记忆身份鉴权",
                )
        return user_id
