"""为记忆身份头生成签名（X-User-Id / X-User-Sig）。

服务端配置 ``MEMORY_AUTH_SECRET`` 后，所有带记忆作用域的请求必须
附带 ``X-User-Sig = HMAC-SHA256(X-User-Id, secret)`` 的十六进制签名。
本脚本供运维/服务端调用方生成这对请求头（密钥不该进浏览器——
浏览器场景请走服务端代理或真实身份体系）。

用法::

    python scripts/memory_user_sig.py --user-id alice --secret "$MEMORY_AUTH_SECRET"
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="生成 X-User-Id / X-User-Sig 请求头（记忆身份鉴权）"
    )
    parser.add_argument("--user-id", required=True, help="用户标识（1-64 字符，[\\w.-]）")
    parser.add_argument("--secret", required=True, help="与服务端 MEMORY_AUTH_SECRET 一致的密钥")
    args = parser.parse_args()
    signature = hmac.new(
        args.secret.encode("utf-8"), args.user_id.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    print(f"X-User-Id: {args.user_id}")
    print(f"X-User-Sig: {signature}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
