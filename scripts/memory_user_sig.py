"""为记忆身份头生成签名（X-User-Id / X-User-Timestamp / X-User-Nonce / X-User-Sig）。

服务端配置 ``MEMORY_AUTH_SECRET`` 后，所有带记忆作用域的请求必须附带
``X-User-Sig = HMAC-SHA256(user_id\\n timestamp\\n nonce, secret)`` 的十六
进制签名；时间戳须落在 ``MEMORY_AUTH_REPLAY_WINDOW_SECONDS``（默认
300s）内，nonce 在窗口期内只允许出现一次（防重放）。本脚本供运维/
服务端调用方生成这组请求头（密钥不该进浏览器——浏览器场景请走服务
端代理或真实身份体系）。

用法::

    python scripts/memory_user_sig.py --user-id alice --secret "$MEMORY_AUTH_SECRET"
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import sys
import time
import uuid


def main() -> int:
    parser = argparse.ArgumentParser(description="生成记忆身份鉴权的请求头（含防重放时间戳/nonce）")
    parser.add_argument("--user-id", required=True, help="用户标识（1-64 字符，[\\w.-]）")
    parser.add_argument("--secret", required=True, help="与服务端 MEMORY_AUTH_SECRET 一致的密钥")
    args = parser.parse_args()
    timestamp = str(int(time.time()))
    nonce = uuid.uuid4().hex
    message = f"{args.user_id}\n{timestamp}\n{nonce}"
    signature = hmac.new(
        args.secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    print(f"X-User-Id: {args.user_id}")
    print(f"X-User-Timestamp: {timestamp}")
    print(f"X-User-Nonce: {nonce}")
    print(f"X-User-Sig: {signature}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
