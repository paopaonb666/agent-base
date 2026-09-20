"""演示：消费 SSE 事件契约——把一次 invoke 的事件流逐帧打印。

演示什么
    README「事件契约（SSE）」节的活文档：连接本地 server 的 invoke 端点，
    解析并打印每一个契约事件（ping/step/delta/tool_call/plan/done/error）。
    事件类型与字段的权威定义在 `extensions/events.py`。

怎么跑
    1) 另开终端启动服务（.env 需已配 LLM_API_KEY）：
       uvicorn agent_base.entrypoints.server:app
    2) python examples/sse_stream_demo.py ["你的问题"] [服务地址]
       默认问题"用一句话介绍你自己"，默认地址 http://localhost:8000。

预期输出（事件序列因模块与模型而异）
    event=step     {"name":"...","status":"running",...}
    event=delta    你
    event=delta    好
    event=done     {"thread_id":"..."}
"""

from __future__ import annotations

import json
import sys

import httpx

DEFAULT_BASE = "http://localhost:8000"


def print_frame(event: str, data: str) -> None:
    """按事件类型挑一个重点字段打印，完整载荷以 JSON 形式跟在后面。"""
    try:
        payload = json.loads(data)
    except ValueError:
        print(f"event={event:<10} {data}")
        return
    if event == "delta":
        print(payload.get("content", ""), end="", flush=True)
        return
    if event == "done":
        print()  # 结束 delta 的未换行输出（delta 之后直接 error 时兜底）
    print(f"event={event:<10} {data}")


def main() -> None:
    message = sys.argv[1] if len(sys.argv) > 1 else "用一句话介绍你自己"
    base = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_BASE
    url = f"{base}/v1/agents/chat/invoke"
    # X-User-Id：开发环境可任意指定；生产（ENV=production）需附 HMAC 签名，
    # 见 README「身份与安全」。
    headers = {"X-User-Id": "demo"}
    event = ""
    prev_delta = False
    with httpx.stream("POST", url, json={"message": message}, headers=headers, timeout=120) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line.startswith("event:"):
                event = line[len("event:") :].strip()
            elif line.startswith("data:") and event:
                if prev_delta and event != "delta":
                    print()  # delta 流结束，换行再打印后续事件
                print_frame(event, line[len("data:") :].strip())
                prev_delta = event == "delta"


if __name__ == "__main__":
    main()
