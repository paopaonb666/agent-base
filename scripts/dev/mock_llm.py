"""开发/验收用的本地 mock LLM（OpenAI 兼容 /chat/completions）。

用途：在没有真实 API key 的环境里跑通 agent-base 的完整链路——
端到端验收、前端联调、树功能演示。对每一次请求轮换一条固定文案，
让多轮对话/分岔/重新生成的每条回复肉眼可辨。

用法：
    python scripts/dev/mock_llm.py [port]     # 默认 9999

配对启动 agent-base：
    LLM_PROVIDER=openai-compatible LLM_API_KEY=test \\
    LLM_BASE_URL=http://127.0.0.1:9999/v1 LLM_MODEL=mock \\
    CHECKPOINTER_BACKEND=sqlite REDIS_ENABLED=false MEMORY_ENABLED=false \\
    uvicorn agent_base.entrypoints.server:app --port 8000
"""

from __future__ import annotations

import asyncio
import itertools
import json
import sys
import time
from collections.abc import AsyncIterator
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()

_REPLIES = (
    "回复一：这是 mock 模型的固定回答。",
    "回复二：这是 mock 模型的另一条回答，便于在消息树里区分分支。",
    "回复三：mock 模型按请求轮换文案。",
    "回复四：mock 模型继续轮换。",
    "回复五：mock 模型的第五条固定回答。",
    "回复六：mock 模型的第六条固定回答。",
)
_counter = itertools.count(1)


def _reply() -> str:
    return _REPLIES[next(_counter) % len(_REPLIES)]


def _chunk(completion_id: str, *, content: str | None, finish: str | None) -> str:
    delta: dict[str, str] = {}
    if content is not None:
        delta["role"] = "assistant"
        delta["content"] = content
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "mock",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Any:
    body = await request.json()
    completion_id = f"cmpl-{next(_counter)}"
    reply = _reply()
    if body.get("stream"):

        async def _stream() -> AsyncIterator[str]:
            # 两个内容块 + 终止块：模拟真实的流式分段。
            yield _chunk(completion_id, content=reply[:4], finish=None)
            await asyncio.sleep(0.02)
            yield _chunk(completion_id, content=reply[4:], finish=None)
            await asyncio.sleep(0.02)
            yield _chunk(completion_id, content=None, finish="stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(_stream(), media_type="text/event-stream")
    return JSONResponse(
        {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "mock",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": reply},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9999
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
