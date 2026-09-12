"""embedding 客户端（M6b）的测试：批量、降级、冷却与维度告警。

全部走 httpx.MockTransport，绝不打真实网络（真实端点只做手动冒烟）。
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from agent_base.core.config import Settings
from agent_base.memory.embeddings import (
    EmbeddingError,
    NullEmbedding,
    OpenAICompatibleEmbedding,
    build_embedding_client,
)


def _settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "llm_api_key": "sk-test",
        "memory_embedding_api_key": "sk-embed",
    }
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


def _transport(handler: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_null_embedding_returns_none() -> None:
    assert await NullEmbedding().embed(["任意"]) is None
    assert await NullEmbedding().embed([]) is None


def test_build_embedding_client_selection() -> None:
    disabled = build_embedding_client(_settings(memory_embedding_enabled=False))
    assert isinstance(disabled, NullEmbedding)
    assert isinstance(
        build_embedding_client(_settings(memory_embedding_api_key="")), NullEmbedding
    )
    client = build_embedding_client(_settings())
    assert isinstance(client, OpenAICompatibleEmbedding)
    assert client.dims == 1024


async def test_embed_success_single_batch() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            200,
            json={"data": [{"index": i, "embedding": [0.1 * (i + 1), 0.2]} for i in range(2)]},
        )

    client = OpenAICompatibleEmbedding(
        base_url="https://embed.example.com/v1",
        api_key="sk-x",
        model="test-model",
        dims=2,
        client=_transport(handler),
    )
    vectors = await client.embed(["你好", "世界"])
    assert vectors == [[0.1, 0.2], [0.2, 0.2]]
    assert len(calls) == 1
    assert calls[0].url.path.endswith("/embeddings")
    assert calls[0].headers["Authorization"] == "Bearer sk-x"
    body = json.loads(calls[0].content)
    assert body["model"] == "test-model" and body["input"] == ["你好", "世界"]


async def test_embed_batches_split_requests() -> None:
    calls: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body["input"])
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": i, "embedding": [1.0, 0.0]} for i in range(len(body["input"]))
                ]
            },
        )

    client = OpenAICompatibleEmbedding(
        base_url="https://embed.example.com/v1",
        api_key="sk-x",
        model="m",
        dims=2,
        batch_size=2,
        client=_transport(handler),
    )
    vectors = await client.embed(["a", "b", "c"])
    assert len(vectors) == 3
    assert calls == [["a", "b"], ["c"]]


async def test_embed_failure_degrades_and_cooldowns() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(500, text="server exploded")

    client = OpenAICompatibleEmbedding(
        base_url="https://embed.example.com/v1",
        api_key="sk-x",
        model="m",
        dims=2,
        cooldown_seconds=3600.0,
        client=_transport(handler),
    )
    # 失败 → None（不抛异常）。
    assert await client.embed(["文本"]) is None
    assert attempts == 1
    # 冷却期内不再打真实网络。
    assert await client.embed(["文本"]) is None
    assert attempts == 1
    # 空输入不触发请求、也不重置冷却。
    assert await client.embed([]) == []
    assert attempts == 1


async def test_embed_network_error_raises_internally() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    client = OpenAICompatibleEmbedding(
        base_url="https://embed.example.com/v1",
        api_key="sk-x",
        model="m",
        dims=2,
        cooldown_seconds=0.0,  # 不冷却：直接观察第二次真实重试
        client=_transport(handler),
    )
    assert await client.embed(["x"]) is None
    assert await client.embed(["x"]) is None


async def test_embed_malformed_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": True})

    client = OpenAICompatibleEmbedding(
        base_url="https://embed.example.com/v1",
        api_key="sk-x",
        model="m",
        dims=2,
        cooldown_seconds=0.0,
        client=_transport(handler),
    )
    assert await client.embed(["x"]) is None
    with pytest.raises(EmbeddingError):
        # 直接调用内部批处理可观察异常类型（embed 对外吞掉它）。
        await client._embed_batch(["x"])


async def test_embed_dims_mismatch_keeps_actual() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.5, 0.5]}]})

    client = OpenAICompatibleEmbedding(
        base_url="https://embed.example.com/v1",
        api_key="sk-x",
        model="m",
        dims=1024,  # 配置与实际不符
        client=_transport(handler),
    )
    vectors = await client.embed(["x"])
    assert vectors == [[0.5, 0.5]]  # 以实际维度为准（告警只发一次）


async def test_embed_long_input_truncated() -> None:
    seen: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.extend(len(text) for text in body["input"])
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})

    client = OpenAICompatibleEmbedding(
        base_url="https://embed.example.com/v1",
        api_key="sk-x",
        model="m",
        dims=1,
        client=_transport(handler),
    )
    await client.embed(["x" * 999_999])
    assert seen == [6000]
