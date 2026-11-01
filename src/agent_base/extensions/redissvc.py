"""Redis 共享状态服务（ADR-012）：连接生命周期 + 带键前缀的原语面。

Redis 在基座里承载的是**治理状态外置**（成本/预算计数、nonce 防重放、
分布式锁、注入快照缓存、SSE 事件镜像），不是对话或记忆的持久化后端。
``REDIS_ENABLED=false`` 时本模块不会被装配，上述能力全部回退进程内
实现（单进程口径，与 0.3 行为一致）。

设计说明：

- **依赖收敛**：基座各处与记忆包只依赖 ``RedisService`` 的原语面，
  不直接触碰 redis 客户端——测试用手写 fake（``tests/fakes.py``）
  替换，记忆发行包也不必 import redis。
- **键前缀**：所有键与频道自动拼接 ``REDIS_KEY_PREFIX``（默认
  ``agent_base:``），同一实例上多应用共享时互不踩踏。
- **命令超时**：``REDIS_SOCKET_TIMEOUT_SECONDS`` 作用于全部命令——
  治理路径（预算门 / nonce 校验）绝不因 Redis 慢而拖垮请求。
- **启用即承诺**：``create`` 握手（ping）失败即抛 ``RedisServiceError``
  （启动 fail-fast），绝不静默降级成"假共享"。
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import avoided at runtime
    from agent_base.core.config import RedisSettings

logger = logging.getLogger(__name__)

# 抢锁重试间隔（秒）：足够细以快速拿到刚释放的锁，又不打满 Redis。
_LOCK_RETRY_INTERVAL_SECONDS = 0.05

# 订阅轮询间隔（秒）：用带超时的 get_message 轮询而非阻塞 listen()，
# 避免连接级 socket_timeout 与阻塞读相互作用，且随时可被取消。
_PUBSUB_POLL_SECONDS = 1.0

# 安全释放锁（compare-and-delete）：只有 token 匹配（锁仍归本持有者）
# 才删除——防止删掉 TTL 过期后他人已抢到的锁。
_RELEASE_LOCK_LUA = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
else
    return 0
end
"""


class RedisServiceError(RuntimeError):
    """Redis 已启用但不可达/装配失败（启用即承诺，启动 fail-fast）。"""


class RedisService:
    """redis.asyncio 客户端的薄封装：键前缀、原语与分布式锁。

    经 ``create`` 装配（握手失败抛错）；原语方法全部自动拼接键前缀，
    调用方使用不带前缀的逻辑键名。
    """

    def __init__(self, client: Any, *, settings: RedisSettings) -> None:
        self._client = client
        self._prefix = settings.key_prefix
        self._release_script: Any = None

    @classmethod
    async def create(cls, settings: RedisSettings) -> RedisService:
        """建立连接池并握手（ping）；不可达时抛 ``RedisServiceError``。"""
        import redis.asyncio as aioredis  # 惰性导入：未启用时零导入开销

        # REDIS_PASSWORD 优先于 URL 内嵌密码；两者都未配则匿名连接。
        password = settings.password.get_secret_value().strip() or None
        try:
            client = aioredis.from_url(
                settings.url,
                password=password,
                max_connections=settings.pool_max_connections,
                socket_timeout=settings.socket_timeout_seconds,
                decode_responses=True,
            )
            await asyncio.wait_for(client.ping(), timeout=settings.socket_timeout_seconds)
        except Exception as exc:
            raise RedisServiceError(
                f"REDIS_ENABLED=true but Redis is unreachable at {settings.url!r} "
                f"({type(exc).__name__}: {exc}); set REDIS_ENABLED=false for "
                "single-process deployments or fix the connection"
            ) from exc
        return cls(client, settings=settings)

    # -- 生命周期 ----------------------------------------------------------

    async def ping(self) -> None:
        """握手探活（/health 与装配复用）。"""
        await self._client.ping()

    async def aclose(self) -> None:
        """释放连接池。"""
        aclose = getattr(self._client, "aclose", None)
        if aclose is not None:
            await aclose()
        else:  # pragma: no cover - redis<5.0.1 只有同步 close
            self._client.close()

    # -- 键 ----------------------------------------------------------------

    def _k(self, key: str) -> str:
        return f"{self._prefix}{key}"

    # -- 字符串原语 ---------------------------------------------------------

    async def get(self, key: str) -> str | None:
        value = await self._client.get(self._k(key))
        return str(value) if value is not None else None

    async def set_ex(self, key: str, value: str, *, ttl_seconds: int) -> None:
        await self._client.set(self._k(key), value, ex=ttl_seconds)

    async def set_nx(self, key: str, value: str, *, ttl_seconds: int) -> bool:
        """``SET key value NX EX ttl``：键不存在才写入（nonce/抢锁语义）。"""
        ok = await self._client.set(self._k(key), value, nx=True, ex=ttl_seconds)
        return bool(ok)

    async def delete(self, *keys: str) -> int:
        if not keys:
            return 0
        return int(await self._client.delete(*(self._k(k) for k in keys)))

    async def delete_prefix(self, prefix: str) -> int:
        """删除匹配 ``<key_prefix><prefix>*`` 的全部键（SCAN，不用 KEYS）。"""
        pattern = f"{self._prefix}{prefix}*"
        matched: list[str] = []
        async for key in self._client.scan_iter(match=pattern, count=100):
            matched.append(str(key))
        if not matched:
            return 0
        return int(await self._client.delete(*matched))

    async def incr_by_float(self, key: str, amount: float) -> float:
        return float(await self._client.incrbyfloat(self._k(key), amount))

    async def expire(self, key: str, *, ttl_seconds: int) -> None:
        await self._client.expire(self._k(key), ttl_seconds)

    # -- Pub/Sub ------------------------------------------------------------

    async def publish(self, channel: str, message: str) -> int:
        """发布消息到频道（返回接收到的订阅者数）；频道同样拼接前缀。"""
        return int(await self._client.publish(self._k(channel), message))

    async def pubsub_messages(self, channel: str) -> AsyncIterator[str]:
        """订阅频道并逐条产出消息文本；消费者取消时自动退订关闭。

        用带超时的 ``get_message`` 轮询而非阻塞 ``listen()``：连接级
        socket_timeout 不会打断空闲订阅，外层取消（客户端断开）也能
        及时生效。
        """
        pubsub = self._client.pubsub()
        await pubsub.subscribe(self._k(channel))
        try:
            while True:
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=_PUBSUB_POLL_SECONDS
                )
                if message is not None and message.get("type") == "message":
                    yield str(message.get("data") or "")
        finally:
            close = getattr(pubsub, "aclose", None)
            if close is not None:
                await close()
            else:  # pragma: no cover - redis<5.0.1
                pubsub.close()

    # -- 分布式锁 ------------------------------------------------------------

    async def acquire_lock(
        self, key: str, *, ttl_seconds: int, acquire_timeout_seconds: float
    ) -> str | None:
        """``SET NX PX`` + 持有者 token；超时未抢到返回 None（由调用方
        决定跳过还是放弃）。TTL 兜底持有者崩溃后的死锁。"""
        token = secrets.token_hex(16)
        deadline = time.monotonic() + acquire_timeout_seconds
        while True:
            if await self.set_nx(key, token, ttl_seconds=ttl_seconds):
                return token
            if time.monotonic() >= deadline:
                return None
            await asyncio.sleep(_LOCK_RETRY_INTERVAL_SECONDS)

    async def release_lock(self, key: str, token: str) -> bool:
        """释放锁（compare-and-delete）；token 不匹配（锁已易主）不删除。"""
        if self._release_script is None:
            self._release_script = self._client.register_script(_RELEASE_LOCK_LUA)
        result = await self._release_script(keys=[self._k(key)], args=[token])
        return bool(result)


__all__ = ["RedisService", "RedisServiceError"]
