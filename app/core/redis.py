import redis.asyncio as redis

from app.core.config import settings


def create_redis_client(url: str) -> redis.Redis:
    return redis.from_url(
        url,
        decode_responses=True,
        socket_timeout=settings.REDIS_SOCKET_TIMEOUT_SECONDS,
        socket_connect_timeout=settings.REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS,
    )


redis_client = create_redis_client(settings.REDIS_URL)
