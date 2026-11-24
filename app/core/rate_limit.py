import hashlib
import ipaddress
import logging
from collections.abc import Callable
from dataclasses import dataclass

from fastapi import Request
from redis.exceptions import RedisError

from app.core.config import settings
from app.core.exceptions import TooManyRequestsError
from app.core.redis import redis_client

logger = logging.getLogger(__name__)

MINUTE = 60
HOUR = 60 * MINUTE


@dataclass(frozen=True)
class Rule:
    """At most `limit()` requests per `window_seconds` for one key. The count is read from the
    settings at each call, so a test can change it."""

    name: str
    window_seconds: int
    limit: Callable[[], int]


LOGIN_IP = Rule("login_ip", MINUTE, lambda: settings.RATE_LIMIT_LOGIN_PER_IP)
LOGIN_EMAIL = Rule("login_email", 15 * MINUTE, lambda: settings.RATE_LIMIT_LOGIN_PER_EMAIL)
REGISTER_IP = Rule("register_ip", HOUR, lambda: settings.RATE_LIMIT_REGISTER_PER_IP)
FORGOT_IP = Rule("forgot_ip", HOUR, lambda: settings.RATE_LIMIT_FORGOT_PER_IP)
FORGOT_EMAIL = Rule("forgot_email", HOUR, lambda: settings.RATE_LIMIT_FORGOT_PER_EMAIL)
GUESS_IP = Rule("guess_ip", MINUTE, lambda: settings.RATE_LIMIT_GUESS_PER_IP)
SEARCH_IP = Rule("search_ip", MINUTE, lambda: settings.RATE_LIMIT_SEARCH_PER_IP)
# Keyed by user id on update-password and by IP on reset-password.
PASSWORD_CHANGE = Rule("password_change", HOUR, lambda: settings.RATE_LIMIT_PASSWORD_CHANGE)


def client_key(request: Request) -> str:
    """The client address. Behind the proxy uvicorn's `--proxy-headers` has already put the real
    address in `request.client`. An IPv6 address counts as its /64, because one customer usually
    owns the whole block."""
    host = request.client.host if request.client else "unknown"
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return host
    if isinstance(address, ipaddress.IPv6Address):
        if address.ipv4_mapped is not None:
            return str(address.ipv4_mapped)
        return str(ipaddress.ip_network(f"{address}/64", strict=False))
    return str(address)


def email_key(email: str) -> str:
    """A hash, so Redis holds no addresses."""
    return hashlib.sha256(email.strip().lower().encode()).hexdigest()


async def enforce(rule: Rule, key: str) -> None:
    """Counts one request and raises `TooManyRequestsError` when the window's count is used up.

    A Redis error lets the request through: sessions need Redis anyway, but guest search and
    guesses do not, and the limiter must not be what takes the site down.
    """
    if not settings.RATE_LIMITS_ENABLED:
        return
    redis_key = f"rl:{rule.name}:{key}"
    try:
        # One transaction: `EXPIRE NX` starts the window on the first request only, and the TTL
        # read is the time left in it.
        async with redis_client.pipeline(transaction=True) as pipe:
            pipe.incr(redis_key)
            pipe.expire(redis_key, rule.window_seconds, nx=True)
            pipe.ttl(redis_key)
            count, _, ttl = await pipe.execute()
    except RedisError:
        logger.warning("Rate limiter skipped, Redis failed (rule %s)", rule.name, exc_info=True)
        return
    if count > rule.limit():
        raise TooManyRequestsError(retry_after=max(1, min(ttl, rule.window_seconds)))
