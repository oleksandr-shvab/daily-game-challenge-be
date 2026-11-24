import asyncio
import time
from types import SimpleNamespace
from typing import Any

import pytest
from httpx import AsyncClient
from redis.exceptions import ConnectionError as RedisConnectionError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import rate_limit
from app.core.config import settings
from app.core.rate_limit import Rule, client_key, email_key, enforce
from app.core.redis import create_redis_client, redis_client
from app.crud.password_reset import PASSWORD_RESET_USER_KEY_PREFIX
from app.crud.session import create_session
from app.crud.user import create_user

GENERIC_FORGOT_PASSWORD_DETAIL = "If an account with that email exists, a reset link has been sent."
PASSWORD_CHANGE_BODY = {
    "old_password": "OldPass123",
    "new_password": "NewPass123",
    "confirm_password": "NewPass123",
}
RESET_BODY = {"token": "x", "new_password": "NewPass123", "confirm_password": "NewPass123"}
GUESS_URL = "/v1/games/no-such-game/puzzles/2026-01-01/guesses"
SEARCH_URL = "/v1/games/no-such-game/titles?q=ab"


@pytest.fixture
def rate_limits_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "RATE_LIMITS_ENABLED", True)


# (rule, method, url, json body, the rule's setting)
ROUTE_RULES: list[tuple[Rule, str, str, Any, str]] = [
    (
        rate_limit.LOGIN_IP,
        "post",
        "/v1/auth/login",
        {"email": "a@example.com", "password": "Whatever123"},
        "RATE_LIMIT_LOGIN_PER_IP",
    ),
    (
        rate_limit.LOGIN_EMAIL,
        "post",
        "/v1/auth/login",
        {"email": "a@example.com", "password": "Whatever123"},
        "RATE_LIMIT_LOGIN_PER_EMAIL",
    ),
    (
        rate_limit.REGISTER_IP,
        "post",
        "/v1/auth/register",
        {"email": "new@example.com", "password": "Whatever123"},
        "RATE_LIMIT_REGISTER_PER_IP",
    ),
    (
        rate_limit.FORGOT_IP,
        "post",
        "/v1/auth/forgot-password",
        {"email": "a@example.com"},
        "RATE_LIMIT_FORGOT_PER_IP",
    ),
    (
        rate_limit.GUESS_IP,
        "post",
        GUESS_URL,
        {"title_id": None, "previous": []},
        "RATE_LIMIT_GUESS_PER_IP",
    ),
    (rate_limit.SEARCH_IP, "get", SEARCH_URL, None, "RATE_LIMIT_SEARCH_PER_IP"),
    (
        rate_limit.PASSWORD_CHANGE,
        "post",
        "/v1/auth/reset-password",
        RESET_BODY,
        "RATE_LIMIT_PASSWORD_CHANGE",
    ),
    (
        rate_limit.PASSWORD_CHANGE,
        "post",
        "/v1/auth/update-password",
        PASSWORD_CHANGE_BODY,
        "RATE_LIMIT_PASSWORD_CHANGE",
    ),
]


@pytest.mark.parametrize(
    ("rule", "method", "url", "body", "setting"),
    ROUTE_RULES,
    ids=[
        f"{rule.name}-{url.rsplit('/', 1)[-1].split('?')[0]}" for rule, _, url, _, _ in ROUTE_RULES
    ],
)
async def test__rate_limit__third_request_is_429_with_retry_after(
    client: AsyncClient,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
    rate_limits_on: None,
    rule: Rule,
    method: str,
    url: str,
    body: Any,
    setting: str,
) -> None:
    monkeypatch.setattr(settings, setting, 2)
    if url.endswith("update-password"):
        user = await create_user(db_session, email="limited@example.com", password="OldPass123")
        client.cookies.set(settings.SESSION_COOKIE_NAME, await create_session(user.id))

    statuses = [(await client.request(method, url, json=body)).status_code for _ in range(2)]
    response = await client.request(method, url, json=body)

    assert 429 not in statuses
    assert response.status_code == 429
    assert response.json()["code"] == "too_many_requests"
    assert 1 <= int(response.headers["Retry-After"]) <= rule.window_seconds


async def test__forgot_password_email_limit__same_202_and_no_new_token(
    client: AsyncClient,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
    rate_limits_on: None,
) -> None:
    monkeypatch.setattr(settings, "RATE_LIMIT_FORGOT_PER_EMAIL", 2)
    user = await create_user(db_session, email="forgot@example.com", password="OldPass123")
    key = f"{PASSWORD_RESET_USER_KEY_PREFIX}{user.id}"
    for _ in range(2):
        assert (
            await client.post("/v1/auth/forgot-password", json={"email": user.email})
        ).status_code == 202
    token_before = await redis_client.get(key)
    assert token_before is not None

    response = await client.post("/v1/auth/forgot-password", json={"email": user.email.upper()})

    assert response.status_code == 202
    assert response.json() == {"detail": GENERIC_FORGOT_PASSWORD_DETAIL}
    assert await redis_client.get(key) == token_before


async def test__login_email_limit__is_per_email_and_holds_no_address(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch, rate_limits_on: None
) -> None:
    monkeypatch.setattr(settings, "RATE_LIMIT_LOGIN_PER_EMAIL", 1)
    credentials = {"password": "Whatever123"}

    first = await client.post("/v1/auth/login", json={"email": "one@example.com", **credentials})
    again = await client.post("/v1/auth/login", json={"email": "ONE@example.com", **credentials})
    other = await client.post("/v1/auth/login", json={"email": "two@example.com", **credentials})

    assert (first.status_code, again.status_code, other.status_code) == (401, 429, 401)
    keys = [key async for key in redis_client.scan_iter("rl:*")]
    assert keys
    assert not any("@" in key or "example" in key for key in keys)


async def test__rate_limit__redis_error_lets_the_request_through(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch, rate_limits_on: None
) -> None:
    monkeypatch.setattr(settings, "RATE_LIMIT_LOGIN_PER_IP", 1)

    def broken_pipeline(*_args: object, **_kwargs: object) -> None:
        raise RedisConnectionError("down")

    monkeypatch.setattr(redis_client, "pipeline", broken_pipeline)
    body = {"email": "a@example.com", "password": "Whatever123"}

    statuses = [(await client.post("/v1/auth/login", json=body)).status_code for _ in range(3)]

    assert statuses == [401, 401, 401]


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("203.0.113.7", "203.0.113.7"),
        ("2001:db8:1:2:aaaa:bbbb:cccc:dddd", "2001:db8:1:2::/64"),
        ("2001:db8:1:2:1111:2222:3333:4444", "2001:db8:1:2::/64"),
        ("::ffff:203.0.113.7", "203.0.113.7"),
        ("testclient", "testclient"),
    ],
)
def test__client_key__ipv6_is_its_64_block(host: str, expected: str) -> None:
    request = SimpleNamespace(client=SimpleNamespace(host=host))

    assert client_key(request) == expected  # type: ignore[arg-type]


def test__email_key__ignores_case_and_spaces() -> None:
    assert email_key(" Some.One@Example.com ") == email_key("some.one@example.com")


async def test__enforce__window_starts_at_the_first_request(rate_limits_on: None) -> None:
    rule = Rule("window_test", 100, lambda: 5)
    await enforce(rule, "key")
    await redis_client.expire("rl:window_test:key", 10)

    await enforce(rule, "key")

    # A fixed window: a later request does not push the end out to a fresh 100 seconds.
    assert 0 < await redis_client.ttl("rl:window_test:key") <= 10


async def test__enforce__a_redis_that_does_not_answer_fails_open_within_the_timeout(
    monkeypatch: pytest.MonkeyPatch, rate_limits_on: None
) -> None:
    async def never_answer(_reader: asyncio.StreamReader, _writer: asyncio.StreamWriter) -> None:
        await asyncio.sleep(30)

    server = await asyncio.start_server(never_answer, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    monkeypatch.setattr(settings, "REDIS_SOCKET_TIMEOUT_SECONDS", 0.3)
    monkeypatch.setattr(settings, "REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS", 0.3)
    hanging = create_redis_client(f"redis://127.0.0.1:{port}/0")
    monkeypatch.setattr(rate_limit, "redis_client", hanging)
    try:
        started = time.monotonic()
        await enforce(rate_limit.LOGIN_IP, "key")
        elapsed = time.monotonic() - started
    finally:
        await hanging.aclose()
        server.close()

    # Without the timeouts redis-py waits its default 5 seconds.
    assert elapsed < 2
