import asyncio
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from redis.exceptions import ConnectionError as RedisConnectionError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import rate_limit, security
from app.core.config import settings
from app.core.rate_limit import Rule, client_key, email_key, enforce
from app.core.redis import create_redis_client, redis_client
from app.crud.password_reset import PASSWORD_RESET_USER_KEY_PREFIX
from app.crud.session import create_session
from app.crud.user import create_user, get_user_by_email
from app.main import app
from tests.conftest import TEST_ORIGIN

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


LOGIN_BODY = {"email": "a@example.com", "password": "Whatever123"}
# ASGITransport reports the client as 127.0.0.1 unless told otherwise.
TEST_CLIENT_LOGIN_KEY = "rl:login_ip:127.0.0.1"


@asynccontextmanager
async def client_at(host: str, session: str | None = None) -> AsyncGenerator[AsyncClient, None]:
    """A client whose requests come from `host`, with a session cookie if given. Relies on the
    `client` fixture's dependency overrides, so use it in a test that also takes `client`."""
    transport = ASGITransport(app=app, client=(host, 50000))
    async with AsyncClient(
        transport=transport, base_url="http://test", headers={"Origin": TEST_ORIGIN}
    ) as ac:
        if session is not None:
            ac.cookies.set(settings.SESSION_COOKIE_NAME, session)
        yield ac


async def rate_limit_keys() -> list[str]:
    return sorted([key async for key in redis_client.scan_iter("rl:*")])


async def test__rate_limit__429_is_a_problem_json_with_the_window_left(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch, rate_limits_on: None
) -> None:
    monkeypatch.setattr(settings, "RATE_LIMIT_LOGIN_PER_IP", 1)
    await client.post("/v1/auth/login", json=LOGIN_BODY)

    response = await client.post("/v1/auth/login", json=LOGIN_BODY)

    assert response.status_code == 429
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-request-id"]
    body = response.json()
    assert body["status"] == 429
    assert body["code"] == "too_many_requests"
    assert body["request_id"] == response.headers["x-request-id"]
    # The window just started: a whole number of seconds, close to the full minute.
    assert response.headers["retry-after"].isdigit()
    assert 55 <= int(response.headers["retry-after"]) <= rate_limit.LOGIN_IP.window_seconds


async def test__rate_limit__retry_after_is_at_least_one_second_at_the_end_of_a_window(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch, rate_limits_on: None
) -> None:
    monkeypatch.setattr(settings, "RATE_LIMIT_LOGIN_PER_IP", 1)
    await client.post("/v1/auth/login", json=LOGIN_BODY)
    # Redis rounds a TTL under half a second down to 0.
    await redis_client.pexpire(TEST_CLIENT_LOGIN_KEY, 400)

    response = await client.post("/v1/auth/login", json=LOGIN_BODY)

    assert response.status_code == 429
    assert response.headers["retry-after"] == "1"


async def test__rate_limit__a_new_window_starts_after_the_counter_expires(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch, rate_limits_on: None
) -> None:
    monkeypatch.setattr(settings, "RATE_LIMIT_LOGIN_PER_IP", 2)
    key = TEST_CLIENT_LOGIN_KEY
    for _ in range(2):
        await client.post("/v1/auth/login", json=LOGIN_BODY)
    assert (await client.post("/v1/auth/login", json=LOGIN_BODY)).status_code == 429

    # Let the counter expire for real, as it does at the end of a window.
    await redis_client.pexpire(key, 100)
    await asyncio.sleep(0.25)
    assert await redis_client.exists(key) == 0
    after = [(await client.post("/v1/auth/login", json=LOGIN_BODY)).status_code for _ in range(3)]

    assert after == [401, 401, 429]
    # The new window is a full one, not what was left of the old one.
    assert 55 <= await redis_client.ttl(key) <= rate_limit.LOGIN_IP.window_seconds


@pytest.fixture
def hash_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every password hash or verify the app starts (each one is a slow argon2 run)."""
    calls: list[str] = []
    real_hash, real_verify = security._hash_password, security._verify_password

    def hash_spy(password: str) -> str:
        calls.append("hash")
        return real_hash(password)

    def verify_spy(password: str, hashed_password: str) -> bool:
        calls.append("verify")
        return real_verify(password, hashed_password)

    monkeypatch.setattr(security, "_hash_password", hash_spy)
    monkeypatch.setattr(security, "_verify_password", verify_spy)
    return calls


async def test__login_over_the_limit__never_reaches_password_hashing(
    client: AsyncClient,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
    rate_limits_on: None,
    hash_calls: list[str],
) -> None:
    monkeypatch.setattr(settings, "RATE_LIMIT_LOGIN_PER_IP", 1)
    await create_user(db_session, email="a@example.com", password="Whatever123")
    hash_calls.clear()
    assert (await client.post("/v1/auth/login", json=LOGIN_BODY)).status_code == 200
    assert hash_calls == ["verify"]

    blocked = [(await client.post("/v1/auth/login", json=LOGIN_BODY)).status_code for _ in range(3)]

    assert blocked == [429, 429, 429]
    assert hash_calls == ["verify"]


async def test__login_over_the_email_limit__never_reaches_password_hashing(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    rate_limits_on: None,
    hash_calls: list[str],
) -> None:
    monkeypatch.setattr(settings, "RATE_LIMIT_LOGIN_PER_EMAIL", 1)
    # An unknown email still costs one dummy verification, which is what makes it slow.
    assert (await client.post("/v1/auth/login", json=LOGIN_BODY)).status_code == 401
    assert hash_calls == ["verify"]

    assert (await client.post("/v1/auth/login", json=LOGIN_BODY)).status_code == 429

    assert hash_calls == ["verify"]


async def test__register_over_the_limit__hashes_nothing_and_creates_no_account(
    client: AsyncClient,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
    rate_limits_on: None,
    hash_calls: list[str],
) -> None:
    monkeypatch.setattr(settings, "RATE_LIMIT_REGISTER_PER_IP", 1)
    first = await client.post(
        "/v1/auth/register", json={"email": "first@example.com", "password": "Whatever123"}
    )
    assert first.status_code == 201
    assert hash_calls == ["hash"]

    second = await client.post(
        "/v1/auth/register", json={"email": "second@example.com", "password": "Whatever123"}
    )

    assert second.status_code == 429
    assert "set-cookie" not in second.headers
    assert hash_calls == ["hash"]
    assert await get_user_by_email(db_session, "second@example.com") is None


@pytest.mark.parametrize(
    ("method", "url", "body"),
    [
        ("post", "/v1/auth/login", {"email": "not-an-email", "password": "Whatever123"}),
        ("post", "/v1/auth/login", {"email": "a@example.com"}),
        ("post", "/v1/auth/register", {"email": "a@example.com", "password": "short"}),
        ("post", "/v1/auth/forgot-password", {"email": "not-an-email"}),
        ("post", "/v1/auth/reset-password", {"token": "x"}),
        ("post", "/v1/games/no-such-game/puzzles/2026-01-01/guesses", {"title_id": "text"}),
        ("get", "/v1/games/no-such-game/titles?q=" + "a" * 101, None),
    ],
    ids=["login-email", "login-no-password", "register", "forgot", "reset", "guess", "search"],
)
async def test__rate_limit__a_422_is_not_counted(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    rate_limits_on: None,
    method: str,
    url: str,
    body: Any,
) -> None:
    for setting in (
        "RATE_LIMIT_LOGIN_PER_IP",
        "RATE_LIMIT_LOGIN_PER_EMAIL",
        "RATE_LIMIT_REGISTER_PER_IP",
        "RATE_LIMIT_FORGOT_PER_IP",
        "RATE_LIMIT_FORGOT_PER_EMAIL",
        "RATE_LIMIT_GUESS_PER_IP",
        "RATE_LIMIT_SEARCH_PER_IP",
        "RATE_LIMIT_PASSWORD_CHANGE",
    ):
        monkeypatch.setattr(settings, setting, 1)

    statuses = [(await client.request(method, url, json=body)).status_code for _ in range(3)]

    assert statuses == [422, 422, 422]
    assert await rate_limit_keys() == []


async def test__update_password__unauthenticated_is_401_and_never_counts(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch, rate_limits_on: None
) -> None:
    monkeypatch.setattr(settings, "RATE_LIMIT_PASSWORD_CHANGE", 1)
    url = "/v1/auth/update-password"

    no_cookie = [(await client.post(url, json=PASSWORD_CHANGE_BODY)).status_code for _ in range(3)]
    client.cookies.set(settings.SESSION_COOKIE_NAME, "no-such-session")
    bad_cookie = [(await client.post(url, json=PASSWORD_CHANGE_BODY)).status_code for _ in range(3)]

    assert no_cookie == [401, 401, 401]
    assert bad_cookie == [401, 401, 401]
    assert await rate_limit_keys() == []


async def test__update_password_limit__counts_per_user_not_per_address(
    client: AsyncClient,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
    rate_limits_on: None,
) -> None:
    monkeypatch.setattr(settings, "RATE_LIMIT_PASSWORD_CHANGE", 2)
    url = "/v1/auth/update-password"
    wrong_old = {**PASSWORD_CHANGE_BODY, "old_password": "NotTheOldOne1"}
    alice = await create_user(db_session, email="alice@example.com", password="OldPass123")
    bob = await create_user(db_session, email="bob@example.com", password="OldPass123")
    alice_session = await create_session(alice.id)
    bob_session = await create_session(bob.id)

    async with (
        client_at("198.51.100.1", alice_session) as alice_home,
        client_at("198.51.100.2", alice_session) as alice_elsewhere,
        client_at("198.51.100.1", bob_session) as bob_home,
    ):
        # Two wrong passwords from one address use up Alice's budget (a 400 still counts).
        first = await alice_home.post(url, json=wrong_old)
        second = await alice_home.post(url, json=wrong_old)
        # The same user from another address is still blocked.
        from_elsewhere = await alice_elsewhere.post(url, json=wrong_old)
        # Another user from the first address has a budget of their own.
        other_user = await bob_home.post(url, json=wrong_old)

    assert (first.status_code, second.status_code) == (400, 400)
    assert from_elsewhere.status_code == 429
    assert other_user.status_code == 400
    assert f"rl:password_change:{alice.id}" in await rate_limit_keys()
    assert await redis_client.get(f"rl:password_change:{bob.id}") == "1"


async def test__rate_limit__disabled_never_touches_redis(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `rate_limits_off` is autouse, so the limit setting is the only thing that could stop these.
    monkeypatch.setattr(settings, "RATE_LIMIT_LOGIN_PER_IP", 1)
    monkeypatch.setattr(settings, "RATE_LIMIT_LOGIN_PER_EMAIL", 1)
    pipelines = 0

    def counting_pipeline(*_args: object, **_kwargs: object) -> None:
        nonlocal pipelines
        pipelines += 1
        raise AssertionError("the limiter used Redis while disabled")

    monkeypatch.setattr(redis_client, "pipeline", counting_pipeline)

    statuses = [
        (await client.post("/v1/auth/login", json=LOGIN_BODY)).status_code for _ in range(3)
    ]
    await enforce(rate_limit.LOGIN_IP, "key")

    assert statuses == [401, 401, 401]
    assert pipelines == 0
    assert await rate_limit_keys() == []


async def test__rate_limit__ipv6_clients_in_one_64_share_a_bucket(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch, rate_limits_on: None
) -> None:
    monkeypatch.setattr(settings, "RATE_LIMIT_LOGIN_PER_IP", 1)

    async def login_from(host: str, email: str) -> int:
        async with client_at(host) as ac:
            body = {"email": email, "password": "Whatever123"}
            return (await ac.post("/v1/auth/login", json=body)).status_code

    first = await login_from("2001:db8:1:2:aaaa:bbbb:cccc:dddd", "one@example.com")
    same_block = await login_from("2001:db8:1:2:1111:2222:3333:4444", "two@example.com")
    next_block = await login_from("2001:db8:1:3::1", "three@example.com")
    ipv4 = await login_from("203.0.113.7", "four@example.com")
    mapped = await login_from("::ffff:203.0.113.7", "five@example.com")

    assert (first, same_block, next_block, ipv4, mapped) == (401, 429, 401, 401, 429)
    ip_keys = [key for key in await rate_limit_keys() if key.startswith("rl:login_ip:")]
    assert ip_keys == [
        "rl:login_ip:2001:db8:1:2::/64",
        "rl:login_ip:2001:db8:1:3::/64",
        "rl:login_ip:203.0.113.7",
    ]
