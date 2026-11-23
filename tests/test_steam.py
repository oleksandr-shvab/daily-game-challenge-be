"""The Steam trailer client (owner decision 75, SPEC-04 AC-36, AC-38, AC-85) over a made-up Steam
(`tests/steam_fakes.py`): which part of a trailer is cut, every way to have no trailer, the allow
list, and what a rebuild changes. No network."""

import asyncio
import random
from typing import Any

import httpx
import pytest

from app.content import fetch, steam
from app.content.fetch import ForbiddenHostError, build_client
from app.content.steam import (
    FallbackReason,
    NoTrailer,
    Trailer,
    TrailerMarker,
    checked_url,
)
from tests.steam_fakes import Clock, FakeSteam, new_client

SECONDS = 12  # the real clip


async def find(
    fake: FakeSteam, app_ids: list[int] | None = None, *, seed: int = 1, avoid: Any = ()
) -> Trailer:
    async with new_client(fake) as client:
        return await client.find(
            [100] if app_ids is None else app_ids,
            avoid=avoid,
            stream=random.Random(seed),
            seconds=SECONDS,
        )


async def test__find__cuts_12_seconds_from_the_middle_of_the_1280x720_variant() -> None:
    fake = FakeSteam({100: [7]}, seconds=60)
    starts = set()
    for seed in range(40):
        fake.requests.clear()
        trailer = await find(fake, seed=seed)

        start = trailer.marker.start
        # Not the first or last 15% of 60 s: from 9 s, and 12 s must end by 51 s.
        assert 9 <= start <= 39
        asked = [i for i, url in enumerate(fake.piece_urls(100, 7)) if url in fake.requests]
        assert asked == list(range(start, start + SECONDS))
        assert trailer.data == fake.init + b"".join(fake.pieces[start : start + SECONDS])
        assert trailer.offset == 0
        assert trailer.marker == TrailerMarker(100, 7, start)
        assert not any("1920x1080" in url or "854x480" in url for url in fake.requests)
        starts.add(start)
    assert len(starts) > 5  # the seed moves it
    assert (await find(fake, seed=3)).marker == (await find(fake, seed=3)).marker


BAD_MASTER = httpx.Response(
    200,
    text="#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=1,RESOLUTION=1280x720\nhttps://evil.example/x.m3u8",
)
UNFINISHED = httpx.Response(200, text='#EXTM3U\n#EXT-X-MAP:URI="init.m4s"\n#EXTINF:3,\np.m4s')


def media_playlist(
    map_lines: str = '#EXT-X-MAP:URI="init.m4s"', *, piece: str | None = None
) -> httpx.Response:
    """A good 60 piece playlist of the fake, with `map_lines` as its map and optionally one piece
    URI replaced: each way a playlist can be bad is a row that changes one thing."""
    names = [piece if piece and n == 20 else f"p{n:05d}.m4s" for n in range(60)]
    body = [f"#EXTINF:1.000,\n{name}" for name in names]
    return httpx.Response(200, text="\n".join(["#EXTM3U", map_lines, *body, "#EXT-X-ENDLIST"]))


def master_with(uri: str) -> httpx.Response:
    return httpx.Response(
        200, text=f"#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=1,RESOLUTION=1280x720\n{uri}"
    )


class Trickle(httpx.AsyncByteStream):
    """A body that sends a byte and then stalls, so only a deadline over the whole request ends it."""

    async def __aiter__(self) -> Any:
        yield b"x"
        await asyncio.sleep(2)


FOREIGN_MOVIE = httpx.Response(
    200,
    json={
        "100": {
            "success": True,
            "data": {"movies": [{"id": 1, "hls_h264": "https://evil.example/m"}]},
        }
    },
)


@pytest.mark.parametrize(
    ("answers", "app_ids", "seconds", "expected"),
    [
        ({}, [], 60, FallbackReason.NO_STEAM_ID),
        ({}, [999], 60, FallbackReason.NO_TRAILER),  # Steam says no such app
        ({"appdetails": FOREIGN_MOVIE}, [100], 60, FallbackReason.NO_TRAILER),
        ({"appdetails": 404}, [100], 60, FallbackReason.NOT_FOUND),
        ({"appdetails": 429}, [100], 60, FallbackReason.STEAM_UNAVAILABLE),
        ({"appdetails": 503}, [100], 60, FallbackReason.STEAM_UNAVAILABLE),
        ({"appdetails": httpx.ConnectError("down")}, [100], 60, FallbackReason.STEAM_UNAVAILABLE),
        ({"appdetails": 403}, [100], 60, FallbackReason.UNREADABLE),
        ({"appdetails": httpx.Response(200, text="<html>")}, [100], 60, FallbackReason.UNREADABLE),
        ({"hls_264_master": 404}, [100], 60, FallbackReason.NOT_FOUND),
        ({"hls_264_master": BAD_MASTER}, [100], 60, FallbackReason.UNREADABLE),
        ({"1280x720/index.m3u8": UNFINISHED}, [100], 60, FallbackReason.UNREADABLE),
        ({"init.m4s": 404}, [100], 60, FallbackReason.NOT_FOUND),
        ({".m4s": 502}, [100], 60, FallbackReason.STEAM_UNAVAILABLE),
        ({".m4s": httpx.Response(302, headers={"location": "https://evil.example/x"})}, [100], 60, FallbackReason.UNREADABLE),
        ({}, [100], 8, FallbackReason.UNREADABLE),  # a trailer shorter than the clip
        ({"hls_264_master": master_with("//[x/y.m3u8")}, [100], 60, FallbackReason.UNREADABLE),
        ({"1280x720/index.m3u8": media_playlist(piece="//[x/p.m4s")}, [100], 60, FallbackReason.UNREADABLE),
        ({"hls_264_master": master_with("a" * 3000 + ".m3u8")}, [100], 60, FallbackReason.UNREADABLE),
        ({"appdetails": httpx.Response(200, text="[" * 200_000)}, [100], 60, FallbackReason.UNREADABLE),
        ({"1280x720/index.m3u8": media_playlist('#EXT-X-MAP:URI="init.m4s",BYTERANGE="720@0"')}, [100], 60, FallbackReason.UNREADABLE),
        ({"1280x720/index.m3u8": media_playlist('#EXT-X-MAP:URI="init.m4s"\n#EXT-X-MAP:URI="other.m4s"')}, [100], 60, FallbackReason.UNREADABLE),
        ({".m4s": lambda request: httpx.Response(200, stream=Trickle())}, [100], 60, FallbackReason.STEAM_UNAVAILABLE),
    ],
    ids=[
        "no-steam-id", "unknown-app", "movie-on-a-foreign-host", "appdetails-404", "appdetails-429",
        "appdetails-503", "no-answer", "appdetails-403", "not-json", "master-404", "variant-on-a-foreign-host",
        "playlist-not-finished", "piece-404", "piece-502", "piece-redirect", "too-short",
        "variant-uri-urljoin-error", "piece-uri-urljoin-error", "very-long-url", "deeply-nested-json",
        "init-byterange", "two-init-segments", "trickling-piece",
    ],
)  # fmt: skip
async def test__find__every_way_to_have_no_trailer_is_a_named_reason_and_never_leaves_the_host_list(
    answers: dict[str, Any],
    app_ids: list[int],
    seconds: int,
    expected: FallbackReason,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fetch, "REQUEST_DEADLINE_SECONDS", 0.2)  # only the trickling row waits
    fake = FakeSteam({100: [1]}, seconds=seconds)
    fake.answers.update(answers)
    clock = Clock()

    with pytest.raises(NoTrailer) as raised:
        async with new_client(fake, clock) as client:
            await client.find(app_ids, avoid=(), stream=random.Random(1), seconds=SECONDS)

    assert raised.value.reason == expected
    assert not any("evil.example" in url for url in fake.requests)  # never asked, nor followed
    assert not app_ids or "appdetails" in fake.requests[0]
    if expected == FallbackReason.STEAM_UNAVAILABLE:
        # Bounded: one try and at most 5 retries of the one call that fails, each wait at most 60 s.
        assert len(fake.requests) <= 3 + 6 and max(clock.sleeps) <= 60
    if not app_ids:
        assert fake.requests == []


async def test__find__honours_retry_after_and_stops_after_five_retries() -> None:
    fake = FakeSteam({100: [1]})
    fake.answers["appdetails"] = httpx.Response(429, headers={"retry-after": "7"})
    clock = Clock()

    with pytest.raises(NoTrailer) as raised:
        async with new_client(fake, clock) as client:
            await client.find([100], avoid=(), stream=random.Random(1), seconds=SECONDS)

    assert raised.value.reason == FallbackReason.STEAM_UNAVAILABLE
    assert len(fake.requests) == 6  # one try and five retries
    assert clock.sleeps.count(7.0) == 5


async def test__find__spaces_the_store_calls_and_after_an_outage_asks_no_more() -> None:
    fake = FakeSteam({})
    clock = Clock()
    async with new_client(fake, clock) as client:
        with pytest.raises(NoTrailer):  # three apps, all unknown to the store
            await client.find([1, 2, 3], avoid=(), stream=random.Random(1), seconds=SECONDS)
        assert clock.sleeps == [pytest.approx(1.6), pytest.approx(1.6)]  # calls 2 and 3 wait

        fake.answers["appdetails"] = 503
        fake.requests.clear()
        with pytest.raises(NoTrailer) as first:
            await client.find([1], avoid=(), stream=random.Random(1), seconds=SECONDS)
        asked = len(fake.requests)
        with pytest.raises(NoTrailer) as second:
            await client.find([1], avoid=(), stream=random.Random(1), seconds=SECONDS)
        with pytest.raises(NoTrailer) as no_id:  # a game with no Steam id is still just that
            await client.find([], avoid=(), stream=random.Random(1), seconds=SECONDS)
    assert no_id.value.reason == FallbackReason.NO_STEAM_ID

    assert first.value.reason == second.value.reason == FallbackReason.STEAM_UNAVAILABLE
    assert asked == 6 and len(fake.requests) == asked  # the second slot made no request


async def test__find__a_rebuild_takes_another_movie_or_a_segment_no_rejected_draft_used() -> None:
    # Two movies: the rejected draft's movie comes last.
    fake = FakeSteam({100: [1, 2]})
    first = await find(fake, seed=5)
    again = await find(fake, seed=5, avoid=["abc123", str(first.marker)])
    assert again.marker.movie_id != first.marker.movie_id

    # Two apps, one movie each: the rejected draft's movie comes after the other app's unused one,
    # whichever app is read first.
    fake = FakeSteam({100: [1], 200: [2]})
    for seed in range(6):
        first = await find(fake, [100, 200], seed=seed)
        again = await find(fake, [100, 200], seed=seed, avoid=[str(first.marker)])
        assert (again.marker.app_id, again.marker.movie_id) != (
            first.marker.app_id,
            first.marker.movie_id,
        )

    # One movie: each rebuild gets a clip that overlaps no rejected one, until none is left.
    fake = FakeSteam({100: [1]})
    used: list[int] = []
    for round_number in range(5):
        avoid = [str(TrailerMarker(100, 1, start)) for start in used]
        try:
            trailer = await find(fake, seed=5, avoid=avoid)
        except NoTrailer as error:
            assert error.reason == FallbackReason.NO_NEW_SEGMENT
            break
        assert all(abs(trailer.marker.start - start) >= SECONDS for start in used)
        used.append(trailer.marker.start)
    else:
        pytest.fail("a 60 s trailer has no room for five clips of 12 s")
    assert 1 <= len(used) <= 3 and round_number == len(used)


async def test__find__one_failing_trailer_object_fails_that_movie_and_only_a_run_of_them_turns_steam_off() -> (
    None
):
    # Movie 1's init segment is gone; movie 2 is fine. Whichever comes first, the clip is movie 2's
    # and the client stays on.
    tried_first = 0
    for seed in range(8):
        fake = FakeSteam({100: [1, 2]})
        fake.answers["/100/1/1280x720/init.m4s"] = 502
        async with new_client(fake) as client:
            trailer = await client.find(
                [100], avoid=(), stream=random.Random(seed), seconds=SECONDS
            )
            assert trailer.marker.movie_id == 2 and not client.is_unavailable
        tried_first += any("/100/1/1280x720/init.m4s" in url for url in fake.requests)
    assert tried_first > 0  # the failing movie really was tried before the good one

    # Three movies that all fail on the trailer host in a row do turn it off for the run.
    fake = FakeSteam({100: [1, 2, 3]})
    fake.answers["init.m4s"] = 502
    async with new_client(fake) as client:
        with pytest.raises(NoTrailer) as raised:
            await client.find([100], avoid=(), stream=random.Random(1), seconds=SECONDS)
        assert raised.value.reason == FallbackReason.STEAM_UNAVAILABLE and client.is_unavailable
        asked = len(fake.requests)
        with pytest.raises(NoTrailer):
            await client.find([100], avoid=(), stream=random.Random(1), seconds=SECONDS)
        assert len(fake.requests) == asked


async def test__get__a_url_httpx_cannot_build_is_unreadable_not_a_crash() -> None:
    async with new_client(FakeSteam({})) as client:
        with pytest.raises(NoTrailer) as raised:
            await client._get("https://video.akamai.steamstatic.com/" + "a" * 70_000, 10)
    assert raised.value.reason == FallbackReason.UNREADABLE


@pytest.mark.parametrize(
    "url",
    [
        "http://video.akamai.steamstatic.com/a.m4s",
        "https://cdn.akamai.steamstatic.com/a.m4s",
        "https://video.akamai.steamstatic.com.evil.example/a.m4s",
        "https://video.akamai.steamstatic.com@evil.example/a.m4s",
        "https://user:pw@video.akamai.steamstatic.com/a.m4s",
        "https://video.akamai.steamstatic.com:8443/a.m4s",
        "https://store.steampowered.com/a.m4s",  # the store is allowed to answer, not to stream
        "https://video.akamai.steamstatic.com/a b.m4s",
        "//video.akamai.steamstatic.com/a.m4s",
        "https://video.akamai.steamstatic.com:99999999/a.m4s",
    ],
)
def test__checked_url__takes_only_a_plain_https_url_on_the_trailer_host(url: str) -> None:
    assert checked_url(url) is None
    assert checked_url("https://video.akamai.steamstatic.com/store_trailers/1/2/x.m3u8?t=1")


async def test__allow_list__is_these_five_hosts_and_the_client_refuses_the_rest() -> None:
    assert fetch.ALLOWED_HOSTS == {
        "api.igdb.com",
        "id.twitch.tv",
        "images.igdb.com",
        "store.steampowered.com",
        "video.akamai.steamstatic.com",
    }
    seen: list[str] = []

    def answer(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200)

    async with build_client(httpx.MockTransport(answer)) as client:
        await client.get("https://store.steampowered.com/api/appdetails")  # allowed
        await client.get("https://video.akamai.steamstatic.com/x")  # allowed
        for refused in (
            "http://store.steampowered.com/x",
            "https://cdn.akamai.steamstatic.com/x",
            "https://steamstatic.com/x",
            "https://store.steampowered.com.evil.example/x",
        ):
            with pytest.raises(ForbiddenHostError):
                await client.get(refused)
    assert len(seen) == 2


def test__marker__round_trips_and_fits_the_32_character_column() -> None:
    marker = TrailerMarker(4_294_967_295, 9_999_999_999, 3_600)
    assert len(str(marker)) <= 32
    assert TrailerMarker.parse(str(marker)) == marker
    assert TrailerMarker.parse("steam:1:2") is None and TrailerMarker.parse("abc123") is None
    assert steam.is_marker(str(marker)) and not steam.is_marker("abc123")
