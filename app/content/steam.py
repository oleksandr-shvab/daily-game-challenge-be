"""Steam trailers for the step-6 clip (owner decision 75, SPEC-04 AC-36, AC-38, AC-85).

    trailer = await steam.find([1145360], avoid=[...], stream=seed.stream(Purpose.CLIP))

The game's Steam app id comes from IGDB (`IgdbGame.steam_app_ids`). The store's `appdetails` call
lists the game's movies; each has an `hls_h264` master playlist (there is no plain mp4 or webm any
more). A variant playlist of it holds an init segment and pieces of a few seconds, with no audio
in the video track. The pieces of the chosen 12 seconds are fetched into one buffer, init first,
which PyAV opens through `io.BytesIO` (opening the HLS URL itself fails: spike of 2026-11-23). The
re-encode is `media.trailer_clip`.

Everything from Steam is untrusted data. Every URL is checked (https, one of the trailer hosts, no
credentials) before it is fetched, on top of the allow-listed client of `fetch.py`, which does not
follow redirects. Sizes are capped, retries are bounded (`Retry-After` honoured), and `appdetails`
calls are spaced (Steam allows about 200 per 5 minutes, so 1.6 s apart). When anything is missing
or wrong the caller gets `NoTrailer` with a stable reason and makes the slideshow instead.

A trailer's choice is deterministic: the movie and the start come from the seed's `clip` stream. The
first and last `TRAILER_MARGIN` of the trailer are skipped (logos and title cards usually sit there).
A rebuild passes the markers of the rejected drafts (`avoid`, taken from the exclusion row's image
ids) and gets another movie, or another segment of the same movie.
"""

import asyncio
import json
import math
import random
import re
import time
from collections.abc import Awaitable, Callable, Collection, Sequence
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from typing import Annotated, Protocol, Self
from urllib.parse import urljoin, urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.content.fetch import FetchError, build_client, send_capped
from app.content.igdb import MAX_RETRIES, RequestSpacer, retry_delay
from app.content.seed import draw_distinct, draw_index

STORE_URL = "https://store.steampowered.com/api/appdetails"
TRAILER_HOSTS = frozenset({"video.akamai.steamstatic.com"})
# Steam allows about 200 `appdetails` calls per 5 minutes: one every 1.6 s stays under that.
APPDETAILS_INTERVAL_SECONDS = 1.6

# The clip is cut this far from both ends of the trailer, where logos and title cards usually sit.
TRAILER_MARGIN = Fraction(15, 100)
# The largest picture the clip may have (SPEC-02 AC-74b); the largest variant at or below it wins.
TRAILER_MAX_SIZE = (1280, 720)

MAX_APPDETAILS_BYTES = 5 * 1024 * 1024
MAX_PLAYLIST_BYTES = 1024 * 1024
MAX_PIECE_BYTES = 8 * 1024 * 1024
MAX_BUFFER_BYTES = 24 * 1024 * 1024
MAX_PIECES_PER_CLIP = 40
MAX_PLAYLIST_PIECES = 2000
MAX_TRAILER_SECONDS = 3600
# How many Steam apps of one game and how many movies of one app a slot tries before it gives up.
MAX_APPS_TRIED = 3
MAX_MOVIES_TRIED = 12  # across all the game's apps
# A playlist or piece that fails this many movies in a row, with no success between, turns Steam off
# for the run like an `appdetails` outage does; one failing object is only that movie failing.
CDN_FAILURES_BEFORE_OFF = 3
# Real trailer URLs are well under this; a longer one is data to refuse, not to fetch.
MAX_URL_LENGTH = 2048

# What the builder stores in step 6's `source_image_ids` (a column of 32-character strings): a
# prefix no IGDB image id can have, then the Steam app id, the movie id and the start in seconds.
MARKER_PREFIX = "steam:"
_MARKER = re.compile(r"steam:([0-9]{1,10}):([0-9]{1,10}):([0-9]{1,4})", re.ASCII)
_RESOLUTION = re.compile(r"RESOLUTION=([0-9]{1,5})x([0-9]{1,5})", re.ASCII)
_MAP_URI = re.compile(r'URI="([^"]+)"')


class FallbackReason(StrEnum):
    """Why a slot's clip is a slideshow. Stable: reports and tests use the names."""

    STEAM_OFF = "steam_off"  # the run has no Steam client
    NO_STEAM_ID = "no_steam_id"  # IGDB has no Steam id for the game
    NO_TRAILER = "no_trailer"  # Steam lists no movie, or says it has no such app
    NOT_FOUND = "not_found"  # a 404 or 410 from Steam or the trailer host
    STEAM_UNAVAILABLE = "steam_unavailable"  # 429, 5xx or no answer, after the retries
    UNREADABLE = "unreadable"  # a bad answer, bad playlist, bad URL, too large or not decodable
    OVER_LIMIT = "over_limit"  # the clip cannot be made to fit the limits
    NO_NEW_SEGMENT = "no_new_segment"  # a rebuild: every segment was in a rejected draft


class NoTrailer(Exception):
    """No usable trailer: the caller makes the slideshow. `reason` is stable, the text is ours."""

    def __init__(self, reason: FallbackReason) -> None:
        super().__init__(reason.value)
        self.reason = reason


@dataclass(frozen=True)
class TrailerMarker:
    """Which trailer segment a draft was cut from. Its text is the id kept in step 6's
    `source_image_ids`; a rebuild reads it back from the rejected draft's row."""

    app_id: int
    movie_id: int
    start: int  # whole seconds from the start of the trailer

    def __str__(self) -> str:
        return f"{MARKER_PREFIX}{self.app_id}:{self.movie_id}:{self.start}"

    @classmethod
    def parse(cls, text: str) -> Self | None:
        match = _MARKER.fullmatch(text)
        return None if match is None else cls(*map(int, match.groups()))


def is_marker(image_id: str) -> bool:
    """True for a trailer marker, false for an IGDB image id."""
    return image_id.startswith(MARKER_PREFIX)


def checked_url(url: str) -> str | None:
    """`url` when it is a plain https URL on a trailer host with no credentials, else `None`."""
    if len(url) > MAX_URL_LENGTH:
        return None
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return None
    ok = (
        parts.scheme == "https"
        and parts.hostname in TRAILER_HOSTS
        and port in (None, 443)
        and parts.username is None
        and parts.password is None
        and not any(ord(char) < 33 or ord(char) == 127 for char in url)
    )
    return url if ok else None


def joined_url(base_url: str, uri: str) -> str | None:
    """`uri` resolved against `base_url` and checked (`checked_url`), or `None`. `urljoin` raises a
    plain `ValueError` on some malformed input (`//[x/y`), which is a refused URL, not a crash."""
    try:
        return checked_url(urljoin(base_url, uri))
    except ValueError:
        return None


class _Record(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore", frozen=True)


class SteamMovie(_Record):
    """One entry of `data.movies`. One whose id or playlist URL is wrong is dropped."""

    id: Annotated[int, Field(gt=0, lt=10**10)]
    hls_h264: str

    @field_validator("hls_h264")
    @classmethod
    def _trailer_url(cls, value: str) -> str:
        if checked_url(value) is None:
            raise ValueError("not a trailer URL")
        return value


def parse_movies(payload: object, app_id: int) -> list[SteamMovie]:
    """The valid movies of an `appdetails` answer, lowest id first. `NoTrailer(NO_TRAILER)` when
    Steam does not know the app or the app has none."""
    entry = payload.get(str(app_id)) if isinstance(payload, dict) else None
    if not isinstance(entry, dict) or entry.get("success") is not True:
        raise NoTrailer(FallbackReason.NO_TRAILER)
    data = entry.get("data")
    raw = data.get("movies") if isinstance(data, dict) else None
    movies: dict[int, SteamMovie] = {}
    for item in raw if isinstance(raw, list) else []:
        try:
            movie = SteamMovie.model_validate(item)
        except ValidationError:
            continue
        movies.setdefault(movie.id, movie)
    if not movies:
        raise NoTrailer(FallbackReason.NO_TRAILER)
    return [movies[movie_id] for movie_id in sorted(movies)]


@dataclass(frozen=True)
class Variant:
    width: int
    height: int
    url: str


@dataclass(frozen=True)
class Piece:
    url: str
    start: float
    duration: float


@dataclass(frozen=True)
class Playlist:
    init_url: str
    pieces: tuple[Piece, ...]

    @property
    def total(self) -> float:
        last = self.pieces[-1]
        return last.start + last.duration


class PlaylistError(ValueError):
    """A playlist this module will not use. The reason is for tests, never for a log with a URL."""


def parse_master(text: str, base_url: str) -> list[Variant]:
    """The video variants of a master playlist: each `#EXT-X-STREAM-INF` with a `RESOLUTION` and
    the URI on the next line. A variant whose URL is not a trailer URL is left out."""
    variants: list[Variant] = []
    lines = text.splitlines()
    for number, line in enumerate(lines):
        if not line.startswith("#EXT-X-STREAM-INF"):
            continue
        resolution = _RESOLUTION.search(line)
        uri = next((later for later in lines[number + 1 :] if later.strip()), "")
        if resolution is None or uri.startswith("#"):
            continue
        url = joined_url(base_url, uri.strip())
        width, height = int(resolution.group(1)), int(resolution.group(2))
        if url is not None and width > 0 and height > 0:
            variants.append(Variant(width, height, url))
    return variants


def choose_variant(variants: Sequence[Variant]) -> Variant | None:
    """The largest variant that fits `TRAILER_MAX_SIZE` (1280x720 itself when there is one); when
    every one is larger, the smallest of them, which the clip scales down."""
    fitting = [
        v for v in variants if v.width <= TRAILER_MAX_SIZE[0] and v.height <= TRAILER_MAX_SIZE[1]
    ]
    if fitting:
        return max(fitting, key=lambda v: (v.width * v.height, v.url))
    return min(variants, key=lambda v: (v.width * v.height, v.url), default=None)


def parse_playlist(text: str, base_url: str) -> Playlist:
    """A variant (media) playlist of fMP4 pieces: one `#EXT-X-MAP` init segment and one URI per
    `#EXTINF`, ending with `#EXT-X-ENDLIST`. Byte ranges, encryption and a live playlist are not
    supported and raise `PlaylistError`, and so does a second, different init segment."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines or lines[0] != "#EXTM3U" or "#EXT-X-ENDLIST" not in lines:
        raise PlaylistError("not a finished media playlist")
    init_url: str | None = None
    pieces: list[Piece] = []
    duration: float | None = None
    clock = 0.0
    for line in lines:
        if line.startswith("#EXT-X-BYTERANGE") or (
            line.startswith("#EXT-X-KEY") and "METHOD=NONE" not in line
        ):
            raise PlaylistError("byte ranges and encryption are not supported")
        if line.startswith("#EXT-X-MAP"):
            match = _MAP_URI.search(line)
            map_url = None if match is None else joined_url(base_url, match.group(1))
            if map_url is None:
                raise PlaylistError("the init segment is not a trailer URL")
            if "BYTERANGE" in line:
                raise PlaylistError("an init segment with a byte range is not supported")
            if init_url is not None and init_url != map_url:
                raise PlaylistError("the playlist has two different init segments")
            init_url = map_url
        elif line.startswith("#EXTINF:"):
            try:
                duration = float(line[len("#EXTINF:") :].split(",")[0])
            except ValueError as error:
                raise PlaylistError("a piece has no duration") from error
            if not 0 < duration <= 60:
                raise PlaylistError("a piece has an odd duration")
        elif not line.startswith("#"):
            url = joined_url(base_url, line)
            if url is None or duration is None:
                raise PlaylistError("a piece is not a trailer URL or has no duration")
            pieces.append(Piece(url, clock, duration))
            clock += duration
            duration = None
            if len(pieces) > MAX_PLAYLIST_PIECES or clock > MAX_TRAILER_SECONDS:
                raise PlaylistError("the playlist is too long")
    if init_url is None or not pieces:
        raise PlaylistError("the playlist has no init segment or no pieces")
    return Playlist(init_url, tuple(pieces))


def candidate_starts(total: float, seconds: int) -> range:
    """The whole-second starts of a clip of `seconds`: out of the first and last `TRAILER_MARGIN` of
    a trailer of `total` seconds. A trailer too short to leave `seconds` between the margins is cut
    in the middle (the centre), when it is long enough at all."""
    first = math.ceil(total * TRAILER_MARGIN)
    last = math.floor(total * (1 - TRAILER_MARGIN) - seconds)
    if last >= first:
        return range(first, last + 1)
    if total >= seconds:
        centre = int((total - seconds) // 2)
        return range(centre, centre + 1)
    return range(0)


def unused_starts(starts: range, seconds: int, previous: Collection[int]) -> list[int]:
    """The starts whose clip does not overlap a rejected one (`previous`: their starts)."""
    return [s for s in starts if all(abs(s - old) >= seconds for old in previous)]


def pieces_for(playlist: Playlist, start: int, seconds: int) -> tuple[list[Piece], float]:
    """The pieces that cover `start` to `start + seconds`, and how far into the first one `start`
    is. A clip that needs more than `MAX_PIECES_PER_CLIP` pieces is refused."""
    first = max(index for index, piece in enumerate(playlist.pieces) if piece.start <= start)
    chosen: list[Piece] = []
    for piece in playlist.pieces[first:]:
        chosen.append(piece)
        if len(chosen) > MAX_PIECES_PER_CLIP:
            raise PlaylistError("the clip needs too many pieces")
        if piece.start + piece.duration >= start + seconds:
            return chosen, start - playlist.pieces[first].start
    raise PlaylistError("the playlist does not cover the clip")


def draw_order(movies: Sequence[SteamMovie], stream: random.Random) -> list[SteamMovie]:
    """Every movie once, in an order drawn from `stream` (lowest ids first before the draw, so the
    order does not depend on what Steam happened to list first)."""
    return draw_distinct(stream, sorted(movies, key=lambda movie: movie.id), len(movies))


@dataclass(frozen=True)
class Trailer:
    """The init segment and the pieces of the chosen clip in one buffer, ready for `media`."""

    marker: TrailerMarker
    data: bytes
    offset: float  # seconds from the start of `data` to the clip's first frame


class TrailerSource(Protocol):
    """What the builder needs: a trailer for one of the game's Steam apps, or `NoTrailer`."""

    async def find(
        self, app_ids: Sequence[int], *, avoid: Collection[str], stream: random.Random, seconds: int
    ) -> Trailer: ...


class SteamClient:
    """Reads Steam's trailers through the allow-listed client (`fetch.py`). The clock and the sleep
    are injected, so a test needs no real time. After the store is unavailable (`appdetails` gives 429
    or 5xx through every retry), or after `CDN_FAILURES_BEFORE_OFF` movies in a row fail on the
    trailer host, the client answers `STEAM_UNAVAILABLE` at once for the rest of the run instead of
    waiting through the retries again for every slot. One failing playlist or piece is only that
    movie failing: the game's other movies are tried."""

    def __init__(
        self,
        transport: httpx.AsyncBaseTransport | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._http = build_client(transport)
        self._sleep = sleep
        self._spacer = RequestSpacer(APPDETAILS_INTERVAL_SECONDS, clock=clock, sleep=sleep)
        self.is_unavailable = False
        self._cdn_failures = 0

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self._http.aclose()

    async def find(
        self, app_ids: Sequence[int], *, avoid: Collection[str], stream: random.Random, seconds: int
    ) -> Trailer:
        """A trailer segment of one of the game's apps. Every movie no rejected draft used is tried
        first, across all the apps (each app in turn, its movies in drawn order), then new segments
        of the movies that were used. `NoTrailer` with the last reason when none works."""
        if not app_ids:
            raise NoTrailer(FallbackReason.NO_STEAM_ID)
        if self.is_unavailable:
            raise NoTrailer(FallbackReason.STEAM_UNAVAILABLE)
        rejected = [m for m in map(TrailerMarker.parse, avoid) if m is not None]
        used = {(m.app_id, m.movie_id) for m in rejected}
        reason = FallbackReason.NO_TRAILER
        tries = 0
        used_movies: list[tuple[int, SteamMovie]] = []
        candidates: list[tuple[int, SteamMovie]] = []
        for app_id in app_ids[:MAX_APPS_TRIED]:
            try:
                movies = draw_order(await self._movies(app_id), stream)
            except NoTrailer as error:
                reason = error.reason
                if error.reason is FallbackReason.STEAM_UNAVAILABLE:
                    self.is_unavailable = True
                    raise
                continue
            candidates += [(app_id, m) for m in movies if (app_id, m.id) not in used]
            used_movies += [(app_id, m) for m in movies if (app_id, m.id) in used]
            # This app's unused movies now; the used ones wait until every app has had its turn.
            while candidates and tries < MAX_MOVIES_TRIED:
                app, movie = candidates.pop(0)
                tries += 1
                try:
                    return await self._attempt(app, movie, rejected, stream, seconds)
                except NoTrailer as error:
                    reason = error.reason
                    if self.is_unavailable:
                        raise
        for app, movie in used_movies[: max(0, MAX_MOVIES_TRIED - tries)]:
            try:
                return await self._attempt(app, movie, rejected, stream, seconds)
            except NoTrailer as error:
                reason = error.reason
                if self.is_unavailable:
                    raise
        raise NoTrailer(reason)

    async def _attempt(
        self,
        app_id: int,
        movie: SteamMovie,
        rejected: Sequence[TrailerMarker],
        stream: random.Random,
        seconds: int,
    ) -> Trailer:
        """One movie. A failure on the trailer host is that movie failing; only
        `CDN_FAILURES_BEFORE_OFF` of them in a row turn Steam off for the run."""
        previous = [m.start for m in rejected if (m.app_id, m.movie_id) == (app_id, movie.id)]
        try:
            trailer = await self._trailer(app_id, movie, previous, stream, seconds)
        except NoTrailer as error:
            if error.reason is FallbackReason.STEAM_UNAVAILABLE:
                self._cdn_failures += 1
                if self._cdn_failures >= CDN_FAILURES_BEFORE_OFF:
                    self.is_unavailable = True
                    raise
            raise
        self._cdn_failures = 0
        return trailer

    async def _movies(self, app_id: int) -> list[SteamMovie]:
        url = f"{STORE_URL}?appids={app_id}&cc=us&l=en"
        body = await self._get(url, MAX_APPDETAILS_BYTES, spaced=True)
        try:
            payload = json.loads(body)
        except (ValueError, RecursionError):  # not JSON, or nested too deeply for the parser
            raise NoTrailer(FallbackReason.UNREADABLE) from None
        return parse_movies(payload, app_id)

    async def _trailer(
        self,
        app_id: int,
        movie: SteamMovie,
        previous: Sequence[int],
        stream: random.Random,
        seconds: int,
    ) -> Trailer:
        master = await self._text(movie.hls_h264)
        try:
            variant = choose_variant(parse_master(master, movie.hls_h264))
            if variant is None:
                raise NoTrailer(FallbackReason.UNREADABLE)
            playlist = parse_playlist(await self._text(variant.url), variant.url)
            starts = unused_starts(candidate_starts(playlist.total, seconds), seconds, previous)
            if not starts:
                raise NoTrailer(
                    FallbackReason.NO_NEW_SEGMENT if previous else FallbackReason.UNREADABLE
                )
            start = starts[draw_index(stream, len(starts))]
            pieces, offset = pieces_for(playlist, start, seconds)
        except PlaylistError:
            raise NoTrailer(FallbackReason.UNREADABLE) from None
        data = await self._get(playlist.init_url, MAX_PIECE_BYTES)
        for piece in pieces:
            data += await self._get(piece.url, MAX_PIECE_BYTES)
            if len(data) > MAX_BUFFER_BYTES:
                raise NoTrailer(FallbackReason.UNREADABLE)
        return Trailer(TrailerMarker(app_id, movie.id, start), data, offset)

    async def _text(self, url: str) -> str:
        if checked_url(url) is None:
            raise NoTrailer(FallbackReason.UNREADABLE)
        body = await self._get(url, MAX_PLAYLIST_BYTES)
        try:
            return body.decode("utf-8")
        except UnicodeDecodeError:
            raise NoTrailer(FallbackReason.UNREADABLE) from None

    async def _get(self, url: str, max_bytes: int, *, spaced: bool = False) -> bytes:
        """One GET with bounded retries: 429, 5xx and a failed transfer are retried at most
        `MAX_RETRIES` times (`Retry-After` honoured), then `STEAM_UNAVAILABLE`. A 404 or 410 is
        `NOT_FOUND`; any other answer that is not 200, a redirect included, is `UNREADABLE`."""
        retries = 0
        while True:
            if spaced:
                await self._spacer.wait()
            status: int | None = None
            retry_after: str | None = None
            try:
                response = await send_capped(
                    self._http, self._http.build_request("GET", url), max_bytes=max_bytes
                )
            except httpx.RequestError:
                pass  # A timeout or a refused connection: retried like a 5xx.
            except (FetchError, httpx.InvalidURL):
                raise NoTrailer(FallbackReason.UNREADABLE) from None
            else:
                status = response.status_code
                retry_after = response.headers.get("retry-after")
                if status == 200:
                    return response.content
                if status in (404, 410):
                    raise NoTrailer(FallbackReason.NOT_FOUND)
                if status != 429 and status < 500:
                    raise NoTrailer(FallbackReason.UNREADABLE)
            if retries >= MAX_RETRIES:
                raise NoTrailer(FallbackReason.STEAM_UNAVAILABLE)
            await self._sleep(retry_delay(retries, retry_after))
            retries += 1
