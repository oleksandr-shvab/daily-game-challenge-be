"""A made-up Steam for the tests: the store's `appdetails`, a master playlist and a variant playlist
per movie, and fMP4 pieces that really decode. No network (`httpx.MockTransport`).

Pieces are one second each (a real trailer's are about three; the code goes by `#EXTINF`). Second
`n` of every trailer is flat grey `4 * n`, so a decoded frame names the second it came from.
"""

import functools
import io
import struct
from collections.abc import Callable
from fractions import Fraction
from urllib.parse import parse_qs, urlsplit

import av
import httpx
from PIL import Image

from app.content.steam import STORE_URL, SteamClient

HOST = "https://video.akamai.steamstatic.com"
FPS = 10
SIZE = (160, 90)


@functools.cache
def fmp4_trailer(seconds: int) -> tuple[bytes, tuple[bytes, ...]]:
    """The init segment and one fragment per second of a silent video, as an HLS server sends them
    (made once per length: encoding takes a moment)."""
    buffer = io.BytesIO()
    options = {"movflags": "frag_keyframe+empty_moov+default_base_moof"}
    with av.open(buffer, "w", format="mp4", options=options) as container:
        stream = container.add_stream("libx264", rate=FPS)
        stream.width, stream.height, stream.pix_fmt = (*SIZE, "yuv420p")
        stream.options = {"g": str(FPS), "crf": "30", "preset": "ultrafast", "bf": "0"}
        for index in range(seconds * FPS):
            grey = 4 * (index // FPS)
            frame = av.VideoFrame.from_image(Image.new("RGB", SIZE, (grey, grey, grey)))
            frame.pts, frame.time_base = index, Fraction(1, FPS)
            container.mux(stream.encode(frame))
        container.mux(stream.encode())
    data = buffer.getvalue()
    init, pieces, position, moof = b"", [], 0, b""
    while position < len(data):
        size, kind = struct.unpack(">I4s", data[position : position + 8])
        box = data[position : position + size]
        if kind in (b"ftyp", b"moov"):
            init += box
        elif kind == b"moof":
            moof = box
        elif kind == b"mdat":
            pieces.append(moof + box)
        position += size
    return init, tuple(pieces)


class FakeSteam:
    """`apps` maps a Steam app id to its movie ids. Every movie is a `seconds` long trailer with
    three variants; only the 1280x720 one is served with pieces (the others answer 404, so a client
    that picked them would fail). `requests` lists every URL asked, in order. `answers` replaces the
    answer to any URL that contains one of its keys: a status code, an exception to raise, a
    ready `httpx.Response` or a function of the request that makes one."""

    def __init__(self, apps: dict[int, list[int]], *, seconds: int = 60) -> None:
        self.apps = apps
        self.seconds = seconds
        self.init, self.pieces = fmp4_trailer(seconds)
        self.requests: list[str] = []
        self.answers: dict[
            str, int | Exception | httpx.Response | Callable[[httpx.Request], httpx.Response]
        ] = {}

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def master_url(self, app_id: int, movie_id: int) -> str:
        return f"{HOST}/store_trailers/{app_id}/{movie_id}/hls_264_master.m3u8"

    def variant_url(self, app_id: int, movie_id: int, resolution: str = "1280x720") -> str:
        return f"{HOST}/store_trailers/{app_id}/{movie_id}/{resolution}/index.m3u8"

    def piece_urls(self, app_id: int, movie_id: int) -> list[str]:
        base = self.variant_url(app_id, movie_id).rsplit("/", 1)[0]
        return [f"{base}/p{number:05d}.m4s" for number in range(len(self.pieces))]

    def details(self, app_id: int) -> dict[str, object]:
        if app_id not in self.apps:
            return {str(app_id): {"success": False}}
        movies = [
            {"id": m, "name": "Trailer", "hls_h264": self.master_url(app_id, m), "highlight": True}
            for m in self.apps[app_id]
        ]
        return {str(app_id): {"success": True, "data": {"name": "Game", "movies": movies}}}

    def _handle(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requests.append(url)
        for part, answer in self.answers.items():
            if part in url:
                if isinstance(answer, Exception):
                    raise answer
                if callable(answer):
                    return answer(request)
                return httpx.Response(answer) if isinstance(answer, int) else answer
        parts = urlsplit(url)
        if url.startswith(STORE_URL):
            app_id = int(parse_qs(parts.query)["appids"][0])
            return httpx.Response(200, json=self.details(app_id))
        path = parts.path
        if path.endswith("hls_264_master.m3u8"):
            lines = ["#EXTM3U"]
            for bandwidth, resolution in (
                (5800000, "1920x1080"),
                (2600000, "1280x720"),
                (1400000, "854x480"),
            ):
                lines.append(
                    f'#EXT-X-STREAM-INF:BANDWIDTH={bandwidth},CODECS="avc1.640029,mp4a.40.2",'
                    f'RESOLUTION={resolution},FRAME-RATE=30,AUDIO="audio"'
                )
                lines.append(f"{resolution}/index.m3u8")
            return httpx.Response(200, text="\n".join(lines))
        if "/1280x720/" in path and path.endswith("index.m3u8"):
            lines = ["#EXTM3U", "#EXT-X-VERSION:7", "#EXT-X-TARGETDURATION:1"]
            lines += ['#EXT-X-MAP:URI="init.m4s"']
            for number in range(len(self.pieces)):
                lines += ["#EXTINF:1.000,", f"p{number:05d}.m4s"]
            return httpx.Response(200, text="\n".join([*lines, "#EXT-X-ENDLIST"]))
        if "/1280x720/" in path and path.endswith("init.m4s"):
            return httpx.Response(200, content=self.init)
        if "/1280x720/" in path and path.endswith(".m4s"):
            number = int(path.rsplit("/p", 1)[1].split(".")[0])
            return httpx.Response(200, content=self.pieces[number])
        return httpx.Response(404)


class Clock:
    """A clock that only moves when the client sleeps."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def new_client(fake: FakeSteam, clock: Clock | None = None) -> SteamClient:
    clock = clock or Clock()
    return SteamClient(fake.transport(), clock=clock.time, sleep=clock.sleep)
