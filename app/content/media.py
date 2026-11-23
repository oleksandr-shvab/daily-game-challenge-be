"""Clue media for the puzzle builder (SPEC-04 AC-34 to AC-40, owner decision 75).

Pure functions over files and bytes: no database, no network. Pillow writes the images (WebP,
metadata dropped, clear cuts of the screenshots) and draws the slideshow clip frame by frame; PyAV
encodes the clips (H.264 `yuv420p`, `+faststart`, no audio), cuts a segment out of a Steam trailer
and reads the files back. Every function that writes a file returns the facts measured
from that file (`measure`), and the builder stores those, never the intended values (AC-39).

The numbers below are start values for the owner to tune after the first real drafts (AC-34).
The limits table equals "Limits for the files of a clue" in `CONTRACTS.md` (SPEC-02 AC-74b).
"""

from __future__ import annotations

import io
import random
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
from av.video.reformatter import Colorspace
from PIL import Image, ImageOps

MB = 1024 * 1024  # The repo counts a megabyte as 2**20 bytes (`tests/test_seed_media.py`).

IMAGE_CONTENT_TYPE = "image/webp"
VIDEO_CONTENT_TYPE = "video/mp4"


@dataclass(frozen=True)
class Limits:
    max_width: int
    max_height: int
    max_bytes: int
    duration_ms: tuple[int, int] | None  # Inclusive range; None for a file with no duration.


LIMITS: dict[str, Limits] = {
    IMAGE_CONTENT_TYPE: Limits(1920, 1080, 1 * MB, None),
    VIDEO_CONTENT_TYPE: Limits(1280, 720, 4 * MB, (8_000, 15_000)),
}

# Quality steps for a WebP, highest first; the first one under the size limit wins (AC-35, AC-39).
WEBP_QUALITIES = (82, 74, 66, 58, 50)

# Steps 1 to 5: clear cuts of five different screenshots, growing with the step (AC-34, decision 75).
# The width of each cut as a fraction of its screenshot's; step 5 is the whole screenshot. A cut keeps
# the screenshot's aspect ratio and its own pixels: it is never enlarged.
CUT_WIDTH_FRACTIONS = (
    Fraction(1, 4),
    Fraction(7, 20),
    Fraction(1, 2),
    Fraction(7, 10),
    Fraction(1),
)
# Where the cuts of steps 1 to 4 sit: how far the cut's centre is from the screenshot's centre, as a
# fraction of the screenshot's width, drawn from (low, high). Step 1 is near a side; each band is
# closer to the centre than the one before, and the bands do not overlap, so a later cut is always
# closer. The side (left or right) is drawn once per puzzle and kept for the four cuts.
CUT_CENTRE_DISTANCES = (
    (0.26, 0.33),
    (0.15, 0.21),
    (0.05, 0.10),
    (0.00, 0.02),
)
# The top and bottom of a screenshot usually hold the HUD (health bars, minimaps, prompts), so a cut
# keeps this fraction of the height clear of both edges, when the cut leaves that much room. Only
# the vertical margin is kept: step 1 sits at a side on purpose.
CUT_VERTICAL_MARGIN = 0.06

# Step 6: a muted looping clip (AC-36). The slideshow of 2 or 3 screenshots is the fallback; the
# first choice is a segment of a Steam trailer, re-encoded at the trailer's own size (at most
# 1280x720) and at most `CLIP_TRAILER_FPS` frames a second. CRF steps are tried in this order (a
# higher CRF is a smaller file) until the clip is under the size limit.
CLIP_SIZE = (1280, 720)
CLIP_TRAILER_FPS = 30
# The largest picture a trailer may have before it is decoded: generous (a 1080p variant passes, the
# coded height of 1080p is 1088), because a small buffer can declare a huge size and a decoder would
# allocate for it. Each decoded frame is checked too, since H.264 can change size mid-stream.
TRAILER_SOURCE_MAX = (1920, 1088)
TRAILER_SOURCE_MAX_PIXELS = 1920 * 1088
CLIP_SECONDS = 12
CLIP_FPS = 24
CLIP_FADE_SECONDS = 1
CLIP_ZOOM_MAX = Fraction(11, 10)
CLIP_CRFS = (23, 26, 29, 32, 35, 38)
CLIP_SHOTS = (2, 3)

# BT.709 in the stream's metadata and in the RGB to YUV conversion, so a player does not guess.
_BT709 = 1


class MediaLimitError(Exception):
    """A file cannot be made to fit the limits, or does not meet them."""


@dataclass(frozen=True)
class MediaFacts:
    """What a file really is, read from the file itself (AC-39)."""

    content_type: str
    byte_size: int
    width: int
    height: int
    duration_ms: int | None


def broken_limits(facts: MediaFacts) -> list[str]:
    """Names of the limits `facts` breaks, each dimension on its own (a portrait clip must fail)."""
    limits = LIMITS.get(facts.content_type)
    if limits is None:
        return ["content_type"]
    broken: list[str] = []
    if limits.duration_ms is None:
        if facts.duration_ms is not None:
            broken.append("duration")
    elif (
        facts.duration_ms is None
        or not limits.duration_ms[0] <= facts.duration_ms <= (limits.duration_ms[1])
    ):
        broken.append("duration")
    if facts.width > limits.max_width:
        broken.append("width")
    if facts.height > limits.max_height:
        broken.append("height")
    if facts.byte_size > limits.max_bytes:
        broken.append("byte_size")
    return broken


def measure(path: Path) -> MediaFacts:
    """Read the content type, size, dimensions and duration from the file's own bytes.

    The kind comes from the file's signature, never from its name. An MP4 must hold exactly one
    stream, an H.264 `yuv420p` video: any audio or other stream is refused (AC-36).
    """
    with path.open("rb") as handle:
        head = handle.read(12)
    byte_size = path.stat().st_size
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        with Image.open(path) as image:
            if image.format != "WEBP":
                raise MediaLimitError("not a WebP image")
            width, height = image.size
        return MediaFacts(IMAGE_CONTENT_TYPE, byte_size, width, height, None)
    if head[4:8] == b"ftyp":
        with av.open(str(path)) as container:
            streams = container.streams
            if len(streams) != 1 or len(streams.video) != 1:
                raise MediaLimitError("the clip must have exactly one video stream and no other")
            video = streams.video[0]
            context = video.codec_context
            if context.name != "h264":
                raise MediaLimitError(f"the clip codec is {context.name}, not h264")
            if context.pix_fmt != "yuv420p":
                raise MediaLimitError(f"the clip pixel format is {context.pix_fmt}, not yuv420p")
            if container.duration is None:
                raise MediaLimitError("the clip has no duration")
            duration_ms = round(container.duration / 1000)
            return MediaFacts(
                VIDEO_CONTENT_TYPE, byte_size, context.width, context.height, duration_ms
            )
    raise MediaLimitError("not a WebP image or an MP4 clip")


def _checked(path: Path) -> MediaFacts:
    """Measure a file we just wrote; delete it and refuse if it breaks a limit (AC-39)."""
    try:
        facts = measure(path)
        broken = broken_limits(facts)
        if broken:
            raise MediaLimitError(f"{path.name} breaks the limits: {', '.join(broken)}")
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return facts


def _save_webp(image: Image.Image, out: Path, *, lossless: bool = False) -> None:
    """Save `image` as WebP with no metadata, lowering the quality until it is under 1 MB.

    An explicit empty `exif` and no `icc_profile` and `xmp` keep every metadata block out, whatever
    `image.info` carries.
    """
    max_bytes = LIMITS[IMAGE_CONTENT_TYPE].max_bytes
    for quality in WEBP_QUALITIES:
        image.save(out, "WEBP", quality=quality, lossless=lossless, exif=b"", method=4)
        if out.stat().st_size <= max_bytes:
            return
        if lossless:
            break
    out.unlink(missing_ok=True)
    raise MediaLimitError(f"{out.name} does not fit in {max_bytes} bytes")


def _open_rgb(source: Path) -> Image.Image:
    with Image.open(source) as image:
        image.load()
        return ImageOps.exif_transpose(image).convert("RGB")


def still(source: Path, out: Path) -> MediaFacts:
    """A full screenshot: fitted within 1920x1080 (never enlarged), WebP, metadata dropped."""
    image = _open_rgb(source)
    limits = LIMITS[IMAGE_CONTENT_TYPE]
    image.thumbnail((limits.max_width, limits.max_height), Image.Resampling.LANCZOS)
    _save_webp(image, out)
    return _checked(out)


@dataclass(frozen=True)
class CutPlacement:
    """Where one cut sits, as fractions, so it applies to a screenshot of any size."""

    dx: float  # The cut's centre minus the image's, in image widths; negative is left of centre.
    dy: float  # 0 to 1: where the cut sits between the top and the bottom of the allowed band.


def cut_placements(rng: random.Random) -> tuple[CutPlacement, ...]:
    """The placements of steps 1 to 4, from the `crop` stream: one side for all four, a distance
    from the centre in each step's band (`CUT_CENTRE_DISTANCES`) and a vertical position. Only
    `random()` is used, so the numbers are stable across Python versions."""
    side = -1.0 if rng.random() < 0.5 else 1.0
    placements = []
    for low, high in CUT_CENTRE_DISTANCES:
        distance = low + rng.random() * (high - low)
        placements.append(CutPlacement(side * distance, rng.random()))
    return tuple(placements)


def cut_box(
    size: tuple[int, int], fraction: Fraction, placement: CutPlacement
) -> tuple[int, int, int, int]:
    """The window of a cut: `fraction` of the width and of the height (the screenshot's aspect
    ratio), its centre `placement.dx` widths from the centre, always inside the image. The window
    stays `CUT_VERTICAL_MARGIN` of the height clear of the top and bottom when there is room."""
    width, height = size
    window_w = round(width * fraction)
    window_h = round(height * fraction)
    left = round(width / 2 + placement.dx * width - window_w / 2)
    left = min(max(left, 0), width - window_w)
    margin = min(CUT_VERTICAL_MARGIN * height, (height - window_h) / 2)
    top = round(margin + placement.dy * (height - window_h - 2 * margin))
    top = min(max(top, 0), height - window_h)
    return (left, top, left + window_w, top + window_h)


def cut(source: Path, out: Path, step: int, placement: CutPlacement) -> MediaFacts:
    """Steps 1 to 4: a clear cut of `source`, `CUT_WIDTH_FRACTIONS[step - 1]` of its width, with its
    own pixels (never enlarged, no pixelation), as WebP with no metadata. The screenshot is fitted
    within 1920x1080 first, as `still` does. `source` must be a screenshot used in no other step
    (AC-34); the builder chooses it. Step 5 is `still`."""
    if not 1 <= step <= len(CUT_CENTRE_DISTANCES):
        raise ValueError("a cut is for steps 1 to 4; step 5 is the whole screenshot (still)")
    image = _open_rgb(source)
    limits = LIMITS[IMAGE_CONTENT_TYPE]
    image.thumbnail((limits.max_width, limits.max_height), Image.Resampling.LANCZOS)
    box = cut_box(image.size, CUT_WIDTH_FRACTIONS[step - 1], placement)
    _save_webp(image.crop(box), out)
    return _checked(out)


@dataclass(frozen=True)
class _Shot:
    """One screenshot of the clip with its pan, drawn from the `clip` stream."""

    image: Image.Image
    base: tuple[float, float, float, float]  # The largest window of the output's aspect.
    zoom_from: float
    zoom_to: float
    pan_from: tuple[float, float]  # Where the window sits in the slack, 0 to 1 per axis.
    pan_to: tuple[float, float]

    def frame(self, progress: float, size: tuple[int, int]) -> Image.Image:
        zoom = self.zoom_from + (self.zoom_to - self.zoom_from) * progress
        base_left, base_top, base_right, base_bottom = self.base
        window_w = (base_right - base_left) / zoom
        window_h = (base_bottom - base_top) / zoom
        fx = self.pan_from[0] + (self.pan_to[0] - self.pan_from[0]) * progress
        fy = self.pan_from[1] + (self.pan_to[1] - self.pan_from[1]) * progress
        left = base_left + ((base_right - base_left) - window_w) * fx
        top = base_top + ((base_bottom - base_top) - window_h) * fy
        box = (left, top, left + window_w, top + window_h)
        return self.image.resize(size, Image.Resampling.BILINEAR, box=box)


def _make_shot(image: Image.Image, size: tuple[int, int], rng: random.Random) -> _Shot:
    """The largest centred window of `image` in the output's aspect, plus a random slow move."""
    width, height = image.size
    aspect = size[0] / size[1]
    window_w = min(float(width), height * aspect)
    window_h = window_w / aspect
    left = (width - window_w) / 2
    top = (height - window_h) / 2
    zoom_max = float(CLIP_ZOOM_MAX)
    zoom_from, zoom_to = (1.0, zoom_max) if rng.random() < 0.5 else (zoom_max, 1.0)
    return _Shot(
        image=image,
        base=(left, top, left + window_w, top + window_h),
        zoom_from=zoom_from,
        zoom_to=zoom_to,
        pan_from=(rng.random(), rng.random()),
        pan_to=(rng.random(), rng.random()),
    )


def _clip_frames(
    shots: list[_Shot], size: tuple[int, int], fps: int, seconds: int, fade: int
) -> Iterator[Image.Image]:
    """Frames of `seconds` at `fps`: each shot pans, the next one fades in over `fade` seconds.

    N shots share the time so that N * shot_length - (N - 1) * fade = seconds. The clip does not
    fade in from black: its first frame is the poster (AC-40).
    """
    count = len(shots)
    shot_len = (seconds + (count - 1) * fade) / count
    step = shot_len - fade  # Time from one shot's start to the next one's.
    for index in range(seconds * fps):
        t = index / fps
        current = min(int(t / step), count - 1) if step > 0 else 0
        started = current * step
        frame = shots[current].frame(min((t - started) / shot_len, 1.0), size)
        if current > 0 and t - started < fade:
            previous = shots[current - 1]
            before = previous.frame(min((t - (started - step)) / shot_len, 1.0), size)
            frame = Image.blend(before, frame, (t - started) / fade)
        yield frame


def _encode(
    out_mp4: Path, frames: Iterator[Image.Image], size: tuple[int, int], fps: int, crf: int
) -> None:
    with av.open(str(out_mp4), "w", format="mp4", options={"movflags": "+faststart"}) as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height = size
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf), "preset": "medium"}
        context = stream.codec_context
        context.colorspace = context.color_primaries = context.color_trc = _BT709
        for index, image in enumerate(frames):
            frame = av.VideoFrame.from_image(image).reformat(
                format="yuv420p", dst_colorspace=Colorspace.ITU709
            )
            frame.pts = index
            frame.time_base = Fraction(1, fps)
            container.mux(stream.encode(frame))
        container.mux(stream.encode())


def _encode_within_limit(
    out_mp4: Path,
    make_frames: Callable[[], Iterator[Image.Image]],
    size: tuple[int, int],
    fps: int,
) -> MediaFacts:
    """Encode `make_frames()` with each CRF in `CLIP_CRFS` until the file is under 4 MB, then
    measure and check it. The frames are drawn again per attempt: 288 of them do not fit in memory.
    `MediaLimitError`, and no file, when even the last CRF is too big."""
    max_bytes = LIMITS[VIDEO_CONTENT_TYPE].max_bytes
    for crf in CLIP_CRFS:
        _encode(out_mp4, make_frames(), size, fps, crf)
        if out_mp4.stat().st_size <= max_bytes:
            return _checked(out_mp4)
    out_mp4.unlink(missing_ok=True)
    raise MediaLimitError(f"{out_mp4.name} does not fit in {max_bytes} bytes")


def clip(
    sources: list[Path],
    out_mp4: Path,
    rng: random.Random,
    *,
    size: tuple[int, int] = CLIP_SIZE,
    fps: int = CLIP_FPS,
    seconds: int = CLIP_SECONDS,
) -> MediaFacts:
    """Step 6, the fallback: a silent MP4 of 2 or 3 screenshots (a slow pan and zoom, 1 s
    cross-fades), for a game with no usable Steam trailer.

    The CRF is raised in steps until the file is under 4 MB. `size`, `fps` and `seconds` default
    to the real clip; the tests make a smaller one.
    """
    if len(sources) not in CLIP_SHOTS:
        raise ValueError(f"a clip needs {CLIP_SHOTS[0]} or {CLIP_SHOTS[-1]} screenshots")
    if size[0] % 2 or size[1] % 2:
        raise ValueError("a yuv420p clip needs even dimensions")
    shots = [_make_shot(_open_rgb(source), size, rng) for source in sources]
    return _encode_within_limit(
        out_mp4, lambda: _clip_frames(shots, size, fps, seconds, CLIP_FADE_SECONDS), size, fps
    )


def _fitted_size(size: tuple[int, int]) -> tuple[int, int]:
    """`size` scaled down (never up) to fit `CLIP_SIZE`, with even sides for yuv420p."""
    scale = min(1.0, CLIP_SIZE[0] / size[0], CLIP_SIZE[1] / size[1])
    width, height = (max(2, int(side * scale) // 2 * 2) for side in size)
    return width, height


def _check_source_size(width: int, height: int) -> None:
    """Refuse a trailer picture that is empty or larger than `TRAILER_SOURCE_MAX`."""
    if width <= 0 or height <= 0:
        raise MediaLimitError("the trailer has an empty picture size")
    if (
        width > TRAILER_SOURCE_MAX[0]
        or height > TRAILER_SOURCE_MAX[1]
        or width * height > TRAILER_SOURCE_MAX_PIXELS
    ):
        raise MediaLimitError("the trailer picture is too large")


def _trailer_frames(
    data: bytes, offset: float, seconds: int, fps: int, size: tuple[int, int]
) -> Iterator[Image.Image]:
    """`seconds * fps` frames from the video of `data`, starting `offset` seconds after its first
    frame, at a constant `fps` (each output frame is the latest source frame at its time), scaled to
    `size`. Only the video stream is read, so an audio track is never carried over, and decoding
    stops as soon as the clip has its frames. Raises `MediaLimitError` when the data ends before the
    clip does or a frame is empty or too large."""
    total = seconds * fps
    current: Image.Image | None = None
    current_time = gap = 0.0
    produced = 0
    first_time: float | None = None
    with av.open(io.BytesIO(data)) as container:
        for frame in container.decode(video=0):
            _check_source_size(frame.width, frame.height)
            if frame.time is None:
                raise MediaLimitError("a trailer frame has no time")
            if first_time is None:
                first_time = frame.time
            at = frame.time - first_time - offset
            while current is not None and produced < total and produced / fps < at:
                yield current
                produced += 1
            if produced >= total:
                return
            gap = at - current_time if current is not None else 0.0
            current, current_time = _scaled(frame, size), at
        # The data ended: its last frame (shown for as long as the one before it was) fills what is
        # left, unless the data stops before the last frame of the clip.
        if current is None or current_time + gap < seconds - 1 / fps:
            raise MediaLimitError("the trailer segment is shorter than the clip")
        while produced < total:
            yield current
            produced += 1


def _scaled(frame: av.VideoFrame, size: tuple[int, int]) -> Image.Image:
    """The frame as RGB at `size`. Trailers are HD, so BT.709 is the matrix, as the clip is tagged."""
    if (frame.width, frame.height) == size:
        return frame.to_image(src_colorspace=Colorspace.ITU709)
    return frame.to_image(width=size[0], height=size[1], src_colorspace=Colorspace.ITU709)


def trailer_clip(
    data: bytes,
    out_mp4: Path,
    *,
    offset: float,
    fps: int = CLIP_TRAILER_FPS,
    seconds: int = CLIP_SECONDS,
) -> MediaFacts:
    """Step 6: `seconds` of the video in `data` (a trailer's init segment and pieces, in one
    buffer), from `offset` seconds after its first frame, re-encoded as H.264 `yuv420p`,
    `+faststart`, no audio, at the trailer's own size fitted to 1280x720. The CRF is raised in steps
    until the file is under 4 MB (`MediaLimitError` when it never is). The poster is `poster`."""
    with av.open(io.BytesIO(data)) as container:
        if not container.streams.video:
            raise MediaLimitError("the trailer has no video stream")
        context = container.streams.video[0].codec_context
        _check_source_size(context.width, context.height)  # before anything is decoded
        size = _fitted_size((context.width, context.height))
    return _encode_within_limit(
        out_mp4, lambda: _trailer_frames(data, offset, seconds, fps, size), size, fps
    )


def first_frame(clip_mp4: Path) -> Image.Image:
    """The first frame, decoded from the encoded file."""
    with av.open(str(clip_mp4)) as container:
        for frame in container.decode(video=0):
            return frame.to_image()
    raise MediaLimitError(f"{clip_mp4.name} has no frames")


def poster(clip_mp4: Path, out: Path) -> MediaFacts:
    """The poster is the clip's first frame as decoded from the file, as WebP (AC-40)."""
    _save_webp(first_frame(clip_mp4), out)
    return _checked(out)
