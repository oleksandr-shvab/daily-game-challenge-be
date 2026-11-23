"""Clue media of the puzzle builder (SPEC-04 AC-34 to AC-40, decision 75): the limits table, the
stills and the growing clear cuts, the slideshow and trailer clips and the poster, and the seed's
part in each. Images and videos are generated here, no network. Slideshow clips are small (640x360
at the real 24 fps and 12 s) to keep the run short."""

import io
import random
from collections.abc import Iterator
from itertools import pairwise
from pathlib import Path
from typing import Self

import av
import pytest
from PIL import Image, ImageChops, ImageStat

from app.content import media
from app.content.media import MB, MediaFacts, MediaLimitError

SMALL_CLIP = {"size": (640, 360), "fps": 24, "seconds": 12}
TINY_CLIP = {"size": (320, 180), "fps": 12, "seconds": 8}


def blobs(seed: int, size: tuple[int, int], noise: float = 0.0) -> Image.Image:
    """A made-up screenshot: smooth colour blobs, optionally mixed with per-pixel noise."""
    rng = random.Random(seed)
    image = Image.frombytes("RGB", (16, 9), rng.randbytes(16 * 9 * 3))
    image = image.resize(size, Image.Resampling.BICUBIC)
    if noise:
        grain = Image.frombytes("RGB", size, rng.randbytes(size[0] * size[1] * 3))
        image = Image.blend(image, grain, noise)
    return image


def striped(seed: int, size: tuple[int, int]) -> Image.Image:
    """Blobs with a fine 4 px checker on top: lossy WebP keeps it, any averaging into blocks of
    about 20 px flattens it."""
    cells = (size[0] // 4 + 1, size[1] // 4 + 1)
    checker = Image.frombytes(
        "L",
        cells,
        bytes(90 if (x + y) % 2 else 0 for y in range(cells[1]) for x in range(cells[0])),
    )
    checker = checker.resize((cells[0] * 4, cells[1] * 4), Image.Resampling.NEAREST).crop(
        (0, 0, *size)
    )
    return ImageChops.add(blobs(seed, size), Image.merge("RGB", (checker, checker, checker)))


@pytest.fixture(scope="module")
def shots(tmp_path_factory: pytest.TempPathFactory) -> list[Path]:
    folder = tmp_path_factory.mktemp("shots")
    paths = [folder / f"shot{i}.png" for i in range(3)]
    for index, path in enumerate(paths):
        blobs(100 + index, (1280, 720)).save(path)
    return paths


def mean_difference(a: Image.Image, b: Image.Image) -> float:
    return sum(ImageStat.Stat(ImageChops.difference(a.convert("RGB"), b.convert("RGB"))).mean) / 3


def decoded_frame(path: Path, index: int) -> Image.Image:
    with av.open(str(path)) as container:
        for position, frame in enumerate(container.decode(video=0)):
            if position == index:
                return frame.to_image()
    raise AssertionError(f"no frame {index}")


@pytest.mark.parametrize(
    ("content_type", "width", "height", "byte_size", "duration_ms", "expected"),
    [
        ("video/mp4", 1280, 720, 4 * MB, 8_000, []),
        ("video/mp4", 1280, 720, 1000, 15_000, []),
        ("video/mp4", 720, 1280, 1000, 12_000, ["height"]),  # A portrait pair fits as a pair.
        ("video/mp4", 1281, 720, 1000, 12_000, ["width"]),
        ("video/mp4", 1280, 721, 1000, 12_000, ["height"]),
        ("video/mp4", 640, 360, 4 * MB + 1, 12_000, ["byte_size"]),
        ("video/mp4", 640, 360, 1000, 7_999, ["duration"]),
        ("video/mp4", 640, 360, 1000, 15_001, ["duration"]),
        ("video/webm", 640, 360, 1000, 12_000, ["content_type"]),
        ("image/webp", 1920, 1080, 1 * MB, None, []),
        ("image/webp", 1921, 1080, 1000, None, ["width"]),
        ("image/webp", 1920, 1081, 1000, None, ["height"]),
        ("image/webp", 640, 360, 1 * MB + 1, None, ["byte_size"]),
        ("image/webp", 640, 360, 1000, 12_000, ["duration"]),
    ],
)
def test__broken_limits__each_limit_is_checked_on_its_own(
    content_type: str,
    width: int,
    height: int,
    byte_size: int,
    duration_ms: int | None,
    expected: list[str],
) -> None:
    facts = MediaFacts(content_type, byte_size, width, height, duration_ms)
    assert media.broken_limits(facts) == expected


def test__still__fits_the_limits_with_no_metadata_and_refuses_what_cannot_fit(
    tmp_path: Path,
) -> None:
    # 2400x1500 of pure grain: too big until it is scaled down, and over 1 MB at quality 82, so the
    # quality steps are needed (about 0.9 MB at 74).
    source = tmp_path / "big.jpg"
    exif = Image.Exif()
    exif[0x010F] = "Test camera"
    exif.get_ifd(0x8825)[1] = "N"  # A GPS tag.
    blobs(1, (2400, 1500), noise=1.0).save(
        source, exif=exif, icc_profile=b"not a real profile", quality=95
    )
    out = tmp_path / "out.webp"

    facts = media.still(source, out)

    assert facts == media.measure(out)
    assert (facts.width, facts.height) == (1728, 1080)  # 16:10, fitted to the height.
    assert facts.byte_size <= 1 * MB
    with Image.open(out) as result:
        assert not {"exif", "icc_profile", "xmp"} & set(result.info)
        assert not result.getexif()

    # Pure noise does not compress under 1 MB at any quality: refused, and no file is left.
    noise = tmp_path / "noise.png"
    Image.frombytes("RGB", (1920, 1080), random.Random(2).randbytes(1920 * 1080 * 3)).save(noise)
    refused = tmp_path / "refused.webp"
    with pytest.raises(MediaLimitError):
        media.still(noise, refused)
    assert not refused.exists()


def fitted(size: tuple[int, int]) -> tuple[int, int]:
    """What `still` makes of a screenshot of `size`: within 1920x1080, never enlarged."""
    scale = min(1.0, 1920 / size[0], 1080 / size[1])
    return round(size[0] * scale), round(size[1] * scale)


@pytest.mark.parametrize("size", [(1280, 720), (1920, 1080), (1500, 1000), (3000, 500), (640, 360)])
@pytest.mark.parametrize(("step", "fraction"), [(1, 0.25), (2, 0.35), (3, 0.5), (4, 0.7)])
def test__cut__grows_with_the_step_keeps_the_aspect_and_its_own_pixels_with_no_pixelation(
    tmp_path: Path, size: tuple[int, int], step: int, fraction: float
) -> None:
    source = tmp_path / "source.png"
    striped(5, size).save(source)  # Fine detail: any averaging into blocks shows.
    width, height = fitted(size)
    placement = media.cut_placements(random.Random(3))[step - 1]

    facts = media.cut(source, tmp_path / "cut.webp", step, placement)

    # "About" the fraction of the width (AC-34), with the screenshot's own aspect ratio, and never
    # larger than the screenshot: a small screenshot gives a small cut, nothing is enlarged.
    assert facts.width == pytest.approx(width * fraction, abs=width * 0.01)
    assert facts.height == pytest.approx(height * fraction, abs=height * 0.01)
    assert facts.width <= width and facts.height <= height
    assert facts == media.measure(tmp_path / "cut.webp")
    # The cut is the screenshot's own pixels at the window the placement names, as it was drawn.
    with Image.open(source) as original:
        fit = original.convert("RGB")
        fit.thumbnail((1920, 1080), Image.Resampling.LANCZOS)
    box = media.cut_box(fit.size, media.CUT_WIDTH_FRACTIONS[step - 1], placement)
    with Image.open(tmp_path / "cut.webp") as result:
        assert result.size == (box[2] - box[0], box[3] - box[1])
        assert mean_difference(result, fit.crop(box)) < 6  # lossy WebP only; blocks would be 40


@pytest.mark.parametrize("size", [(1920, 1080), (1280, 720), (1500, 1000), (3000, 500)])
def test__cut_placements__step_1_sits_near_a_side_and_every_later_cut_is_closer_to_the_centre(
    size: tuple[int, int],
) -> None:
    width, height = size
    sides = set()
    for seed in range(300):
        placements = media.cut_placements(random.Random(seed))
        boxes = [
            media.cut_box(size, fraction, placement)
            for fraction, placement in zip(media.CUT_WIDTH_FRACTIONS, placements, strict=False)
        ]
        distances = []
        for (left, top, right, bottom), fraction in zip(
            boxes, media.CUT_WIDTH_FRACTIONS, strict=False
        ):
            assert 0 <= left < right <= width and 0 <= top < bottom <= height  # always inside
            assert (right - left) / (bottom - top) == pytest.approx(width / height, abs=0.05)
            distances.append(abs((left + right) / 2 - width / 2))
            if bottom - top < height * 0.85:  # a cut with room keeps clear of the HUD strips
                assert top >= height * 0.05 and bottom <= height * 0.95
        assert distances[0] >= width * 0.2  # Step 1: near a side, away from the centre.
        assert all(near < far for far, near in pairwise(distances))  # then closer each step
        signs = {left + right > width for left, _, right, _ in boxes[:3]}  # step 4 is at the centre
        assert len(signs) == 1  # one side for the cuts
        sides.add(signs.pop())
    assert sides == {True, False}  # both sides occur


def test__clip__is_a_silent_12_second_h264_mp4_and_the_poster_is_its_first_frame(
    tmp_path: Path, shots: list[Path]
) -> None:
    out = tmp_path / "clip.mp4"

    facts = media.clip(shots, out, random.Random(1), **SMALL_CLIP)

    assert facts == media.measure(out)
    assert facts.content_type == "video/mp4"
    assert media.broken_limits(facts) == []
    assert abs(facts.duration_ms - 12_000) < 100
    assert facts.byte_size <= 4 * MB
    with av.open(str(out)) as container:
        assert len(container.streams) == 1
        assert not container.streams.audio
        video = container.streams.video[0]
        assert (video.codec_context.name, video.codec_context.pix_fmt) == ("h264", "yuv420p")
        pictures = [frame.to_image().resize((64, 36)) for frame in container.decode(video=0)]
    assert len(pictures) == 12 * 24
    # Shots fade into each other over a second: with a hard cut one step would jump by about 70.
    steps = [mean_difference(a, b) for a, b in pairwise(pictures)]
    assert max(steps) < 8
    data = out.read_bytes()
    assert data.index(b"moov") < data.index(b"mdat")  # +faststart

    poster_out = tmp_path / "poster.webp"
    poster_facts = media.poster(out, poster_out)
    assert poster_facts == media.measure(poster_out)
    assert poster_facts.content_type == "image/webp"
    with Image.open(poster_out) as poster_image:
        assert mean_difference(poster_image, decoded_frame(out, 0)) < 3
        assert mean_difference(poster_image, decoded_frame(out, 150)) > 10


def test__clip__slow_full_size_default_clip_fits_the_limits(
    tmp_path: Path, shots: list[Path]
) -> None:
    """The real clip: no size, fps or length given, so the defaults are what is checked (about 5 s)."""
    out = tmp_path / "clip.mp4"

    facts = media.clip(shots, out, random.Random(1))

    assert (facts.width, facts.height) == (1280, 720)
    assert facts.duration_ms == 12_000
    assert facts.byte_size <= 4 * MB
    assert media.broken_limits(facts) == []
    with av.open(str(out)) as container:
        assert len(container.streams) == 1
        assert container.streams.video[0].codec_context.name == "h264"


def test__clip__over_the_size_limit_is_refused_and_leaves_no_file(
    tmp_path: Path, shots: list[Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    limits = media.LIMITS["video/mp4"]
    monkeypatch.setitem(
        media.LIMITS, "video/mp4", media.Limits(1280, 720, 1000, limits.duration_ms)
    )
    out = tmp_path / "clip.mp4"

    with pytest.raises(MediaLimitError):
        media.clip(shots[:2], out, random.Random(1), **TINY_CLIP)

    assert not out.exists()


def test__measure__refuses_a_clip_with_an_audio_track(tmp_path: Path) -> None:
    path = tmp_path / "with-audio.mp4"
    with av.open(str(path), "w", format="mp4") as container:
        video = container.add_stream("libx264", rate=12)
        video.width, video.height, video.pix_fmt = 64, 64, "yuv420p"
        audio = container.add_stream("aac", rate=44100)
        for index in range(12):
            frame = av.VideoFrame.from_image(Image.new("RGB", (64, 64), (index * 10, 0, 0)))
            frame.pts = index
            container.mux(video.encode(frame))
        silence = av.AudioFrame(format="fltp", layout="mono", samples=1024)
        for plane in silence.planes:
            plane.update(bytes(plane.buffer_size))
        silence.sample_rate = 44100
        for index in range(20):
            silence.pts = index * 1024
            container.mux(audio.encode(silence))
        container.mux(video.encode())
        container.mux(audio.encode())

    with pytest.raises(MediaLimitError):
        media.measure(path)


def test__seed__decides_the_cuts_and_the_slideshow(tmp_path: Path, shots: list[Path]) -> None:
    assert media.cut_placements(random.Random(9)) == media.cut_placements(random.Random(9))
    assert media.cut_placements(random.Random(9)) != media.cut_placements(random.Random(10))

    # The encoder is not asked to give identical bytes, so the clips are compared by their pictures.
    def clip_frame(seed: int, name: str) -> Image.Image:
        out = tmp_path / name
        media.clip(shots, out, random.Random(seed), **TINY_CLIP)
        return decoded_frame(out, 30)

    same = mean_difference(clip_frame(4, "a.mp4"), clip_frame(4, "b.mp4"))
    other = mean_difference(clip_frame(4, "a.mp4"), clip_frame(5, "c.mp4"))
    assert same < 3
    assert other > 10


def fake_trailer(
    path: Path, seconds: int, *, fps: int = 10, size: tuple[int, int] = (320, 180), audio: bool
) -> None:
    """A trailer made up: second n is flat grey 15 * n, so a decoded frame names its second. With
    `audio`, a silent AAC track next to the video, as a real trailer's stream may have."""
    with av.open(str(path), "w", format="mp4") as container:
        video = container.add_stream("libx264", rate=fps)
        video.width, video.height, video.pix_fmt = (*size, "yuv420p")
        track = container.add_stream("aac", rate=44100) if audio else None
        for index in range(seconds * fps):
            grey = 15 * (index // fps)
            frame = av.VideoFrame.from_image(Image.new("RGB", size, (grey, grey, grey)))
            frame.pts = index
            container.mux(video.encode(frame))
        container.mux(video.encode())
        if track is not None:
            silence = av.AudioFrame(format="fltp", layout="mono", samples=1024)
            for plane in silence.planes:
                plane.update(bytes(plane.buffer_size))
            silence.sample_rate = 44100
            for index in range(seconds * 44100 // 1024):
                silence.pts = index * 1024
                container.mux(track.encode(silence))
            container.mux(track.encode())


def grey_of(image: Image.Image) -> float:
    return ImageStat.Stat(image.convert("L")).mean[0]


def test__trailer_clip__is_the_asked_segment_as_a_silent_h264_mp4_and_refuses_a_short_one(
    tmp_path: Path,
) -> None:
    source = tmp_path / "trailer.mp4"
    fake_trailer(source, 16, audio=True)
    data = source.read_bytes()
    with av.open(str(source)) as container:
        assert container.streams.audio  # the source has sound; the clip must not

    out = tmp_path / "clip.mp4"
    facts = media.trailer_clip(data, out, offset=3.0, fps=10, seconds=8)

    assert facts == media.measure(out)
    assert media.broken_limits(facts) == []
    assert (facts.width, facts.height) == (320, 180)  # the trailer's own size, not enlarged
    assert abs(facts.duration_ms - 8_000) < 100
    with av.open(str(out)) as container:
        assert len(container.streams) == 1 and not container.streams.audio  # no audio (AC-36)
        assert container.streams.video[0].codec_context.name == "h264"
        pictures = [frame.to_image() for frame in container.decode(video=0)]
    assert len(pictures) == 8 * 10
    # Seconds 3 to 11 of the trailer: grey 45 first, 90 after 3.9 s (second 6), 150 at the end.
    assert grey_of(pictures[0]) == pytest.approx(45, abs=8)
    assert grey_of(pictures[39]) == pytest.approx(90, abs=8)
    assert grey_of(pictures[79]) == pytest.approx(150, abs=8)
    data_out = out.read_bytes()
    assert data_out.index(b"moov") < data_out.index(b"mdat")  # +faststart
    poster_out = tmp_path / "poster.webp"
    media.poster(out, poster_out)
    with Image.open(poster_out) as poster_image:
        assert grey_of(poster_image) == pytest.approx(45, abs=8)  # the clip's first frame (AC-40)

    # Data that ends before the clip does is refused, and no file is left.
    with pytest.raises(MediaLimitError):
        media.trailer_clip(data, tmp_path / "short.mp4", offset=12.0, fps=10, seconds=8)
    assert not (tmp_path / "short.mp4").exists()


class FakeFrame:
    def __init__(self, time: float, width: int, height: int) -> None:
        self.time, self.width, self.height = time, width, height

    def to_image(self, **kwargs: int) -> Image.Image:
        return Image.new(
            "RGB", (kwargs.get("width", self.width), kwargs.get("height", self.height))
        )


class FakeContainer:
    """Stands in for an opened trailer: a header size, and frames that are counted as decoded."""

    def __init__(self, header: tuple[int, int], frames: list[FakeFrame], *, video: bool = True):
        context = type("Context", (), {"width": header[0], "height": header[1]})()
        self.streams = type(
            "Streams", (), {"video": [type("Stream", (), {"codec_context": context})()] * video}
        )()
        self.frames, self.decoded = frames, 0

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def decode(self, video: int) -> Iterator[FakeFrame]:
        for frame in self.frames:
            self.decoded += 1
            yield frame


def patch_trailer_open(monkeypatch: pytest.MonkeyPatch, container: FakeContainer) -> None:
    """The buffer opens as `container`; the encoder's own file opens as usual."""
    real_open = av.open

    def fake_open(file: object, *args: object, **kwargs: object) -> object:
        return container if isinstance(file, io.BytesIO) else real_open(file, *args, **kwargs)

    monkeypatch.setattr(media.av, "open", fake_open)


@pytest.mark.parametrize(
    ("header", "frames", "video", "message"),
    [
        ((320, 180), [FakeFrame(0.0, 320, 180)], False, "no video stream"),
        ((0, 180), [FakeFrame(0.0, 320, 180)], True, "empty picture"),
        ((320, 0), [FakeFrame(0.0, 320, 180)], True, "empty picture"),
        ((8000, 8000), [FakeFrame(0.0, 320, 180)], True, "too large"),  # declared, never decoded
        ((320, 180), [FakeFrame(0.0, 320, 180), FakeFrame(0.1, 7680, 4320)], True, "too large"),
        ((320, 180), [FakeFrame(0.0, 0, 180)], True, "empty picture"),
    ],
    ids=[
        "no-video",
        "zero-width",
        "zero-height",
        "huge-header",
        "size-changes-mid-stream",
        "empty-frame",
    ],
)
def test__trailer_clip__refuses_what_it_cannot_safely_decode_with_a_media_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    header: tuple[int, int],
    frames: list[FakeFrame],
    video: bool,
    message: str,
) -> None:
    container = FakeContainer(header, frames, video=video)
    patch_trailer_open(monkeypatch, container)

    with pytest.raises(MediaLimitError, match=message):
        media.trailer_clip(b"", tmp_path / "clip.mp4", offset=0.0, fps=10, seconds=1)

    assert container.decoded <= 2  # a declared size is refused before a frame is decoded
    assert not (tmp_path / "clip.mp4").exists()


def test__trailer_clip__a_real_buffer_with_no_video_or_a_huge_picture_is_refused(
    tmp_path: Path,
) -> None:
    audio_only = tmp_path / "audio.mp4"
    with av.open(str(audio_only), "w", format="mp4") as container:
        track = container.add_stream("aac", rate=44100)
        silence = av.AudioFrame(format="fltp", layout="mono", samples=1024)
        for plane in silence.planes:
            plane.update(bytes(plane.buffer_size))
        silence.sample_rate = 44100
        for index in range(20):
            silence.pts = index * 1024
            container.mux(track.encode(silence))
        container.mux(track.encode())
    with pytest.raises(MediaLimitError, match="no video stream"):
        media.trailer_clip(audio_only.read_bytes(), tmp_path / "a.mp4", offset=0.0, seconds=1)

    big = tmp_path / "big.mp4"
    fake_trailer(big, 2, size=(2000, 1100), audio=False)  # over 1920 wide: refused up front
    with pytest.raises(MediaLimitError, match="too large"):
        media.trailer_clip(big.read_bytes(), tmp_path / "b.mp4", offset=0.0, fps=10, seconds=1)


def test__trailer_frames__stop_decoding_as_soon_as_the_clip_has_its_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = [FakeFrame(index / 10, 320, 180) for index in range(1000)]  # 100 s of video
    container = FakeContainer((320, 180), frames)
    patch_trailer_open(monkeypatch, container)

    pictures = list(media._trailer_frames(b"", 5.0, 1, 10, (320, 180)))

    assert len(pictures) == 10
    assert container.decoded <= 5 * 10 + 10 + 2  # the lead-in and the clip, not the rest
