"""The puzzle builder (M1-22d, SPEC-04 AC-2, AC-29, AC-32, AC-33, AC-41 to AC-46, AC-50, AC-52,
AC-85, AC-86, decision 71).

No network: IGDB is a fake list of games, screenshots are small generated JPEGs, storage and the
working folder are in `tmp_path`, the clip is shortened to the smallest the limits allow."""

import datetime as dt
import hashlib
import importlib.util
import random
from collections import Counter
from collections.abc import AsyncIterator
from dataclasses import dataclass
from functools import partial
from io import BytesIO
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest
from botocore.exceptions import ClientError
from PIL import Image
from pydantic import SecretStr
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.content import builder as builder_module
from app.content import media
from app.content.builder import (
    GENERATOR_VERSION,
    Builder,
    ImageFetcher,
    SlotRequest,
    SlotStatus,
    give_up_slot,
)
from app.content.candidates import never_igdb_ids
from app.content.fetch import FetchError, ForbiddenHostError, ResponseTooLargeError
from app.content.igdb import IgdbGame, parse_game
from app.content.seed import TEST_SALT, SaltError
from app.content.settings import ContentSettings
from app.content.target import TargetRefusedError
from app.core.config import settings as app_settings
from app.crud import puzzle as crud_puzzle
from app.crud.exclusion import list_exclusions
from app.crud.title import create_title
from app.media.writer import LocalMediaWriter, MediaWriteError, S3MediaWriter
from app.models.exclusion import AnswerExclusion
from app.models.game import Game
from app.models.progress import Progress
from app.models.puzzle import ClueMediaFile, Puzzle, PuzzleClue, PuzzleStatus
from app.models.title import Title
from app.models.user import User
from tests.factories import create_game, create_puzzle

NOW = dt.datetime(2026, 11, 16, 12, 0, tzinfo=dt.UTC)
LONG_AGO = int(dt.datetime(2015, 1, 1, tzinfo=dt.UTC).timestamp())
# The shortest clip the limits allow (8 s), small and slow, so the run stays quick.
TINY_CLIP = {"size": (320, 180), "fps": 12, "seconds": 8}


def make_game(igdb_id: int, *, screenshots: int = 8, **fields: Any) -> IgdbGame:
    """A game that passes every filter and has every hint's data."""
    raw = {
        "id": igdb_id,
        "name": f"Test Game {igdb_id}",
        "slug": f"test-game-{igdb_id}",
        "game_type": 0,
        "first_release_date": LONG_AGO + igdb_id * 86400 * 400,
        "total_rating_count": 100,
        "aggregated_rating": 84.5,
        "aggregated_rating_count": 10,
        "genres": [{"name": "Adventure"}],
        "involved_companies": [{"developer": True, "company": {"name": "Test Studio"}}],
        "release_dates": [{"date": LONG_AGO, "platform": {"name": "PC"}}],
        "screenshots": [
            {"image_id": f"g{igdb_id}s{number:02d}", "width": 1920, "height": 1080}
            for number in range(screenshots)
        ],
        **fields,
    }
    game = parse_game(raw, max_year=2027, skips=Counter(), drops=Counter())
    assert game is not None
    return game


class FakeIgdb:
    """The games source of a run. `calls` counts how often a run asked IGDB to be read."""

    def __init__(self, games: list[IgdbGame]) -> None:
        self.games = games
        self.calls = 0

    def __call__(self) -> AsyncIterator[IgdbGame]:
        self.calls += 1
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[IgdbGame]:
        for game in self.games:
            yield game


class FakeImages:
    """Answers for `fetch_image`: a small JPEG for each image id, and a list of what was asked."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    async def __call__(self, image_id: str) -> bytes:
        self.asked.append(image_id)
        rng = random.Random(image_id)
        image = Image.frombytes("RGB", (16, 9), rng.randbytes(16 * 9 * 3))
        out = BytesIO()
        image.resize((320, 180), Image.Resampling.BICUBIC).save(out, "JPEG")
        return out.getvalue()


@dataclass
class Env:
    db: AsyncSession
    settings: ContentSettings
    storage: Path
    images: FakeImages
    game_id: int

    def builder(
        self,
        igdb: FakeIgdb,
        *,
        dry_run: bool = False,
        target: str = "local",
        settings: ContentSettings | None = None,
        has_writer: bool = True,
    ) -> Builder:
        return Builder(
            self.db,
            game_slug="game-guesser",
            target=target,
            settings=settings or self.settings,
            app_settings=app_settings,
            now=NOW,
            games=igdb,
            fetch_image=self.images,
            writer=LocalMediaWriter(self.storage) if has_writer else None,
            dry_run=dry_run,
        )


def content_settings(tmp_path: Path, **fields: Any) -> ContentSettings:
    values: dict[str, Any] = {
        "PUZZLE_SEED_SALT": SecretStr(TEST_SALT),
        "CONTENT_WORK_DIR": tmp_path / "work",
        **fields,
    }
    return ContentSettings(_env_file=None, **values)


@pytest.fixture
async def env(db_session: AsyncSession, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Env:
    monkeypatch.setattr(media, "clip", partial(media.clip, **TINY_CLIP))
    game = await create_game(db_session)
    await db_session.commit()
    return Env(db_session, content_settings(tmp_path), tmp_path / "storage", FakeImages(), game.id)


async def count(db: AsyncSession, model: type) -> int:
    return (await db.execute(select(func.count()).select_from(model))).scalar_one()


async def clues_of(db: AsyncSession, puzzle: Puzzle) -> list[PuzzleClue]:
    result = await db.execute(
        select(PuzzleClue).where(PuzzleClue.puzzle_id == puzzle.id).order_by(PuzzleClue.step)
    )
    return list(result.scalars())


async def files_of(db: AsyncSession, clue: PuzzleClue) -> list[ClueMediaFile]:
    result = await db.execute(
        select(ClueMediaFile).where(ClueMediaFile.clue_id == clue.id).order_by(ClueMediaFile.id)
    )
    return list(result.scalars())


async def puzzle_in_slot(db: AsyncSession, slot: int) -> Puzzle:
    return (await db.execute(select(Puzzle).where(Puzzle.build_slot == slot))).scalar_one()


# (a) A real build ---------------------------------------------------------------------------


async def test__build__two_slots_write_in_review_drafts_and_a_second_run_adds_nothing(
    env: Env,
) -> None:
    igdb = FakeIgdb([make_game(1), make_game(2), make_game(3)])

    report = await env.builder(igdb).run(SlotRequest(slots=range(1, 3)))

    assert [outcome.status for outcome in report.outcomes] == [SlotStatus.BUILT] * 2
    puzzles = list((await env.db.execute(select(Puzzle).order_by(Puzzle.build_slot))).scalars())
    assert [(p.build_slot, p.status, p.date) for p in puzzles] == [
        (1, PuzzleStatus.IN_REVIEW, None),
        (2, PuzzleStatus.IN_REVIEW, None),
    ]
    assert {p.generator_version for p in puzzles} == {GENERATOR_VERSION}
    assert len({p.answer_title_id for p in puzzles}) == 2  # AC-25: slot 2 sees slot 1's answer
    for puzzle in puzzles:
        title = await env.db.get(Title, puzzle.answer_title_id)
        assert title is not None and title.igdb_id is not None
        clues = await clues_of(env.db, puzzle)
        assert [c.step for c in clues] == [1, 2, 3, 4, 5, 6]
        assert [c.media_type for c in clues] == ["image"] * 5 + ["video"]
        assert [c.hint_kind for c in clues] == [
            None, "critic_score", "platform", "genre", "year", "developer",
        ]  # fmt: skip
        assert clues[1].hint_value == "85"  # 84.5 rounds half up
        assert {c.source_url for c in clues} == {
            f"https://www.igdb.com/games/test-game-{title.igdb_id}"
        }
        image_ids = [image_id for c in clues for image_id in c.source_image_ids or []]
        assert len(image_ids) == len(set(image_ids)) >= 7  # no screenshot twice (AC-34, AC-37)
        stored = [f for c in clues for f in await files_of(env.db, c)]
        assert [(f.role, f.content_type) for f in stored] == [("source", "image/webp")] * 5 + [
            ("source", "video/mp4"),
            ("poster", "image/webp"),
        ]
        for file in stored:  # AC-39, AC-41: measured facts, keyed by the file's own SHA-256
            data = (env.storage / file.key).read_bytes()
            assert hashlib.sha256(data).hexdigest() == file.key
            assert len(data) == file.byte_size
            assert (file.duration_ms is not None) == (file.content_type == "video/mp4")
    asked_before, reads_before = list(env.images.asked), igdb.calls

    again = await env.builder(igdb).run(SlotRequest(slots=range(1, 3)))

    assert [outcome.status for outcome in again.outcomes] == [SlotStatus.EXISTS] * 2
    assert (env.images.asked, igdb.calls) == (asked_before, reads_before)  # nothing fetched or read
    assert await count(env.db, Puzzle) == 2 and await count(env.db, Title) == 2


# (b) Atomic write, dry run, played puzzles --------------------------------------------------


async def test__build__a_failure_while_writing_the_fifth_media_row_leaves_files_and_no_row(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = crud_puzzle.add_clue_media_file
    calls = 0

    async def fail_on_the_fifth(db: AsyncSession, clue_id: int, file: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == 5:
            raise RuntimeError("the disk is full")
        await real(db, clue_id, file)

    monkeypatch.setattr(crud_puzzle, "add_clue_media_file", fail_on_the_fifth)
    igdb = FakeIgdb([make_game(1)])

    with pytest.raises(RuntimeError, match="disk is full"):
        await env.builder(igdb).run(SlotRequest(slots=range(1, 2)))

    # No half draft: no puzzle, clue, media row, and not the new title either (AC-46).
    assert [await count(env.db, m) for m in (Puzzle, PuzzleClue, ClueMediaFile, Title)] == [0] * 4
    # The files went first and stay (non-goal: clean-up).
    assert len(list(env.storage.iterdir())) == 7

    monkeypatch.setattr(crud_puzzle, "add_clue_media_file", real)
    report = await env.builder(igdb).run(SlotRequest(slots=range(1, 2)))

    assert report.outcomes[0].status == SlotStatus.BUILT
    assert [await count(env.db, m) for m in (Puzzle, PuzzleClue, ClueMediaFile, Title)] == [
        1, 6, 7, 1,
    ]  # fmt: skip


async def test__dry_run__does_every_step_but_writes_nothing_and_hides_the_answer(
    env: Env,
) -> None:
    igdb = FakeIgdb([make_game(1)])

    report = await env.builder(igdb, dry_run=True).run(SlotRequest(next_count=1))

    outcome = report.outcomes[0]
    assert outcome.status == SlotStatus.PLANNED and len(outcome.files) == 7
    assert [await count(env.db, m) for m in (Puzzle, PuzzleClue, ClueMediaFile, Title)] == [0] * 4
    assert not env.storage.exists()  # AC-45: not even the folder
    assert len(list((env.settings.content_work_path / "out" / "1").iterdir())) == 7
    assert "Test Game 1" not in "\n".join(report.lines())
    assert '"Test Game 1"' in "\n".join(report.lines(show_answers=True))


async def test__build__never_touches_a_slot_that_has_a_puzzle_with_stored_games(env: Env) -> None:
    game = await env.db.get(Game, env.game_id)
    assert game is not None
    answer = await create_title(env.db, name="Hades", igdb_id=1)
    played = await create_puzzle(env.db, game=game, answer=answer, puzzle_date=dt.date(2026, 11, 1))
    played.build_slot = 3
    account = User(email="player@example.com", hashed_password="not-a-real-hash")
    env.db.add(account)
    await env.db.flush()
    env.db.add(
        Progress(
            user_id=account.id,
            puzzle_id=played.id,
            status="in_progress",
            started_at=NOW,
            counts_for_streak=True,
            origin="play",
        )
    )
    await env.db.commit()
    clues_before = [(c.id, c.hint_value, c.source_url) for c in await clues_of(env.db, played)]
    igdb = FakeIgdb([make_game(1), make_game(2)])

    report = await env.builder(igdb).run(SlotRequest(slots=range(3, 4)))

    assert report.outcomes[0].status == SlotStatus.EXISTS
    assert (igdb.calls, env.images.asked) == (0, [])
    await env.db.refresh(played)
    assert (played.answer_title_id, played.status, played.date) == (
        answer.id, PuzzleStatus.SCHEDULED, dt.date(2026, 11, 1),
    )  # fmt: skip
    assert [(c.id, c.hint_value, c.source_url) for c in await clues_of(env.db, played)] == (
        clues_before
    )
    assert (await count(env.db, Puzzle), await count(env.db, Title)) == (1, 1)


async def test__build__skips_and_reports_a_candidate_that_would_be_a_near_duplicate(
    env: Env,
) -> None:
    first, second = make_game(1), make_game(2)
    # A hand-made title with the name and year of the first game: the import dropped it (R-20).
    await create_title(env.db, name=first.name.upper(), year=first.release_year)
    await env.db.commit()

    report = await env.builder(FakeIgdb([first])).run(SlotRequest(slots=range(1, 2)))

    outcome = report.outcomes[0]
    assert (outcome.status, outcome.detail, outcome.near_duplicates) == (
        SlotStatus.FAILED, "no_candidate", 1,
    )  # fmt: skip
    assert (await count(env.db, Title), await count(env.db, Puzzle)) == (1, 0)

    report = await env.builder(FakeIgdb([first, second])).run(SlotRequest(slots=range(1, 2)))

    answer = await env.db.get(Title, (await puzzle_in_slot(env.db, 1)).answer_title_id)
    assert answer is not None and answer.igdb_id == second.id
    assert await count(env.db, Title) == 2


# (c) Gates ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("target", "fields", "error", "message"),
    [
        ("staging", {}, TargetRefusedError, "IGDB_IMAGES_CLEARED_BY"),
        ("local", {"PUZZLE_SEED_SALT": None}, SaltError, "PUZZLE_SEED_SALT is not set"),
        (
            "local",
            {"PUZZLE_SEED_SALT": SecretStr("short")},
            SaltError,
            "placeholder or too short",
        ),
    ],
    ids=["images_gate_closed", "no_salt", "weak_salt"],
)
async def test__builder__refuses_before_any_read_or_fetch(
    env: Env,
    tmp_path: Path,
    target: str,
    fields: dict[str, Any],
    error: type[Exception],
    message: str,
) -> None:
    igdb = FakeIgdb([make_game(1)])
    refused = content_settings(tmp_path, **fields)

    with pytest.raises(error, match=message):
        await env.builder(igdb, target=target, settings=refused).run(SlotRequest(next_count=1))

    assert (igdb.calls, env.images.asked, await count(env.db, Puzzle)) == (0, [], 0)
    # The gate is the only thing that stopped it: with the decision named, the run goes on.
    cleared = content_settings(tmp_path, IGDB_IMAGES_CLEARED_BY="SPEC.md decision 72")
    if error is TargetRefusedError:
        report = await env.builder(igdb, target=target, settings=cleared, dry_run=True).run(
            SlotRequest(next_count=1)
        )
        assert report.outcomes[0].status == SlotStatus.PLANNED


def load_script() -> ModuleType:
    path = Path(__file__).resolve().parent.parent / "scripts" / "build_puzzles.py"
    spec = importlib.util.spec_from_file_location("p3_build_puzzles", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("argv", "environ", "message"),
    [
        (["--next", "3"], {"ENVIRONMENT": "local"}, "--target is required"),
        (["--dry-run", "--next", "3"], {"ENVIRONMENT": "local"}, "--target is required"),
        (["--give-up", "3"], {"ENVIRONMENT": "local"}, "--target is required"),
        (["--target", "local", "--next", "3"], {"ENVIRONMENT": "local", "CI": "true"}, "CI"),
        (
            ["--target", "local", "--next", "3"],
            {"ENVIRONMENT": "staging"},
            "ENVIRONMENT variable is 'staging'",
        ),
        (
            ["--target", "staging", "--confirm-target", "staging", "--next", "3"],
            {"ENVIRONMENT": "staging"},
            "IGDB_IMAGES_CLEARED_BY",
        ),
    ],
    ids=["no_target", "dry_run_no_target", "give_up_no_target", "ci", "environment", "images_gate"],
)
def test__build_script__guard_runs_first_and_prints_the_target_before_it_refuses(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    environ: dict[str, str],
    message: str,
) -> None:
    for name in ("CI", "ENVIRONMENT", "IGDB_IMAGES_CLEARED_BY"):
        monkeypatch.delenv(name, raising=False)
    for name, value in environ.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr("sys.argv", ["build_puzzles.py", *argv])

    code = load_script().main()

    captured = capsys.readouterr()
    assert code == 1 and message in captured.err
    # A refusal by the target guard prints nothing first; a refusal by the image gate comes after
    # the target line (AC-44).
    assert captured.out.startswith("Target: staging;") == (message == "IGDB_IMAGES_CLEARED_BY")


# AC-85: the fetch ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url_template", "error", "status", "body_size"),
    [
        (None, None, 200, 10),
        ("https://evil.example/{image_id}.jpg", ForbiddenHostError, 200, 10),
        ("http://images.igdb.com/{image_id}.jpg", ForbiddenHostError, 200, 10),
        (None, ResponseTooLargeError, 200, 5000),
        (None, FetchError, 302, 10),
    ],
    ids=["ok", "foreign_host", "plain_http", "too_large", "redirect_is_not_followed"],
)
async def test__image_fetcher__goes_only_to_the_image_host_within_the_size_cap(
    monkeypatch: pytest.MonkeyPatch,
    url_template: str | None,
    error: type[Exception] | None,
    status: int,
    body_size: int,
) -> None:
    requests: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        headers = {"Location": "https://evil.example/x"} if status == 302 else {}
        return httpx.Response(status, content=b"x" * body_size, headers=headers)

    if url_template is not None:
        monkeypatch.setattr(builder_module, "IMAGE_URL", url_template)
    monkeypatch.setattr(builder_module, "MAX_IMAGE_BYTES", 1000)

    async with ImageFetcher(httpx.MockTransport(answer)) as fetcher:
        if error is None:
            assert await fetcher.fetch("abc123") == b"x" * 10
            assert [str(r.url) for r in requests] == [
                "https://images.igdb.com/igdb/image/upload/t_1080p/abc123.jpg"
            ]
        else:
            with pytest.raises(error):
                await fetcher.fetch("abc123")
            if error is ForbiddenHostError:
                assert requests == []  # refused before the request left


# (d) Rebuilds, decision 71 and the give-up command ------------------------------------------


async def reject_as_rebuild(db: AsyncSession, slot: int) -> list[str]:
    """What the review page's reject does (M1-24a): the exclusion row gets the title, the slot and
    every screenshot of the draft, then the draft is deleted. Returns the image ids."""
    puzzle = await puzzle_in_slot(db, slot)
    image_ids = sorted(
        {i for clue in await clues_of(db, puzzle) for i in clue.source_image_ids or []}
    )
    db.add(
        AnswerExclusion(
            game_id=puzzle.game_id,
            variant=puzzle.variant,
            title_id=puzzle.answer_title_id,
            reason="rebuild",
            build_slot=slot,
            image_ids=image_ids,
        )
    )
    await db.execute(delete(Puzzle).where(Puzzle.id == puzzle.id))
    await db.commit()
    return image_ids


async def test__rebuild__avoids_every_rejected_screenshot_and_the_title_stays_in_its_slot(
    env: Env,
) -> None:
    # One game with 14 screenshots: exactly one rebuild fits (decision 71).
    first = make_game(1, screenshots=14)
    other = make_game(2)
    igdb = FakeIgdb([first])
    await env.builder(igdb).run(SlotRequest(slots=range(1, 2)))
    answer_id = (await puzzle_in_slot(env.db, 1)).answer_title_id
    first_ids = await reject_as_rebuild(env.db, 1)

    # Another slot cannot draw the title that waits for slot 1 (AC-25).
    report = await env.builder(igdb).run(SlotRequest(slots=range(2, 3)))
    assert (report.outcomes[0].status, report.outcomes[0].detail) == (
        SlotStatus.FAILED, "no_candidate",
    )  # fmt: skip

    report = await env.builder(igdb).run(SlotRequest(rebuilds=True))
    assert report.outcomes[0].status == SlotStatus.BUILT and report.outcomes[0].is_rebuild
    rebuilt = await puzzle_in_slot(env.db, 1)
    assert rebuilt.answer_title_id == answer_id
    second_ids = await reject_as_rebuild(env.db, 1)
    assert not set(first_ids) & set(second_ids)  # AC-52: other screenshots, other crop

    # The union of both rows leaves nothing: not rebuildable, and the row stays.
    report = await env.builder(igdb).run(SlotRequest(rebuilds=True))
    assert (report.outcomes[0].status, report.outcomes[0].detail) == (
        SlotStatus.FAILED, "cannot_rebuild",
    )  # fmt: skip
    assert await count(env.db, Puzzle) == 0

    # Giving up turns both rows into one "never" and frees the slot.
    assert await give_up_slot(env.db, "game-guesser", 1) == 2
    rows = await list_exclusions(env.db, env.game_id)
    assert [(row.reason, row.build_slot) for row in rows] == [("never", None)]
    assert never_igdb_ids(rows) == {first.id}
    assert await give_up_slot(env.db, "game-guesser", 1) == 0  # nothing left to give up

    igdb = FakeIgdb([first, other])
    report = await env.builder(igdb).run(SlotRequest(slots=range(1, 2)))
    assert report.outcomes[0].status == SlotStatus.BUILT and not report.outcomes[0].is_rebuild
    new_answer = await env.db.get(Title, (await puzzle_in_slot(env.db, 1)).answer_title_id)
    assert new_answer is not None and new_answer.igdb_id == other.id


# Storage writer -----------------------------------------------------------------------------


def test__local_writer__stores_once_by_key_and_leaves_no_half_file(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"clue bytes")
    key = hashlib.sha256(b"clue bytes").hexdigest()
    writer = LocalMediaWriter(tmp_path / "storage")

    assert (writer.put(key, source, "image/webp"), writer.put(key, source, "image/webp")) == (
        True, False,
    )  # fmt: skip
    assert [p.name for p in (tmp_path / "storage").iterdir()] == [key]  # no temp file left
    assert (tmp_path / "storage" / key).read_bytes() == b"clue bytes"
    with pytest.raises(MediaWriteError):
        writer.put("../outside", source, "image/webp")
    # A write that fails half way leaves neither the key nor a temp file.
    broken_key = hashlib.sha256(b"never stored").hexdigest()
    with pytest.raises(OSError):
        writer.put(broken_key, tmp_path, "image/webp")  # a folder cannot be read as a file
    assert [p.name for p in (tmp_path / "storage").iterdir()] == [key]


def test__s3_writer__skips_a_key_that_exists_and_puts_a_private_object(tmp_path: Path) -> None:
    writer = S3MediaWriter(
        endpoint_url="https://s3.test",
        region="eu",
        bucket="b",
        access_key_id="a",
        secret_access_key="s",
    )
    stored: list[dict[str, Any]] = []

    class StubClient:
        def head_object(self, **kwargs: Any) -> None:
            if kwargs["Key"] not in {obj["Key"] for obj in stored}:
                raise ClientError({"Error": {"Code": "404"}}, "HeadObject")

        def put_object(self, **kwargs: Any) -> None:
            kwargs["Body"].read()
            stored.append(kwargs)

    writer.client = StubClient()
    source = tmp_path / "f"
    source.write_bytes(b"x")
    key = hashlib.sha256(b"x").hexdigest()

    assert (writer.put(key, source, "video/mp4"), writer.put(key, source, "video/mp4")) == (
        True, False,
    )  # fmt: skip
    assert len(stored) == 1
    assert stored[0]["ContentType"] == "video/mp4" and "ACL" not in stored[0]
