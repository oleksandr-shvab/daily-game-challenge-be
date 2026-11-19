"""Promote approved puzzles to another environment (SPEC-04 AC-2, AC-25, AC-44, AC-86; R-10, R-12).

Two real databases on the test server: the target is `db_session`, the source is a second database
(`<test database>_src`). Both sides are rolled back after each test. The files are the placeholder
media of `scripts/seed_media/`, stored under their SHA-256 in two `tmp_path` folders."""

import datetime as dt
import hashlib
import importlib.util
import shutil
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import asyncpg
import pytest
from PIL import Image
from sqlalchemy import func, select, text, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine

from app.content import promote as promote_module
from app.content.media import measure
from app.content.promote import (
    PromoteRefusedError,
    PromoteReport,
    PromoteStatus,
    SourceDatabaseFailure,
    promote,
    read_source,
    run_promote,
    source_session,
)
from app.content.target import TargetRefusedError
from app.core.config import settings
from app.core.db import Base
from app.crud.puzzle import ClueInput, MediaFileInput, add_built_puzzle
from app.crud.title import create_title
from app.games.enums import ClueRole, MediaType
from app.media.writer import LocalMediaWriter, MediaWriteError
from app.models.game import Game
from app.models.progress import Progress
from app.models.puzzle import ClueMediaFile, Puzzle, PuzzleClue, PuzzleStatus
from app.models.title import Title
from app.models.user import User
from tests.conftest import _maintenance_url_and_db_name
from tests.factories import create_game

BACKEND_ROOT = Path(__file__).resolve().parent.parent
SEED_MEDIA = BACKEND_ROOT / "scripts" / "seed_media"
APPROVED_AT = dt.datetime(2026, 11, 1, 12, 0, tzinfo=dt.UTC)
SOURCE_URL = settings.TEST_DATABASE_URL + "_src"
SOURCE_ANSWER = "Source Answer Name"
# A WebP header with nothing behind it: Pillow raises a plain `OSError` for it.
JUNK_WEBP = b"RIFF\x10\x00\x00\x00WEBPVP8 junk-junk-junk"


@pytest.fixture(scope="module")
async def source_engine() -> AsyncGenerator[AsyncEngine, None]:
    maintenance_url, db_name = _maintenance_url_and_db_name(SOURCE_URL)
    connection = await asyncpg.connect(maintenance_url)
    try:
        await connection.execute(f'CREATE DATABASE "{db_name}"')
    except asyncpg.DuplicateDatabaseError:
        pass
    engine = create_async_engine(SOURCE_URL)
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()
    # The throwaway database goes away with the run.
    await connection.execute(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)')
    await connection.close()


@pytest.fixture
async def source_db(source_engine: AsyncEngine) -> AsyncGenerator[AsyncSession, None]:
    async with source_engine.connect() as conn:
        trans = await conn.begin()
        session = AsyncSession(
            bind=conn, join_transaction_mode="create_savepoint", expire_on_commit=False
        )
        try:
            yield session
        finally:
            await session.close()
            await trans.rollback()


class CountingWriter:
    """The target's writer: records every call, and `put`s into a real folder."""

    kind = "local"

    def __init__(self, directory: Path) -> None:
        self.inner = LocalMediaWriter(directory)
        self.directory = directory
        self.puts: list[str] = []

    def exists(self, key: str) -> bool:
        return self.inner.exists(key)

    def put(self, key: str, path: Path, content_type: str) -> bool:
        self.puts.append(key)
        return self.inner.put(key, path, content_type)


@dataclass
class World:
    source_db: AsyncSession
    target_db: AsyncSession
    source_dir: Path
    writer: CountingWriter
    source_game: Game
    target_game: Game
    stored: dict[str, MediaFileInput]  # the seed files by name, as stored in the source folder
    own: dict[str, MediaFileInput]  # each puzzle's own step-5 file, by the puzzle's answer name

    def own_file(self, name: str) -> MediaFileInput:
        """A WebP only this puzzle has (the seed files are shared by every puzzle of the tests), so
        a test can break one puzzle's file and leave the others' alone."""
        path = self.source_dir / name
        Image.new("RGB", (640, 360), (sum(name.encode()) % 256, 40, 90)).save(path, "WEBP")
        key = hashlib.sha256(path.read_bytes()).hexdigest()
        path.rename(self.source_dir / key)
        facts = measure(self.source_dir / key)
        return MediaFileInput(
            ClueRole.SOURCE,
            key,
            facts.content_type,
            facts.byte_size,
            facts.width,
            facts.height,
            None,
        )

    async def add_source_puzzle(self, slot: int, igdb_id: int, name: str) -> None:
        """Steps 1 to 4 and 6 use the seed files; step 5 is the puzzle's own file."""
        title = await create_title(self.source_db, name=name, igdb_id=igdb_id)
        own = self.own_file(name)
        self.own[name] = own
        clues = [
            ClueInput(
                step=step,
                media_type=MediaType.VIDEO if step == 6 else MediaType.IMAGE,
                hint_kind=None if step == 3 else "year",
                hint_value=None if step == 3 else str(2000 + step),
                source_url=f"https://www.igdb.com/games/{name.lower().replace(' ', '-')}",
                source_image_ids=None if step == 3 else [f"img{step}a", f"img{step}b"],
                files=[own] if step == 5 else [self.stored[n] for n in files_of_step(step)],
            )
            for step in range(1, 7)
        ]
        await add_built_puzzle(
            self.source_db,
            game_id=self.source_game.id,
            variant="default",
            build_slot=slot,
            generator_version="1",
            answer_title_id=title.id,
            status=PuzzleStatus.APPROVED,
            clues=clues,
            approved_at=APPROVED_AT,
        )

    def files_of(self, name: str) -> list[MediaFileInput]:
        """The seven files of the puzzle named `name`."""
        shared = [file for file_name, file in self.stored.items() if file_name != "step-5.webp"]
        return [*shared, self.own[name]]

    async def add_target_title(self, igdb_id: int, name: str) -> Title:
        return await create_title(self.target_db, name=name, igdb_id=igdb_id)

    async def run(self) -> PromoteReport:
        # `promote` ends its reads with a rollback; the setup must be committed (it is not durable:
        # the fixture's outer transaction is rolled back).
        await self.target_db.commit()
        source = await read_source(self.source_db, "game-guesser")
        return await promote(
            self.target_db,
            source,
            slug="game-guesser",
            source_dir=self.source_dir,
            writer=self.writer,
        )

    async def count(self, model: type) -> int:
        result = await self.target_db.execute(select(func.count()).select_from(model))
        return result.scalar_one()


def files_of_step(step: int) -> list[str]:
    return ["clip.mp4", "poster.webp"] if step == 6 else [f"step-{step}.webp"]


@pytest.fixture
async def world(source_db: AsyncSession, db_session: AsyncSession, tmp_path: Path) -> World:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    stored = {}
    for path in sorted(SEED_MEDIA.glob("*.*")):
        if path.suffix not in (".webp", ".mp4"):
            continue
        key = hashlib.sha256(path.read_bytes()).hexdigest()
        shutil.copyfile(path, source_dir / key)
        facts = measure(source_dir / key)
        role = ClueRole.POSTER if path.name == "poster.webp" else ClueRole.SOURCE
        stored[path.name] = MediaFileInput(
            role, key, facts.content_type, facts.byte_size, facts.width, facts.height,
            facts.duration_ms,
        )  # fmt: skip
    return World(
        source_db,
        db_session,
        source_dir,
        CountingWriter(tmp_path / "target"),
        await create_game(source_db),
        await create_game(db_session),
        stored,
        {},
    )


async def media_rows(db: AsyncSession) -> list[tuple[Any, ...]]:
    result = await db.execute(
        select(
            Puzzle.build_slot,
            PuzzleClue.step,
            PuzzleClue.hint_kind,
            PuzzleClue.hint_value,
            PuzzleClue.source_url,
            PuzzleClue.source_image_ids,
            ClueMediaFile.role,
            ClueMediaFile.key,
            ClueMediaFile.content_type,
            ClueMediaFile.byte_size,
            ClueMediaFile.width,
            ClueMediaFile.height,
            ClueMediaFile.duration_ms,
        )
        .join(PuzzleClue, PuzzleClue.puzzle_id == Puzzle.id)
        .join(ClueMediaFile, ClueMediaFile.clue_id == PuzzleClue.id)
        .order_by(Puzzle.build_slot, PuzzleClue.step, ClueMediaFile.id)
    )
    return [tuple(row) for row in result.all()]


# (a) A puzzle is copied whole; a second run adds nothing ----------------------------------------


async def test__promote__copies_an_approved_puzzle_whole_and_a_second_run_adds_nothing(
    world: World,
) -> None:
    await world.add_source_puzzle(41, 10, SOURCE_ANSWER)
    title_id = (await world.add_target_title(10, "Target Name Stays")).id

    first = await world.run()

    assert [(o.build_slot, o.status) for o in first.outcomes] == [(41, PromoteStatus.PROMOTED)]
    puzzle = (await world.target_db.execute(select(Puzzle))).scalar_one()
    assert (puzzle.status, puzzle.date, puzzle.approved_at, puzzle.generator_version) == (
        PuzzleStatus.APPROVED,
        None,
        APPROVED_AT,
        "1",
    )
    assert puzzle.answer_title_id == title_id
    # Rows are the source's, step by step, file by file; every file is in the target's folder.
    assert await media_rows(world.target_db) == await media_rows(world.source_db)
    assert len(await media_rows(world.target_db)) == 7
    every_file = world.files_of(SOURCE_ANSWER)
    assert all(world.writer.exists(file.key) for file in every_file)
    assert sorted(world.writer.puts) == sorted(file.key for file in every_file)
    # An existing title is never changed (the target's name stays).
    names = (await world.target_db.execute(select(Title.name))).scalars().all()
    assert names == ["Target Name Stays"]
    # No answer in the lines unless asked for.
    assert SOURCE_ANSWER not in "\n".join(first.lines(show_answers=False))
    assert SOURCE_ANSWER in "\n".join(first.lines(show_answers=True))

    world.writer.puts.clear()
    again = await world.run()

    assert [(o.build_slot, o.status) for o in again.outcomes] == [(41, PromoteStatus.ALREADY)]
    assert world.writer.puts == []
    assert (await world.count(Puzzle), await world.count(PuzzleClue)) == (1, 6)
    assert await world.count(ClueMediaFile) == 7

    # The target lost its files: a re-run puts them back and leaves every row as it was.
    rows_before = await media_rows(world.target_db)
    for file in every_file:
        (world.writer.directory / file.key).unlink()
    repaired = await world.run()

    assert [(o.status, o.repaired) for o in repaired.outcomes] == [(PromoteStatus.ALREADY, 7)]
    assert "7 file(s) put back" in "\n".join(repaired.lines(show_answers=False))
    assert all(world.writer.exists(file.key) for file in every_file)
    assert await media_rows(world.target_db) == rows_before
    assert (await world.count(Puzzle), await world.count(ClueMediaFile)) == (1, 7)


# (b) Refusals: the puzzle is left out, the others go on ------------------------------------------


async def make_slot_taken_by_a_played_puzzle(world: World) -> None:
    other = await world.add_target_title(99, "Another Answer")
    await world.add_target_title(10, "Wanted")
    puzzle = await add_built_puzzle(
        world.target_db,
        game_id=world.target_game.id,
        variant="default",
        build_slot=41,
        generator_version="1",
        answer_title_id=other.id,
        status=PuzzleStatus.APPROVED,
        clues=[],
    )
    account = User(email="player@example.com", hashed_password="not-a-real-hash")
    world.target_db.add(account)
    await world.target_db.flush()
    world.target_db.add(
        Progress(
            user_id=account.id,
            puzzle_id=puzzle.id,
            status="in_progress",
            started_at=APPROVED_AT,
            counts_for_streak=True,
            origin="play",
        )
    )
    await world.target_db.flush()


async def make_answer_used(world: World) -> None:
    title = await world.add_target_title(10, "Wanted")
    await add_built_puzzle(
        world.target_db,
        game_id=world.target_game.id,
        variant="default",
        build_slot=7,
        generator_version="1",
        answer_title_id=title.id,
        status=PuzzleStatus.APPROVED,
        clues=[],
    )


async def make_file_missing(world: World) -> None:
    await world.add_target_title(10, "Wanted")
    (world.source_dir / world.own[SOURCE_ANSWER].key).unlink()


async def make_file_corrupt(world: World) -> None:
    # The same size and the same header, so `measure` and the row's facts agree; only the SHA-256
    # that is the key shows the bytes are not the ones the key names.
    await world.add_target_title(10, "Wanted")
    path = world.source_dir / world.own[SOURCE_ANSWER].key
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(bytes(data))


async def make_file_undecodable_and_hash_mismatch(world: World) -> None:
    # Junk under a key it does not match: refused on the hash, before `measure` can raise.
    await world.add_target_title(10, "Wanted")
    (world.source_dir / world.own[SOURCE_ANSWER].key).write_bytes(JUNK_WEBP)


async def make_file_undecodable_but_hash_matches(world: World) -> None:
    # The key is the junk's own SHA-256, so the hash passes and `measure` fails with a decoder
    # `OSError`: still a bad file, not a storage error that stops the run.
    await world.add_target_title(10, "Wanted")
    old_key = world.own[SOURCE_ANSWER].key
    new_key = hashlib.sha256(JUNK_WEBP).hexdigest()
    (world.source_dir / new_key).write_bytes(JUNK_WEBP)
    await world.source_db.execute(
        update(ClueMediaFile).where(ClueMediaFile.key == old_key).values(key=new_key)
    )


@pytest.mark.parametrize(
    ("prepare", "reason", "slot"),
    [
        (make_slot_taken_by_a_played_puzzle, "slot_has_another_answer", 41),
        (make_answer_used, "answer_already_in_target", 41),
        (make_file_missing, "not_ready (missing_file at step 5)", 41),
        (make_file_corrupt, "bad_file (at step 5)", 41),
        (make_file_undecodable_and_hash_mismatch, "bad_file (at step 5)", 41),
        (make_file_undecodable_but_hash_matches, "bad_file (at step 5)", 41),
        (None, "title_missing_in_target (run the title import on the target first)", 41),
    ],
    ids=[
        "played_slot_taken",
        "answer_used",
        "file_missing",
        "file_corrupt",
        "undecodable_hash_mismatch",
        "undecodable_hash_matches",
        "title_missing",
    ],
)
async def test__promote__refuses_the_puzzle_touches_nothing_and_goes_on_with_the_next(
    world: World, prepare: Callable[[World], Any] | None, reason: str, slot: int
) -> None:
    await world.add_source_puzzle(slot, 10, SOURCE_ANSWER)
    await world.add_source_puzzle(42, 11, "Fine")
    await world.add_target_title(11, "Fine")
    if prepare is not None:
        await prepare(world)
    before = (await world.count(Puzzle), await media_rows(world.target_db))
    other_before = (
        await world.target_db.execute(
            select(Puzzle.build_slot, Puzzle.answer_title_id, Puzzle.status, Puzzle.date)
            .where(Puzzle.build_slot != 42)
            .order_by(Puzzle.id)
        )
    ).all()

    report = await world.run()

    first, second = report.outcomes
    assert (first.build_slot, first.status, first.detail) == (slot, PromoteStatus.REFUSED, reason)
    assert (second.build_slot, second.status) == (42, PromoteStatus.PROMOTED)
    assert not report.is_ok
    # The refused puzzle changed nothing: only slot 42's puzzle and its files arrived.
    assert await world.count(Puzzle) == before[0] + 1
    assert [row for row in await media_rows(world.target_db) if row[0] != 42] == before[1]
    other_after = (
        await world.target_db.execute(
            select(Puzzle.build_slot, Puzzle.answer_title_id, Puzzle.status, Puzzle.date)
            .where(Puzzle.build_slot != 42)
            .order_by(Puzzle.id)
        )
    ).all()
    assert other_after == other_before


# (c) Files go first; a failing write rolls the rows back ---------------------------------------


async def test__promote__a_row_failure_rolls_back_the_puzzle_and_leaves_the_files(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    await world.add_source_puzzle(41, 10, SOURCE_ANSWER)
    await world.add_target_title(10, "Wanted")

    async def insert_then_fail(*args: Any, **kwargs: Any) -> None:
        await add_built_puzzle(*args, **kwargs)
        raise SQLAlchemyError("boom")

    monkeypatch.setattr(promote_module, "add_built_puzzle", insert_then_fail)
    report = PromoteReport()
    source = await read_source(world.source_db, "game-guesser")
    await world.target_db.commit()

    with pytest.raises(SQLAlchemyError):
        await promote(
            world.target_db,
            source,
            slug="game-guesser",
            source_dir=world.source_dir,
            writer=world.writer,
            report=report,
        )

    # The rows were flushed and then rolled back: no puzzle, clue or media row. The files, which go
    # first, are all there (a swapped order would leave none), and no outcome claims success.
    assert (await world.count(Puzzle), await world.count(PuzzleClue)) == (0, 0)
    assert await world.count(ClueMediaFile) == 0
    assert all(world.writer.exists(file.key) for file in world.files_of(SOURCE_ANSWER))
    assert report.outcomes == []


# (c2) Another writer takes the slot while the files go up; a storage read error stops the run -----


@pytest.mark.parametrize(
    ("rival_igdb_id", "expected"),
    [(10, PromoteStatus.ALREADY), (99, PromoteStatus.REFUSED), (None, None)],
    ids=["same_answer", "other_answer", "not_the_slot"],
)
async def test__promote__a_slot_taken_during_the_run_is_reported_and_the_run_goes_on(
    world: World,
    monkeypatch: pytest.MonkeyPatch,
    rival_igdb_id: int | None,
    expected: PromoteStatus | None,
) -> None:
    await world.add_source_puzzle(41, 10, SOURCE_ANSWER)
    await world.add_source_puzzle(42, 11, "Fine")
    wanted = await world.add_target_title(10, "Wanted")
    rival = await world.add_target_title(99, "Rival")
    await world.add_target_title(11, "Fine")
    wanted_id, rival_id = wanted.id, rival.id
    real_add = add_built_puzzle
    is_first = True

    async def add_after_a_rival_commits(db: AsyncSession, **kwargs: Any) -> Any:
        nonlocal is_first
        if kwargs["build_slot"] == 41 and is_first:
            is_first = False
            if rival_igdb_id is not None:
                # The rival is committed (to the test's outer transaction) before our insert.
                title_id = wanted_id if rival_igdb_id == 10 else rival_id
                await real_add(
                    db,
                    **{**kwargs, "answer_title_id": title_id, "clues": [], "approved_at": None},
                )
                await db.commit()
            else:
                # An integrity error that is not the slot's unique key.
                raise IntegrityError("INSERT", {}, Exception("some other constraint"))
        return await real_add(db, **kwargs)

    monkeypatch.setattr(promote_module, "add_built_puzzle", add_after_a_rival_commits)
    hashed: list[str] = []
    real_sha = promote_module.sha256_file

    def counting_sha(path: Path) -> str:
        hashed.append(path.name)
        return real_sha(path)

    monkeypatch.setattr(promote_module, "sha256_file", counting_sha)

    if expected is None:
        with pytest.raises(IntegrityError):
            await world.run()
        return
    report = await world.run()

    first, second = report.outcomes
    assert (first.build_slot, first.status) == (41, expected)
    assert (second.build_slot, second.status) == (42, PromoteStatus.PROMOTED)
    # Each file is read once for the whole run: the two puzzles share six of their seven.
    assert len(hashed) == len(set(hashed)) == 8
    # Our rows were rolled back: slot 41 holds the rival's empty puzzle only.
    assert await world.count(PuzzleClue) == 6  # slot 42's
    assert await world.count(Puzzle) == 2


async def test__promote__a_file_that_cannot_be_read_is_a_storage_error_not_a_bad_file(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    await world.add_source_puzzle(41, 10, SOURCE_ANSWER)
    await world.add_target_title(10, "Wanted")

    def unreadable(path: Path) -> str:
        raise PermissionError(13, "denied")

    monkeypatch.setattr(promote_module, "sha256_file", unreadable)

    with pytest.raises(MediaWriteError, match="PermissionError"):
        await world.run()


# (d) The run's own checks come before anything is read ---------------------------------------


@dataclass
class Setup:
    target: str = "staging"
    cleared_by: str | None = None
    kind: str = "local"
    source_url: str = "postgresql+asyncpg://postgres:x@localhost:1/none"
    target_url: str = "postgresql+asyncpg://postgres:x@localhost:2/none"


@pytest.mark.parametrize(
    ("setup", "error"),
    [
        (Setup(), TargetRefusedError),
        (Setup(target="local", kind="s3"), TargetRefusedError),
        (Setup(cleared_by="SPEC.md decision 72"), SourceDatabaseFailure),
        (Setup(target="local"), SourceDatabaseFailure),
        (Setup(cleared_by="x", source_url=settings.TEST_DATABASE_URL + "_missing"), SourceDatabaseFailure),
        (Setup(cleared_by="x", target_url="postgresql+asyncpg://u:y@localhost:1/none"), PromoteRefusedError),
        (Setup(cleared_by="x", target_url="postgresql+asyncpg://u:y@127.0.0.1:1/none"), PromoteRefusedError),
        (Setup(cleared_by="x", source_url="postgresql+asyncpg://u@db.example.com/none"), PromoteRefusedError),
    ],
    ids=["gate_closed", "gate_closed_for_a_bucket", "gate_open", "local_needs_no_gate", "no_such_database", "same_database", "same_database_other_spelling", "remote_source"],
)  # fmt: skip
async def test__run_promote__the_gate_and_the_setup_come_before_the_source_is_read(
    tmp_path: Path, setup: Setup, error: type[Exception]
) -> None:
    writer = CountingWriter(tmp_path / "target")
    writer.kind = setup.kind
    report = PromoteReport()

    with pytest.raises(error) as raised:
        await run_promote(
            slug="game-guesser",
            source_url=setup.source_url,
            source_dir=tmp_path,
            target_name=setup.target,
            target_url=setup.target_url,
            cleared_by=setup.cleared_by,
            writer=writer,
            report=report,
        )

    # No message holds the URL's password.
    assert "x@" not in str(raised.value) and ":y@" not in str(raised.value)
    assert (writer.puts, report.outcomes) == ([], [])


async def test__source_session__never_writes_to_the_source(source_engine: AsyncEngine) -> None:
    with pytest.raises(SourceDatabaseFailure):
        async with source_session(SOURCE_URL) as db:
            await db.execute(text("DELETE FROM games"))


# (e) The script: the guard and the gate show before anything else --------------------------------


def load_script() -> ModuleType:
    path = Path(__file__).resolve().parent.parent / "scripts" / "promote_puzzles.py"
    spec = importlib.util.spec_from_file_location("p3_promote_puzzles", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("argv", "environ", "message", "has_target_line"),
    [
        ([], {"ENVIRONMENT": "local"}, "--target is required", False),
        (["--target", "local"], {"ENVIRONMENT": "staging"}, "ENVIRONMENT variable is", False),
        (["--target", "staging"], {"ENVIRONMENT": "staging"}, "--confirm-target staging", False),
        (
            ["--target", "staging", "--confirm-target", "staging"],
            {"ENVIRONMENT": "staging"},
            "IGDB_IMAGES_CLEARED_BY",
            True,
        ),
    ],
    ids=["no_target", "wrong_environment", "no_confirmation", "gate_closed"],
)
def test__promote_script__guard_and_gate_refuse_with_exit_1(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    environ: dict[str, str],
    message: str,
    has_target_line: bool,
) -> None:
    for name in ("CI", "ENVIRONMENT", "IGDB_IMAGES_CLEARED_BY"):
        monkeypatch.delenv(name, raising=False)
    for name, value in environ.items():
        monkeypatch.setenv(name, value)
    source = ["--source-database-url", "postgresql+asyncpg://u:secret@localhost:1/x"]
    monkeypatch.setattr(
        "sys.argv",
        ["promote_puzzles.py", *argv, *source, "--source-media-dir", "var/p3_24c_relative"],
    )

    code = load_script().main()

    captured = capsys.readouterr()
    assert code == 1 and message in captured.err
    assert captured.out.startswith("Target: staging;") == has_target_line
    # A relative folder is read from the repo root, whatever the working directory is.
    assert (str(BACKEND_ROOT / "var/p3_24c_relative") in captured.out) == has_target_line
    assert "secret" not in captured.out + captured.err
