"""Promote approved puzzles to another environment (SPEC-04 AC-2, AC-25, AC-44, AC-86; R-10, R-12;
owner decisions 68 to 71).

Review stays in `local`. A promote run reads the `approved` puzzles of the source database (the
reviewed one, read-only) and copies each, rows and files, to the target as `approved` with no date.
Scheduling then runs on the target.

Rules, in the order a puzzle meets them:
- The IGDB image gate comes before anything is read or copied: a target other than `local`, or any
  bucket, is refused while `IGDB_IMAGES_CLEARED_BY` is empty (`run_promote`, AC-2).
- A puzzle needs a build slot and an answer with an IGDB id; those are its keys.
- The slot is the key of idempotence: the unique key `(game, variant, build_slot)`. A slot the
  target already has is never touched, so a re-run adds nothing, and a puzzle with stored games is
  never edited (AC-86), because promote only inserts. When the target's puzzle in the slot has the
  same answer the slot is reported as already promoted; with another answer it is a conflict and is
  refused.
- The answer's title must already be in the target, found by `igdb_id`. Promote never creates or
  changes a title: the target's list comes from its own title import (AC-10, AC-18), and a title
  made here from the source row would skip that import's near-duplicate check (AC-14). A missing
  title is refused with "run the title import on the target first".
- An answer the target already has, in any slot, status or variant, is refused (AC-25).
- The puzzle must be ready in the source (`puzzle_problems`: six clues, the AC-13b rule, every file
  there), and each file must be what its row says: its SHA-256 is its key, and `measure` gives the
  row's facts and meets the limits.
- Files go first (content-addressed and idempotent), then the rows of one puzzle in one
  transaction, rolled back on any failure. A failure stops the run; puzzles finished before it stay.

`never` and `rebuild` exclusions do not travel: the target does not build (review is local, and
the builder refuses the target while the gate is closed), so nothing there reads them, and scheduling
guards a repeated answer by itself. The target keeps its own `approved_at` order from the source's.

Library code: it returns a `PromoteReport` and the script prints it. No answer is in the report's
lines unless asked for.
"""

import asyncio
import contextlib
import datetime as dt
import hashlib
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from av.error import FFmpegError
from PIL import Image, UnidentifiedImageError
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.content.builder import DatabaseFailure, UnknownGameError, database_errors, database_session
from app.content.media import MediaFacts, MediaLimitError, broken_limits, measure
from app.content.readiness import puzzle_problems
from app.content.target import LOCAL_HOSTS, check_image_gate
from app.crud.game import get_game_by_slug
from app.crud.puzzle import (
    ClueInput,
    MediaFileInput,
    add_built_puzzle,
    list_answer_igdb_ids,
    list_approved_puzzles,
    list_slot_answers,
)
from app.crud.title import get_titles_by_igdb_ids
from app.media.writer import LocalMediaWriter, MediaWriteError, MediaWriter
from app.models.puzzle import Puzzle, PuzzleStatus

HASH_CHUNK_BYTES = 1024 * 1024
# A file that is not what it claims to be: this is a refusal of the puzzle, not a failed run. A plain
# `OSError` (the file cannot be read) is a storage error, so `UnidentifiedImageError`, which is an
# `OSError` too, is named first.
BAD_FILE_ERRORS = (
    MediaLimitError,
    FFmpegError,
    Image.DecompressionBombError,
    UnidentifiedImageError,
)


class PromoteRefusedError(Exception):
    """The run cannot start: the setup is wrong. The message says why and holds no credential."""


class SourceDatabaseFailure(DatabaseFailure):
    """A call to the source database failed. The target's failures are plain `DatabaseFailure`s,
    named by `database_session`."""


class PromoteStatus(StrEnum):
    PROMOTED = "promoted"
    ALREADY = "already_promoted"
    REFUSED = "refused"


class RefusalReason(StrEnum):
    NO_SLOT = "no_build_slot"
    NO_IGDB_ID = "answer_has_no_igdb_id"
    SLOT_TAKEN = "slot_has_another_answer"
    ANSWER_USED = "answer_already_in_target"
    TITLE_MISSING = "title_missing_in_target"
    NOT_READY = "not_ready"
    BAD_FILE = "bad_file"


@dataclass(frozen=True)
class SourceClue:
    """A clue read from the source. `media_files` is what readiness and `clue_layout_problem`
    read."""

    step: int
    media_type: str
    hint_kind: str | None
    hint_value: str | None
    source_url: str | None
    source_image_ids: list[str]
    media_files: tuple[MediaFileInput, ...]


@dataclass(frozen=True)
class SourcePuzzle:
    build_slot: int | None
    generator_version: str | None
    approved_at: dt.datetime | None
    answer_igdb_id: int | None
    answer_name: str
    clues: tuple[SourceClue, ...]


@dataclass(frozen=True)
class SourceGame:
    variant: str
    puzzles: tuple[SourcePuzzle, ...]


@dataclass
class PuzzleOutcome:
    build_slot: int | None
    status: PromoteStatus
    detail: str = ""
    answer_name: str = ""

    def line(self, *, show_answers: bool) -> str:
        slot = "no slot" if self.build_slot is None else f"slot {self.build_slot}"
        answer = f" ({self.answer_name})" if show_answers and self.answer_name else ""
        if self.status == PromoteStatus.PROMOTED:
            return f"promoted {slot}{answer}"
        if self.status == PromoteStatus.ALREADY:
            return f"already in the target, left alone: {slot}{answer}"
        return f"refused {slot}{answer}: {self.detail}"


@dataclass
class PromoteReport:
    outcomes: list[PuzzleOutcome] = field(default_factory=list)

    @property
    def has_content(self) -> bool:
        return bool(self.outcomes)

    @property
    def is_ok(self) -> bool:
        return all(outcome.status != PromoteStatus.REFUSED for outcome in self.outcomes)

    def count(self, status: PromoteStatus) -> int:
        return sum(outcome.status == status for outcome in self.outcomes)

    def lines(self, *, show_answers: bool) -> list[str]:
        lines = [outcome.line(show_answers=show_answers) for outcome in self.outcomes]
        lines.append(
            f"promoted {self.count(PromoteStatus.PROMOTED)},"
            f" already in the target {self.count(PromoteStatus.ALREADY)},"
            f" refused {self.count(PromoteStatus.REFUSED)}"
        )
        return lines


# ---- The source: read-only, and its own failures named ----


@contextlib.asynccontextmanager
async def source_session(url: str) -> AsyncIterator[AsyncSession]:
    """A read-only session on the source database, for its whole lifetime. Its failures are
    `SourceDatabaseFailure`, so the report can tell the two databases apart."""
    engine = None
    try:
        with database_errors():
            engine = create_async_engine(url).execution_options(postgresql_readonly=True)
            async with AsyncSession(engine, expire_on_commit=False) as session:
                yield session
    except DatabaseFailure as error:
        raise SourceDatabaseFailure(str(error)) from error.__cause__
    finally:
        if engine is not None:
            with contextlib.suppress(Exception):
                await engine.dispose()


async def read_source(db: AsyncSession, slug: str) -> SourceGame:
    """The game's approved puzzles as plain data, in slot order (a puzzle with no slot last)."""
    game = await get_game_by_slug(db, slug)
    if game is None:
        raise UnknownGameError(f"no game with the slug {slug!r} in the source database")
    rows = await list_approved_puzzles(db, game.id, game.default_variant)
    puzzles = [_source_puzzle(row) for row in rows]
    puzzles.sort(key=lambda puzzle: (puzzle.build_slot is None, puzzle.build_slot or 0))
    return SourceGame(game.default_variant, tuple(puzzles))


def _source_puzzle(row: Puzzle) -> SourcePuzzle:
    return SourcePuzzle(
        build_slot=row.build_slot,
        generator_version=row.generator_version,
        approved_at=row.approved_at,
        answer_igdb_id=row.answer_title.igdb_id,
        answer_name=row.answer_title.name,
        clues=tuple(
            SourceClue(
                step=clue.step,
                media_type=clue.media_type,
                hint_kind=clue.hint_kind,
                hint_value=clue.hint_value,
                source_url=clue.source_url,
                source_image_ids=list(clue.source_image_ids or []),
                media_files=tuple(
                    MediaFileInput(
                        role=file.role,
                        key=file.key,
                        content_type=file.content_type,
                        byte_size=file.byte_size,
                        width=file.width,
                        height=file.height,
                        duration_ms=file.duration_ms,
                    )
                    for file in clue.media_files
                ),
            )
            for clue in sorted(row.clues, key=lambda clue: clue.step)
        ),
    )


# ---- Checking and copying files ----


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def bad_file_step(puzzle: SourcePuzzle, source_dir: Path) -> int | None:
    """The first step whose file is not what its row says, or None. The key is the SHA-256 of the
    bytes (AC-41), `measure` reads the real facts from the bytes (AC-39), and they must be the
    row's and meet the limits. Blocking: call it in a thread. A file that cannot be read raises
    `OSError`."""
    for clue in puzzle.clues:
        for file in clue.media_files:
            path = source_dir / file.key
            try:
                facts = measure(path)
            except BAD_FILE_ERRORS:
                return clue.step
            row_facts = MediaFacts(
                file.content_type, file.byte_size, file.width, file.height, file.duration_ms
            )
            if _sha256(path) != file.key or facts != row_facts or broken_limits(facts):
                return clue.step
    return None


def _clue_inputs(puzzle: SourcePuzzle) -> list[ClueInput]:
    return [
        ClueInput(
            step=clue.step,
            media_type=clue.media_type,
            hint_kind=clue.hint_kind,
            hint_value=clue.hint_value,
            source_url=clue.source_url,
            source_image_ids=clue.source_image_ids,
            files=clue.media_files,
        )
        for clue in puzzle.clues
    ]


async def _storage[T, **P](call: Callable[P, T], *args: P.args) -> T:
    """A blocking storage or file call in a thread. An `OSError` becomes a `MediaWriteError`
    here, before the target session's wrapper could take it for a database error."""
    try:
        return await asyncio.to_thread(call, *args)
    except OSError as error:
        raise MediaWriteError(f"a file cannot be read or stored: {type(error).__name__}") from error


# ---- The promote run ----


async def promote(
    db: AsyncSession,
    source: SourceGame,
    *,
    slug: str,
    source_dir: Path,
    writer: MediaWriter,
    report: PromoteReport | None = None,
) -> PromoteReport:
    """Copy the source's puzzles to the target `db`. Pass a `report` to keep what a run that
    stops half way has finished: outcomes are added to it puzzle by puzzle. The caller has run
    the target guard and the image gate (`run_promote` does)."""
    report = PromoteReport() if report is None else report
    game = await get_game_by_slug(db, slug)
    if game is None:
        raise UnknownGameError(
            f"no game with the slug {slug!r} in the target database: run scripts/manage_game.py"
        )
    if game.default_variant != source.variant:
        raise PromoteRefusedError("the target's default variant is not the source's")
    game_id, variant = game.id, game.default_variant
    igdb_ids = {p.answer_igdb_id for p in source.puzzles if p.answer_igdb_id is not None}
    title_ids = {
        igdb: title.id for igdb, title in (await get_titles_by_igdb_ids(db, igdb_ids)).items()
    }
    slot_answers = await list_slot_answers(db, game_id, variant)
    used = await list_answer_igdb_ids(db, game_id)
    await db.rollback()  # No transaction stays open through the file work.
    source_files = LocalMediaWriter(source_dir)

    def refuse(puzzle: SourcePuzzle, reason: RefusalReason, extra: str = "") -> PuzzleOutcome:
        detail = f"{reason}{extra}"
        return PuzzleOutcome(puzzle.build_slot, PromoteStatus.REFUSED, detail, puzzle.answer_name)

    for puzzle in source.puzzles:
        slot, igdb_id = puzzle.build_slot, puzzle.answer_igdb_id
        done = PuzzleOutcome(slot, PromoteStatus.PROMOTED, answer_name=puzzle.answer_name)
        if slot is None:
            done = refuse(puzzle, RefusalReason.NO_SLOT)
        elif igdb_id is None:
            done = refuse(puzzle, RefusalReason.NO_IGDB_ID)
        elif slot in slot_answers:
            done.status = PromoteStatus.ALREADY
            if slot_answers[slot] != igdb_id:
                done = refuse(puzzle, RefusalReason.SLOT_TAKEN)
        elif igdb_id in used:
            done = refuse(puzzle, RefusalReason.ANSWER_USED)
        elif igdb_id not in title_ids:
            done = refuse(
                puzzle, RefusalReason.TITLE_MISSING, " (run the title import on the target first)"
            )
        else:
            problems = await _storage(
                lambda puzzle=puzzle: puzzle_problems(puzzle.clues, file_exists=source_files.exists)
            )
            bad_step = None if problems else await _storage(bad_file_step, puzzle, source_dir)
            if problems:
                steps = ", ".join(map(str, problems))
                done = refuse(puzzle, RefusalReason.NOT_READY, f" ({steps})")
            elif bad_step is not None:
                done = refuse(puzzle, RefusalReason.BAD_FILE, f" (at step {bad_step})")
            else:
                await _write(db, puzzle, game_id, variant, title_ids[igdb_id], source_dir, writer)
                slot_answers[slot] = igdb_id
                used.add(igdb_id)
        report.outcomes.append(done)
    return report


async def _write(
    db: AsyncSession,
    puzzle: SourcePuzzle,
    game_id: int,
    variant: str,
    title_id: int,
    source_dir: Path,
    writer: MediaWriter,
) -> None:
    """Files first, then the rows of one puzzle in one transaction: it exists whole or not at all.
    A failure rolls the rows back and stops the run; the files stay (non-goal: clean-up)."""
    assert puzzle.build_slot is not None
    keys = {file.key: file for clue in puzzle.clues for file in clue.media_files}
    for key, file in keys.items():
        await _storage(writer.put, key, source_dir / key, file.content_type)
    try:
        await add_built_puzzle(
            db,
            game_id=game_id,
            variant=variant,
            build_slot=puzzle.build_slot,
            generator_version=puzzle.generator_version,
            answer_title_id=title_id,
            status=PuzzleStatus.APPROVED,
            clues=_clue_inputs(puzzle),
            approved_at=puzzle.approved_at,
        )
        await db.commit()
    except BaseException:
        # A failing rollback (the database is gone) never replaces the original error.
        with contextlib.suppress(Exception):
            await db.rollback()
        raise


def _same_database(first: str, second: str) -> bool:
    one, two = make_url(first), make_url(second)
    return (one.host, one.port or 5432, one.database) == (two.host, two.port or 5432, two.database)


async def run_promote(
    *,
    slug: str,
    source_url: str,
    source_dir: Path,
    target_name: str,
    target_url: str,
    cleared_by: str | None,
    writer: MediaWriter,
    report: PromoteReport,
) -> None:
    """One run. The IGDB image gate comes first, before the source is opened or a file is read
    (R-12, AC-2), and checks the storage the files go to as well as the target's name. The source
    is read once, then closed; the target session lives to the end, and its failures are named by
    `database_session` and the source's by `source_session`."""
    check_image_gate(target_name, cleared_by, writer.kind)
    try:
        parsed = make_url(source_url)
    except ArgumentError:
        # The message of the library would hold the URL, and the URL a password.
        raise PromoteRefusedError("the source database URL cannot be read") from None
    if parsed.drivername != "postgresql+asyncpg":
        raise PromoteRefusedError("the source database URL must start with postgresql+asyncpg://")
    if parsed.host not in LOCAL_HOSTS:
        raise PromoteRefusedError("the source database must be on this machine: review is local")
    if _same_database(source_url, target_url):
        raise PromoteRefusedError("the source and the target are the same database")
    if not source_dir.is_dir():
        raise PromoteRefusedError("--source-media-dir is not a folder")
    async with source_session(source_url) as source_db:
        source = await read_source(source_db, slug)
    async with database_session() as db:
        await promote(db, source, slug=slug, source_dir=source_dir, writer=writer, report=report)
