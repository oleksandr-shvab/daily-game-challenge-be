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

import contextlib
import datetime as dt
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from av.error import FFmpegError
from PIL import Image
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError, IntegrityError
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
    list_slot_media_keys,
)
from app.crud.title import get_titles_by_igdb_ids
from app.media.hashing import sha256_file
from app.media.writer import LocalMediaWriter, MediaWriteError, MediaWriter, storage_call
from app.models.puzzle import Puzzle, PuzzleStatus


class PromoteRefusedError(Exception):
    """The run cannot start: the setup is wrong. The message says why and holds no credential."""


class SlotTakenError(Exception):
    """Another writer took the slot between the plan and the insert. `igdb_id` is the IGDB id of
    the answer it holds (None for a title that did not come from IGDB)."""

    def __init__(self, igdb_id: int | None) -> None:
        super().__init__("the slot was taken while the puzzle was being written")
        self.igdb_id = igdb_id


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
    source_image_ids: list[str] | None  # NULL stays NULL
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
    # Files of an already promoted puzzle that the target had lost and this run put back.
    repaired: int = 0
    # Files the target has lost that the source does not have either (a rebuilt source).
    lost: int = 0
    # The bucket cannot say what it has (a 403 on HEAD): repair was skipped.
    is_unchecked: bool = False

    def line(self, *, show_answers: bool) -> str:
        slot = "no slot" if self.build_slot is None else f"slot {self.build_slot}"
        answer = f" ({self.answer_name})" if show_answers and self.answer_name else ""
        if self.status == PromoteStatus.PROMOTED:
            return f"promoted {slot}{answer}"
        if self.status == PromoteStatus.ALREADY:
            notes = ""
            if self.repaired:
                notes += f"; {self.repaired} file(s) put back"
            if self.lost:
                notes += f"; {self.lost} file(s) lost, not in the source"
            if self.is_unchecked:
                notes += "; this bucket does not let us check, repair skipped"
            return f"already in the target, rows left alone: {slot}{answer}{notes}"
        return f"refused {slot}{answer}: {self.detail}"


@dataclass
class PromoteReport:
    outcomes: list[PuzzleOutcome] = field(default_factory=list)

    @property
    def has_content(self) -> bool:
        return bool(self.outcomes)

    @property
    def is_ok(self) -> bool:
        """False (exit code 1) when a puzzle was refused, or when a file the target lost cannot
        be put back because the source has no file with that key (decision 72)."""
        return self.lost_files == 0 and all(
            outcome.status != PromoteStatus.REFUSED for outcome in self.outcomes
        )

    @property
    def lost_files(self) -> int:
        return sum(outcome.lost for outcome in self.outcomes)

    def count(self, status: PromoteStatus) -> int:
        return sum(outcome.status == status for outcome in self.outcomes)

    def lines(self, *, show_answers: bool) -> list[str]:
        lines = [outcome.line(show_answers=show_answers) for outcome in self.outcomes]
        lines.append(
            f"promoted {self.count(PromoteStatus.PROMOTED)},"
            f" already in the target {self.count(PromoteStatus.ALREADY)},"
            f" refused {self.count(PromoteStatus.REFUSED)},"
            f" files put back {sum(outcome.repaired for outcome in self.outcomes)},"
            f" files lost and not in the source {self.lost_files}"
        )
        if self.lost_files:
            lines.append(
                "the run fails (exit 1): the target lacks files that the source no longer has"
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
                source_image_ids=clue.source_image_ids,
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

# What a file really is, by key: its facts, or None when its bytes are not what the key names or
# are not a readable WebP or MP4. One run reads each file once, however many clues name it.
Verified = dict[str, MediaFacts | None]


def _measured(path: Path, key: str) -> MediaFacts | None:
    """Hash first: bytes that are not the ones the key names are bad whatever they decode to.
    `measure` then reads the facts from the bytes (AC-39). Only content errors make the file bad:
    `MediaLimitError`, `ValueError`, a Pillow or PyAV decode error, and an `OSError` that carries
    no `errno` (a decoder's, not the operating system's). An `OSError` with an `errno` is a read
    error and propagates (a storage error). Anything else is a bug or the environment and stops
    the run as a `MediaWriteError` naming the error class; nothing is cached for it. Blocking."""
    if sha256_file(path) != key:
        return None
    try:
        return measure(path)
    except OSError as error:
        if error.errno is None:
            return None
        raise
    except (MediaLimitError, ValueError, FFmpegError, Image.DecompressionBombError):
        return None
    except Exception as error:
        raise MediaWriteError(
            f"a source file could not be checked: {type(error).__name__}"
        ) from error


def bad_file_step(puzzle: SourcePuzzle, source_dir: Path, verified: Verified) -> int | None:
    """The first step whose file is not what its row says, or None: the facts must be the row's
    and meet the limits. Blocking: call it in a thread. A file that cannot be read raises
    `OSError`."""
    for clue in puzzle.clues:
        for file in clue.media_files:
            if file.key not in verified:
                verified[file.key] = _measured(source_dir / file.key, file.key)
            facts = verified[file.key]
            row_facts = MediaFacts(
                file.content_type, file.byte_size, file.width, file.height, file.duration_ms
            )
            if facts is None or facts != row_facts or broken_limits(facts):
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


def _file_types(puzzle: SourcePuzzle) -> dict[str, str]:
    """The puzzle's distinct files: key to content type."""
    return {file.key: file.content_type for clue in puzzle.clues for file in clue.media_files}


def _source_file_with_key(source_dir: Path, key: str) -> Path | None:
    """The file of the source folder named `key`, when its bytes really have that SHA-256. Blocking."""
    path = source_dir / key
    return path if path.is_file() and sha256_file(path) == key else None


@dataclass
class _Run:
    """What one promote run knows while it walks the source's puzzles."""

    db: AsyncSession
    game_id: int
    variant: str
    source_dir: Path
    writer: MediaWriter
    title_ids: dict[int, int]
    slot_answers: dict[int, int | None]
    # The media keys the target's own rows name, by slot, each with its content type.
    slot_files: dict[int, dict[str, str]]
    used: set[int]
    verified: Verified = field(default_factory=dict)

    async def source_problem(self, puzzle: SourcePuzzle) -> tuple[RefusalReason, str] | None:
        """Why the puzzle's source cannot be copied: not ready, or a file that is not what its
        row says. None when it can."""
        source_files = LocalMediaWriter(self.source_dir)
        problems = await storage_call(
            "a source file cannot be read",
            puzzle_problems,
            puzzle.clues,
            file_exists=source_files.exists,
        )
        if problems:
            return RefusalReason.NOT_READY, f" ({', '.join(map(str, problems))})"
        step = await storage_call(
            "a source file cannot be read", bad_file_step, puzzle, self.source_dir, self.verified
        )
        return None if step is None else (RefusalReason.BAD_FILE, f" (at step {step})")

    async def put_files(self, files: dict[str, str]) -> int:
        """Store these files (key to content type) from the source folder in the target; returns
        how many were written. `put` does nothing for a key that is already there."""
        written = 0
        for key, content_type in files.items():
            written += await storage_call(
                "a file cannot be read or stored",
                self.writer.put,
                key,
                self.source_dir / key,
                content_type,
            )
        return written

    async def repair(self, puzzle: SourcePuzzle) -> PuzzleOutcome:
        """The slot has this answer already, so it is "already in the target" whatever its source
        looks like now (a rebuilt or broken source copy never refuses a healthy slot), and its rows
        stay as they are (AC-86). Files the target's own rows name and the store has lost are put
        back from a source file whose SHA-256 is that key; one the source lacks is counted."""
        assert puzzle.build_slot is not None
        done = PuzzleOutcome(
            puzzle.build_slot, PromoteStatus.ALREADY, answer_name=puzzle.answer_name
        )
        types = self.slot_files.get(puzzle.build_slot, {})
        lost: dict[str, str] = {}
        for key, content_type in types.items():
            known = await storage_call(
                "a file cannot be read or stored", self.writer.exists_or_unknown, key
            )
            if known is None:
                done.is_unchecked = True
                return done
            if not known:
                lost[key] = content_type
        restorable: dict[str, str] = {}
        for key, content_type in lost.items():
            path = await storage_call(
                "a source file cannot be read", _source_file_with_key, self.source_dir, key
            )
            if path is None:
                done.lost += 1
            else:
                restorable[key] = content_type
        done.repaired = await self.put_files(restorable)
        return done

    async def write(self, puzzle: SourcePuzzle, title_id: int) -> None:
        """Files first, then the rows of one puzzle in one transaction: it exists whole or not at
        all. A failure rolls the rows back and stops the run; the files stay (non-goal:
        clean-up). The one failure that does not stop it: another writer took the slot while the
        files were going up, which raises `SlotTakenError`."""
        assert puzzle.build_slot is not None
        await self.put_files(_file_types(puzzle))
        try:
            await add_built_puzzle(
                self.db,
                game_id=self.game_id,
                variant=self.variant,
                build_slot=puzzle.build_slot,
                generator_version=puzzle.generator_version,
                answer_title_id=title_id,
                status=PuzzleStatus.APPROVED,
                clues=_clue_inputs(puzzle),
                approved_at=puzzle.approved_at,
            )
            await self.db.commit()
        except IntegrityError:
            await self._rollback()
            try:
                now_there = await list_slot_answers(self.db, self.game_id, self.variant)
            finally:
                await self._rollback()  # Also when the read fails: it ends its transaction.
            if puzzle.build_slot not in now_there:
                raise  # Not the slot's unique key: something else is wrong.
            raise SlotTakenError(now_there[puzzle.build_slot]) from None
        except BaseException:
            await self._rollback()
            raise

    async def _rollback(self) -> None:
        # A failing rollback (the database is gone) never replaces the original error.
        with contextlib.suppress(Exception):
            await self.db.rollback()


def _refused(puzzle: SourcePuzzle, reason: RefusalReason, extra: str = "") -> PuzzleOutcome:
    detail = f"{reason}{extra}"
    return PuzzleOutcome(puzzle.build_slot, PromoteStatus.REFUSED, detail, puzzle.answer_name)


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
    igdb_ids = {p.answer_igdb_id for p in source.puzzles if p.answer_igdb_id is not None}
    title_ids = {
        igdb: title.id for igdb, title in (await get_titles_by_igdb_ids(db, igdb_ids)).items()
    }
    run = _Run(
        db=db,
        game_id=game.id,
        variant=game.default_variant,
        source_dir=source_dir,
        writer=writer,
        title_ids=title_ids,
        slot_answers=await list_slot_answers(db, game.id, game.default_variant),
        slot_files=await list_slot_media_keys(db, game.id, game.default_variant),
        used=await list_answer_igdb_ids(db, game.id),
    )
    await db.rollback()  # No transaction stays open through the file work.
    for puzzle in source.puzzles:
        report.outcomes.append(await _promote_one(run, puzzle))
    return report


async def _promote_one(run: _Run, puzzle: SourcePuzzle) -> PuzzleOutcome:
    slot, igdb_id = puzzle.build_slot, puzzle.answer_igdb_id
    if slot is None:
        return _refused(puzzle, RefusalReason.NO_SLOT)
    if igdb_id is None:
        return _refused(puzzle, RefusalReason.NO_IGDB_ID)
    if slot in run.slot_answers:
        # The target has a puzzle in the slot: the same answer is "already promoted" (its files
        # are repaired, its rows never touched), another answer is a conflict.
        if run.slot_answers[slot] != igdb_id:
            return _refused(puzzle, RefusalReason.SLOT_TAKEN)
        return await run.repair(puzzle)
    if igdb_id in run.used:
        return _refused(puzzle, RefusalReason.ANSWER_USED)
    if igdb_id not in run.title_ids:
        return _refused(
            puzzle, RefusalReason.TITLE_MISSING, " (run the title import on the target first)"
        )
    problem = await run.source_problem(puzzle)
    if problem is not None:
        return _refused(puzzle, *problem)
    try:
        await run.write(puzzle, run.title_ids[igdb_id])
    except SlotTakenError as taken:
        # Another writer took the slot while the files went up. Its answer is used from now on.
        run.slot_answers[slot] = taken.igdb_id
        if taken.igdb_id is not None:
            run.used.add(taken.igdb_id)
        if taken.igdb_id != igdb_id:
            return _refused(puzzle, RefusalReason.SLOT_TAKEN)
        return PuzzleOutcome(slot, PromoteStatus.ALREADY, answer_name=puzzle.answer_name)
    run.slot_answers[slot] = igdb_id
    run.used.add(igdb_id)
    return PuzzleOutcome(slot, PromoteStatus.PROMOTED, answer_name=puzzle.answer_name)


def _same_database(first: str, second: str) -> bool:
    """The same server and database name. Every spelling of this machine is one host, so
    `localhost` and `127.0.0.1` on one port are the same server."""

    def where(url: str) -> tuple[str, int, str | None]:
        parsed = make_url(url)
        host = LOCAL_HOSTS[0] if parsed.host in (None, *LOCAL_HOSTS) else parsed.host
        return host, parsed.port or 5432, parsed.database

    return where(first) == where(second)


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
