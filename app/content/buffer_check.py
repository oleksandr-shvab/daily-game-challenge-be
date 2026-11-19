"""The buffer and tomorrow check (SPEC-04 AC-73 to AC-82; owner decisions 68 to 71).

For every enabled game, in its default variant, it says whether the first day to be played is ready
and how many days in a row are ready from there. It only reads (AC-78): no puzzle, file or status is
written, so it needs no target guard and may run against production.

- The first day is `first_schedulable_day`: the later of tomorrow and the launch date (AC-82). It is
  ready when a puzzle dated that day exists, is `scheduled` and has no readiness problem (AC-74:
  six clues, the media rule of AC-13b, every file present in storage; `puzzle_problems`).
- The buffer is the ready days in a row from the first day, counted up to `BUFFER_CAP` (AC-75). It is
  `ready_buffer` of `schedule.py`, the function `schedule_puzzles.py` reports with, so both give the
  same number on the same data.
- Until `LAUNCH_WINDOW_DAYS` days before the launch date a game is "not launched": reported and never
  a failure (AC-82).
- "Today" is the clock the caller passes (AC-81).

Exit codes (`ExitCode`; the schedule and the alert that read them are M1-52): 2 when the first day is
not ready for any launched game (page), else 1 when any launched game's buffer is under
`MIN_BUFFER_DAYS` (digest), else 0. 3 is "could not check" (database or storage down, a usage error,
a crash): Python's own 1 and 2 never mean "buffer low" or "tomorrow missing".

The output goes to alert channels (AC-79): game slug, dates, problem kinds with step numbers and the
buffer. Never a title, an answer, a hint, a media key or a credential. Library code: it returns data;
`scripts/check_puzzle_buffer.py` prints it. It imports nothing that needs the content libraries
(Pillow, PyAV, httpx), so a server with runtime dependencies only can run it.
"""

import contextlib
import datetime as dt
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from urllib.parse import urlsplit

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.content.readiness import Problem
from app.content.schedule import (
    BUFFER_CAP,
    ONE_DAY,
    first_schedulable_day,
    readiness_problems,
    ready_buffer,
)
from app.content.target import _storage_description
from app.core.clock import utc_today
from app.core.config import Settings
from app.crud.game import list_enabled_games
from app.crud.puzzle import get_puzzle_on_date
from app.media.writer import MediaWriteError
from app.models.game import Game
from app.models.puzzle import PuzzleStatus

# Under this many ready days in a row the check exits 1 (AC-77).
MIN_BUFFER_DAYS = 7
# A game is "not launched" until this many days before its launch date (AC-82).
LAUNCH_WINDOW_DAYS = 7


class ExitCode(IntEnum):
    OK = 0
    BUFFER_LOW = 1
    TOMORROW_NOT_READY = 2
    CANNOT_CHECK = 3


class DayState(StrEnum):
    """The state of a game's first day. `NOT_READY` means the puzzle exists and is scheduled but has
    `Problem`s."""

    READY = "ready"
    MISSING = "missing"
    WRONG_STATUS = "wrong_status"
    NOT_READY = "not_ready"
    NOT_LAUNCHED = "not_launched"


@dataclass(frozen=True)
class GameCheck:
    slug: str
    first_day: dt.date
    state: DayState
    problems: list[Problem] = field(default_factory=list)
    buffer: int = 0


class CheckUnavailableError(Exception):
    """The check could not run (exit 3). The message names the cause and holds no credential."""


class DatabaseUnavailableError(CheckUnavailableError):
    pass


class StorageUnavailableError(CheckUnavailableError):
    pass


@contextlib.contextmanager
def database_errors() -> Iterator[None]:
    """Name an `SQLAlchemyError` or `OSError` raised in the block a `DatabaseUnavailableError` (the
    message is the error's type only: no statement, no value). An unreachable Postgres raises
    `ConnectionRefusedError`, an `OSError`. Storage errors reach here as `MediaWriteError`
    (`readiness_problems` turns an `OSError` from the storage into one), so they pass through."""
    try:
        yield
    except (SQLAlchemyError, OSError) as error:
        raise DatabaseUnavailableError(f"database error: {type(error).__name__}") from error


@contextlib.asynccontextmanager
async def database_session() -> AsyncIterator[AsyncSession]:
    """The session of a check run: opening it, every call made with it and closing it are
    classified by `database_errors`. A check never commits."""
    from app.core import db as core_db  # Looked up at call time, so a test can replace the maker.

    try:
        with database_errors():
            async with core_db.async_session_maker() as session:
                yield session
    finally:
        with contextlib.suppress(Exception):
            await core_db.engine.dispose()


async def check_game(
    db: AsyncSession, game: Game, file_exists: Callable[[str], bool], *, today: dt.date
) -> GameCheck:
    first = first_schedulable_day(game, today)
    if today < game.launch_date - LAUNCH_WINDOW_DAYS * ONE_DAY:
        return GameCheck(game.slug, first, DayState.NOT_LAUNCHED)
    puzzle = await get_puzzle_on_date(db, game.id, game.default_variant, first)
    if puzzle is None:
        return GameCheck(game.slug, first, DayState.MISSING)
    if puzzle.status != PuzzleStatus.SCHEDULED:
        return GameCheck(game.slug, first, DayState.WRONG_STATUS)
    problems = await readiness_problems(puzzle, file_exists)
    # The first day's problems are known, so the buffer does not look at its files again.
    buffer = await ready_buffer(db, game, first, file_exists, {puzzle.id: problems})
    state = DayState.NOT_READY if problems else DayState.READY
    return GameCheck(game.slug, first, state, problems, buffer)


async def check_buffer(
    db: AsyncSession, file_exists: Callable[[str], bool], *, now: Callable[[], dt.datetime]
) -> list[GameCheck]:
    """One `GameCheck` per enabled game, by slug. Reads the clock once. A storage that cannot be
    read raises `StorageUnavailableError`; database errors are named by `database_session`."""
    today = utc_today(now())
    try:
        return [
            await check_game(db, game, file_exists, today=today)
            for game in await list_enabled_games(db)
        ]
    except MediaWriteError as error:
        raise StorageUnavailableError(f"storage error: {error}") from error


def exit_code(checks: list[GameCheck]) -> ExitCode:
    """2 beats 1 beats 0. A game that is not launched never fails (AC-82)."""
    launched = [check for check in checks if check.state != DayState.NOT_LAUNCHED]
    if any(check.state != DayState.READY for check in launched):
        return ExitCode.TOMORROW_NOT_READY
    if any(check.buffer < MIN_BUFFER_DAYS for check in launched):
        return ExitCode.BUFFER_LOW
    return ExitCode.OK


def describe_environment(app_settings: Settings) -> str:
    """Which database and storage are checked, with no password or key: the host, port and name of
    the database and the folder or bucket of the storage."""
    url = urlsplit(app_settings.DATABASE_URL)
    host = f"{url.hostname}:{url.port}" if url.port else str(url.hostname)
    return (
        f"Environment: {app_settings.ENVIRONMENT}; database: {host}{url.path};"
        f" storage: {_storage_description(app_settings)}"
    )


def _game_line(check: GameCheck) -> str:
    if check.state == DayState.NOT_LAUNCHED:
        return (
            f"{check.slug}: not launched, first day {check.first_day};"
            f" checked from {LAUNCH_WINDOW_DAYS} days before launch"
        )
    cap = "+" if check.buffer >= BUFFER_CAP else ""
    buffer = f"buffer {check.buffer}{cap} day(s)"
    if check.state == DayState.READY:
        return f"{check.slug}: {check.first_day} ready; {buffer}"
    if check.state == DayState.NOT_READY:
        why = ", ".join(str(problem) for problem in check.problems)
    else:
        why = check.state.value
    return f"{check.slug}: {check.first_day} NOT READY ({why}); {buffer}"


def report_lines(checks: list[GameCheck]) -> list[str]:
    """What the check prints on success: one line per game, then the result."""
    code = exit_code(checks)
    lines = [_game_line(check) for check in checks] or ["no enabled game"]
    lines.append(f"result: {code.name.lower()} (exit {int(code)})")
    return lines
