"""The buffer and tomorrow check (SPEC-04 AC-73 to AC-82). Real Postgres through `db_session`, a
fake `file_exists`, an injected clock. The rows go through the script's `run`, so what is pinned is
what the owner and the alert see: the exit code and the printed text."""

import contextlib
import datetime as dt
import importlib.util
import re
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from urllib.parse import urlsplit

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.content.buffer_check import ExitCode
from app.content.schedule import schedule
from app.core.config import settings
from app.crud.title import create_title
from app.models.game import Game
from app.models.puzzle import Puzzle, PuzzleStatus
from tests.factories import create_game, create_puzzle, media_key

TODAY = dt.date(2026, 11, 10)
ANSWER = "Zorblax Prime"
APPROVED_AT = dt.datetime(2026, 11, 1, tzinfo=dt.UTC)


def day(offset: int) -> dt.date:
    return TODAY + dt.timedelta(days=offset)


def clock_at(moment: dt.datetime) -> Callable[[], dt.datetime]:
    return lambda: moment


NOON = clock_at(dt.datetime.combine(TODAY, dt.time(12), tzinfo=dt.UTC))


def load_script() -> ModuleType:
    path = Path(__file__).resolve().parent.parent / "scripts" / "check_puzzle_buffer.py"
    spec = importlib.util.spec_from_file_location("p3_check_puzzle_buffer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def use_session(monkeypatch: pytest.MonkeyPatch, db: AsyncSession) -> None:
    """The script opens `core_db.async_session_maker()`; give it the test's session."""

    @contextlib.asynccontextmanager
    async def maker() -> AsyncIterator[AsyncSession]:
        yield db

    monkeypatch.setattr("app.core.db.async_session_maker", maker)


async def add_puzzles(
    db: AsyncSession, game: Game, offsets: list[int], status: PuzzleStatus
) -> dict[int, Puzzle]:
    """One puzzle per offset, all with the same answer name (answers are not checked here)."""
    title = await create_title(db, name=ANSWER)
    return {
        offset: await create_puzzle(
            db, game=game, answer=title, puzzle_date=day(offset), status=status
        )
        for offset in offsets
    }


@dataclass
class Row:
    days: list[int]
    code: ExitCode
    shown: str
    launch_offset: int | None = None
    # Puzzles on these days start as `approved` instead of `scheduled`.
    approved_days: list[int] = field(default_factory=list)
    # (day offset, step, file name): files the storage does not have.
    missing_files: list[tuple[int, int, str]] = field(default_factory=list)
    # How the run breaks: "database", "storage", "usage" or "crash".
    fault: str | None = None
    has_game: bool = True


ROWS = {
    "30_ready_days": Row(list(range(1, 31)), ExitCode.OK, "ready; buffer 30 day(s)"),
    "exactly_7_days_is_fine": Row(list(range(1, 8)), ExitCode.OK, "buffer 7 day(s)"),
    "6_days_is_low": Row(list(range(1, 7)), ExitCode.BUFFER_LOW, "buffer 6 day(s)"),
    "a_gap_ends_the_count": Row(
        [*range(1, 6), *range(7, 13)], ExitCode.BUFFER_LOW, "ready; buffer 5 day(s)"
    ),
    "a_gap_at_day_2": Row([1, *range(3, 10)], ExitCode.BUFFER_LOW, "buffer 1 day(s)"),
    "tomorrow_missing_beats_a_low_buffer": Row([2, 3], ExitCode.TOMORROW_NOT_READY, "missing"),
    "tomorrow_step_6_file_missing": Row(
        list(range(1, 11)),
        ExitCode.TOMORROW_NOT_READY,
        "missing_file at step 6); buffer 0 day(s)",
        missing_files=[(1, 6, "source")],
    ),
    "tomorrow_step_6_poster_missing": Row(
        list(range(1, 11)),
        ExitCode.TOMORROW_NOT_READY,
        "missing_file at step 6",
        missing_files=[(1, 6, "poster")],
    ),
    "tomorrow_approved_not_scheduled": Row(
        list(range(1, 11)),
        ExitCode.TOMORROW_NOT_READY,
        "wrong_status",
        approved_days=[1],
    ),
    "a_later_file_missing_ends_the_buffer": Row(
        list(range(1, 11)),
        ExitCode.BUFFER_LOW,
        "buffer 3 day(s)",
        missing_files=[(4, 2, "source")],
    ),
    "a_later_day_approved_ends_the_buffer": Row(
        list(range(1, 11)), ExitCode.BUFFER_LOW, "buffer 2 day(s)", approved_days=[3]
    ),
    # The launch date is the first day to be played (AC-82).
    "launch_in_20_days_is_not_launched": Row([], ExitCode.OK, "not launched", launch_offset=20),
    "launch_in_8_days_is_not_launched": Row([], ExitCode.OK, "not launched", launch_offset=8),
    "launch_in_7_days_is_checked": Row(
        [], ExitCode.TOMORROW_NOT_READY, f"{day(7)} NOT READY (missing)", launch_offset=7
    ),
    "launch_in_5_days_nothing_scheduled": Row(
        [], ExitCode.TOMORROW_NOT_READY, f"{day(5)} NOT READY (missing)", launch_offset=5
    ),
    "launch_in_5_days_counts_from_launch": Row(
        list(range(5, 12)), ExitCode.OK, f"{day(5)} ready; buffer 7 day(s)", launch_offset=5
    ),
    "no_enabled_game_prints_it": Row([], ExitCode.OK, "no enabled game", has_game=False),
    # Could not check: always 3, never Python's 1 or 2.
    "database_down": Row([], ExitCode.CANNOT_CHECK, "database error", fault="database"),
    "storage_unreadable": Row(
        [1], ExitCode.CANNOT_CHECK, "storage error: the storage cannot be read", fault="storage"
    ),
    "usage_error": Row([], ExitCode.CANNOT_CHECK, "usage error", fault="usage"),
    "crash": Row([], ExitCode.CANNOT_CHECK, "crashed: RuntimeError", fault="crash"),
}


@pytest.mark.parametrize("row", ROWS.values(), ids=ROWS.keys())
async def test__check__exit_code_and_output(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    row: Row,
) -> None:
    script = load_script()
    puzzles: dict[int, Puzzle] = {}
    if row.has_game:
        launch = day(row.launch_offset) if row.launch_offset is not None else dt.date(2026, 1, 1)
        game = await create_game(db_session, launch_date=launch)
        scheduled = [d for d in row.days if d not in row.approved_days]
        puzzles = await add_puzzles(db_session, game, scheduled, PuzzleStatus.SCHEDULED)
        puzzles |= await add_puzzles(db_session, game, row.approved_days, PuzzleStatus.APPROVED)
    # A disabled game is never checked (AC-73).
    await create_game(db_session, slug="disabled-game", is_enabled=False)
    use_session(monkeypatch, db_session)
    missing = {
        media_key(puzzles[offset].id, step, name) for offset, step, name in row.missing_files
    }
    argv: list[str] = []
    now = NOON

    def file_exists(key: str) -> bool:
        return key not in missing

    if row.fault == "database":

        def down() -> contextlib.AbstractAsyncContextManager[AsyncSession]:
            raise ConnectionRefusedError("down")

        monkeypatch.setattr("app.core.db.async_session_maker", down)
    elif row.fault == "storage":

        def unreadable(key: str) -> bool:
            raise PermissionError(f"no access to {key}")

        file_exists = unreadable  # type: ignore[assignment]
    elif row.fault == "usage":
        argv = ["--no-such-option"]
    elif row.fault == "crash":

        def broken() -> dt.datetime:
            raise RuntimeError("boom")

        now = broken

    code = await script.run(argv, now=now, file_exists=file_exists)

    output = capsys.readouterr()
    text = output.out + output.err
    assert (code, row.shown in text) == (row.code, True), text
    # The output goes to alert channels (AC-79): no answer, source, hint, media key or password.
    every_key = re.findall(r"[0-9a-f]{64}", text)
    password = urlsplit(settings.DATABASE_URL).password
    leaks = [
        leak
        for leak in (ANSWER, "example.com", "boom", "no access", *every_key, password)
        if leak and leak in text
    ]
    assert leaks == []
    assert not re.search(r"200[1-6]", text)
    # Whatever the run does, the first line says what it checked (a usage error stops before).
    assert row.fault == "usage" or text.startswith(f"Environment: {settings.ENVIRONMENT}; ")


async def test__check__tomorrow_moves_at_midnight_utc(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    game = await create_game(db_session)
    await add_puzzles(db_session, game, [1, 2], PuzzleStatus.SCHEDULED)
    use_session(monkeypatch, db_session)
    script = load_script()

    async def first_day_at(moment: dt.datetime) -> str:
        code = await script.run([], now=clock_at(moment), file_exists=lambda key: True)
        lines = capsys.readouterr().out.splitlines()
        return f"{code}: {lines[1]}"

    before = dt.datetime.combine(TODAY, dt.time(23, 59, 59), tzinfo=dt.UTC)
    at_midnight = dt.datetime.combine(day(1), dt.time(0, 0, 0), tzinfo=dt.UTC)

    # Tomorrow of 23:59:59 is day 1, with a buffer of 2; at 00:00:00 tomorrow is day 2: a buffer of 1.
    assert await first_day_at(before) == f"1: game-guesser: {day(1)} ready; buffer 2 day(s)"
    assert await first_day_at(at_midnight) == f"1: game-guesser: {day(2)} ready; buffer 1 day(s)"


async def test__check__reports_the_same_buffer_as_scheduling(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    game = await create_game(db_session)
    await add_puzzles(db_session, game, [1, 2, 3, 5, 6], PuzzleStatus.SCHEDULED)
    spare = await create_puzzle(
        db_session,
        game=game,
        answer=await create_title(db_session, name="Other"),
        puzzle_date=None,
        status=PuzzleStatus.APPROVED,
    )
    spare.approved_at = APPROVED_AT
    await db_session.flush()
    use_session(monkeypatch, db_session)

    def exists(key: str) -> bool:
        return True

    result = await schedule(db_session, game, exists, now=NOON)
    code = await load_script().run([], now=NOON, file_exists=exists)

    # Day 4 is a hole: scheduling gives it to the approved spare, so both count days 1 to 6.
    assert result.buffer == 6
    assert f"buffer {result.buffer} day(s)" in capsys.readouterr().out
    assert code == ExitCode.BUFFER_LOW
