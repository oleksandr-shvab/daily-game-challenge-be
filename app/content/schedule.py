"""Scheduling approved puzzles onto dates (SPEC-04 AC-55 to AC-60, AC-62, AC-86; decisions 68 to 71).

`schedule` walks the dates from the first schedulable day (the later of tomorrow and the launch
date) and gives each free date to the first approved, ready puzzle, in the order they were approved
(`approved_at`, then id; a NULL `approved_at` counts as the oldest). A puzzle is skipped for a date
when its answer was already scheduled (an answer never repeats) or when another scheduled answer of
the same series, the same non-null normalized franchise, lies within 14 days on either side. When
no puzzle fits a date the walk stops there and says so: it never skips the date and leaves a hole,
because a hole ends the buffer (AC-75). Today and past dates are never touched.

A date is free when no puzzle of the game and variant has it, in any status. A scheduled puzzle
takes a date, and a legacy puzzle that has a date but is not scheduled (the old seed draft) holds it
too, so the unique key `(game_id, variant, date)` is never hit; only scheduled puzzles count for
the series and repeat rules.

Library code: it flushes and never commits, so a run that fails half way changes nothing, and it
prints nothing. The caller owns the transaction and shows the `ScheduleResult`. `file_exists` is
blocking (`MediaWriter.exists`), so it is called in `asyncio.to_thread`.
"""

import asyncio
import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

from sqlalchemy.ext.asyncio import AsyncSession

from app.content.readiness import Problem, puzzle_problems
from app.crud.puzzle import (
    clear_puzzle_schedule,
    get_scheduled_puzzle_on_date,
    has_stored_games,
    list_approved_puzzles,
    list_dated_puzzles,
    list_scheduled_puzzles_from,
    set_puzzle_schedule,
)
from app.models.game import Game
from app.models.puzzle import Puzzle, PuzzleStatus

# Two answers of one series are at least this many days apart (AC-57): 14 days later is allowed.
SERIES_GAP_DAYS = 14
# The buffer is counted up to here (AC-75).
BUFFER_CAP = 60
ONE_DAY = dt.timedelta(days=1)


class StopReason(StrEnum):
    POOL_EMPTY = "pool_empty"
    NOTHING_FITS = "nothing_fits"
    UNTIL = "until"


class UnscheduleRefusedError(Exception):
    """The puzzle must not be moved. The message says why and holds no answer."""


@dataclass(frozen=True)
class ScheduledPuzzle:
    date: dt.date
    puzzle_id: int
    build_slot: int | None
    answer_name: str


@dataclass(frozen=True)
class NotReadyPuzzle:
    puzzle_id: int
    build_slot: int | None
    problems: list[Problem]


@dataclass
class ScheduleResult:
    first_day: dt.date
    scheduled: list[ScheduledPuzzle] = field(default_factory=list)
    # Approved puzzles that are not complete or miss a file (AC-58): they stay approved.
    not_ready: list[NotReadyPuzzle] = field(default_factory=list)
    # Ready approved puzzles whose answer is already scheduled: they never fit.
    repeats: list[int] = field(default_factory=list)
    # Dates the walk passed because a puzzle that is not scheduled holds them.
    blocked_dates: list[dt.date] = field(default_factory=list)
    stop_reason: StopReason = StopReason.POOL_EMPTY
    # The date nothing fit (`NOTHING_FITS`), or the first date past `until` (`UNTIL`).
    stopped_on: dt.date | None = None
    buffer: int = 0


def first_schedulable_day(game: Game, today: dt.date) -> dt.date:
    """AC-56: the later of tomorrow and the launch date. Never today or earlier."""
    return max(today + ONE_DAY, game.launch_date)


@dataclass(frozen=True)
class _Placed:
    date: dt.date
    title_id: int
    franchise: str | None


def _placed(puzzle: Puzzle, puzzle_date: dt.date) -> _Placed:
    return _Placed(
        puzzle_date, puzzle.answer_title_id, puzzle.answer_title.normalized_franchise or None
    )


def _fits(puzzle: Puzzle, date: dt.date, placed: list[_Placed]) -> bool:
    franchise = puzzle.answer_title.normalized_franchise or None
    for other in placed:
        if other.title_id == puzzle.answer_title_id:
            return False
        if (
            franchise is not None
            and other.franchise == franchise
            and abs((other.date - date).days) < SERIES_GAP_DAYS
        ):
            return False
    return True


async def _problems(puzzle: Puzzle, file_exists: Callable[[str], bool]) -> list[Problem]:
    return await asyncio.to_thread(puzzle_problems, puzzle.clues, file_exists=file_exists)


async def schedule(
    db: AsyncSession,
    game: Game,
    file_exists: Callable[[str], bool],
    *,
    today: dt.date,
    until: dt.date | None = None,
) -> ScheduleResult:
    variant = game.default_variant
    # `stop_reason` stays `POOL_EMPTY` unless the walk breaks out below.
    result = ScheduleResult(first_day=first_schedulable_day(game, today))
    dated = await list_dated_puzzles(db, game.id, variant)
    taken = {puzzle.date for puzzle in dated}
    non_scheduled_dates = {p.date for p in dated if p.status != PuzzleStatus.SCHEDULED}
    placed = [
        _placed(puzzle, puzzle.date)
        for puzzle in dated
        if puzzle.status == PuzzleStatus.SCHEDULED and puzzle.date is not None
    ]

    # Readiness is checked once per approved puzzle, before any date is given (AC-58).
    problems_by_id: dict[int, list[Problem]] = {}
    pool: list[Puzzle] = []
    for puzzle in await list_approved_puzzles(db, game.id, variant):
        problems = await _problems(puzzle, file_exists)
        problems_by_id[puzzle.id] = problems
        if problems:
            result.not_ready.append(NotReadyPuzzle(puzzle.id, puzzle.build_slot, problems))
        else:
            pool.append(puzzle)

    day = result.first_day
    while pool:
        if until is not None and day > until:
            result.stop_reason, result.stopped_on = StopReason.UNTIL, day
            break
        if day in taken:
            if day in non_scheduled_dates:
                result.blocked_dates.append(day)
            day += ONE_DAY
            continue
        chosen = next((puzzle for puzzle in pool if _fits(puzzle, day, placed)), None)
        if chosen is None:
            # Never skip the date: a hole ends the buffer (decision 69).
            result.stop_reason, result.stopped_on = StopReason.NOTHING_FITS, day
            break
        await set_puzzle_schedule(db, chosen, day)
        pool.remove(chosen)
        taken.add(day)
        placed.append(_placed(chosen, day))
        result.scheduled.append(
            ScheduledPuzzle(day, chosen.id, chosen.build_slot, chosen.answer_title.name)
        )
        day += ONE_DAY

    scheduled_titles = {other.title_id for other in placed}
    result.repeats = [puzzle.id for puzzle in pool if puzzle.answer_title_id in scheduled_titles]
    result.buffer = await _buffer(db, game, result.first_day, file_exists, problems_by_id)
    return result


async def _buffer(
    db: AsyncSession,
    game: Game,
    first_day: dt.date,
    file_exists: Callable[[str], bool],
    problems_by_id: dict[int, list[Problem]],
) -> int:
    """The consecutive days from `first_day` that have a ready scheduled puzzle, up to
    `BUFFER_CAP` (AC-60, AC-75). A gap, or a puzzle that is not ready, ends the count."""
    by_date = {
        puzzle.date: puzzle
        for puzzle in await list_scheduled_puzzles_from(
            db, game.id, game.default_variant, first_day
        )
    }
    days = 0
    day = first_day
    while days < BUFFER_CAP and day in by_date:
        puzzle = by_date[day]
        if puzzle.id not in problems_by_id:
            problems_by_id[puzzle.id] = await _problems(puzzle, file_exists)
        if problems_by_id[puzzle.id]:
            break
        days += 1
        day += ONE_DAY
    return days


async def unschedule(
    db: AsyncSession, game: Game, puzzle_date: dt.date, *, today: dt.date
) -> Puzzle:
    """Move the scheduled puzzle of `puzzle_date` back to `approved`, which frees the date (AC-59).
    Today's and past puzzles are visible and may have stored games: never moved, whatever else is
    true (AC-86). Flushes, never commits."""
    if puzzle_date <= today:
        raise UnscheduleRefusedError(f"{puzzle_date} is today or past: it is locked (AC-59)")
    puzzle = await get_scheduled_puzzle_on_date(db, game.id, game.default_variant, puzzle_date)
    if puzzle is None:
        raise UnscheduleRefusedError(f"no scheduled puzzle on {puzzle_date}")
    if await has_stored_games(db, puzzle.id):
        raise UnscheduleRefusedError(f"the puzzle on {puzzle_date} has stored games (AC-86)")
    await clear_puzzle_schedule(db, puzzle)
    return puzzle
