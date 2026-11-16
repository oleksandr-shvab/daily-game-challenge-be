"""Scheduling approved puzzles onto dates (SPEC-04 AC-55 to AC-60, AC-62, AC-86; decisions 68 to 71).

`schedule` walks the dates from the first schedulable day (the later of tomorrow and the launch
date) and gives each free date to the first approved, ready puzzle, in the order they were approved
(`approved_at`, then id; a NULL `approved_at` counts as the oldest). A puzzle is skipped for a date
when another scheduled answer of the same series, the same non-null normalized franchise, lies less
than 14 days away on either side. An answer never repeats: a puzzle whose answer is already used
leaves the pool and is reported. Used means the answer of a scheduled puzzle of any variant, past
or future, and of a dated puzzle that is not scheduled (an old draft). Approved and in-review
puzzles with the same answer are kept apart by the builder's funnel (AC-25); here the second of two
approved puzzles with one answer becomes a repeat as soon as the first gets a date.

When no puzzle fits a date the walk stops there and says so: it never skips the date and leaves a
hole, because a hole ends the buffer (AC-75). Today and past dates are never touched.

A date is free when no puzzle of the game and variant has it, in any status. A scheduled puzzle
takes a date, and a legacy puzzle that has a date but is not scheduled (the old seed draft) holds it
too, so the unique key `(game_id, variant, date)` is never hit; only scheduled puzzles count for
the series rule.

Readiness (files in storage) is checked lazily, when the walk reaches a candidate for a date it
could take, and once per puzzle. A not-ready puzzle is reported and leaves the pool; one the walk
never reaches (after `until`, or kept back by the series rule) is not checked or reported.

Library code: it flushes and never commits, so a run that fails half way changes nothing, and it
prints nothing. The caller owns the transaction and shows the `ScheduleResult`. `file_exists` is
blocking (`MediaWriter.exists`), so it is called in `asyncio.to_thread`; an `OSError` from it is a
`MediaWriteError`, a storage error, not a database one.
"""

import asyncio
import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

from sqlalchemy.ext.asyncio import AsyncSession

from app.content.readiness import Problem, puzzle_problems
from app.core.clock import utc_today
from app.crud.puzzle import (
    clear_puzzle_schedule,
    has_stored_games,
    list_approved_puzzles,
    list_dated_answers,
    list_dated_puzzles,
    list_scheduled_puzzles_between,
    lock_scheduled_puzzle_on_date,
    set_puzzle_schedule,
)
from app.media.writer import MediaWriteError
from app.models.game import Game
from app.models.puzzle import Puzzle, PuzzleStatus

# Two answers of one series are at least this many days apart (AC-57): 14 days later is allowed.
SERIES_GAP_DAYS = 14
# The buffer is counted up to here (AC-75), and no further day is read for it.
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
    # Approved puzzles the walk reached that are not complete or miss a file (AC-58): they stay
    # approved.
    not_ready: list[NotReadyPuzzle] = field(default_factory=list)
    # Approved puzzles whose answer is already used (ids): they never get a date.
    repeats: list[int] = field(default_factory=list)
    # Dates the walk passed because a puzzle that is not scheduled holds them.
    blocked_dates: list[dt.date] = field(default_factory=list)
    stop_reason: StopReason = StopReason.POOL_EMPTY
    # The date nothing fit (`NOTHING_FITS`), or the first date past `until` (`UNTIL`).
    stopped_on: dt.date | None = None
    # Consecutive days from tomorrow with a ready scheduled puzzle (AC-75, as the M1-25 check
    # counts it), and, only while the launch date is later than tomorrow, the same count from the
    # launch date.
    buffer: int = 0
    buffer_from_launch: int | None = None


def first_schedulable_day(game: Game, today: dt.date) -> dt.date:
    """AC-56: the later of tomorrow and the launch date. Never today or earlier."""
    return max(today + ONE_DAY, game.launch_date)


@dataclass(frozen=True)
class _Placed:
    date: dt.date
    franchise: str | None


def _fits(puzzle: Puzzle, date: dt.date, placed: list[_Placed]) -> bool:
    """The series rule (AC-57): no other scheduled answer of the same series less than
    `SERIES_GAP_DAYS` days away, either side."""
    franchise = puzzle.answer_title.normalized_franchise or None
    return franchise is None or all(
        other.franchise != franchise or abs((other.date - date).days) >= SERIES_GAP_DAYS
        for other in placed
    )


def _drop_repeats(pool: list[Puzzle], used_titles: set[int], repeats: list[int]) -> list[Puzzle]:
    """The pool without the puzzles whose answer is used (decision 68), which join `repeats`."""
    repeats += [puzzle.id for puzzle in pool if puzzle.answer_title_id in used_titles]
    return [puzzle for puzzle in pool if puzzle.answer_title_id not in used_titles]


async def _problems(puzzle: Puzzle, file_exists: Callable[[str], bool]) -> list[Problem]:
    try:
        return await asyncio.to_thread(puzzle_problems, puzzle.clues, file_exists=file_exists)
    except OSError as error:
        # A folder that cannot be read is a storage error, not a database one (the caller's
        # session wrapper would name it so).
        raise MediaWriteError(f"the storage cannot be read: {type(error).__name__}") from error


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
        _Placed(puzzle.date, puzzle.answer_title.normalized_franchise or None)
        for puzzle in dated
        if puzzle.status == PuzzleStatus.SCHEDULED and puzzle.date is not None
    ]

    pool = await list_approved_puzzles(db, game.id, variant)
    # An approved puzzle that has a legacy date is a candidate, not a user of its own answer.
    pool_ids = {puzzle.id for puzzle in pool}
    used_titles = {
        title_id
        for puzzle_id, title_id in await list_dated_answers(db, game.id)
        if puzzle_id not in pool_ids
    }
    pool = _drop_repeats(pool, used_titles, result.repeats)

    problems_by_id: dict[int, list[Problem]] = {}
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
        chosen = None
        for puzzle in list(pool):
            if not _fits(puzzle, day, placed):
                continue
            problems = await _problems(puzzle, file_exists)
            problems_by_id[puzzle.id] = problems
            if problems:
                # AC-58: it stays approved and is reported; it does not hold the date back.
                pool.remove(puzzle)
                result.not_ready.append(NotReadyPuzzle(puzzle.id, puzzle.build_slot, problems))
                continue
            chosen = puzzle
            break
        if chosen is None:
            if pool:
                # Never skip the date: a hole ends the buffer (decision 69).
                result.stop_reason, result.stopped_on = StopReason.NOTHING_FITS, day
            break
        await set_puzzle_schedule(db, chosen, day)
        pool.remove(chosen)
        taken.add(day)
        placed.append(_Placed(day, chosen.answer_title.normalized_franchise or None))
        used_titles.add(chosen.answer_title_id)
        pool = _drop_repeats(pool, used_titles, result.repeats)
        result.scheduled.append(
            ScheduledPuzzle(day, chosen.id, chosen.build_slot, chosen.answer_title.name)
        )
        day += ONE_DAY

    tomorrow = today + ONE_DAY
    result.buffer = await _buffer(db, game, tomorrow, file_exists, problems_by_id)
    if result.first_day > tomorrow:
        result.buffer_from_launch = await _buffer(
            db, game, result.first_day, file_exists, problems_by_id
        )
    return result


async def _buffer(
    db: AsyncSession,
    game: Game,
    start: dt.date,
    file_exists: Callable[[str], bool],
    problems_by_id: dict[int, list[Problem]],
) -> int:
    """The consecutive days from `start` that have a ready scheduled puzzle, up to `BUFFER_CAP`
    (AC-60, AC-75). A gap, or a puzzle that is not ready, ends the count. Only the days that can
    count are read."""
    by_date = {
        puzzle.date: puzzle
        for puzzle in await list_scheduled_puzzles_between(
            db, game.id, game.default_variant, start, start + BUFFER_CAP * ONE_DAY
        )
    }
    days = 0
    day = start
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
    db: AsyncSession, game: Game, puzzle_date: dt.date, *, now: Callable[[], dt.datetime]
) -> Puzzle:
    """Move the scheduled puzzle of `puzzle_date` back to `approved`, which frees the date (AC-59),
    and put it at the back of the approval queue (`approved_at` becomes now), so the next
    scheduling run gives the date to another puzzle. Today's and past puzzles are visible and may
    have stored games: never moved, whatever else is true (AC-86). Flushes, never commits.

    `now` is a clock, called twice: once for the early refusal, and again after the puzzle's row is
    locked, so a run that crosses 00:00 UTC (the puzzle has just become visible) is checked against
    the time of the write, not the time of the start. Stored games are looked up under the same
    lock, inside the same transaction. The row lock also holds back a new `progress` row for the
    puzzle (its foreign key needs a share lock) until the transaction ends."""
    if puzzle_date <= utc_today(now()):
        raise UnscheduleRefusedError(f"{puzzle_date} is today or past: it is locked (AC-59)")
    puzzle = await lock_scheduled_puzzle_on_date(db, game.id, game.default_variant, puzzle_date)
    if puzzle is None:
        raise UnscheduleRefusedError(f"no scheduled puzzle on {puzzle_date}")
    write_time = now()
    if puzzle_date <= utc_today(write_time):
        raise UnscheduleRefusedError(f"{puzzle_date} is today or past: it is locked (AC-59)")
    if await has_stored_games(db, puzzle.id):
        raise UnscheduleRefusedError(f"the puzzle on {puzzle_date} has stored games (AC-86)")
    await clear_puzzle_schedule(db, puzzle, write_time)
    return puzzle
