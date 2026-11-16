"""Scheduling and unscheduling (SPEC-04 AC-55 to AC-60, AC-86; decisions 68 to 71) and the
readiness rule they use (AC-58). Real Postgres through `db_session`, a fake `file_exists`, an
injected `today`. Rows say what is approved, what is already dated and what the walk must do."""

import datetime as dt
import importlib.util
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.content import schedule as schedule_module
from app.content.readiness import Problem, ProblemKind
from app.content.schedule import (
    ScheduledPuzzle,
    ScheduleResult,
    StopReason,
    UnscheduleRefusedError,
    schedule,
    unschedule,
)
from app.crud.title import create_title
from app.media.writer import MediaWriteError
from app.models.game import Game
from app.models.progress import Progress
from app.models.puzzle import Puzzle, PuzzleStatus
from app.models.title import Title
from app.models.user import User
from tests.factories import create_game, create_puzzle, media_key

TODAY = dt.date(2026, 11, 10)
APPROVED_AT = dt.datetime(2026, 11, 1, tzinfo=dt.UTC)
FAR_PAST_LAUNCH = dt.date(2026, 1, 1)


def day(offset: int) -> dt.date:
    return TODAY + dt.timedelta(days=offset)


@dataclass
class Puz:
    """One puzzle of a row. `label` names it in the expectations; `answer` defaults to the label,
    and two puzzles with one answer share one title. `rank` orders `approved_at` (None is NULL).
    `offset` is the date relative to TODAY and `status` is what the row starts as."""

    label: str
    answer: str | None = None
    franchise: str | None = None
    rank: int | None = 1
    offset: int | None = None
    status: PuzzleStatus = PuzzleStatus.APPROVED
    variant: str | None = None
    missing_step_6_file: bool = False
    has_media: bool = True


@dataclass
class Row:
    puzzles: list[Puz]
    scheduled: list[tuple[int, str]]
    stop: StopReason
    stopped_on: int | None = None
    launch_offset: int | None = None
    until_offset: int | None = None
    not_ready: dict[str, list[Problem]] = field(default_factory=dict)
    repeats: list[str] = field(default_factory=list)
    blocked: list[int] = field(default_factory=list)
    buffer: int = 0
    buffer_from_launch: int | None = None


def scheduled_puz(label: str, offset: int, franchise: str | None = None) -> Puz:
    return Puz(label, franchise=franchise, offset=offset, status=PuzzleStatus.SCHEDULED)


def approved(*labels: str, franchise: str | None = None) -> list[Puz]:
    return [Puz(label, franchise=franchise, rank=index) for index, label in enumerate(labels)]


ZELDA = "zelda"
ROWS = {
    # The order of approval, not of id; a NULL `approved_at` is the oldest (R-16).
    "approval_order": Row(
        [Puz("a", rank=3), Puz("b", rank=1), Puz("c", rank=2), Puz("d", rank=None)],
        [(1, "d"), (2, "b"), (3, "c"), (4, "a")],
        StopReason.POOL_EMPTY,
        buffer=4,
    ),
    # The first day is the later of tomorrow and the launch date (AC-56, decision 71).
    "launch_in_10_days": Row(
        approved("a", "b"),
        [(10, "a"), (11, "b")],
        StopReason.POOL_EMPTY,
        launch_offset=10,
        buffer=0,
        buffer_from_launch=2,
    ),
    "launch_in_the_past": Row(
        approved("a", "b"), [(1, "a"), (2, "b")], StopReason.POOL_EMPTY, launch_offset=-30, buffer=2
    ),
    # A dated puzzle holds its date; a draft is not a scheduled answer, so it is no series (AC-55).
    "dated_draft_holds_its_date": Row(
        [
            Puz("old", franchise=ZELDA, rank=None, offset=1, status=PuzzleStatus.DRAFT),
            Puz("a", franchise=ZELDA),
        ],
        [(2, "a")],
        StopReason.POOL_EMPTY,
        blocked=[1],
        buffer=0,
    ),
    # Series: 14 days apart is fine, 13 is not, on either side (AC-57, decision 68).
    "series_14_days_after_is_fine": Row(
        [scheduled_puz("s", 15, ZELDA), *approved("a", franchise=ZELDA)],
        [(1, "a")],
        StopReason.POOL_EMPTY,
        buffer=1,
    ),
    "series_13_days_after_waits": Row(
        [scheduled_puz("s", 14, ZELDA), *approved("a", franchise=ZELDA)],
        [],
        StopReason.NOTHING_FITS,
        stopped_on=1,
    ),
    "series_6_days_before_waits": Row(
        [scheduled_puz("s", -5, ZELDA), *approved("a", franchise=ZELDA)],
        [],
        StopReason.NOTHING_FITS,
        stopped_on=1,
    ),
    "series_14_days_before_is_fine": Row(
        [scheduled_puz("s", -13, ZELDA), *approved("a", franchise=ZELDA)],
        [(1, "a")],
        StopReason.POOL_EMPTY,
        buffer=1,
    ),
    # Another series takes the day; the second of the pair waits, and the walk stops at the first
    # date nothing fits, never leaving a hole (decision 69). `until` bounds a run that would not.
    "stops_at_the_first_date_nothing_fits": Row(
        [Puz("z1", franchise=ZELDA, rank=1), Puz("z2", franchise=ZELDA, rank=2), Puz("m", rank=3)],
        [(1, "z1"), (2, "m")],
        StopReason.NOTHING_FITS,
        stopped_on=3,
        until_offset=20,
        buffer=2,
    ),
    "stops_at_until": Row(
        approved("a", "b", "c"),
        [(1, "a"), (2, "b")],
        StopReason.UNTIL,
        stopped_on=3,
        until_offset=2,
        buffer=2,
    ),
    # An answer never repeats (decision 68): not against a past answer, nor within the run, nor
    # against another variant or an old dated draft. Repeats leave the pool, so a pool of repeats
    # is empty, not "nothing fits".
    "an_answer_is_never_repeated": Row(
        [
            scheduled_puz("past", -40),
            Puz("again", answer="past", rank=1),
            Puz("x1", answer="x", rank=2),
            Puz("x2", answer="x", rank=3),
            Puz("y", rank=4),
        ],
        [(1, "x1"), (2, "y")],
        StopReason.POOL_EMPTY,
        repeats=["again", "x2"],
        buffer=2,
    ),
    "a_pool_of_only_repeats_is_empty": Row(
        [scheduled_puz("past", -40), Puz("again", answer="past")],
        [],
        StopReason.POOL_EMPTY,
        repeats=["again"],
    ),
    "an_answer_scheduled_in_another_variant_is_used": Row(
        [
            Puz("other", answer="x", offset=-3, status=PuzzleStatus.SCHEDULED, variant="v2"),
            Puz("again", answer="x", rank=1),
            Puz("y", rank=2),
        ],
        [(1, "y")],
        StopReason.POOL_EMPTY,
        repeats=["again"],
        buffer=1,
    ),
    "an_answer_on_an_old_dated_draft_is_used": Row(
        [
            Puz("old", answer="x", rank=None, offset=5, status=PuzzleStatus.DRAFT),
            Puz("again", answer="x"),
        ],
        [],
        StopReason.POOL_EMPTY,
        repeats=["again"],
    ),
    # A puzzle that is not ready stays approved and is reported with its step (AC-58).
    "not_ready_stays_approved": Row(
        [
            Puz("a", rank=1, missing_step_6_file=True),
            Puz("b", rank=2, has_media=False),
            Puz("c", rank=3),
        ],
        [(1, "c")],
        StopReason.POOL_EMPTY,
        not_ready={
            "a": [Problem(ProblemKind.MISSING_FILE, 6)],
            "b": [Problem(ProblemKind.BAD_MEDIA, step) for step in range(1, 7)],
        },
        buffer=1,
    ),
    # A pool that holds only not-ready puzzles ends empty, not "nothing fits".
    "only_not_ready_puzzles": Row(
        [Puz("a", missing_step_6_file=True)],
        [],
        StopReason.POOL_EMPTY,
        not_ready={"a": [Problem(ProblemKind.MISSING_FILE, 6)]},
    ),
    # Files are checked only for a puzzle the walk reaches: `until` ends it before "a".
    "readiness_is_checked_lazily": Row(
        [Puz("c", rank=1), Puz("a", rank=2, missing_step_6_file=True)],
        [(1, "c")],
        StopReason.UNTIL,
        stopped_on=2,
        until_offset=1,
        buffer=1,
    ),
}


async def create_row(db: AsyncSession, game: Game, row: Row) -> tuple[dict[str, Puzzle], set[str]]:
    puzzles: dict[str, Puzzle] = {}
    titles: dict[str, Title] = {}
    missing: set[str] = set()
    for slot, puz in enumerate(row.puzzles, start=1):
        answer = puz.answer or puz.label
        if answer not in titles:
            titles[answer] = await create_title(db, name=answer, franchise=puz.franchise)
        puzzle = await create_puzzle(
            db,
            game=game,
            answer=titles[answer],
            puzzle_date=None if puz.offset is None else day(puz.offset),  # type: ignore[arg-type]
            status=puz.status,
            variant=puz.variant,
            with_media=puz.has_media,
        )
        puzzle.build_slot = slot
        if puz.status == PuzzleStatus.APPROVED and puz.rank is not None:
            puzzle.approved_at = APPROVED_AT + dt.timedelta(hours=puz.rank)
        puzzles[puz.label] = puzzle
        if puz.missing_step_6_file:
            missing.add(media_key(puzzle.id, 6, "source"))
    await db.flush()
    return puzzles, missing


@pytest.mark.parametrize("row", ROWS.values(), ids=ROWS.keys())
async def test__schedule__fills_free_days_by_the_rules(db_session: AsyncSession, row: Row) -> None:
    launch = day(row.launch_offset) if row.launch_offset is not None else FAR_PAST_LAUNCH
    game = await create_game(db_session, launch_date=launch)
    puzzles, missing = await create_row(db_session, game, row)
    labels = {puzzle.id: label for label, puzzle in puzzles.items()}

    result = await schedule(
        db_session,
        game,
        lambda key: key not in missing,
        today=TODAY,
        until=day(row.until_offset) if row.until_offset is not None else None,
    )

    assert [(item.date, labels[item.puzzle_id]) for item in result.scheduled] == [
        (day(offset), label) for offset, label in row.scheduled
    ]
    assert result.stop_reason == row.stop
    assert result.stopped_on == (day(row.stopped_on) if row.stopped_on is not None else None)
    assert {labels[p.puzzle_id]: p.problems for p in result.not_ready} == row.not_ready
    assert sorted(labels[puzzle_id] for puzzle_id in result.repeats) == sorted(row.repeats)
    assert result.blocked_dates == [day(offset) for offset in row.blocked]
    assert (result.buffer, result.buffer_from_launch) == (row.buffer, row.buffer_from_launch)
    # What was given a date is `scheduled` with it; everything else is as the row made it.
    given = {labels[item.puzzle_id]: item.date for item in result.scheduled}
    for puz in row.puzzles:
        puzzle = puzzles[puz.label]
        if puz.label in given:
            assert (puzzle.status, puzzle.date) == (PuzzleStatus.SCHEDULED, given[puz.label])
        else:
            assert (puzzle.status, puzzle.date) == (
                puz.status,
                None if puz.offset is None else day(puz.offset),
            )


def clock(*offsets: int) -> Callable[[], dt.datetime]:
    """A clock that reads noon UTC of TODAY plus each offset in turn, then stays on the last."""
    readings = [dt.datetime.combine(day(offset), dt.time(12), tzinfo=dt.UTC) for offset in offsets]
    index = 0

    def now() -> dt.datetime:
        nonlocal index
        reading = readings[min(index, len(readings) - 1)]
        index += 1
        return reading

    return now


@pytest.mark.parametrize(
    ("offset", "days_on_the_clock", "is_allowed"),
    [
        (1, (0,), True),
        (0, (0,), False),
        (-1, (0,), False),
        # Midnight passes between the first check and the write: the puzzle is today's now.
        (1, (0, 1), False),
    ],
    ids=["tomorrow", "today", "yesterday", "midnight_passes_while_it_runs"],
)
async def test__unschedule__moves_back_only_a_future_puzzle(
    db_session: AsyncSession, offset: int, days_on_the_clock: tuple[int, ...], is_allowed: bool
) -> None:
    game = await create_game(db_session)
    answer = await create_title(db_session, name="Portal")
    puzzle = await create_puzzle(db_session, game=game, answer=answer, puzzle_date=day(offset))
    puzzle.approved_at = APPROVED_AT
    now = clock(*days_on_the_clock)

    if is_allowed:
        await unschedule(db_session, game, day(offset), now=now)
        # It goes to the back of the queue: `approved_at` is the time of the write.
        assert (puzzle.status, puzzle.date, puzzle.approved_at) == (
            PuzzleStatus.APPROVED,
            None,
            now(),
        )
    else:
        with pytest.raises(UnscheduleRefusedError, match="locked"):
            await unschedule(db_session, game, day(offset), now=now)
        assert (puzzle.status, puzzle.date) == (PuzzleStatus.SCHEDULED, day(offset))


async def test__unschedule__then_schedule_gives_the_date_to_another_puzzle(
    db_session: AsyncSession,
) -> None:
    game = await create_game(db_session)
    first, second = [await create_title(db_session, name=name) for name in ("Portal", "Doom")]
    moved = await create_puzzle(db_session, game=game, answer=first, puzzle_date=day(1))
    moved.approved_at = APPROVED_AT
    waiting = await create_puzzle(
        db_session, game=game, answer=second, puzzle_date=None, status=PuzzleStatus.APPROVED
    )
    waiting.approved_at = APPROVED_AT + dt.timedelta(hours=1)

    await unschedule(db_session, game, day(1), now=clock(0))
    result = await schedule(db_session, game, lambda key: True, today=TODAY)

    assert [(item.date, item.puzzle_id) for item in result.scheduled] == [
        (day(1), waiting.id),
        (day(2), moved.id),
    ]


async def test__unschedule__refuses_a_puzzle_with_stored_games(db_session: AsyncSession) -> None:
    game = await create_game(db_session)
    answer = await create_title(db_session, name="Portal")
    puzzle = await create_puzzle(db_session, game=game, answer=answer, puzzle_date=day(3))
    account = User(email="player@example.com", hashed_password="not-a-real-hash")
    db_session.add(account)
    await db_session.flush()
    db_session.add(
        Progress(
            user_id=account.id,
            puzzle_id=puzzle.id,
            status="in_progress",
            started_at=APPROVED_AT,
            counts_for_streak=True,
            origin="play",
        )
    )
    await db_session.flush()

    with pytest.raises(UnscheduleRefusedError, match="stored games"):
        await unschedule(db_session, game, day(3), now=clock(0))

    assert (puzzle.status, puzzle.date) == (PuzzleStatus.SCHEDULED, day(3))


async def test__schedule__the_buffer_is_counted_from_tomorrow_up_to_the_cap(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(schedule_module, "BUFFER_CAP", 2)
    game = await create_game(db_session)
    for offset, name in enumerate(("a", "b", "c"), start=1):
        await create_puzzle(
            db_session,
            game=game,
            answer=await create_title(db_session, name=name),
            puzzle_date=day(offset),
        )

    result = await schedule(db_session, game, lambda key: True, today=TODAY)

    # Days 1 and 2 are ready; day 3 is past the cap. The first puzzle is dated tomorrow.
    assert result.buffer == 2


async def test__schedule__a_storage_error_is_a_media_write_error(db_session: AsyncSession) -> None:
    game = await create_game(db_session)
    puzzle = await create_puzzle(
        db_session,
        game=game,
        answer=await create_title(db_session, name="Portal"),
        puzzle_date=None,
        status=PuzzleStatus.APPROVED,
    )
    puzzle.approved_at = APPROVED_AT

    def unreadable_folder(key: str) -> bool:
        raise PermissionError(f"no access to {key}")

    with pytest.raises(MediaWriteError) as error:
        await schedule(db_session, game, unreadable_folder, today=TODAY)

    # The message names the class only: no path, key or errno text.
    assert str(error.value) == "the storage cannot be read: PermissionError"


def load_script() -> object:
    path = Path(__file__).resolve().parent.parent / "scripts" / "schedule_puzzles.py"
    spec = importlib.util.spec_from_file_location("p3_schedule_puzzles", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test__report__names_puzzles_by_id_hides_answers_and_explains_a_held_date() -> None:
    result = ScheduleResult(
        first_day=day(1),
        scheduled=[ScheduledPuzzle(day(1), 7, None, "Secret Answer")],
        blocked_dates=[day(2)],
    )
    report_lines = load_script().report_lines  # type: ignore[attr-defined]

    hidden = "\n".join(report_lines(result, show_answers=False))
    shown = "\n".join(report_lines(result, show_answers=True))

    assert "puzzle 7" in hidden and "slot None" in hidden
    assert "Secret Answer" not in hidden and "Secret Answer" in shown
    assert "set that row's date to NULL" in hidden
