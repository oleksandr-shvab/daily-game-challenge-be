"""Scheduling and unscheduling (SPEC-04 AC-55 to AC-60, AC-86; decisions 68 to 71) and the
readiness rule they use (AC-58). Real Postgres through `db_session`, a fake `file_exists`, an
injected `today`. Rows say what is approved, what is already dated and what the walk must do."""

import datetime as dt
from dataclasses import dataclass, field

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.content.readiness import Problem, ProblemKind
from app.content.schedule import (
    StopReason,
    UnscheduleRefusedError,
    schedule,
    unschedule,
)
from app.crud.title import create_title
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
        buffer=2,
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
    # An answer never repeats (decision 68): not against a past answer, nor within the run.
    "an_answer_is_never_repeated": Row(
        [
            scheduled_puz("past", -40),
            Puz("again", answer="past", rank=1),
            Puz("x1", answer="x", rank=2),
            Puz("x2", answer="x", rank=3),
            Puz("y", rank=4),
        ],
        [(1, "x1"), (2, "y")],
        StopReason.NOTHING_FITS,
        stopped_on=3,
        repeats=["again", "x2"],
        buffer=2,
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
    assert result.buffer == row.buffer
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


@pytest.mark.parametrize(
    ("offset", "is_allowed"),
    [(1, True), (0, False), (-1, False)],
    ids=["tomorrow", "today", "yesterday"],
)
async def test__unschedule__moves_back_only_a_future_puzzle(
    db_session: AsyncSession, offset: int, is_allowed: bool
) -> None:
    game = await create_game(db_session)
    answer = await create_title(db_session, name="Portal")
    puzzle = await create_puzzle(db_session, game=game, answer=answer, puzzle_date=day(offset))
    puzzle.approved_at = APPROVED_AT

    if is_allowed:
        await unschedule(db_session, game, day(offset), today=TODAY)
        assert (puzzle.status, puzzle.date) == (PuzzleStatus.APPROVED, None)
    else:
        with pytest.raises(UnscheduleRefusedError, match="locked"):
            await unschedule(db_session, game, day(offset), today=TODAY)
        assert (puzzle.status, puzzle.date) == (PuzzleStatus.SCHEDULED, day(offset))


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
        await unschedule(db_session, game, day(3), today=TODAY)

    assert (puzzle.status, puzzle.date) == (PuzzleStatus.SCHEDULED, day(3))
