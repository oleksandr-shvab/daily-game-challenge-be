"""The launch date and the enabled flag (SPEC-04 AC-56, AC-61, Q17), through the library and the
script's guard. Real Postgres through `db_session`."""

import datetime as dt
import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from app.content.game_settings import GameSettingsError, apply_game_settings
from app.crud.game import get_game_by_slug
from app.crud.title import create_title
from app.models.game import Game
from app.models.progress import Progress
from app.models.puzzle import PuzzleStatus
from app.models.user import User
from tests.factories import create_game, create_puzzle

LAUNCH = dt.date(2027, 1, 15)
LATER = dt.date(2027, 2, 1)
TODAY = dt.date(2027, 1, 20)


def clock(*offsets: int) -> Callable[[], dt.datetime]:
    """A clock that reads noon UTC of TODAY plus each offset in turn, then stays on the last."""
    readings = [
        dt.datetime.combine(TODAY + dt.timedelta(days=offset), dt.time(12), tzinfo=dt.UTC)
        for offset in offsets
    ]
    index = 0

    def now() -> dt.datetime:
        nonlocal index
        reading = readings[min(index, len(readings) - 1)]
        index += 1
        return reading

    return now


async def state(db: AsyncSession) -> tuple[dt.date, bool]:
    game = await get_game_by_slug(db, "game-guesser")
    assert game is not None
    return game.launch_date, game.is_enabled


async def test__game_settings__creates_a_disabled_game_then_sets_the_date_and_the_switch(
    db_session: AsyncSession,
) -> None:
    # A missing game needs a launch date.
    with pytest.raises(GameSettingsError, match="--launch-date"):
        await apply_game_settings(
            db_session, "game-guesser", launch_date=None, is_enabled=None, now=clock(0)
        )
    with pytest.raises(GameSettingsError, match="slug"):
        await apply_game_settings(
            db_session, "Bad Slug", launch_date=LAUNCH, is_enabled=None, now=clock(0)
        )

    created = await apply_game_settings(
        db_session, "game-guesser", launch_date=LAUNCH, is_enabled=None, now=clock(0)
    )
    assert (created.is_created, await state(db_session)) == (True, (LAUNCH, False))

    # Each option changes only its own column; no option changes nothing.
    await apply_game_settings(
        db_session, "game-guesser", launch_date=None, is_enabled=True, now=clock(0)
    )
    assert await state(db_session) == (LAUNCH, True)
    await apply_game_settings(
        db_session, "game-guesser", launch_date=LATER, is_enabled=None, now=clock(0)
    )
    assert await state(db_session) == (LATER, True)
    await apply_game_settings(
        db_session, "game-guesser", launch_date=None, is_enabled=False, now=clock(0)
    )
    assert await state(db_session) == (LATER, False)
    unchanged = await apply_game_settings(
        db_session, "game-guesser", launch_date=None, is_enabled=None, now=clock(0)
    )
    assert unchanged.lines()[-1] == "nothing changed"


@dataclass
class Move:
    """One try to move the launch date of a game with one scheduled puzzle (or none)."""

    puzzle_offset: int | None = None  # days from TODAY
    has_account_game: bool = False
    new_date: dt.date = LATER
    message: str | None = None  # the refusal, or None when the move is allowed
    is_game_enabled: bool = True
    enable_arg: bool | None = True
    clock_days: tuple[int, ...] = (0,)  # the days the clock reads in turn
    hidden: int = 0  # scheduled puzzles left before the new date, for an allowed move


@pytest.mark.parametrize(
    "move",
    [
        Move(),
        Move(puzzle_offset=30),  # scheduled well after the new date
        Move(puzzle_offset=-3, has_account_game=True, message="stored games"),
        Move(puzzle_offset=-3, message="today or earlier"),
        Move(puzzle_offset=0, message="today or earlier"),
        Move(puzzle_offset=5, new_date=TODAY + dt.timedelta(days=9), message="would be hidden"),
        # A disabled game shows nothing: the move is allowed and the puzzles before it are counted.
        Move(puzzle_offset=-3, is_game_enabled=False, enable_arg=False, hidden=1),
        Move(puzzle_offset=-3, has_account_game=True, is_game_enabled=False, enable_arg=False, message="stored games"),
        # ... unless the same command enables it.
        Move(puzzle_offset=-3, is_game_enabled=False, enable_arg=True, message="today or earlier"),
        # Today moves forward between the early check and the write: the puzzle is visible now.
        Move(puzzle_offset=1, new_date=LAUNCH - dt.timedelta(days=10), clock_days=(0, 1), message="today or earlier"),
    ],
    ids=["no_puzzle", "scheduled_after", "stored_games", "visible", "dated_today", "hidden", "disabled_counts_the_hidden", "disabled_stored_games", "disabled_but_enabling", "today_moves_forward"],
)  # fmt: skip
async def test__game_settings__the_launch_date_stays_when_it_would_change_what_players_saw(
    db_session: AsyncSession, move: Move
) -> None:
    game = await create_game(db_session, launch_date=LAUNCH, is_enabled=move.is_game_enabled)
    if move.puzzle_offset is not None:
        title = await create_title(db_session, name="Answer")
        puzzle = await create_puzzle(
            db_session,
            game=game,
            answer=title,
            puzzle_date=TODAY + dt.timedelta(days=move.puzzle_offset),
            status=PuzzleStatus.SCHEDULED,
        )
        if move.has_account_game:
            account = User(email="player@example.com", hashed_password="not-a-real-hash")
            db_session.add(account)
            await db_session.flush()
            db_session.add(
                Progress(
                    user_id=account.id,
                    puzzle_id=puzzle.id,
                    status="in_progress",
                    started_at=dt.datetime(2027, 1, 15, tzinfo=dt.UTC),
                    counts_for_streak=True,
                    origin="play",
                )
            )
            await db_session.flush()

    if move.message is None:
        change = await apply_game_settings(
            db_session,
            "game-guesser",
            launch_date=move.new_date,
            is_enabled=move.enable_arg,
            now=clock(*move.clock_days),
        )
        assert await state(db_session) == (move.new_date, move.enable_arg)
        assert change.hidden_scheduled == move.hidden
        assert any("--unschedule" in line for line in change.lines()) == bool(move.hidden)
        return
    with pytest.raises(GameSettingsError, match=move.message):
        await apply_game_settings(
            db_session,
            "game-guesser",
            launch_date=move.new_date,
            is_enabled=move.enable_arg,
            now=clock(*move.clock_days),
        )

    # Refused as a whole, and the same date and the switch alone are still allowed.
    assert await state(db_session) == (LAUNCH, move.is_game_enabled)
    await apply_game_settings(
        db_session, "game-guesser", launch_date=LAUNCH, is_enabled=True, now=clock(0)
    )
    assert await state(db_session) == (LAUNCH, True)


async def test__game_settings__reads_the_game_again_under_the_lock(
    db_session: AsyncSession,
) -> None:
    game = await create_game(db_session, launch_date=LATER, is_enabled=False)
    title = await create_title(db_session, name="Answer")
    await create_puzzle(
        db_session, game=game, answer=title, puzzle_date=TODAY - dt.timedelta(days=1)
    )
    # Another session moved the date to LAUNCH after this one loaded the game: the object is stale
    # and says the date is already LATER, so a move to it looks like no move at all.
    await db_session.execute(
        update(Game).where(Game.id == game.id).values(launch_date=LAUNCH),
        execution_options={"synchronize_session": False},
    )
    assert game.launch_date == LATER

    with pytest.raises(GameSettingsError, match="today or earlier"):
        await apply_game_settings(
            db_session, "game-guesser", launch_date=LATER, is_enabled=True, now=clock(0)
        )

    assert await state(db_session) == (LAUNCH, False)


def load_script() -> ModuleType:
    path = Path(__file__).resolve().parent.parent / "scripts" / "manage_game.py"
    spec = importlib.util.spec_from_file_location("p3_manage_game", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("argv", "environ", "message"),
    [
        ([], {"ENVIRONMENT": "local"}, "--target is required"),
        (["--target", "production"], {"ENVIRONMENT": "production"}, "--confirm-target production"),
        (["--target", "local"], {"ENVIRONMENT": "local", "CI": "true"}, "CI"),
    ],
    ids=["no_target", "no_confirmation", "ci"],
)
def test__manage_game_script__the_guard_refuses_before_any_database_call(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    environ: dict[str, str],
    message: str,
) -> None:
    for name in ("CI", "ENVIRONMENT"):
        monkeypatch.delenv(name, raising=False)
    for name, value in environ.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr("sys.argv", ["manage_game.py", *argv, "--launch-date", "2027-01-15"])

    code = load_script().main()

    captured = capsys.readouterr()
    assert code == 1 and message in captured.err and captured.out == ""
