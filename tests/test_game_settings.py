"""The launch date and the enabled flag (SPEC-04 AC-56, AC-61, Q17), through the library and the
script's guard. Real Postgres through `db_session`."""

import datetime as dt
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.content.game_settings import GameSettingsError, apply_game_settings
from app.crud.game import get_game_by_slug
from app.crud.title import create_title
from app.models.progress import Progress
from app.models.puzzle import PuzzleStatus
from app.models.user import User
from tests.factories import create_game, create_puzzle

LAUNCH = dt.date(2027, 1, 15)
LATER = dt.date(2027, 2, 1)
TODAY = dt.date(2027, 1, 20)


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
            db_session, "game-guesser", launch_date=None, is_enabled=None, today=TODAY
        )
    with pytest.raises(GameSettingsError, match="slug"):
        await apply_game_settings(
            db_session, "Bad Slug", launch_date=LAUNCH, is_enabled=None, today=TODAY
        )

    created = await apply_game_settings(
        db_session, "game-guesser", launch_date=LAUNCH, is_enabled=None, today=TODAY
    )
    assert (created.is_created, await state(db_session)) == (True, (LAUNCH, False))

    # Each option changes only its own column; no option changes nothing.
    await apply_game_settings(
        db_session, "game-guesser", launch_date=None, is_enabled=True, today=TODAY
    )
    assert await state(db_session) == (LAUNCH, True)
    await apply_game_settings(
        db_session, "game-guesser", launch_date=LATER, is_enabled=None, today=TODAY
    )
    assert await state(db_session) == (LATER, True)
    await apply_game_settings(
        db_session, "game-guesser", launch_date=None, is_enabled=False, today=TODAY
    )
    assert await state(db_session) == (LATER, False)
    unchanged = await apply_game_settings(
        db_session, "game-guesser", launch_date=None, is_enabled=None, today=TODAY
    )
    assert unchanged.lines()[-1] == "nothing changed"


@pytest.mark.parametrize(
    ("puzzle_offset", "has_account_game", "new_date", "message"),
    [
        # The date moves while no puzzle is scheduled before it and nothing has been played.
        (None, False, LATER, None),
        (30, False, LATER, None),  # scheduled well after the new date
        (-3, True, LATER, "stored games"),
        (-3, False, LATER, "today or earlier"),
        (0, False, LATER, "today or earlier"),
        (5, False, TODAY + dt.timedelta(days=9), "would be hidden"),
    ],
    ids=["no_puzzle", "scheduled_after", "stored_games", "visible", "dated_today", "hidden"],
)
async def test__game_settings__the_launch_date_stays_when_it_would_change_what_players_saw(
    db_session: AsyncSession,
    puzzle_offset: int | None,
    has_account_game: bool,
    new_date: dt.date,
    message: str | None,
) -> None:
    game = await create_game(db_session, launch_date=LAUNCH, is_enabled=False)
    if puzzle_offset is not None:
        title = await create_title(db_session, name="Answer")
        puzzle = await create_puzzle(
            db_session,
            game=game,
            answer=title,
            puzzle_date=TODAY + dt.timedelta(days=puzzle_offset),
            status=PuzzleStatus.SCHEDULED,
        )
        if has_account_game:
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

    if message is None:
        await apply_game_settings(
            db_session, "game-guesser", launch_date=new_date, is_enabled=True, today=TODAY
        )
        assert await state(db_session) == (new_date, True)
        return
    with pytest.raises(GameSettingsError, match=message):
        await apply_game_settings(
            db_session, "game-guesser", launch_date=new_date, is_enabled=True, today=TODAY
        )

    # Refused as a whole, and the same date and the switch alone are still allowed.
    assert await state(db_session) == (LAUNCH, False)
    await apply_game_settings(
        db_session, "game-guesser", launch_date=LAUNCH, is_enabled=True, today=TODAY
    )
    assert await state(db_session) == (LAUNCH, True)


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
