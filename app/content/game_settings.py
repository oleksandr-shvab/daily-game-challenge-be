"""A game's launch date and its enabled flag (SPEC-04 AC-56, AC-61, Q17). The guards on moving the
launch date protect AC-86 (stored games) and AC-62 (visibility); the rule for a disabled game is
owner decision 72.

`scripts/manage_game.py` is the command. The owner sets the production launch date with it before
Phase 5: scheduling starts from the later of tomorrow and that date (AC-56), and the view number
counts from it. A game this script creates is disabled until `--enable`, so a half-ready game is
never public.

Library code: it flushes and never commits, and it prints nothing. The game's row is locked
`FOR UPDATE` before the date is changed, and the guards run again under the lock with the clock
read again, the way `unschedule` does, so scheduling (which reads the launch date under
`FOR SHARE`) and a change of the date cannot interleave.
"""

import datetime as dt
import re
from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.clock import utc_today
from app.crud.game import (
    add_game,
    count_scheduled_puzzles_before,
    game_has_stored_games,
    get_game_by_slug,
    has_scheduled_puzzle_on_or_before,
    lock_game,
    update_game_settings,
)
from app.models.game import Game

DEFAULT_VARIANT = "default"
# The same rule as the `slug_format` CHECK of `games`.
SLUG_PATTERN = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*", re.ASCII)


class GameSettingsError(Exception):
    """The settings cannot be applied. The message says why."""


@dataclass(frozen=True)
class GameChange:
    """What `apply_game_settings` did. `before_*` are None for a game it created.
    `hidden_scheduled` is the number of scheduled puzzles dated before the new launch date of a
    disabled game: they stay scheduled and hidden, and their answers stay used (decision 72)."""

    slug: str
    is_created: bool
    launch_date: dt.date
    is_enabled: bool
    before_launch_date: dt.date | None
    before_is_enabled: bool | None
    hidden_scheduled: int = 0

    def lines(self) -> list[str]:
        state = "enabled" if self.is_enabled else "disabled"
        if self.is_created:
            return [f"{self.slug}: created, launch date {self.launch_date}, {state}"]
        was_state = "enabled" if self.before_is_enabled else "disabled"
        changed = (self.before_launch_date, self.before_is_enabled) != (
            self.launch_date,
            self.is_enabled,
        )
        lines = [
            f"{self.slug}: launch date {self.launch_date}"
            + (
                f" (was {self.before_launch_date})"
                if self.before_launch_date != self.launch_date
                else ""
            )
            + f", {state}"
            + (f" (was {was_state})" if self.before_is_enabled != self.is_enabled else ""),
            "changed" if changed else "nothing changed",
        ]
        if self.hidden_scheduled:
            lines.append(
                f"{self.hidden_scheduled} scheduled puzzle(s) are dated before the new launch date:"
                " they stay scheduled and hidden, and their answers stay used (AC-59: no tool"
                " unschedules a puzzle of today or the past, even for a disabled game)."
            )
        return lines


async def apply_game_settings(
    db: AsyncSession,
    slug: str,
    *,
    launch_date: dt.date | None,
    is_enabled: bool | None,
    now: Callable[[], dt.datetime],
) -> GameChange:
    """Create the game when it is missing (it then needs a launch date, and is disabled unless
    `is_enabled`), or change what is given. A launch date that moves is refused when it would
    change what players have seen (AC-86, AC-62), and the message says which case applies:
    - accounts have stored games on the game's puzzles (in every case);
    - a scheduled puzzle is dated today or earlier: it is visible, and guests may have played it
      (their games live in the browser until they sign in);
    - a scheduled puzzle is dated before the new launch date: it would be hidden.
    A game that is disabled now and stays disabled shows nothing, so only the first case refuses;
    the number of scheduled puzzles before the new date is returned instead (`hidden_scheduled`).
    The view number counts from the launch date, so moving it renumbers every puzzle.

    `now` is a clock, called twice for a move: for the early refusal, and again after the game's
    row is locked, so a run that crosses 00:00 UTC is checked against the time of the write."""
    game = await get_game_by_slug(db, slug)
    if game is None:
        if SLUG_PATTERN.fullmatch(slug) is None:
            raise GameSettingsError("a slug has lower-case letters, digits and single hyphens")
        if launch_date is None:
            raise GameSettingsError(f"no game {slug!r} yet: creating it needs --launch-date")
        created = await add_game(
            db,
            slug=slug,
            default_variant=DEFAULT_VARIANT,
            launch_date=launch_date,
            is_enabled=bool(is_enabled),
        )
        return GameChange(slug, True, created.launch_date, created.is_enabled, None, None)
    if launch_date is not None and launch_date != game.launch_date:
        await _check_the_move(db, game, launch_date, is_enabled, utc_today(now()))
    game = await lock_game(db, game.id, is_shared=False)
    before = (game.launch_date, game.is_enabled)
    hidden = 0
    if launch_date is not None and launch_date != game.launch_date:
        hidden = await _check_the_move(db, game, launch_date, is_enabled, utc_today(now()))
    await update_game_settings(db, game, launch_date=launch_date, is_enabled=is_enabled)
    return GameChange(slug, False, game.launch_date, game.is_enabled, *before, hidden)


async def _check_the_move(
    db: AsyncSession, game: Game, new_date: dt.date, is_enabled: bool | None, today: dt.date
) -> int:
    """Raise `GameSettingsError` when moving the launch date to `new_date` would change what
    players saw. Returns the scheduled puzzles it would leave before the date, for a game that is
    disabled now and stays disabled (0 otherwise)."""
    if await game_has_stored_games(db, game.id):
        raise GameSettingsError(
            "the launch date cannot move: accounts have stored games on this game's puzzles"
        )
    stays_disabled = not game.is_enabled and not (
        game.is_enabled if is_enabled is None else is_enabled
    )
    hidden = await count_scheduled_puzzles_before(db, game.id, new_date)
    if stays_disabled:
        return hidden
    if await has_scheduled_puzzle_on_or_before(db, game.id, today):
        raise GameSettingsError(
            "the launch date cannot move: a scheduled puzzle is dated today or earlier, so it is"
            " visible and guests may have played it"
        )
    if hidden:
        raise GameSettingsError(
            f"the launch date cannot move to {new_date}: a scheduled puzzle is dated before it and"
            " would be hidden (unschedule it first)"
        )
    return 0
