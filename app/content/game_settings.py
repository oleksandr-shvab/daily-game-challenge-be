"""A game's launch date and its enabled flag (SPEC-04 AC-56, AC-61, Q17; owner decision 70).

`scripts/manage_game.py` is the command. The owner sets the production launch date with it before
Phase 5: scheduling starts from the later of tomorrow and that date (AC-56), and the view number
counts from it. A game this script creates is disabled until `--enable`, so a half-ready game is
never public.

Library code: it flushes and never commits, and it prints nothing.
"""

import datetime as dt
import re
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.crud.game import add_game, game_has_stored_games, get_game_by_slug, update_game_settings

DEFAULT_VARIANT = "default"
# The same rule as the `slug_format` CHECK of `games`.
SLUG_PATTERN = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*", re.ASCII)


class GameSettingsError(Exception):
    """The settings cannot be applied. The message says why."""


@dataclass(frozen=True)
class GameChange:
    """What `apply_game_settings` did. `before_*` are None for a game it created."""

    slug: str
    is_created: bool
    launch_date: dt.date
    is_enabled: bool
    before_launch_date: dt.date | None
    before_is_enabled: bool | None

    def lines(self) -> list[str]:
        state = "enabled" if self.is_enabled else "disabled"
        if self.is_created:
            return [f"{self.slug}: created, launch date {self.launch_date}, {state}"]
        was_state = "enabled" if self.before_is_enabled else "disabled"
        changed = (self.before_launch_date, self.before_is_enabled) != (
            self.launch_date,
            self.is_enabled,
        )
        return [
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


async def apply_game_settings(
    db: AsyncSession, slug: str, *, launch_date: dt.date | None, is_enabled: bool | None
) -> GameChange:
    """Create the game when it is missing (it then needs a launch date, and is disabled unless
    `is_enabled`), or change what is given. A launch date that moves is refused once any puzzle of
    the game has stored games: the view number counts from it and a dated puzzle before it is
    hidden, so a player's finished game would change under them (AC-86, AC-62)."""
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
    before = (game.launch_date, game.is_enabled)
    if (
        launch_date is not None
        and launch_date != game.launch_date
        and await game_has_stored_games(db, game.id)
    ):
        raise GameSettingsError(
            "the launch date cannot move: accounts have stored games on this game's puzzles"
        )
    await update_game_settings(db, game, launch_date=launch_date, is_enabled=is_enabled)
    return GameChange(slug, False, game.launch_date, game.is_enabled, *before)
