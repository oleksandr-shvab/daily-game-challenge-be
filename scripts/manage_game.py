"""Create a game, set its launch date, enable or disable it (SPEC-04 AC-56, AC-61, Q17).

    ENVIRONMENT=local uv run python scripts/manage_game.py --target local
    ENVIRONMENT=local uv run python scripts/manage_game.py --target local --launch-date 2026-12-01
    ENVIRONMENT=production uv run python scripts/manage_game.py --target production \\
        --confirm-target production --launch-date 2027-01-15 --enable

This is how the owner sets the production launch date before Phase 5: scheduling starts from the
later of tomorrow and that date. A game that is missing is created (`default_variant` `default`,
video games, disabled unless `--enable`) and then needs `--launch-date`. With no option the
command only prints the game's settings. The launch date does not move once any account has stored
games on the game's puzzles.

`--target` is required, must equal the ENVIRONMENT variable, is printed before anything else, and
a target other than `local` also needs `--confirm-target <name>`. Nothing runs in CI (AC-44).

Exit code 0 on success, 1 when the guard refuses, the settings are refused or the database fails.
"""

import argparse
import asyncio
import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.content.builder import DatabaseFailure, database_session
from app.content.game_settings import GameSettingsError, apply_game_settings
from app.content.target import TargetRefusedError, check_target
from app.crud.game import get_game_by_slug

DEFAULT_SLUG = "game-guesser"


async def run(args: argparse.Namespace) -> list[str]:
    is_enabled = True if args.enable else False if args.disable else None
    async with database_session() as db:
        if args.launch_date is None and is_enabled is None:
            game = await get_game_by_slug(db, args.slug)
            if game is None:
                raise GameSettingsError(
                    f"no game {args.slug!r} yet: creating it needs --launch-date"
                )
            state = "enabled" if game.is_enabled else "disabled"
            return [f"{game.slug}: launch date {game.launch_date}, {state}"]
        change = await apply_game_settings(
            db, args.slug, launch_date=args.launch_date, is_enabled=is_enabled
        )
        await db.commit()
    return change.lines()


def parse_date(text: str) -> dt.date:
    try:
        return dt.date.fromisoformat(text)
    except ValueError:
        raise argparse.ArgumentTypeError("use a date such as 2027-01-15") from None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--slug", default=DEFAULT_SLUG, help="the game to change")
    parser.add_argument("--target", help="where to write; must equal ENVIRONMENT")
    parser.add_argument("--confirm-target", help="the target name again, for a non-local target")
    parser.add_argument("--launch-date", type=parse_date, metavar="DATE", help="YYYY-MM-DD")
    switch = parser.add_mutually_exclusive_group()
    switch.add_argument("--enable", action="store_true", help="make the game public")
    switch.add_argument("--disable", action="store_true", help="hide the game")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        # No media is touched, so the storage kind does not matter.
        target = check_target(
            args.target,
            args.confirm_target,
            ask=input if sys.stdin.isatty() else None,
            needs_local_storage=False,
        )
        print(target.describe())
        print("\n".join(asyncio.run(run(args))))
    except (TargetRefusedError, GameSettingsError) as error:
        print(f"manage_game: refusing to run: {error}", file=sys.stderr)
        return 1
    except DatabaseFailure as error:
        print(f"manage_game: database error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
