"""Schedule approved puzzles onto dates, or move one back (SPEC-04 M1-24, AC-55 to AC-60).

    ENVIRONMENT=local uv run python scripts/schedule_puzzles.py --target local
    ENVIRONMENT=local uv run python scripts/schedule_puzzles.py --target local --until 2026-12-31
    ENVIRONMENT=local uv run python scripts/schedule_puzzles.py --target local --unschedule 2026-12-05
    ENVIRONMENT=staging uv run python scripts/schedule_puzzles.py --target staging --confirm-target staging

Scheduling gives the approved, ready puzzles the free dates from the later of tomorrow and the
game's launch date, in the order they were approved. No two answers of one series fall within 14
days, an answer never repeats, and the run stops at the first date nothing fits instead of leaving
a hole (decision 69); approve more puzzles of other series and run it again. A puzzle that is not
complete, or has a file missing in the target's storage, stays `approved` and is reported with the
step (AC-58); files are checked only for a puzzle the walk reaches. A puzzle whose answer is already
used (scheduled in any variant, or on an old dated row) is never scheduled and is reported. Today
and past dates are never touched (AC-59).

`--unschedule DATE` moves the scheduled puzzle of that date back to `approved` and frees the date.
It refuses today and past dates, unless the game is disabled (nothing shows then, so a launch date
that moves later can free the puzzles before it), and a puzzle with stored games (AC-59, AC-86). The puzzle goes
last in the approval queue, so the next scheduling run does not give it the same date again.

`--target` is required, must equal the ENVIRONMENT variable, is printed before anything else, and a
target other than `local` also needs `--confirm-target <name>`. Nothing runs in CI (AC-44). The
storage is the one `.env` names, local folder or bucket, and is only read. Dates and counts are
printed; answers only with `--show-answers`; never a key or a credential (AC-83).

Exit code 0 on success (also when the walk stopped early: the report says where), 1 when the guard
refuses, the game is unknown, an unschedule is refused, or the database or storage fails.
"""

import argparse
import asyncio
import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.content.builder import DatabaseFailure, UnknownGameError, database_session
from app.content.schedule import (
    ScheduleRefusedError,
    ScheduleResult,
    StopReason,
    UnscheduleRefusedError,
    schedule,
    unschedule,
)
from app.content.target import TargetRefusedError, check_target
from app.core.clock import get_now
from app.core.config import settings as app_settings
from app.crud.game import get_game_by_slug
from app.media.writer import MediaWriteError, build_media_writer

DEFAULT_SLUG = "game-guesser"


def report_lines(result: ScheduleResult, *, show_answers: bool) -> list[str]:
    lines = [f"first schedulable day: {result.first_day}"]
    for item in result.scheduled:
        answer = f"  {item.answer_name}" if show_answers else ""
        lines.append(
            f"scheduled {item.date}: puzzle {item.puzzle_id}, slot {item.build_slot}{answer}"
        )
    lines.append(f"scheduled {len(result.scheduled)} puzzle(s)")
    for problem in result.not_ready:
        steps = ", ".join(str(item) for item in problem.problems)
        lines.append(
            f"not ready, stays approved: puzzle {problem.puzzle_id}, slot {problem.build_slot}"
            f" ({steps})"
        )
    for puzzle_id in result.repeats:
        lines.append(f"not scheduled, its answer is already used: puzzle {puzzle_id}")
    for date in result.blocked_dates:
        lines.append(
            f"passed {date}: an old puzzle that is not scheduled holds that date"
            " (re-run the dev seed, or set that row's date to NULL)"
        )
    if result.stop_reason == StopReason.NOTHING_FITS:
        lines.append(
            f"stopped at {result.stopped_on}: no approved, ready puzzle fits that date"
            " (approve more puzzles of other series, then run again)"
        )
    elif result.stop_reason == StopReason.UNTIL:
        lines.append(f"stopped at --until: next date would be {result.stopped_on}")
    else:
        lines.append("stopped: no approved, ready puzzle is left")
    lines.append(f"buffer: {result.buffer} day(s) with a ready puzzle from tomorrow")
    if result.buffer_from_launch is not None:
        lines.append(
            f"before launch: {result.buffer_from_launch} day(s) with a ready puzzle from the"
            f" launch date {result.first_day} (the check counts from the launch date)"
        )
    return lines


async def run(args: argparse.Namespace) -> list[str]:
    async with database_session() as db:
        game = await get_game_by_slug(db, args.slug)
        if game is None:
            raise UnknownGameError(f"no game with the slug {args.slug!r} in the database")
        if args.unschedule is not None:
            puzzle = await unschedule(db, game, args.unschedule, now=get_now)
            await db.commit()
            moved = f"puzzle {puzzle.id}, slot {puzzle.build_slot}"
            return [
                f"{args.unschedule}: back to approved ({moved}); the puzzle is last in the queue"
            ]
        # The storage is needed to schedule only, so an unschedule builds no writer.
        writer = build_media_writer(app_settings)
        # A `ScheduleRefusedError` leaves before the commit, so the whole run is rolled back.
        result = await schedule(db, game, writer.exists, now=get_now, until=args.until)
        await db.commit()
    return report_lines(result, show_answers=args.show_answers)


def parse_date(text: str) -> dt.date:
    try:
        return dt.date.fromisoformat(text)
    except ValueError:
        raise argparse.ArgumentTypeError("use a date such as 2026-12-31") from None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--slug", default=DEFAULT_SLUG, help="the game to schedule for")
    parser.add_argument("--target", help="where to write; must equal ENVIRONMENT")
    parser.add_argument("--confirm-target", help="the target name again, for a non-local target")
    parser.add_argument("--until", type=parse_date, metavar="DATE", help="schedule up to this day")
    parser.add_argument(
        "--unschedule", type=parse_date, metavar="DATE", help="move this day's puzzle to approved"
    )
    parser.add_argument("--show-answers", action="store_true", help="print the answers' names")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.unschedule is not None and (args.until is not None or args.show_answers):
        parser.error("--unschedule takes neither --until nor --show-answers")
    try:
        # The storage may be a bucket: this script only reads it, so no local folder is required.
        target = check_target(
            args.target,
            args.confirm_target,
            ask=input if sys.stdin.isatty() else None,
            needs_local_storage=False,
        )
        print(target.describe())
        print("\n".join(asyncio.run(run(args))))
    except (TargetRefusedError, UnknownGameError, UnscheduleRefusedError) as error:
        print(f"schedule_puzzles: refusing to run: {error}", file=sys.stderr)
        return 1
    except ScheduleRefusedError as error:
        print(
            f"schedule_puzzles: {error}. Nothing was saved. Run it again.",
            file=sys.stderr,
        )
        return 1
    except DatabaseFailure as error:
        print(f"schedule_puzzles: database error: {error}", file=sys.stderr)
        return 1
    except MediaWriteError as error:
        print(f"schedule_puzzles: storage error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
