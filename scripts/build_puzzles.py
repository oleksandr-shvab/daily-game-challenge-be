"""The puzzle builder's command line (SPEC-04, M1-22).

    uv run python scripts/build_puzzles.py --counts [--slug game-guesser]
    ENVIRONMENT=local uv run python scripts/build_puzzles.py --target local --dry-run --next 5
    ENVIRONMENT=local uv run python scripts/build_puzzles.py --target local --next 40
    ENVIRONMENT=local uv run python scripts/build_puzzles.py --target local --slots 41-50
    ENVIRONMENT=local uv run python scripts/build_puzzles.py --target local --rebuilds
    ENVIRONMENT=local uv run python scripts/build_puzzles.py --target local --give-up 17

Build modes (`--slots A-B`, `--next N`, `--rebuilds`) write one draft per slot, in ascending order,
each whole or not at all (AC-46), as `in_review` puzzles with no date. A slot that already has a
puzzle is skipped, so a run again adds nothing. `--target` is required, must equal the ENVIRONMENT
variable, is printed before anything else (AC-44), and a target other than `local` also needs
`--confirm-target <name>`. Nothing runs in CI. While `IGDB_IMAGES_CLEARED_BY` is empty a target
other than `local` is refused (AC-2). `--dry-run` takes `--target` too (the plan's usage line), runs
every step but the writes, keeps its files in the working folder (`CONTENT_WORK_DIR`) and writes
nothing to the database or storage (AC-45). Answers are printed only with `--show-answers`.
Needs PUZZLE_SEED_SALT, IGDB_CLIENT_ID and IGDB_CLIENT_SECRET in `.env` (never commit or print
them) and the app's own settings. Step 6 is cut from the middle of the game's Steam trailer
(`store.steampowered.com`, no key needed; the calls are spaced 1.6 s apart); a game with no Steam id
or no readable trailer gets a slideshow of screenshots instead, and the report says which and why
(a Steam outage does not stop the run, it makes slideshows and is counted).

`--give-up SLOT` gives up rebuilding a slot: its `rebuild` rows become one `never` exclusion of the
title, which frees the slot for a new answer. It needs the target but not IGDB or the salt, and
prints no title.

`--counts` (AC-24) reads the candidate games from IGDB and prints how many pass each filter, in
order, so the owner can tune the thresholds (`CANDIDATE_*` in `.env`, AC-22) before building.
It writes nothing and prints no game name. It asks IGDB only for the candidate game types with at
least one rating, so "games read" is that number. It needs IGDB_CLIENT_ID and IGDB_CLIENT_SECRET in
`.env` (never commit or print them) and the app's own settings (`DATABASE_URL`, `SECRET_KEY`). The
database is read for the game's earlier answers and its "never" and "rebuild" exclusions; the game
must be in it (an unknown `--slug` exits 1). It takes about as long as the title import.

Exit code 0 on success, 1 when the guard refuses, a setting is missing, IGDB or the database cannot
be read or a slot fails.
"""

import argparse
import asyncio
import logging
import re
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.content.builder import (
    MAX_SLOT_NUMBER,
    MAX_SLOTS_PER_RUN,
    Builder,
    BuildReport,
    DatabaseFailure,
    GiveUpError,
    ImageFetcher,
    SlotLimitError,
    SlotRequest,
    UnknownGameError,
    database_session,
    give_up_slot,
)
from app.content.candidates import (
    Funnel,
    FunnelCounter,
    candidates_where,
    never_igdb_ids,
    rebuild_reserved_igdb_ids,
)
from app.content.fetch import FetchError
from app.content.igdb import DropReason, IgdbClient, IgdbError, SkipReason
from app.content.seed import SaltError
from app.content.settings import ContentSettings, MissingSettingError
from app.content.steam import SteamClient
from app.content.target import TargetRefusedError, check_image_gate, check_target
from app.core.clock import get_now, utc_today
from app.media.writer import MediaWriteError, build_media_writer

DEFAULT_SLUG = "game-guesser"


@dataclass
class CountsReport:
    funnel: Funnel
    skipped: Counter[SkipReason]
    dropped: Counter[DropReason]

    def lines(self) -> list[str]:
        """The records IGDB sent that the client skipped come first (AC-8), then the funnel, which
        starts from the games that were kept. No game name, no id."""
        funnel = self.funnel
        lines = [
            f"records skipped as invalid (AC-8): {sum(self.skipped.values())}"
            + _breakdown(self.skipped)
        ]
        lines += [f"{step.name}: passed {step.passed}" for step in funnel.steps]
        lines.append(f"candidates that survive one rebuild: {funnel.survive_one_rebuild}")
        lines.append(
            f"items dropped from records that were kept: {sum(self.dropped.values())}"
            + _breakdown(self.dropped)
        )
        return lines


def _breakdown(counter: Counter[Any]) -> str:
    parts = [f"{reason}: {count}" for reason, count in sorted(counter.items()) if count]
    return f" ({', '.join(parts)})" if parts else ""


async def run_counts(settings: ContentSettings, slug: str) -> CountsReport:
    from app.crud.exclusion import list_exclusions
    from app.crud.game import get_game_by_slug
    from app.crud.puzzle import list_answer_igdb_ids

    now = get_now()
    # The whole session, its opening and closing included, is database work (`DatabaseFailure`).
    async with database_session() as db:
        game = await get_game_by_slug(db, slug)
        if game is None:
            raise UnknownGameError(f"no game with the slug {slug!r} in the database")
        earlier = await list_answer_igdb_ids(db, game.id)
        exclusions = await list_exclusions(db, game.id)
    counter = FunnelCounter(
        today=utc_today(now),
        settings=settings,
        earlier_answer_igdb_ids=earlier,
        never_igdb_ids=never_igdb_ids(exclusions),
        reserved_igdb_ids=rebuild_reserved_igdb_ids(exclusions),
        keep_candidates=False,
    )
    async with IgdbClient.from_settings(settings) as igdb:
        # One game at a time: only the games that pass are kept.
        async for igdb_game in igdb.games(now=now, where=candidates_where(settings)):
            counter.add(igdb_game)
        return CountsReport(counter.result(), igdb.skipped, igdb.dropped)


SLOT_RANGE = re.compile(r"(\d{1,6})(?:-(\d{1,6}))?", re.ASCII)


def parse_slots(text: str) -> range:
    """`41-50` or `41`, both ends included."""
    match = SLOT_RANGE.fullmatch(text)
    if match is None:
        raise argparse.ArgumentTypeError("use A-B or one number, such as 41-50")
    first = int(match.group(1))
    last = int(match.group(2) or first)
    if first < 1 or last < first:
        raise argparse.ArgumentTypeError("slots start at 1 and A-B needs A <= B")
    if last - first + 1 > MAX_SLOTS_PER_RUN:
        raise argparse.ArgumentTypeError(f"a run builds at most {MAX_SLOTS_PER_RUN} slots")
    return range(first, last + 1)


def bounded_int(low: int, high: int) -> Callable[[str], int]:
    """An argparse type: plain digits, from `low` to `high`."""

    def parse(text: str) -> int:
        if not text.isascii() or not text.isdigit() or not low <= int(text) <= high:
            raise argparse.ArgumentTypeError(f"use a whole number from {low} to {high}")
        return int(text)

    return parse


# `--slots` takes the same: six digits at most.
slot_number = bounded_int(1, MAX_SLOT_NUMBER)
slot_count = bounded_int(1, MAX_SLOTS_PER_RUN)


async def run_build(
    settings: ContentSettings,
    args: argparse.Namespace,
    request: SlotRequest,
    report: BuildReport,
) -> None:
    """Fills `report` slot by slot, so a run that stops half way still shows what it finished."""
    from app.core.config import settings as app_settings

    now = get_now()
    writer = None if args.dry_run else build_media_writer(app_settings)
    async with (
        IgdbClient.from_settings(settings) as igdb,
        ImageFetcher() as images,
        SteamClient() as steam,
        database_session() as db,
    ):
        builder = Builder(
            db,
            game_slug=args.slug,
            target=args.target,
            settings=settings,
            app_settings=app_settings,
            now=now,
            games=lambda: igdb.games(now=now, where=candidates_where(settings)),
            fetch_image=images.fetch,
            writer=writer,
            dry_run=args.dry_run,
            trailers=steam,
        )
        await builder.run(request, report)


async def run_give_up(slug: str, slot: int) -> int:
    async with database_session() as db:
        return await give_up_slot(db, slug, slot)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--counts", action="store_true", help="print how many games pass each filter"
    )
    parser.add_argument("--slug", default=DEFAULT_SLUG, help="the game to build for")
    parser.add_argument("--target", help="where to write; must equal ENVIRONMENT (build modes)")
    parser.add_argument("--confirm-target", help="the target name again, for a non-local target")
    modes = parser.add_argument_group("build modes (exactly one)")
    modes.add_argument("--slots", type=parse_slots, metavar="A-B", help="build these slots")
    modes.add_argument(
        "--next", type=slot_count, metavar="N", help="build the next N slots after the highest"
    )
    modes.add_argument("--rebuilds", action="store_true", help="build the slots waiting for one")
    parser.add_argument("--dry-run", action="store_true", help="do everything but write")
    parser.add_argument("--show-answers", action="store_true", help="print the answers' names")
    parser.add_argument(
        "--give-up",
        type=slot_number,
        metavar="SLOT",
        help="give up rebuilding SLOT: its title is excluded for good and the slot is free",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    build_modes = [args.slots, args.next, args.rebuilds or None]
    is_build = any(mode is not None for mode in build_modes)
    if args.counts + (args.give_up is not None) + is_build != 1:
        parser.error("use exactly one of --counts, --give-up, and --slots, --next or --rebuilds")
    if sum(mode is not None for mode in build_modes) > 1:
        parser.error("use only one of --slots, --next and --rebuilds")
    if (args.dry_run or args.show_answers) and not is_build:
        parser.error("--dry-run and --show-answers belong to the build modes")
    # httpx logs every request URL at INFO; keep the run quiet.
    logging.basicConfig(level=logging.WARNING)
    settings = ContentSettings()
    report = BuildReport()
    # The message of a run that stops early, printed after the report of what it finished.
    error_message: str | None = None
    is_started = False
    try:
        if args.counts:
            settings.require("IGDB_CLIENT_ID", "IGDB_CLIENT_SECRET")
            print("\n".join(asyncio.run(run_counts(settings, args.slug)).lines()))
            return 0
        # The writing modes (and the dry run, which names its target too): the guard first, and
        # the target is the first thing printed (AC-44).
        target = check_target(
            args.target,
            args.confirm_target,
            ask=input if sys.stdin.isatty() else None,
            needs_local_storage=True,
        )
        print(target.describe())
        print(f"IGDB images cleared by: {settings.IGDB_IMAGES_CLEARED_BY or 'nobody (local only)'}")
        if args.give_up is not None:
            count = asyncio.run(run_give_up(args.slug, args.give_up))
            print(
                f"slot {args.give_up}: gave up {count} rebuild row(s); the title is excluded for"
                " good and the slot is free"
                if count
                else f"slot {args.give_up}: no rebuild rows to give up; nothing changed"
            )
            return 0
        # `check_target` has already refused a bucket for the target `local`; the builder checks
        # the writer's storage kind as well.
        check_image_gate(args.target, settings.IGDB_IMAGES_CLEARED_BY)
        settings.require("IGDB_CLIENT_ID", "IGDB_CLIENT_SECRET")
        request = SlotRequest(slots=args.slots, next_count=args.next, rebuilds=args.rebuilds)
        is_started = True
        asyncio.run(run_build(settings, args, request, report))
    except (
        TargetRefusedError,
        MissingSettingError,
        SaltError,
        UnknownGameError,
        GiveUpError,
        SlotLimitError,
    ) as error:
        error_message = f"refusing to run: {error}"
    except (IgdbError, FetchError, httpx.HTTPError) as error:
        error_message = str(error)
    except DatabaseFailure as error:
        # The draft being written was rolled back; drafts of earlier slots stay.
        error_message = f"database error: {error}"
    except (MediaWriteError, OSError) as error:
        # Database errors were named by `database_session`.
        detail = str(error) if isinstance(error, MediaWriteError) else type(error).__name__
        error_message = f"storage or file error: {detail}"
    finally:
        # One place for every way out of a build run, Ctrl-C and a bug included: what the run
        # finished is shown first.
        if is_started:
            if args.dry_run:
                print("Dry run: nothing was written to the database or storage.")
            if report.has_content:
                print("\n".join(report.lines(show_answers=args.show_answers)))
    if error_message is not None:
        print(f"build_puzzles: {error_message}", file=sys.stderr)
        return 1
    return 0 if report.is_ok else 1


if __name__ == "__main__":
    sys.exit(main())
