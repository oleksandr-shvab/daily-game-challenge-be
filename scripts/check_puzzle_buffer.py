"""Check that tomorrow's puzzle is ready and how many days are scheduled ahead (SPEC-04 M1-25,
AC-73 to AC-82).

    uv run python scripts/check_puzzle_buffer.py

It reads the database and the media storage that `.env` and the environment name, for every enabled
game's default variant, and never writes (AC-78): there is no `--target` and no guard, so it may run
against production on the server. The first line says which environment, database and storage it
checked (no password or key). Then one line per game: the first day to be played (the later of
tomorrow and the launch date), whether its puzzle is ready, and the buffer, the ready days in a row
from that day, counted up to 60. Answers, titles, hints and media keys are never printed (AC-79).

Exit code, which the alert of M1-52 reads:
    0  everything is fine
    1  the buffer is under 7 days (a digest)
    2  tomorrow's puzzle is missing, not scheduled, incomplete or missing a file (a page)
    3  the check could not run: the database or storage is down, a usage error or a crash (unknown)
2 wins over 1. A game before the 7 days ahead of its launch date is "not launched" and never fails.
Python's own exit codes 1 (an uncaught error) and 2 (a usage error) are mapped to 3 here, so they
never mean "buffer low" or "tomorrow missing".
"""

import argparse
import asyncio
import datetime as dt
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# This script imports nothing from the app at module level: building the app's settings can fail,
# and that must exit 3 like any other crash, not with Python's 1. Nor does it import
# `app/content/media.py`, `igdb.py`, `fetch.py` or `builder.py` (they need Pillow, PyAV and httpx),
# so it runs on a server with runtime dependencies only.


class UsageError(Exception):
    pass


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # type: ignore[override]
        # argparse would print the usage and exit 2, which means "tomorrow missing" here.
        raise UsageError(message)


def build_parser() -> argparse.ArgumentParser:
    return _Parser(description=__doc__.split("\n")[0])


async def check(
    now: Callable[[], dt.datetime] | None, file_exists: Callable[[str], bool] | None
) -> int:
    from app.content.buffer_check import (
        CheckUnavailableError,
        ExitCode,
        check_buffer,
        database_session,
        describe_environment,
        exit_code,
        report_lines,
    )
    from app.core.clock import get_now
    from app.core.config import settings
    from app.media.writer import build_media_writer

    exists = file_exists or build_media_writer(settings).exists
    print(describe_environment(settings))
    try:
        async with database_session() as db:
            checks = await check_buffer(db, exists, now=now or get_now)
    except CheckUnavailableError as error:
        print(f"check_puzzle_buffer: could not check: {error}", file=sys.stderr)
        return ExitCode.CANNOT_CHECK
    print("\n".join(report_lines(checks)))
    return exit_code(checks)


async def run(
    argv: Sequence[str] | None = None,
    *,
    now: Callable[[], dt.datetime] | None = None,
    file_exists: Callable[[str], bool] | None = None,
) -> int:
    """The exit code of one run. Every way out is a code: 0 to 2 from the check, 3 for the rest.
    `now` and `file_exists` are for tests; a run uses the project clock and the storage `.env`
    names."""
    try:
        build_parser().parse_args(argv)
    except UsageError as error:
        print(f"check_puzzle_buffer: usage error: {error}", file=sys.stderr)
        return 3
    try:
        return await check(now, file_exists)
    except Exception as error:  # noqa: BLE001 - any crash is exit 3, never Python's 1
        # The type only: a settings or driver message could hold a value.
        print(f"check_puzzle_buffer: crashed: {type(error).__name__}", file=sys.stderr)
        return 3


def main() -> int:
    try:
        return asyncio.run(run(sys.argv[1:]))
    except Exception as error:  # noqa: BLE001 - as above
        print(f"check_puzzle_buffer: crashed: {type(error).__name__}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
