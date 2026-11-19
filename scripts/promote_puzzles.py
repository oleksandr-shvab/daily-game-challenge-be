"""Copy the approved puzzles of the reviewed (local) database to another environment (SPEC-04
AC-2, AC-44; owner decision 68, R-10).

    ENVIRONMENT=staging uv run python scripts/promote_puzzles.py --target staging \\
        --confirm-target staging \\
        --source-database-url postgresql+asyncpg://user:pass@localhost:5544/content_db \\
        --source-media-dir var/content-media

Review stays in `local`; this is how an approved puzzle reaches the target, as `approved` with no
date (rows and files). The target is the environment the `.env` and the ENVIRONMENT variable name
(`DATABASE_URL`, `MEDIA_STORAGE`, `S3_*`); the source is only read, and a relative
`--source-media-dir` is read from the backend repo root (as `MEDIA_LOCAL_DIR` is), not from the
working directory. Run the title import on the
target first, and `manage_game.py` if the game is not there yet. A slot the target already has is
left alone, so a re-run adds nothing, and a puzzle with stored games is never touched (AC-86). A
puzzle that is not ready, whose answer is missing from the target's titles or is already used there
is refused and reported by slot; the others go on.

`--target` is required, must equal the ENVIRONMENT variable, is printed before anything else, and
a target other than `local` also needs `--confirm-target <name>`. Nothing runs in CI (AC-44). While
`IGDB_IMAGES_CLEARED_BY` is empty a target other than `local`, and any bucket, is refused before
the source is opened (AC-2). Answers are printed only with `--show-answers`; keys, URLs with a
password and credentials never.

Exit code 0 on success, 1 when the guard refuses, a puzzle is refused, or a database or storage
call fails (the report says which database).
"""

import argparse
import asyncio
import sys
from pathlib import Path

from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.content.builder import DatabaseFailure, UnknownGameError
from app.content.promote import (
    PromoteRefusedError,
    PromoteReport,
    SourceDatabaseFailure,
    run_promote,
)
from app.content.settings import ContentSettings
from app.content.target import TargetRefusedError, check_target
from app.core.config import settings as app_settings
from app.media.writer import MediaWriteError, build_media_writer

DEFAULT_SLUG = "game-guesser"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--slug", default=DEFAULT_SLUG, help="the game to promote for")
    parser.add_argument("--target", help="where to write; must equal ENVIRONMENT")
    parser.add_argument("--confirm-target", help="the target name again, for a non-local target")
    parser.add_argument(
        "--source-database-url", required=True, help="the reviewed database, postgresql+asyncpg://"
    )
    parser.add_argument(
        "--source-media-dir",
        required=True,
        type=Path,
        help="the folder of its media files; a relative path is read from the backend repo root",
    )
    parser.add_argument("--show-answers", action="store_true", help="print the answers' names")
    return parser


def source_description(url: str) -> str:
    """No user name, password or query: the host, port and database name only."""
    try:
        parsed = make_url(url)
    except ArgumentError:
        return "an unreadable URL"
    return f"{parsed.host}:{parsed.port or 5432}/{parsed.database}"


def main() -> int:
    args = build_parser().parse_args()
    # Like `settings.media_local_path`: a relative folder is read from the repo root, not from the
    # working directory the command happens to run in.
    source_dir = ROOT / args.source_media_dir
    cleared_by = ContentSettings().IGDB_IMAGES_CLEARED_BY
    report = PromoteReport()
    error_message: str | None = None
    is_started = False
    try:
        # The target's files may be in a bucket; the source's are always a folder.
        target = check_target(
            args.target,
            args.confirm_target,
            ask=input if sys.stdin.isatty() else None,
            needs_local_storage=False,
        )
        print(target.describe())
        print(f"IGDB images cleared by: {cleared_by or 'nobody (local only)'}")
        print(
            f"Source: database {source_description(args.source_database_url)}; folder {source_dir}"
        )
        writer = build_media_writer(app_settings)
        is_started = True
        asyncio.run(
            run_promote(
                slug=args.slug,
                source_url=args.source_database_url,
                source_dir=source_dir,
                target_name=target.name,
                target_url=app_settings.DATABASE_URL,
                cleared_by=cleared_by,
                writer=writer,
                report=report,
            )
        )
    except (TargetRefusedError, PromoteRefusedError, UnknownGameError) as error:
        error_message = f"refusing to run: {error}"
    except SourceDatabaseFailure as error:
        error_message = f"source database error: {error}"
    except DatabaseFailure as error:
        # The puzzle being written was rolled back; puzzles finished before it stay.
        error_message = f"target database error: {error}"
    except (MediaWriteError, OSError) as error:
        detail = str(error) if isinstance(error, MediaWriteError) else type(error).__name__
        error_message = f"storage or file error: {detail}"
    finally:
        if is_started and report.has_content:
            print("\n".join(report.lines(show_answers=args.show_answers)))
    if error_message is not None:
        print(f"promote_puzzles: {error_message}", file=sys.stderr)
        return 1
    return 0 if report.is_ok else 1


if __name__ == "__main__":
    sys.exit(main())
