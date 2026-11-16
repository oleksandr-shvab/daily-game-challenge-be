"""The local review page for the puzzle drafts (SPEC-04 AC-47 to AC-54, M1-24a).

    ENVIRONMENT=local uv run python scripts/review_puzzles.py [--port 8765]

Then open http://localhost:8765/ (or http://127.0.0.1:8765/) in a browser. Each draft in review is
shown as the player will see it, with its answer and sources. Approve, or reject as "rebuild" (the
answer is fine, the clues are not: `build_puzzles.py --rebuilds` makes the slot again with other
screenshots) or "never" (the answer is unsuitable).

The page runs only when the ENVIRONMENT variable is `local`, the database host is this machine and
MEDIA_STORAGE is `local`, and never in CI. It binds 127.0.0.1 and answers only requests that name
itself (Host, Origin and a form token made when it starts). It shows answers on purpose. Use the
content database of CLAUDE.md ("Title import"), not the one the dev seed rewrites.

Exit code 0 when stopped, 1 when it refuses to start.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import uvicorn

from app.content.review.app import create_review_app
from app.content.target import TargetRefusedError
from app.core import db as core_db
from app.core.config import settings

HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def port_number(text: str) -> int:
    if not text.isascii() or not text.isdigit() or not 1024 <= int(text) <= 65535:
        raise argparse.ArgumentTypeError("use a port from 1024 to 65535")
    return int(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--port", type=port_number, default=DEFAULT_PORT)
    args = parser.parse_args()
    try:
        app = create_review_app(
            session_factory=core_db.async_session_maker,
            media_dir=settings.media_local_path,
            port=args.port,
        )
    except TargetRefusedError as error:
        print(f"review_puzzles: refusing to run: {error}", file=sys.stderr)
        return 1
    print(f"Review page: http://localhost:{args.port}/  (answers are shown; local only)")
    uvicorn.run(app, host=HOST, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
