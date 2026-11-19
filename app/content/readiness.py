"""When a puzzle is ready (SPEC-04 AC-58, AC-73, AC-74): six clues, each meeting the media rule of
SPEC-02 AC-13b, and every file it names present in storage.

Scheduling uses it before a puzzle gets a date, the tomorrow check (M1-25) and promote (M1-24c)
use it too. It is pure apart from the `file_exists` calls it is given, and it reads the clues it is
handed: no session, no clock, no write. `file_exists` is blocking (`MediaWriter.exists`), so async
code calls this function in `asyncio.to_thread`.
"""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from app.games.modules.game_guesser import STEP_COUNT, MediaFileLike, clue_layout_problem
from app.media.signing import is_media_key


class ProblemKind(StrEnum):
    MISSING_CLUE = "missing_clue"
    BAD_MEDIA = "bad_media"
    MISSING_FILE = "missing_file"


@dataclass(frozen=True)
class Problem:
    """What is wrong at which step. Never a key, a name or a hint: it is safe to print."""

    kind: ProblemKind
    step: int

    def __str__(self) -> str:
        return f"{self.kind} at step {self.step}"


class _ClueLike(Protocol):
    """A `PuzzleClue` row with its media files loaded, or any object shaped like it."""

    @property
    def step(self) -> int: ...

    @property
    def media_type(self) -> str: ...

    @property
    def media_files(self) -> Iterable[MediaFileLike]: ...


def puzzle_problems(
    clues: Sequence[_ClueLike], *, file_exists: Callable[[str], bool]
) -> list[Problem]:
    """Every reason the puzzle is not ready, by step; an empty list means ready. A step that is
    absent is `missing_clue`; a clue whose files break AC-13b is `bad_media`; a clue with a key the
    storage does not have is `missing_file` (once per step)."""
    problems = []
    steps = {clue.step for clue in clues}
    problems += [
        Problem(ProblemKind.MISSING_CLUE, step)
        for step in range(1, STEP_COUNT + 1)
        if step not in steps
    ]
    for clue in sorted(clues, key=lambda clue: clue.step):
        files = list(clue.media_files)
        if clue_layout_problem(clue.step, clue.media_type, files) is not None:
            problems.append(Problem(ProblemKind.BAD_MEDIA, clue.step))
        # A key that is not a SHA-256 digest names no stored file, and a writer refuses to look.
        if not all(is_media_key(file.key) and file_exists(file.key) for file in files):
            problems.append(Problem(ProblemKind.MISSING_FILE, clue.step))
    return sorted(problems, key=lambda problem: problem.step)
