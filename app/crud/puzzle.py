import datetime as dt
import uuid
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import NamedTuple

from sqlalchemy import ColumnElement, and_, exists, func, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.models.game import Game
from app.models.progress import Progress, ProgressGuess
from app.models.puzzle import ClueMediaFile, Puzzle, PuzzleClue, PuzzleStatus
from app.models.title import Title


def visible_puzzle_condition(today: dt.date) -> ColumnElement[bool]:
    """The only visibility rule for puzzles (AC-12). Every puzzle query takes it; do not copy it
    into Python.

    The game terms are a correlated `EXISTS`, so it works in any `select(Puzzle)` with no join to
    `Game`. A later admin preview (Q12) widens the rule here and nowhere else.
    """
    return and_(
        Puzzle.status == PuzzleStatus.SCHEDULED,
        Puzzle.date <= today,
        exists(
            select(Game.id)
            .where(
                Game.id == Puzzle.game_id,
                Game.is_enabled.is_(True),
                Puzzle.date >= Game.launch_date,
            )
            # Correlate only to `puzzles`, so the condition also works when the outer query
            # already joins `games`.
            .correlate(Puzzle)
        ),
    )


async def get_visible_puzzle_with_details(
    db: AsyncSession, *, game_id: int, variant: str, puzzle_date: dt.date, today: dt.date
) -> Puzzle | None:
    result = await db.execute(
        select(Puzzle)
        .where(
            Puzzle.game_id == game_id,
            Puzzle.variant == variant,
            Puzzle.date == puzzle_date,
            visible_puzzle_condition(today),
        )
        .options(
            joinedload(Puzzle.answer_title),
            joinedload(Puzzle.clues).joinedload(PuzzleClue.media_files),
        )
    )
    return result.unique().scalar_one_or_none()


class ArchiveRow(NamedTuple):
    """Only an account's query sets `status` and `guesses_used`; a guest's returns the date alone.
    `status` is None when the account never started the puzzle."""

    date: dt.date
    status: str | None = None
    guesses_used: int | None = None


async def list_archive_puzzles(
    db: AsyncSession,
    *,
    game_id: int,
    variant: str,
    before: dt.date,
    today: dt.date,
    limit: int,
    user_id: uuid.UUID | None,
) -> list[ArchiveRow]:
    """A keyset page (AC-63): the caller asks for `limit + 1` to learn whether a next page exists.
    Days with no visible puzzle are absent (AC-65). Nothing of the progress but the status and
    the guess count leaves this query.

    `before` must not be after `today`: the caller clamps it, and `visible_puzzle_condition` is a
    second guard."""
    where = (
        Puzzle.game_id == game_id,
        Puzzle.variant == variant,
        Puzzle.date < before,
        visible_puzzle_condition(today),
    )
    if user_id is None:
        query = select(Puzzle.date).where(*where)
    else:
        guesses_used = (
            select(func.count(ProgressGuess.id))
            .where(ProgressGuess.progress_id == Progress.id)
            .scalar_subquery()
        )
        query = (
            select(Puzzle.date, Progress.status, guesses_used)
            .outerjoin(Progress, and_(Progress.puzzle_id == Puzzle.id, Progress.user_id == user_id))
            .where(*where)
        )
    result = await db.execute(query.order_by(Puzzle.date.desc()).limit(limit))
    return [ArchiveRow(*row) for row in result.all()]


async def list_visible_puzzles(
    db: AsyncSession, *, game_id: int, keys: Collection[tuple[str, dt.date]], today: dt.date
) -> list[Puzzle]:
    """Loads the answer title only, no clues or media: enough to replay guesses, never enough to
    show a view."""
    if not keys:
        return []
    result = await db.execute(
        select(Puzzle)
        .where(
            Puzzle.game_id == game_id,
            tuple_(Puzzle.variant, Puzzle.date).in_(list(keys)),
            visible_puzzle_condition(today),
        )
        .options(joinedload(Puzzle.answer_title))
    )
    return list(result.scalars())


async def list_answer_igdb_ids(db: AsyncSession, game_id: int) -> set[int]:
    """The IGDB ids of every answer the game has had, in any status and variant (AC-25). A title
    that did not come from IGDB has no id and is left out."""
    result = await db.execute(
        select(Title.igdb_id)
        .join(Puzzle, Puzzle.answer_title_id == Title.id)
        .where(Puzzle.game_id == game_id, Title.igdb_id.is_not(None))
        .distinct()
    )
    return set(result.scalars())


async def list_puzzle_slots(db: AsyncSession, game_id: int, variant: str) -> set[int]:
    """The build slots that have a puzzle, in any status."""
    result = await db.execute(
        select(Puzzle.build_slot).where(
            Puzzle.game_id == game_id, Puzzle.variant == variant, Puzzle.build_slot.is_not(None)
        )
    )
    return set(result.scalars())


@dataclass(frozen=True)
class MediaFileInput:
    role: str
    key: str
    content_type: str
    byte_size: int
    width: int
    height: int
    duration_ms: int | None


@dataclass(frozen=True)
class ClueInput:
    step: int
    media_type: str
    hint_kind: str | None
    hint_value: str | None
    source_url: str
    source_image_ids: list[str]
    files: Sequence[MediaFileInput]


async def add_clue_media_file(db: AsyncSession, clue_id: int, file: MediaFileInput) -> None:
    db.add(
        ClueMediaFile(
            clue_id=clue_id,
            role=file.role,
            key=file.key,
            content_type=file.content_type,
            byte_size=file.byte_size,
            width=file.width,
            height=file.height,
            duration_ms=file.duration_ms,
        )
    )
    await db.flush()


async def add_built_puzzle(
    db: AsyncSession,
    *,
    game_id: int,
    variant: str,
    build_slot: int,
    generator_version: str,
    answer_title_id: int,
    status: PuzzleStatus,
    clues: Sequence[ClueInput],
) -> Puzzle:
    """The builder's insert: a puzzle with no date, its clues and their media rows. It flushes and
    never commits; the caller owns the transaction, so a draft exists whole or not at all."""
    puzzle = Puzzle(
        game_id=game_id,
        variant=variant,
        date=None,
        status=status,
        build_slot=build_slot,
        generator_version=generator_version,
        answer_title_id=answer_title_id,
    )
    db.add(puzzle)
    await db.flush()
    for clue_input in clues:
        clue = PuzzleClue(
            puzzle_id=puzzle.id,
            step=clue_input.step,
            media_type=clue_input.media_type,
            hint_kind=clue_input.hint_kind,
            hint_value=clue_input.hint_value,
            source_url=clue_input.source_url,
            source_image_ids=clue_input.source_image_ids,
        )
        db.add(clue)
        await db.flush()
        for file in clue_input.files:
            await add_clue_media_file(db, clue.id, file)
    return puzzle


# --- Scheduling (M1-24b, SPEC-04 AC-55 to AC-60, AC-86) ---

_WITH_ANSWER_AND_CLUES = (
    joinedload(Puzzle.answer_title),
    joinedload(Puzzle.clues).joinedload(PuzzleClue.media_files),
)


async def list_dated_puzzles(db: AsyncSession, game_id: int, variant: str) -> list[Puzzle]:
    """Every puzzle of the game and variant that has a date, in any status, with its answer title
    and no clues. A scheduled one counts for the series and repeat rules, any of them occupies its
    date (the unique key `(game_id, variant, date)`)."""
    result = await db.execute(
        select(Puzzle)
        .where(Puzzle.game_id == game_id, Puzzle.variant == variant, Puzzle.date.is_not(None))
        .options(joinedload(Puzzle.answer_title))
        .order_by(Puzzle.date)
    )
    return list(result.scalars())


async def list_approved_puzzles(db: AsyncSession, game_id: int, variant: str) -> list[Puzzle]:
    """The approved puzzles in the order they were approved, then by id (R-16), with the answer
    title, clues and media files. A NULL `approved_at` (no CHECK forbids one, and a plain
    `ORDER BY` would put it last) counts as the oldest: it goes first."""
    result = await db.execute(
        select(Puzzle)
        .where(
            Puzzle.game_id == game_id,
            Puzzle.variant == variant,
            Puzzle.status == PuzzleStatus.APPROVED,
        )
        .options(*_WITH_ANSWER_AND_CLUES)
        .order_by(Puzzle.approved_at.asc().nulls_first(), Puzzle.id)
    )
    return list(result.unique().scalars())


async def list_dated_answers(db: AsyncSession, game_id: int) -> list[tuple[int, int]]:
    """`(puzzle id, answer title id)` of every puzzle of the game that has a date, in every
    variant and any status: the answers that are used, for the repeat rule. A dated puzzle that is
    not scheduled (an old draft) counts, so its answer is never given a second date."""
    result = await db.execute(
        select(Puzzle.id, Puzzle.answer_title_id).where(
            Puzzle.game_id == game_id, Puzzle.date.is_not(None)
        )
    )
    return [(puzzle_id, title_id) for puzzle_id, title_id in result.all()]


async def list_scheduled_puzzles_between(
    db: AsyncSession, game_id: int, variant: str, start: dt.date, end: dt.date
) -> list[Puzzle]:
    """The scheduled puzzles dated from `start` up to but not including `end`, by date, with clues
    and media files."""
    result = await db.execute(
        select(Puzzle)
        .where(
            Puzzle.game_id == game_id,
            Puzzle.variant == variant,
            Puzzle.status == PuzzleStatus.SCHEDULED,
            Puzzle.date >= start,
            Puzzle.date < end,
        )
        .options(*_WITH_ANSWER_AND_CLUES)
        .order_by(Puzzle.date)
    )
    return list(result.unique().scalars())


async def lock_scheduled_puzzle_on_date(
    db: AsyncSession, game_id: int, variant: str, puzzle_date: dt.date
) -> Puzzle | None:
    """The scheduled puzzle of the date, its row locked until the transaction ends and read fresh,
    so the caller's checks and its write see the same row."""
    result = await db.execute(
        select(Puzzle)
        .where(
            Puzzle.game_id == game_id,
            Puzzle.variant == variant,
            Puzzle.date == puzzle_date,
            Puzzle.status == PuzzleStatus.SCHEDULED,
        )
        .with_for_update(of=Puzzle)
        .execution_options(populate_existing=True)
    )
    return result.scalar_one_or_none()


async def has_stored_games(db: AsyncSession, puzzle_id: int) -> bool:
    """True when any account has progress on the puzzle (AC-86): its answer, clues and date must
    never change then."""
    result = await db.execute(select(exists().where(Progress.puzzle_id == puzzle_id)))
    return bool(result.scalar())


async def set_puzzle_schedule(db: AsyncSession, puzzle: Puzzle, puzzle_date: dt.date) -> None:
    """Approved to scheduled on `puzzle_date`. Flushes, never commits: the caller owns the
    transaction."""
    puzzle.date = puzzle_date
    puzzle.status = PuzzleStatus.SCHEDULED
    await db.flush()


async def clear_puzzle_schedule(db: AsyncSession, puzzle: Puzzle, now: dt.datetime) -> None:
    """Scheduled back to approved, with no date. `approved_at` becomes `now`: the puzzle goes to
    the back of the queue, so the next scheduling run does not put it straight back."""
    puzzle.date = None
    puzzle.approved_at = now
    puzzle.status = PuzzleStatus.APPROVED
    await db.flush()


# ---- The review page (M1-24a). Kept in one block, apart from the builder's functions above. ----


async def list_in_review_puzzles(db: AsyncSession) -> list[Puzzle]:
    """The drafts waiting for the reviewer, lowest build slot first (a puzzle with no slot last),
    with the answer title only."""
    result = await db.execute(
        select(Puzzle)
        .where(Puzzle.status == PuzzleStatus.IN_REVIEW)
        .order_by(Puzzle.build_slot.asc().nulls_last(), Puzzle.id)
        .options(joinedload(Puzzle.answer_title))
    )
    return list(result.scalars())


async def get_puzzle_with_clues(db: AsyncSession, puzzle_id: int) -> Puzzle | None:
    """The puzzle with its answer, clues and media rows, for showing it to the reviewer."""
    result = await db.execute(
        select(Puzzle)
        .where(Puzzle.id == puzzle_id)
        .options(
            joinedload(Puzzle.answer_title),
            joinedload(Puzzle.clues).joinedload(PuzzleClue.media_files),
        )
    )
    return result.unique().scalar_one_or_none()


async def get_puzzle_for_update(db: AsyncSession, puzzle_id: int) -> Puzzle | None:
    """The puzzle row alone, locked until the transaction ends, so two reviewers' clicks (or a
    click and a player's first guess) run one after the other."""
    result = await db.execute(
        select(Puzzle)
        .where(Puzzle.id == puzzle_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    return result.scalar_one_or_none()


async def has_stored_game(db: AsyncSession, puzzle_id: int) -> bool:
    """Whether any account has a game on the puzzle (AC-86)."""
    return bool(await db.scalar(select(exists().where(Progress.puzzle_id == puzzle_id))))


async def list_clue_image_ids(db: AsyncSession, puzzle_id: int) -> list[list[str]]:
    """The `source_image_ids` of each clue, in step order. A clue with none gives an empty list."""
    result = await db.execute(
        select(PuzzleClue.source_image_ids)
        .where(PuzzleClue.puzzle_id == puzzle_id)
        .order_by(PuzzleClue.step)
    )
    return [list(ids or []) for ids in result.scalars()]


async def delete_puzzle(db: AsyncSession, puzzle: Puzzle) -> None:
    """Deletes the puzzle; its clues and their media rows go with it (`ON DELETE CASCADE`). It
    flushes and never commits."""
    await db.delete(puzzle)
    await db.flush()


async def get_media_file_by_key(db: AsyncSession, key: str) -> ClueMediaFile | None:
    """Any media row with this key, whatever the puzzle's status (the key is not unique, but the
    bytes are the same, so the first by id is stable)."""
    result = await db.execute(
        select(ClueMediaFile).where(ClueMediaFile.key == key).order_by(ClueMediaFile.id).limit(1)
    )
    return result.scalar_one_or_none()
