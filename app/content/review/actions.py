"""What the reviewer can do to a draft (AC-49 to AC-53, AC-86).

Each action is one unit of work: it locks the puzzle row, checks the rules, writes, commits, and on
any error rolls back and re-raises, so a refused or failed action leaves nothing behind. The page
(`app.py`) maps the errors to answers.
"""

import datetime as dt

from sqlalchemy.ext.asyncio import AsyncSession

from app.crud.exclusion import add_exclusion, delete_rebuild_rows
from app.crud.puzzle import (
    delete_puzzle,
    get_puzzle_for_update,
    has_stored_game,
    list_clue_image_ids,
)
from app.games.enums import ExclusionReason
from app.models.puzzle import Puzzle, PuzzleStatus


class ReviewError(Exception):
    """An action the rules refuse. The message is ours and holds no IGDB text."""


class PuzzleNotFoundError(ReviewError):
    pass


class NotInReviewError(ReviewError):
    pass


class StoredGameError(ReviewError):
    """An account has a game on this puzzle, so it is never edited (AC-86)."""


class NoBuildSlotError(ReviewError):
    """A puzzle with no build slot cannot be rebuilt; reject it as `never` or leave it."""


async def _load_for_review(db: AsyncSession, puzzle_id: int) -> Puzzle:
    puzzle = await get_puzzle_for_update(db, puzzle_id)
    if puzzle is None:
        raise PuzzleNotFoundError(f"no puzzle {puzzle_id}")
    if puzzle.status != PuzzleStatus.IN_REVIEW:
        raise NotInReviewError(f"puzzle {puzzle_id} is not in review")
    if await has_stored_game(db, puzzle_id):
        raise StoredGameError(f"puzzle {puzzle_id} has stored games and is never edited")
    return puzzle


async def approve(db: AsyncSession, puzzle_id: int, now: dt.datetime) -> None:
    """Only a puzzle in review is approved. The date stays empty until it is scheduled (AC-49)."""
    try:
        puzzle = await _load_for_review(db, puzzle_id)
        puzzle.status = PuzzleStatus.APPROVED
        puzzle.approved_at = now
        await db.commit()
    except BaseException:
        await db.rollback()
        raise


def _image_ids_in_step_order(per_clue: list[list[str]]) -> list[str]:
    """Every screenshot the draft used: the crop's, steps 2 to 5, then the clip's, each once."""
    seen: dict[str, None] = {}
    for ids in per_clue:
        seen.update(dict.fromkeys(ids))
    return list(seen)


async def reject(db: AsyncSession, puzzle_id: int, reason: ExclusionReason) -> None:
    """Delete the draft and record why (AC-51 to AC-53), all in one transaction.

    `rebuild` keeps the slot and the draft's screenshot ids, so the next build of the slot forces
    this answer and leaves those screenshots out (decision 71). `never` has no slot (the table
    forbids one) and also deletes the slot's earlier `rebuild` rows: the title is out for good and
    the slot is free, so the builder need not give it up later."""
    try:
        puzzle = await _load_for_review(db, puzzle_id)
        if reason == ExclusionReason.REBUILD and puzzle.build_slot is None:
            raise NoBuildSlotError(f"puzzle {puzzle_id} has no build slot to rebuild")
        image_ids = _image_ids_in_step_order(await list_clue_image_ids(db, puzzle_id))
        is_never = reason == ExclusionReason.NEVER
        await add_exclusion(
            db,
            game_id=puzzle.game_id,
            variant=puzzle.variant,
            title_id=puzzle.answer_title_id,
            reason=reason,
            build_slot=None if is_never else puzzle.build_slot,
            image_ids=image_ids,
        )
        if is_never and puzzle.build_slot is not None:
            await delete_rebuild_rows(
                db, game_id=puzzle.game_id, variant=puzzle.variant, slot=puzzle.build_slot
            )
        await delete_puzzle(db, puzzle)
        await db.commit()
    except BaseException:
        await db.rollback()
        raise
