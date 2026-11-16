from typing import NamedTuple

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.games.enums import ExclusionReason
from app.models.exclusion import AnswerExclusion
from app.models.title import Title


class ExclusionRow(NamedTuple):
    """One exclusion with its title's IGDB id (`None` for a title that did not come from IGDB)."""

    title_id: int
    igdb_id: int | None
    reason: str
    variant: str
    build_slot: int | None
    image_ids: list[str]


async def list_exclusions(db: AsyncSession, game_id: int) -> list[ExclusionRow]:
    """Every exclusion of the game, oldest first. Rows can repeat for one title and slot: a title
    rejected twice, or rebuilt and then rejected."""
    result = await db.execute(
        select(
            AnswerExclusion.title_id,
            Title.igdb_id,
            AnswerExclusion.reason,
            AnswerExclusion.variant,
            AnswerExclusion.build_slot,
            AnswerExclusion.image_ids,
        )
        .join(Title, Title.id == AnswerExclusion.title_id)
        .where(AnswerExclusion.game_id == game_id)
        .order_by(AnswerExclusion.id)
    )
    return [ExclusionRow(*row) for row in result.all()]


async def give_up_rebuild(db: AsyncSession, *, game_id: int, variant: str, slot: int) -> int:
    """Turn the slot's `rebuild` rows into one `never` exclusion for their title (decision 71).
    That frees the slot for a new answer and keeps the title out for good. Returns the number of
    rebuild rows it replaced (0: there was nothing to give up). It flushes and never commits.

    This is the only place that deletes `rebuild` rows: a rebuild keeps them, because the next one
    leaves out the screenshots of every rejected draft of the slot, the union of all its rows
    (decision 71). A slot's rebuild rows all name one title; more than one is a bad state and raises
    `ValueError` before anything changes."""
    rows = (
        await db.execute(
            select(AnswerExclusion.id, AnswerExclusion.title_id).where(
                AnswerExclusion.game_id == game_id,
                AnswerExclusion.variant == variant,
                AnswerExclusion.build_slot == slot,
                AnswerExclusion.reason == ExclusionReason.REBUILD,
            )
        )
    ).all()
    if not rows:
        return 0
    title_ids = {title_id for _, title_id in rows}
    if len(title_ids) != 1:
        raise ValueError(f"the rebuild rows of slot {slot} name more than one title")
    (title_id,) = title_ids
    await db.execute(
        delete(AnswerExclusion).where(AnswerExclusion.id.in_([row_id for row_id, _ in rows]))
    )
    already_never = await db.scalar(
        select(AnswerExclusion.id)
        .where(
            AnswerExclusion.game_id == game_id,
            AnswerExclusion.variant == variant,
            AnswerExclusion.title_id == title_id,
            AnswerExclusion.reason == ExclusionReason.NEVER,
        )
        .limit(1)
    )
    if already_never is None:
        db.add(
            AnswerExclusion(
                game_id=game_id,
                variant=variant,
                title_id=title_id,
                reason=ExclusionReason.NEVER,
                build_slot=None,
            )
        )
    await db.flush()
    return len(rows)
