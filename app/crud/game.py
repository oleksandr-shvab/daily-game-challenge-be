from collections.abc import Sequence
from datetime import date

from sqlalchemy import exists, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.game import Game
from app.models.progress import Progress
from app.models.puzzle import Puzzle, PuzzleStatus
from app.models.title import TitleKind


async def list_enabled_games(db: AsyncSession) -> Sequence[Game]:
    result = await db.execute(select(Game).where(Game.is_enabled.is_(True)).order_by(Game.slug))
    return result.scalars().all()


async def get_enabled_game_by_slug(db: AsyncSession, slug: str) -> Game | None:
    result = await db.execute(select(Game).where(Game.slug == slug, Game.is_enabled.is_(True)))
    return result.scalar_one_or_none()


async def get_game_by_slug(db: AsyncSession, slug: str) -> Game | None:
    """The game whether it is enabled or not: the content tools work on a disabled game too."""
    result = await db.execute(select(Game).where(Game.slug == slug))
    return result.scalar_one_or_none()


async def lock_game(db: AsyncSession, game_id: int, *, is_shared: bool) -> Game:
    """The game row read fresh and locked until the transaction ends. `is_shared` is `FOR SHARE`:
    scheduling and unscheduling read the launch date and the enabled flag under it, and a change
    of either (`FOR UPDATE`, `manage_game.py`) waits for them, and they for it."""
    result = await db.execute(
        select(Game)
        .where(Game.id == game_id)
        .with_for_update(read=is_shared)
        .execution_options(populate_existing=True)
    )
    return result.scalar_one()


async def add_game(
    db: AsyncSession, *, slug: str, default_variant: str, launch_date: date, is_enabled: bool
) -> Game:
    """A new game row (`manage_game.py`). It flushes and never commits."""
    game = Game(
        slug=slug,
        default_variant=default_variant,
        title_kind=TitleKind.VIDEO_GAME,
        launch_date=launch_date,
        is_enabled=is_enabled,
    )
    db.add(game)
    await db.flush()
    return game


async def update_game_settings(
    db: AsyncSession, game: Game, *, launch_date: date | None, is_enabled: bool | None
) -> None:
    """Set the launch date and the enabled flag; a None leaves that column as it is. It flushes
    and never commits."""
    if launch_date is not None:
        game.launch_date = launch_date
    if is_enabled is not None:
        game.is_enabled = is_enabled
    await db.flush()


async def game_has_stored_games(db: AsyncSession, game_id: int) -> bool:
    """True when any account has progress on any puzzle of the game (AC-86)."""
    result = await db.execute(
        select(exists().where(Progress.puzzle_id == Puzzle.id, Puzzle.game_id == game_id))
    )
    return bool(result.scalar())


async def has_scheduled_puzzle_on_or_before(db: AsyncSession, game_id: int, day: date) -> bool:
    """True when any variant of the game has a scheduled puzzle dated `day` or earlier."""
    result = await db.execute(
        select(
            exists().where(
                Puzzle.game_id == game_id,
                Puzzle.status == PuzzleStatus.SCHEDULED,
                Puzzle.date <= day,
            )
        )
    )
    return bool(result.scalar())


async def count_scheduled_puzzles_before(db: AsyncSession, game_id: int, day: date) -> int:
    """The game's scheduled puzzles, any variant, dated before `day`."""
    result = await db.execute(
        select(func.count()).where(
            Puzzle.game_id == game_id,
            Puzzle.status == PuzzleStatus.SCHEDULED,
            Puzzle.date < day,
        )
    )
    return result.scalar_one()
