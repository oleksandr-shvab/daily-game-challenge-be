"""The local review page (M1-24a, SPEC-04 AC-47 to AC-54, AC-86).

The app is driven through httpx over its own ASGI app, with the test session as its database
session (`actions` commit, which only releases the test's savepoint)."""

import datetime as dt
from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.content.review import actions
from app.content.review.app import create_review_app
from app.content.target import TargetRefusedError
from app.crud.exclusion import list_exclusions
from app.games.enums import ExclusionReason
from app.models.exclusion import AnswerExclusion
from app.models.game import Game
from app.models.progress import Progress
from app.models.puzzle import ClueMediaFile, Puzzle, PuzzleClue, PuzzleStatus
from app.models.title import Title
from app.models.user import User
from tests.factories import create_game, create_puzzle, create_titles, media_key

PORT = 8765
ORIGIN = f"http://localhost:{PORT}"
TOKEN = "t" * 43
NOW = dt.datetime(2026, 11, 20, 12, 0, tzinfo=dt.UTC)
LOCAL = {"ENVIRONMENT": "local"}
HOSTILE = "<script>alert(1)</script>"


@dataclass
class World:
    db: AsyncSession
    game: Game
    titles: list[Title]
    media_dir: Path
    game_id: int  # read once: a rollback expires the ORM objects

    async def draft(
        self,
        slot: int | None,
        *,
        title: int = 0,
        image_ids: list[list[str]] | None = None,
        status: PuzzleStatus = PuzzleStatus.IN_REVIEW,
    ) -> Puzzle:
        puzzle = await create_puzzle(
            self.db,
            game=self.game,
            answer=self.titles[title],
            puzzle_date=dt.date(2000, 1, 1),
            status=status,
        )
        puzzle.date = None
        puzzle.build_slot = slot
        clues = (
            await self.db.execute(select(PuzzleClue).where(PuzzleClue.puzzle_id == puzzle.id))
        ).scalars()
        for clue in clues:
            clue.source_image_ids = (image_ids or [[f"img{clue.step}"] for _ in range(6)])[
                clue.step - 1
            ]
        await self.db.commit()
        return puzzle

    async def count(self, model: type) -> int:
        return await self.db.scalar(select(func.count()).select_from(model)) or 0

    async def status_of(self, puzzle_id: int) -> str | None:
        return await self.db.scalar(select(Puzzle.status).where(Puzzle.id == puzzle_id))


@pytest.fixture
async def world(db_session: AsyncSession, tmp_path: Path) -> World:
    game = await create_game(db_session)
    titles = await create_titles(db_session, "Portal", "Doom", "Quake")
    await db_session.commit()
    return World(db_session, game, titles, tmp_path, game.id)


def make_app(world: World, environ: Mapping[str, str] = LOCAL):
    @asynccontextmanager
    async def session_factory() -> AsyncIterator[AsyncSession]:
        yield world.db

    return create_review_app(
        session_factory=session_factory,
        media_dir=world.media_dir,
        port=PORT,
        environ=environ,
        clock=lambda: NOW,
        token=TOKEN,
    )


@pytest.fixture
async def page(world: World) -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=make_app(world))
    async with AsyncClient(transport=transport, base_url=ORIGIN, headers={"Origin": ORIGIN}) as ac:
        yield ac


# (a) one happy path per route


async def test__queue__lists_the_draft(world: World, page: AsyncClient) -> None:
    puzzle = await world.draft(7)
    response = await page.get("/")
    assert response.status_code == 200
    assert f"/drafts/{puzzle.id}" in response.text
    assert "Portal" in response.text


async def test__draft_page__shows_six_clues_the_answer_and_escapes_igdb_text(
    world: World, page: AsyncClient
) -> None:
    world.titles[0].name = f"Portal {HOSTILE}"
    world.titles[0].franchise = '"><img src=x onerror=alert(2)>'
    puzzle = await world.draft(7)
    clue = (await world.db.execute(select(PuzzleClue).where(PuzzleClue.step == 2))).scalars().one()
    clue.hint_kind, clue.hint_value = "developer", f"{HOSTILE} & Co"
    clue.source_url = "javascript:alert(3)"
    await world.db.commit()

    response = await page.get(f"/drafts/{puzzle.id}")

    body = response.text
    assert response.status_code == 200
    assert body.count("<figure>") == 6
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in body  # the answer is shown, escaped
    assert "Developer: &lt;script&gt;" in body
    assert "<script" not in body
    assert "<img src=x" not in body
    assert 'href="javascript:' not in body
    assert "default-src 'self'" in response.headers["content-security-policy"]


async def test__file__is_served_with_the_stored_type(world: World, page: AsyncClient) -> None:
    puzzle = await world.draft(7)
    key = media_key(puzzle.id, 1, "source")
    (world.media_dir / key).write_bytes(b"RIFFxxxxWEBP")

    response = await page.get(f"/files/{key}")

    assert response.status_code == 200
    assert response.content == b"RIFFxxxxWEBP"
    assert response.headers["content-type"] == "image/webp"
    assert response.headers["x-content-type-options"] == "nosniff"
    # Only a SHA-256 key is a file name, even if a row says otherwise.
    row = (await world.db.execute(select(ClueMediaFile).limit(1))).scalars().one()
    row.key = "notes.txt"
    await world.db.commit()
    (world.media_dir / "notes.txt").write_text("not a media file")
    assert (await page.get("/files/notes.txt")).status_code == 404


async def test__approve__moves_the_draft_to_approved_at_the_clock(
    world: World, page: AsyncClient
) -> None:
    puzzle = await world.draft(7)
    other = await world.draft(8, title=1)

    response = await page.post(f"/drafts/{puzzle.id}/approve", data={"token": TOKEN})

    assert response.status_code == 303
    assert response.headers["location"] == f"/drafts/{other.id}"
    row = (await world.db.execute(select(Puzzle).where(Puzzle.id == puzzle.id))).scalar_one()
    assert (row.status, row.approved_at, row.date) == (PuzzleStatus.APPROVED, NOW, None)


async def test__reject_never__leaves_an_exclusion_and_no_puzzle(
    world: World, page: AsyncClient
) -> None:
    puzzle = await world.draft(7)

    response = await page.post(
        f"/drafts/{puzzle.id}/reject", data={"token": TOKEN, "reason": "never"}
    )

    assert (response.status_code, response.headers["location"]) == (303, "/")
    assert [await world.count(m) for m in (Puzzle, PuzzleClue, ClueMediaFile)] == [0, 0, 0]
    (row,) = await list_exclusions(world.db, world.game_id)
    assert (row.title_id, row.reason) == (world.titles[0].id, ExclusionReason.NEVER)


# (b) the rules, at the service level


async def test__reject__records_variant_slot_and_every_image_id_in_step_order(
    world: World,
) -> None:
    image_ids = [["c"], ["b"], ["a"], ["e"], ["d"], ["z", "y", "x"]]
    puzzle = await world.draft(7, image_ids=image_ids)
    await world.db.commit()

    await actions.reject(world.db, puzzle.id, ExclusionReason.REBUILD)

    (row,) = (await world.db.execute(select(AnswerExclusion))).scalars()
    assert (row.game_id, row.variant, row.title_id) == (
        world.game_id,
        "default",
        world.titles[0].id,
    )
    assert (row.reason, row.build_slot) == (ExclusionReason.REBUILD, 7)
    assert row.image_ids == ["c", "b", "a", "e", "d", "z", "y", "x"]
    assert await world.count(Puzzle) == 0


async def test__reject_never__deletes_only_its_slots_rebuild_rows(world: World) -> None:
    first = await world.draft(7)
    await actions.reject(world.db, first.id, ExclusionReason.REBUILD)
    other = await world.draft(8, title=1)
    await actions.reject(world.db, other.id, ExclusionReason.REBUILD)
    rebuilt = await world.draft(7)  # the rebuilt draft of slot 7, same answer
    await actions.reject(world.db, rebuilt.id, ExclusionReason.NEVER)

    rows = await list_exclusions(world.db, world.game_id)

    assert [(r.title_id, r.reason, r.build_slot) for r in rows] == [
        (world.titles[1].id, ExclusionReason.REBUILD, 8),
        (world.titles[0].id, ExclusionReason.NEVER, None),
    ]


async def test__reject__is_one_transaction(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    earlier = await world.draft(7)
    await actions.reject(world.db, earlier.id, ExclusionReason.REBUILD)
    puzzle_id = (await world.draft(7)).id

    async def failing_delete(db: AsyncSession, puzzle: Puzzle) -> None:
        raise RuntimeError("the delete failed")

    monkeypatch.setattr(actions, "delete_puzzle", failing_delete)
    with pytest.raises(RuntimeError):
        await actions.reject(world.db, puzzle_id, ExclusionReason.NEVER)

    assert await world.status_of(puzzle_id) == PuzzleStatus.IN_REVIEW
    rows = await list_exclusions(world.db, world.game_id)
    assert [r.reason for r in rows] == [ExclusionReason.REBUILD]  # no never row, no row deleted


async def test__approve__a_failed_commit_leaves_the_draft_in_review(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    puzzle_id = (await world.draft(7)).id

    async def failing_commit() -> None:
        raise RuntimeError("the commit failed")

    monkeypatch.setattr(world.db, "commit", failing_commit)
    with pytest.raises(RuntimeError):
        await actions.approve(world.db, puzzle_id, NOW)
    monkeypatch.undo()

    assert await world.status_of(puzzle_id) == PuzzleStatus.IN_REVIEW


@pytest.mark.parametrize("action", ["approve", "reject"])
@pytest.mark.parametrize(
    ("case", "error"),
    [
        ("approved", actions.NotInReviewError),
        ("stored_game", actions.StoredGameError),
        ("missing", actions.PuzzleNotFoundError),
    ],
)
async def test__action__refused_cases_change_nothing(
    world: World, action: str, case: str, error: type[Exception]
) -> None:
    puzzle = await world.draft(
        7, status=PuzzleStatus.APPROVED if case == "approved" else PuzzleStatus.IN_REVIEW
    )
    if case == "stored_game":
        account = User(email="p@example.com", hashed_password="x")
        world.db.add(account)
        await world.db.flush()
        world.db.add(
            Progress(
                user_id=account.id,
                puzzle_id=puzzle.id,
                status="in_progress",
                started_at=NOW,
                counts_for_streak=True,
                origin="play",
            )
        )
        await world.db.commit()
    real_id = puzzle.id
    puzzle_id = 0 if case == "missing" else real_id

    with pytest.raises(error):
        if action == "approve":
            await actions.approve(world.db, puzzle_id, NOW)
        else:
            await actions.reject(world.db, puzzle_id, ExclusionReason.NEVER)

    if case != "missing":
        assert await world.status_of(real_id) == (
            PuzzleStatus.APPROVED if case == "approved" else PuzzleStatus.IN_REVIEW
        )
    assert await world.count(AnswerExclusion) == 0


async def test__reject_rebuild__needs_a_slot(world: World) -> None:
    puzzle_id = (await world.draft(None)).id
    with pytest.raises(actions.NoBuildSlotError):
        await actions.reject(world.db, puzzle_id, ExclusionReason.REBUILD)
    assert await world.status_of(puzzle_id) == PuzzleStatus.IN_REVIEW


# (c) the page takes requests only from itself


@pytest.mark.parametrize(
    ("headers", "data"),
    [
        ({"Origin": "http://evil.example"}, {"token": TOKEN}),
        ({"Origin": "http://localhost:9999"}, {"token": TOKEN}),
        ({"Origin": "null"}, {"token": TOKEN}),
        ({"Origin": ""}, {"token": TOKEN}),
        ({}, {"token": TOKEN}),
        ({"Origin": ORIGIN}, {"other": "1"}),
        ({"Origin": ORIGIN}, {"token": "wrong"}),
        ({"Origin": ORIGIN, "Host": "evil.example:8765"}, {"token": TOKEN}),
        ({"Origin": ORIGIN, "Host": "localhost:9999"}, {"token": TOKEN}),
        ({"Origin": ORIGIN, "Host": "localhost"}, {"token": TOKEN}),
    ],
    ids=[
        "foreign-origin",
        "own-host-other-port-origin",
        "null-origin",
        "empty-origin",
        "no-origin",
        "no-token",
        "wrong-token",
        "foreign-host",
        "other-port-host",
        "no-port-host",
    ],
)
async def test__post__from_anywhere_but_the_page_is_refused_and_changes_nothing(
    world: World,
    page: AsyncClient,
    headers: dict[str, str],
    data: dict[str, str],
) -> None:
    puzzle = await world.draft(7)
    page.headers.pop("Origin")

    for path, form in (("approve", data), ("reject", {**data, "reason": "never"})):
        response = await page.post(f"/drafts/{puzzle.id}/{path}", data=form, headers=headers)
        assert response.status_code == 403

    assert await world.status_of(puzzle.id) == PuzzleStatus.IN_REVIEW
    assert await world.count(AnswerExclusion) == 0


async def test__get__with_a_foreign_host_is_refused(world: World, page: AsyncClient) -> None:
    puzzle = await world.draft(7)
    for path in ("/", f"/drafts/{puzzle.id}", f"/files/{media_key(puzzle.id, 1, 'source')}"):
        response = await page.get(path, headers={"Host": "evil.example:8765"})
        assert response.status_code == 403
        assert "Portal" not in response.text


async def test__reject__with_another_reason_is_422_and_changes_nothing(
    world: World, page: AsyncClient
) -> None:
    puzzle = await world.draft(7)
    response = await page.post(
        f"/drafts/{puzzle.id}/reject", data={"token": TOKEN, "reason": "delete"}
    )
    assert response.status_code == 422
    assert await world.status_of(puzzle.id) == PuzzleStatus.IN_REVIEW


@pytest.mark.parametrize(
    "environ",
    [
        {"ENVIRONMENT": "staging"},
        {"ENVIRONMENT": "production"},
        {},
        {"ENVIRONMENT": "local", "CI": "true"},
    ],
    ids=["staging", "production", "unset", "ci"],
)
async def test__app__refuses_to_start_outside_local(
    world: World, environ: Mapping[str, str]
) -> None:
    with pytest.raises(TargetRefusedError):
        make_app(world, environ)
