"""The puzzle builder (SPEC-04 AC-26 to AC-33, AC-39 to AC-46, AC-50 to AC-52, AC-86).

For each slot, in ascending order: choose the answer and the screenshots from the seed (a rebuild
forces its excluded title and leaves out the screenshots of every rejected draft of the slot), fetch
the screenshots from IGDB's image host through the allow-listed client, make the six clues' files
with `app/content/media.py`, then store the files under their SHA-256 and write the draft whole or
not at all: one transaction with the title (when the list does not hold it yet), the puzzle in
`in_review` with no date, six clues and their media rows measured from the files. The files go
first, so a failed write leaves files and no row (non-goal: clean-up). A slot that already has a
puzzle is skipped and never touched, so a re-run adds nothing and a puzzle with stored games is
never edited (AC-46, AC-86). `dry_run` does everything but write to the database and storage
(AC-45).

The `rebuild` exclusion rows of a slot are never deleted here: the next rebuild leaves out the
screenshots of every rejected draft of the slot, the union of all its rows (decision 71). A slot
needs a rebuild while it has such a row and no puzzle; `give_up_slot` ends that.

Library code: it returns a `BuildReport` and the script prints it. Nothing here prints, and no
answer is in the report's lines unless asked for.
"""

import asyncio
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Self

import httpx
from av.error import FFmpegError
from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession

from app.content import media
from app.content.candidates import (
    FunnelCounter,
    NotBuildableError,
    ScreenshotPlan,
    assign_screenshots,
    choose_answer,
    funnel,
    never_igdb_ids,
    rebuild_avoid_ids,
    rebuild_reserved_igdb_ids,
)
from app.content.fetch import FetchError, build_client, send_capped
from app.content.hints import hints, source_url
from app.content.igdb import IgdbGame
from app.content.media import MediaFacts, MediaLimitError
from app.content.seed import Purpose, SlotSeed, check_salt, reserved_secrets
from app.content.settings import ContentSettings
from app.content.target import check_image_gate
from app.content.titles_import import ImportRecord, to_record
from app.core.clock import utc_today
from app.core.config import Settings
from app.crud.exclusion import ExclusionRow, give_up_rebuild, list_exclusions
from app.crud.game import get_game_by_slug
from app.crud.puzzle import (
    ClueInput,
    MediaFileInput,
    add_built_puzzle,
    list_answer_igdb_ids,
    list_puzzle_slots,
)
from app.crud.title import create_title, find_key_holders, get_titles_by_igdb_ids
from app.games.enums import ClueContentType, ClueRole, ExclusionReason, MediaType
from app.media.writer import MediaWriter
from app.models.puzzle import PuzzleStatus
from app.models.title import TitleKind

# Raise this whenever the choice logic changes (R-8): the same inputs must then give other choices.
# The golden tests of `tests/test_content_seed.py` and `tests/test_candidates.py` go red first.
GENERATOR_VERSION = "1"

# IGDB's image host, the size the clues are made from (`t_1080p`).
IMAGE_URL = "https://images.igdb.com/igdb/image/upload/t_1080p/{image_id}.jpg"
IMAGE_ID = re.compile(r"[a-z0-9]+", re.ASCII)
MAX_IMAGE_BYTES = 15 * 1024 * 1024
# A typo like `--slots 1-99999999` must not start a run that never ends.
MAX_SLOTS_PER_RUN = 500
STEPS = 6

FetchImage = Callable[[str], Awaitable[bytes]]
GamesSource = Callable[[], AsyncIterator[IgdbGame]]


class SlotError(Exception):
    """A slot cannot be built; the run goes on with the next one. `reason` is stable."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


class UnknownGameError(Exception):
    """The database has no game with the slug that was asked for."""


class ImageFetcher:
    """Fetches IGDB screenshots through the allow-listed client (AC-85): `images.igdb.com` only,
    https only, no redirect, a size cap. The bytes are untrusted; `media` decodes and re-encodes
    them."""

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._http = build_client(transport)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self._http.aclose()

    async def fetch(self, image_id: str) -> bytes:
        if IMAGE_ID.fullmatch(image_id) is None:
            raise FetchError("an IGDB image id has only lowercase letters and digits")
        request = self._http.build_request("GET", IMAGE_URL.format(image_id=image_id))
        response = await send_capped(self._http, request, max_bytes=MAX_IMAGE_BYTES)
        if response.status_code != 200:
            raise FetchError(f"IGDB's image host answered {response.status_code}")
        return response.content


class SlotStatus(StrEnum):
    BUILT = "built"
    PLANNED = "planned"  # dry run: everything but the writes
    EXISTS = "exists"  # the slot already has a puzzle
    FAILED = "failed"


@dataclass(frozen=True)
class FilePlan:
    step: int
    role: str
    facts: MediaFacts


@dataclass
class SlotOutcome:
    slot: int
    status: SlotStatus
    detail: str = ""
    answer_name: str | None = None
    is_rebuild: bool = False
    files: list[FilePlan] = field(default_factory=list)
    hint_steps: list[int] = field(default_factory=list)
    near_duplicates: int = 0


@dataclass
class BuildReport:
    slots: list[int]
    outcomes: list[SlotOutcome] = field(default_factory=list)

    @property
    def is_ok(self) -> bool:
        return all(outcome.status != SlotStatus.FAILED for outcome in self.outcomes)

    def lines(self, *, show_answers: bool = False) -> list[str]:
        """What the script prints. The answer's name only with `show_answers` (AC-45, AC-83), and
        then as JSON text, so a control character in IGDB's data cannot reach the terminal. Never a
        media key, a path or a screenshot id."""
        lines = [f"slots: {_range_text(self.slots)}"]
        for outcome in self.outcomes:
            head = f"slot {outcome.slot}: {outcome.status}"
            if outcome.is_rebuild:
                head += " (rebuild)"
            if outcome.detail:
                head += f": {outcome.detail}"
            lines.append(head)
            if outcome.status in (SlotStatus.BUILT, SlotStatus.PLANNED):
                lines.append(f"  hints at steps: {', '.join(map(str, outcome.hint_steps)) or '-'}")
                for plan in outcome.files:
                    facts = plan.facts
                    size = f"{facts.byte_size / media.MB:.2f} MB, {facts.width}x{facts.height}"
                    duration = "" if facts.duration_ms is None else f", {facts.duration_ms} ms"
                    lines.append(
                        f"  step {plan.step} {plan.role}: {facts.content_type}, {size}{duration}"
                    )
                if show_answers and outcome.answer_name is not None:
                    lines.append(f"  answer: {json.dumps(outcome.answer_name)}")
            if outcome.near_duplicates:
                lines.append(
                    f"  candidates left out (their name and year are taken by another title,"
                    f" or the name does not fit the list): {outcome.near_duplicates}"
                )
        counts = {status: sum(o.status == status for o in self.outcomes) for status in SlotStatus}
        lines.append(", ".join(f"{status}: {count}" for status, count in counts.items() if count))
        return lines


def _range_text(slots: Sequence[int]) -> str:
    if not slots:
        return "none"
    return str(slots[0]) if len(slots) == 1 else f"{slots[0]} to {slots[-1]}"


@dataclass(frozen=True)
class SlotRequest:
    """Which slots a run builds: a range (`--slots`), the next N after the highest slot in use
    (`--next`), or every slot waiting for a rebuild (`--rebuilds`). Exactly one."""

    slots: range | None = None
    next_count: int | None = None
    rebuilds: bool = False

    def __post_init__(self) -> None:
        if (self.slots is not None) + (self.next_count is not None) + self.rebuilds != 1:
            raise ValueError("ask for exactly one of a slot range, a next count and the rebuilds")
        count = len(self.slots) if self.slots is not None else self.next_count or 0
        if self.slots is not None and (self.slots.start < 1 or count < 1):
            raise ValueError("slots start at 1 and a range has at least one")
        if not 0 <= count <= MAX_SLOTS_PER_RUN:
            raise ValueError(f"a run builds at most {MAX_SLOTS_PER_RUN} slots")

    def pick(self, *, puzzle_slots: set[int], rebuild_slots: set[int]) -> list[int]:
        if self.rebuilds:
            return sorted(rebuild_slots - puzzle_slots)
        if self.next_count is not None:
            highest = max(puzzle_slots | rebuild_slots, default=0)
            return list(range(highest + 1, highest + 1 + self.next_count))
        assert self.slots is not None
        return list(self.slots)


@dataclass(frozen=True)
class BuiltFile:
    step: int
    role: str
    path: Path
    key: str
    facts: MediaFacts


@dataclass(frozen=True)
class SlotPlan:
    slot: int
    game: IgdbGame
    shots: ScreenshotPlan
    title_id: int | None  # None: the answer is not in the title list yet
    record: ImportRecord | None  # what a new title is made from
    is_rebuild: bool


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def make_files(
    shots: ScreenshotPlan, raw: dict[str, Path], seed: SlotSeed, out_dir: Path
) -> list[BuiltFile]:
    """The six clues' files, in step order: the step-1 crop, the four stills, the clip and its
    poster. Sync and CPU-bound, so the caller runs it in a thread. Each function measures its file
    and refuses one that breaks a limit (AC-39)."""
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    built: list[BuiltFile] = []

    def add(step: int, role: str, path: Path, facts: MediaFacts) -> None:
        built.append(BuiltFile(step, role, path, _sha256(path), facts))

    path = out_dir / "step1.webp"
    add(
        1,
        ClueRole.SOURCE,
        path,
        media.pixelated_crop(raw[shots.crop.image_id], path, seed.stream(Purpose.CROP)),
    )
    for step, shot in enumerate(shots.steps, start=2):
        path = out_dir / f"step{step}.webp"
        add(step, ClueRole.SOURCE, path, media.still(raw[shot.image_id], path))
    clip_path = out_dir / "step6.mp4"
    sources = [raw[shot.image_id] for shot in shots.clip]
    add(6, ClueRole.SOURCE, clip_path, media.clip(sources, clip_path, seed.stream(Purpose.CLIP)))
    poster_path = out_dir / "step6-poster.webp"
    add(6, ClueRole.POSTER, poster_path, media.poster(clip_path, poster_path))
    return built


def clue_inputs(plan: SlotPlan, files: Sequence[BuiltFile]) -> list[ClueInput]:
    """The clue rows. `source_image_ids` follow the step order: step 1 the crop's screenshot,
    steps 2 to 5 their own, step 6 the clip's screenshots in clip order."""
    hint_of = hints(plan.game)
    url = source_url(plan.game)
    image_ids = {
        1: [plan.shots.crop.image_id],
        **{step: [shot.image_id] for step, shot in enumerate(plan.shots.steps, start=2)},
        6: [shot.image_id for shot in plan.shots.clip],
    }
    clues = []
    for step in range(1, STEPS + 1):
        hint = hint_of[step]
        clues.append(
            ClueInput(
                step=step,
                media_type=MediaType.VIDEO if step == STEPS else MediaType.IMAGE,
                hint_kind=None if hint is None else hint.kind,
                hint_value=None if hint is None else hint.value,
                source_url=url,
                source_image_ids=image_ids[step],
                files=[
                    MediaFileInput(
                        role=file.role,
                        key=file.key,
                        content_type=file.facts.content_type,
                        byte_size=file.facts.byte_size,
                        width=file.facts.width,
                        height=file.facts.height,
                        duration_ms=file.facts.duration_ms,
                    )
                    for file in files
                    if file.step == step
                ],
            )
        )
    return clues


def layout_problem(clues: Sequence[ClueInput]) -> str | None:
    """The media rule of SPEC-02 AC-13b for a draft about to be written: steps 1 to 5 each one WebP
    and no poster; step 6 one MP4 and one WebP poster. `None` when it holds."""
    if [clue.step for clue in clues] != list(range(1, STEPS + 1)):
        return "a draft has exactly the steps 1 to 6"
    for clue in clues:
        sources = sorted(f.content_type for f in clue.files if f.role == ClueRole.SOURCE)
        posters = [f for f in clue.files if f.role == ClueRole.POSTER]
        if clue.step < STEPS:
            is_valid = (
                clue.media_type == MediaType.IMAGE
                and sources == [ClueContentType.IMAGE_WEBP]
                and not posters
            )
        else:
            is_valid = (
                clue.media_type == MediaType.VIDEO
                and sources == [ClueContentType.VIDEO_MP4]
                and [f.content_type for f in posters] == [ClueContentType.IMAGE_WEBP]
            )
        if not is_valid:
            return f"step {clue.step} breaks the media rules"
    return None


def _write_atomically(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=path.parent, prefix=".raw-")
    try:
        with os.fdopen(handle, "wb") as out:
            out.write(data)
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


# What a slot's media step can raise for a screenshot or a file that is bad. Network errors are not
# here: a refused host or a failing fetch stops the run.
MEDIA_ERRORS = (MediaLimitError, OSError, Image.DecompressionBombError, FFmpegError)


class Builder:
    """One run. The checks come first: the IGDB image gate (R-12, AC-2), then the salt (AC-27), both
    before IGDB is read or anything is fetched. `target` is the target's name; the script has
    already run the target guard (`check_target`)."""

    def __init__(
        self,
        db: AsyncSession,
        *,
        game_slug: str,
        target: str,
        settings: ContentSettings,
        app_settings: Settings,
        now: dt.datetime,
        games: GamesSource,
        fetch_image: FetchImage,
        writer: MediaWriter | None,
        dry_run: bool = False,
    ) -> None:
        if writer is None and not dry_run:
            raise ValueError("a run that writes needs a media writer")
        self.db = db
        self.game_slug = game_slug
        self.target = target
        self.settings = settings
        self.app_settings = app_settings
        self.today = utc_today(now)
        self.games = games
        self.fetch_image = fetch_image
        self.writer = writer
        self.dry_run = dry_run
        # Set by `run`, which owns the game's rows.
        self.game_id = 0
        self.variant = ""
        self.exclusions: list[ExclusionRow] = []
        self.earlier: set[int] = set()
        # Candidates the slot being built left out as near-duplicates of a stored title (R-20).
        self.near_duplicates = 0
        self.salt = settings.PUZZLE_SEED_SALT

    async def run(self, request: SlotRequest) -> BuildReport:
        check_image_gate(self.target, self.settings.IGDB_IMAGES_CLEARED_BY)
        self.salt = check_salt(
            self.settings.PUZZLE_SEED_SALT,
            reserved_secrets(self.app_settings, self.settings.IGDB_CLIENT_SECRET),
        )
        game = await get_game_by_slug(self.db, self.game_slug)
        if game is None:
            raise UnknownGameError(f"no game with the slug {self.game_slug!r} in the database")
        self.game_id, self.variant = game.id, game.default_variant
        puzzle_slots = await list_puzzle_slots(self.db, self.game_id, self.variant)
        self.exclusions = await list_exclusions(self.db, self.game_id)
        self.earlier = await list_answer_igdb_ids(self.db, self.game_id)
        rebuild_slots = {
            row.build_slot
            for row in self.exclusions
            if row.reason == ExclusionReason.REBUILD
            and row.variant == self.variant
            and row.build_slot is not None
        }
        slots = request.pick(puzzle_slots=puzzle_slots, rebuild_slots=rebuild_slots)
        report = BuildReport(slots)
        # A slot that has a puzzle is never touched, whatever its status or games (AC-46, AC-86).
        # When every slot has one, IGDB is not even read.
        todo = [slot for slot in slots if slot not in puzzle_slots]
        for slot in slots:
            if slot in puzzle_slots:
                report.outcomes.append(
                    SlotOutcome(slot, SlotStatus.EXISTS, "the slot already has a puzzle")
                )
        pool = await self._read_pool() if todo else []
        for slot in todo:
            report.outcomes.append(await self._build_slot(slot, pool))
        report.outcomes.sort(key=lambda outcome: outcome.slot)
        return report

    async def _read_pool(self) -> list[IgdbGame]:
        """The games that pass the filters that do not depend on the slot. Read once for the run;
        the exclusions are applied per slot."""
        counter = FunnelCounter(
            today=self.today,
            settings=self.settings,
            earlier_answer_igdb_ids=(),
            never_igdb_ids=(),
        )
        async for game in self.games():
            counter.add(game)
        return list(counter.result().candidates)

    def _seed(self, slot: int) -> SlotSeed:
        assert self.salt is not None
        return SlotSeed(
            self.salt,
            self.game_slug,
            self.variant,
            slot,
            GENERATOR_VERSION,
            other_secrets=reserved_secrets(self.app_settings, self.settings.IGDB_CLIENT_SECRET),
        )

    async def _build_slot(self, slot: int, pool: list[IgdbGame]) -> SlotOutcome:
        self.near_duplicates = 0
        try:
            plan = await self._plan(slot, pool)
            files = await self._make_files(plan)
        except (NotBuildableError, SlotError) as error:
            return SlotOutcome(
                slot, SlotStatus.FAILED, error.reason, near_duplicates=self.near_duplicates
            )
        except MEDIA_ERRORS as error:
            return SlotOutcome(
                slot,
                SlotStatus.FAILED,
                f"media: {type(error).__name__}",
                near_duplicates=self.near_duplicates,
            )
        # The next slot of the run must not draw this answer (AC-25), written or only planned.
        self.earlier.add(plan.game.id)
        outcome = SlotOutcome(
            slot,
            SlotStatus.PLANNED if self.dry_run else SlotStatus.BUILT,
            answer_name=plan.game.name,
            is_rebuild=plan.is_rebuild,
            files=[FilePlan(file.step, file.role, file.facts) for file in files],
            hint_steps=[step for step, hint in hints(plan.game).items() if hint is not None],
            near_duplicates=self.near_duplicates,
        )
        if not self.dry_run:
            await self._write(plan, files)
        return outcome

    async def _plan(self, slot: int, pool: list[IgdbGame]) -> SlotPlan:
        seed = self._seed(slot)
        own_rows = [
            row
            for row in self.exclusions
            if row.reason == ExclusionReason.REBUILD
            and (row.variant, row.build_slot) == (self.variant, slot)
        ]
        candidates = funnel(
            pool,
            today=self.today,
            settings=self.settings,
            earlier_answer_igdb_ids=self.earlier,
            never_igdb_ids=never_igdb_ids(self.exclusions),
            reserved_igdb_ids=rebuild_reserved_igdb_ids(
                self.exclusions, own_slot=(self.variant, slot)
            ),
        ).candidates
        # Raises `cannot_rebuild` when fewer than 7 screenshots are left (decision 71).
        avoid = rebuild_avoid_ids(self.exclusions, variant=self.variant, slot=slot)
        if own_rows:
            game, title_id, record = self._forced(own_rows, candidates)
        else:
            game, title_id, record = await self._draw(candidates, seed)
        shots = assign_screenshots(
            game, seed.stream(Purpose.SCREENSHOTS), avoid, settings=self.settings
        )
        return SlotPlan(slot, game, shots, title_id, record, bool(own_rows))

    def _forced(
        self, rows: list[ExclusionRow], candidates: Sequence[IgdbGame]
    ) -> tuple[IgdbGame, int, None]:
        """A rebuild builds the title its slot was rejected with (R-9)."""
        titles = {(row.title_id, row.igdb_id) for row in rows}
        if len(titles) != 1 or next(iter(titles))[1] is None:
            raise SlotError("bad_rebuild_row", "the slot's rebuild rows do not name one IGDB title")
        title_id, igdb_id = next(iter(titles))
        game = next((candidate for candidate in candidates if candidate.id == igdb_id), None)
        if game is None:
            raise SlotError(
                "rebuild_title_not_a_candidate",
                "IGDB's data no longer makes the slot's title a candidate",
            )
        return game, title_id, None

    async def _draw(
        self, candidates: Sequence[IgdbGame], seed: SlotSeed
    ) -> tuple[IgdbGame, int | None, ImportRecord | None]:
        """Draw an answer. One whose name and year another stored title holds is a near-duplicate
        the import dropped: it is left out and counted, and the draw goes on (R-20)."""
        remaining = list(candidates)
        stream = seed.stream(Purpose.ANSWER)
        while True:
            game = choose_answer(remaining, stream)
            stored = (await get_titles_by_igdb_ids(self.db, [game.id])).get(game.id)
            if stored is not None:
                return game, stored.id, None
            record = to_record(game)
            if isinstance(record, ImportRecord) and not await find_key_holders(
                self.db, kind=TitleKind.VIDEO_GAME, keys=[record.key]
            ):
                return game, None, record
            remaining = [candidate for candidate in remaining if candidate.id != game.id]
            self.near_duplicates += 1

    async def _raw_path(self, image_id: str) -> Path:
        """The screenshot in the working folder, fetched once and kept by its IGDB image id."""
        path = self.settings.content_work_path / "raw" / image_id
        if not path.is_file():
            await asyncio.to_thread(_write_atomically, path, await self.fetch_image(image_id))
        return path

    async def _make_files(self, plan: SlotPlan) -> list[BuiltFile]:
        raw = {image_id: await self._raw_path(image_id) for image_id in plan.shots.image_ids}
        out_dir = self.settings.content_work_path / "out" / str(plan.slot)
        return await asyncio.to_thread(make_files, plan.shots, raw, self._seed(plan.slot), out_dir)

    async def _write(self, plan: SlotPlan, files: list[BuiltFile]) -> None:
        """Files first, then the rows in one transaction (AC-46). A failure rolls the rows back and
        stops the run; the files stay."""
        clues = clue_inputs(plan, files)
        problem = layout_problem(clues)
        if problem is not None:
            raise SlotError("media_layout", problem)
        assert self.writer is not None
        for file in files:
            await asyncio.to_thread(self.writer.put, file.key, file.path, file.facts.content_type)
        try:
            title_id = plan.title_id
            if title_id is None:
                assert plan.record is not None
                title = await create_title(
                    self.db,
                    name=plan.record.name,
                    franchise=plan.record.franchise,
                    year=plan.record.year,
                    igdb_id=plan.record.igdb_id,
                )
                title_id = title.id
            await add_built_puzzle(
                self.db,
                game_id=self.game_id,
                variant=self.variant,
                build_slot=plan.slot,
                generator_version=GENERATOR_VERSION,
                answer_title_id=title_id,
                status=PuzzleStatus.IN_REVIEW,
                clues=clues,
            )
            await self.db.commit()
        except BaseException:
            await self.db.rollback()
            raise


async def give_up_slot(db: AsyncSession, game_slug: str, slot: int) -> int:
    """Give up rebuilding a slot (decision 71): its `rebuild` rows become one `never` exclusion of
    their title, which frees the slot for a new answer. Returns the number of rebuild rows it
    replaced, 0 when there was nothing to give up. A slot that has a puzzle is left alone (AC-86)."""
    game = await get_game_by_slug(db, game_slug)
    if game is None:
        raise UnknownGameError(f"no game with the slug {game_slug!r} in the database")
    if slot in await list_puzzle_slots(db, game.id, game.default_variant):
        return 0
    try:
        count = await give_up_rebuild(db, game_id=game.id, variant=game.default_variant, slot=slot)
        await db.commit()
    except BaseException:
        await db.rollback()
        raise
    return count
