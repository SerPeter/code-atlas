"""End-to-end regression test for the original bug: "the graph doesn't get updated".

Full index → REAL FileWatcher + REAL EventBus + REAL ASTConsumer (wired like the
daemon) → body-only edit on disk → the graph's stored source refreshes and an
EmbedDirty event is published.

This exercises every link of the historically broken chain at once:
- the watcher's debounce-fired flush must publish through the real Redis-backed
  bus (the self-cancel bug killed every timer flush at its first ``await``, so
  it only ever passed with fake non-yielding buses);
- a body-only edit (signature/docstring unchanged) must classify as modified
  (content_hash previously excluded ``entity.source``, sealing staleness in
  permanently via the file-hash write-back).

Requires the isolated test infra (``docker compose --profile test up -d``),
provided via the conftest fixtures.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import pytest

from code_atlas.events import EmbedDirty, Topic, decode_event
from code_atlas.indexing.consumers import ASTConsumer
from code_atlas.indexing.orchestrator import FileScope, detect_sub_projects, index_project
from code_atlas.indexing.watcher import FileWatcher
from code_atlas.settings import derive_project_name

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from code_atlas.events import EventBus
    from code_atlas.graph.client import GraphClient
    from code_atlas.settings import AtlasSettings

pytestmark = pytest.mark.integration

_BODY_V1 = '''def work(x: int) -> int:
    """Compute the thing."""
    return x + 1
'''

_BODY_V2 = '''def work(x: int) -> int:
    """Compute the thing."""
    return x + 2
'''


async def _wait_for(
    predicate: Callable[[], Awaitable[bool]],
    *,
    timeout_s: float,
    interval_s: float = 0.25,
) -> bool:
    """Poll an async *predicate* until it returns True or *timeout_s* elapses."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if await predicate():
            return True
        await asyncio.sleep(interval_s)
    return False


async def test_live_update_body_only_edit_reaches_graph_and_embed_stream(  # noqa: PLR0915
    settings: AtlasSettings,
    graph_client: GraphClient,
    event_bus: EventBus,
) -> None:
    """A body-only edit saved to disk propagates to the graph via the live pipeline.

    1. Full-index a tmp project (orchestrator API), assert entities landed.
    2. Start the REAL FileWatcher + REAL ASTConsumer on the REAL EventBus,
       wired exactly like DaemonManager (minus the EmbedConsumer — embeddings
       stay enabled so the EmbedDirty publish branch runs, but no embedding
       endpoint is required).
    3. Modify ONLY a function body (signature/docstring unchanged) on disk.
    4. Assert the watcher's debounce-fired flush published a FileChanged while
       still running (the historic self-cancel bug), the graph's stored source
       contains the new body, and an EmbedDirty for the entity was published.
    """
    # --- Phase 1: full index (embeddings off: no embedding endpoint in tests) ---
    settings.embeddings.enabled = False
    await graph_client.ensure_schema()

    project_name = derive_project_name(settings.project_root)
    (settings.project_root / "live.py").write_text(_BODY_V1, encoding="utf-8")

    result = await index_project(settings, graph_client, event_bus, drain_timeout_s=120.0)
    assert result.files_published >= 1
    assert result.drained is True
    assert result.entities_total >= 1

    rows = await graph_client.execute(
        "MATCH (c:Callable {project_name: $p, name: 'work'}) RETURN c.source AS src, c.uid AS uid",
        {"p": project_name},
    )
    assert rows, "full index did not create the Callable entity"
    assert "return x + 1" in (rows[0]["src"] or "")

    # Baselines: everything published from here on comes from the live pipeline
    fc_key = event_bus._stream_key(Topic.FILE_CHANGED)
    ed_key = event_bus._stream_key(Topic.EMBED_DIRTY)
    fc_len_before = await event_bus._redis.xlen(fc_key)
    ed_len_before = await event_bus._redis.xlen(ed_key)

    # --- Phase 2: live watcher + AST consumer, wired like DaemonManager ---
    settings.embeddings.enabled = True  # EmbedDirty publish branch must run
    settings.watcher.debounce_s = 0.5
    settings.watcher.max_wait_s = 10.0

    scope = FileScope(settings.project_root, settings)
    subs = detect_sub_projects(settings.project_root, settings.monorepo)
    watcher = FileWatcher(
        settings.project_root,
        event_bus,
        scope,
        settings.watcher,
        sub_projects=subs or None,
        root_name=project_name,
    )
    consumer = ASTConsumer(event_bus, graph_client, settings, cooldown_s=settings.watcher.cooldown_s)

    watcher_task = asyncio.create_task(watcher.run())
    consumer_task = asyncio.create_task(consumer.run())
    try:
        # Let awatch initialize before touching the file
        await asyncio.sleep(1.5)

        # --- Phase 3: body-only edit (signature/docstring unchanged) ---
        async def _fc_grew() -> bool:
            return await event_bus._redis.xlen(fc_key) > fc_len_before

        published = False
        for _attempt in range(3):
            (settings.project_root / "live.py").write_text(_BODY_V2, encoding="utf-8")
            published = await _wait_for(_fc_grew, timeout_s=15.0)
            if published:
                break

        # The historic self-cancel bug: every timer-fired flush died at its
        # first await into the real bus, so nothing was ever XADDed.
        assert published, "watcher's debounce-fired flush never published a FileChanged to the real bus"
        assert not watcher_task.done(), "watcher task died — publish did not come from a live timer-fired flush"

        # --- Phase 4: the stored source must reflect the new body ---
        async def _source_updated() -> bool:
            rows = await graph_client.execute(
                "MATCH (c:Callable {project_name: $p, name: 'work'}) RETURN c.source AS src",
                {"p": project_name},
            )
            return bool(rows) and "return x + 2" in (rows[0]["src"] or "")

        assert await _wait_for(_source_updated, timeout_s=60.0), (
            "graph still serves the old body — body-only edit was not classified as modified"
        )

        # --- Phase 5: an EmbedDirty for the edited entity was published ---
        async def _embed_dirty_published() -> bool:
            return await event_bus._redis.xlen(ed_key) > ed_len_before

        assert await _wait_for(_embed_dirty_published, timeout_s=30.0), (
            "no EmbedDirty event was published for the body-only edit"
        )

        entries = await event_bus._redis.xrange(ed_key)
        dirty_qns = {
            ev.entity.qualified_name
            for _mid, fields in entries
            if isinstance(ev := decode_event(Topic.EMBED_DIRTY, fields), EmbedDirty)
        }
        assert f"{project_name}:live.work" in dirty_qns

        assert consumer.stats.entities_modified >= 1
    finally:
        watcher.stop()
        consumer.stop()
        _done, still_pending = await asyncio.wait({watcher_task, consumer_task}, timeout=10.0)
        for task in still_pending:
            task.cancel()
        await asyncio.gather(watcher_task, consumer_task, return_exceptions=True)
