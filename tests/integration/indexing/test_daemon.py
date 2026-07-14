"""Integration tests for DaemonManager durability: supervision + startup catch-up.

Requires Memgraph + Valkey (provided by conftest fixtures).
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import pytest

from code_atlas.events import FileChanged, Topic
from code_atlas.indexing.daemon import DaemonManager
from code_atlas.indexing.orchestrator import index_project
from code_atlas.settings import ExtraVaultSettings, derive_project_name

if TYPE_CHECKING:
    from code_atlas.events import EventBus
    from code_atlas.graph.client import GraphClient
    from code_atlas.settings import AtlasSettings

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_python_file(root, rel_path: str, content: str) -> None:
    """Write a Python file under *root* at the given relative path."""
    full = root / rel_path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content, encoding="utf-8")


async def _wait_for_rows(
    graph_client: GraphClient,
    query: str,
    params: dict,
    *,
    timeout_s: float = 30.0,
) -> list:
    """Poll *query* until it returns rows or the deadline passes."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        rows = await graph_client.execute(query, params)
        if rows:
            return rows
        await asyncio.sleep(0.5)
    return []


# ---------------------------------------------------------------------------
# Supervision: consumer crash → restart
# ---------------------------------------------------------------------------


async def test_consumer_restarts_after_group_destroyed(
    settings: AtlasSettings,
    graph_client: GraphClient,
    event_bus: EventBus,
) -> None:
    """One uncaught consumer exception must not permanently kill consumption.

    Destroying the 'ast' group out from under a blocked XREADGROUP raises
    NOGROUP inside TierConsumer.run(). Before supervision, the consumer died
    once and the daemon starved forever; now the supervised restart loop
    re-runs run(), whose ensure_group() recreates the group, and the
    published change is consumed.
    """
    settings.embeddings.enabled = False
    await graph_client.ensure_schema()

    daemon = DaemonManager()
    started = await daemon.start(settings, graph_client, include_watcher=False, catchup=False)
    assert started is True
    try:
        bus = daemon._bus
        assert bus is not None
        # Let the consumer enter its read loop (run() creates the group)
        await asyncio.sleep(0.5)

        # Simulate the pre-fix bus.flush() / Valkey data loss
        key = bus._stream_key(Topic.FILE_CHANGED)
        await bus._redis.xgroup_destroy(key, "ast")

        # Publish a change for a real file on the daemon's (project-scoped) bus
        _write_python_file(settings.project_root, "hello.py", "def greet():\n    return 'hi'\n")
        project_name = derive_project_name(settings.project_root)
        await bus.publish(
            Topic.FILE_CHANGED,
            FileChanged(
                path="hello.py",
                change_type="created",
                project_name=project_name,
                project_root=str(settings.project_root),
            ),
        )

        rows = await _wait_for_rows(
            graph_client,
            "MATCH (c:Callable {name: 'greet'}) WHERE c.uid STARTS WITH $prefix RETURN c.uid AS uid",
            {"prefix": f"{project_name}:"},
            timeout_s=30.0,
        )
        assert rows, "entity never appeared — consumer was not restarted after NOGROUP"
        assert daemon.status()["crash_counts"].get("ast-0", 0) >= 1
    finally:
        await daemon.stop()


# ---------------------------------------------------------------------------
# Startup catch-up: edits made while the daemon was down
# ---------------------------------------------------------------------------


async def test_daemon_start_runs_catchup_delta(
    settings: AtlasSettings,
    graph_client: GraphClient,
    event_bus: EventBus,
) -> None:
    """Files changed while nothing was running are indexed by start(catchup=True).

    Before the fix nothing ever indexed them: the watcher only sees future
    changes and StalenessChecker only annotates query results.
    """
    settings.embeddings.enabled = False
    await graph_client.ensure_schema()

    # Initial index of a small project (several files keep the later add
    # under the delta threshold)
    for i in range(5):
        _write_python_file(settings.project_root, f"mod_{i}.py", f"def fn_{i}():\n    return {i}\n")
    await index_project(settings, graph_client, event_bus, drain_timeout_s=60.0)

    # Edit while nothing is running — no watcher, no daemon, no consumers
    _write_python_file(settings.project_root, "added_later.py", "def beta():\n    return 42\n")

    daemon = DaemonManager()
    started = await daemon.start(settings, graph_client, include_watcher=False, catchup=True)
    assert started is True
    try:
        # Catch-up is awaited inside start(): the new file's entities must
        # already be in the graph by the time start() returns.
        project_name = derive_project_name(settings.project_root)
        rows = await graph_client.execute(
            "MATCH (c:Callable {name: 'beta'}) WHERE c.uid STARTS WITH $prefix RETURN c.uid AS uid",
            {"prefix": f"{project_name}:"},
        )
        assert rows, "file added while the daemon was down was not indexed by startup catch-up"
    finally:
        await daemon.stop()


# ---------------------------------------------------------------------------
# Extra-vault indexing (Phase 2 catch-up scan + Phase 5 live watching)
# ---------------------------------------------------------------------------


async def test_daemon_indexes_extra_vault(
    settings: AtlasSettings,
    graph_client: GraphClient,
    event_bus: EventBus,
    tmp_path,
) -> None:
    """A note already in a configured extra vault becomes a searchable Note node
    via the daemon's one-time startup catch-up scan for that vault."""
    settings.embeddings.enabled = False
    await graph_client.ensure_schema()

    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    _write_python_file(
        vault_dir,
        "test-vault-note.md",
        "---\nid: test-vault-note\nkind: note\ntags: [test]\n---\n\n# Test Vault Note\n\nSome content.\n",
    )
    settings.knowledge.extra_vaults = [ExtraVaultSettings(path=str(vault_dir), project_name="test-vault")]

    daemon = DaemonManager()
    started = await daemon.start(settings, graph_client, include_watcher=False, catchup=True)
    assert started is True
    try:
        rows = await _wait_for_rows(
            graph_client,
            "MATCH (n:Note {uid: $uid}) RETURN n.kind AS kind, n.docstring AS docstring",
            {"uid": "test-vault:note:test-vault-note"},
            timeout_s=30.0,
        )
        assert rows, "vault note never became a Note node — vault catch-up or consumer wiring is broken"
        assert rows[0]["kind"] == "note"
        assert "Some content." in rows[0]["docstring"]
    finally:
        await daemon.stop()


async def test_daemon_live_watches_extra_vault(
    settings: AtlasSettings,
    graph_client: GraphClient,
    event_bus: EventBus,
    tmp_path,
) -> None:
    """A note added to an extra vault AFTER the daemon starts is indexed live —
    multi-root watching (Phase 5), not a re-scan on the next poll cycle."""
    settings.embeddings.enabled = False
    settings.watcher.debounce_s = 0.5
    settings.watcher.max_wait_s = 10.0
    await graph_client.ensure_schema()

    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    settings.knowledge.extra_vaults = [ExtraVaultSettings(path=str(vault_dir), project_name="test-vault")]

    daemon = DaemonManager()
    started = await daemon.start(settings, graph_client, include_watcher=False, catchup=True)
    assert started is True
    try:
        assert len(daemon._vault_watchers) == 1

        # Let awatch initialize its OS-level watch handle before touching the file
        # (the same settle window test_live_update.py uses for the main watcher).
        await asyncio.sleep(1.5)

        # Written after start() returns — only a live watcher (not the one-time
        # catch-up scan already completed above) can pick this up.
        _write_python_file(
            vault_dir,
            "later-note.md",
            "---\nid: later-note\nkind: note\ntags: [test]\n---\n\n# Later Note\n\nAppeared after startup.\n",
        )

        rows = await _wait_for_rows(
            graph_client,
            "MATCH (n:Note {uid: $uid}) RETURN n.docstring AS docstring",
            {"uid": "test-vault:note:later-note"},
            timeout_s=20.0,
        )
        assert rows, "note added after daemon startup was never indexed — vault live-watching is broken"
        assert "Appeared after startup." in rows[0]["docstring"]
    finally:
        await daemon.stop()
