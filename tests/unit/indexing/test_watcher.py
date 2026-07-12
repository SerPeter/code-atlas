"""Unit tests for the file watcher with hybrid debounce."""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

from watchfiles import Change

from code_atlas.events import FileChanged
from code_atlas.indexing.orchestrator import DetectedProject
from code_atlas.indexing.watcher import FileWatcher
from code_atlas.settings import WatcherSettings

if TYPE_CHECKING:
    from pathlib import Path

    from code_atlas.events import Topic


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class StubScope:
    """Minimal FileScope substitute that accepts/rejects based on a set."""

    def __init__(self, *, excluded: set[str] | None = None) -> None:
        self._excluded = excluded or set()

    def is_included(self, rel_path: str) -> bool:
        for pat in self._excluded:
            if pat.endswith("/") and rel_path.startswith(pat):
                return False
            if rel_path == pat:
                return False
        return True


class ExtensionStubScope:
    """FileScope substitute that mimics the real include-spec: only files
    with a recognized extension pass. A bare directory path (no extension)
    is rejected — exactly like the real ``FileScope.is_included()``.
    """

    def __init__(self, *, ext: str = ".py") -> None:
        self._ext = ext

    def is_included(self, rel_path: str) -> bool:
        return rel_path.endswith(self._ext)


class RecordingBus:
    """Fake EventBus that records published events.

    ``publish`` yields control before recording — the real bus suspends on
    network I/O, which is exactly where a pending cancellation of the
    flushing timer task would be delivered. A non-yielding fake masks the
    _flush self-cancel bug entirely.
    """

    def __init__(self) -> None:
        self.published: list[tuple[Topic, FileChanged]] = []

    async def publish(self, topic: Topic, event: FileChanged) -> bytes:
        await asyncio.sleep(0)
        self.published.append((topic, event))
        return b"fake-id"


class FlakyBus(RecordingBus):
    """Bus whose publish raises a configurable number of times, then succeeds."""

    def __init__(self, failures: int = 1) -> None:
        super().__init__()
        self._failures = failures

    async def publish(self, topic: Topic, event: FileChanged) -> bytes:
        await asyncio.sleep(0)
        if self._failures > 0:
            self._failures -= 1
            raise ConnectionError("bus down")
        return await super().publish(topic, event)


def _make_watcher(
    tmp_path: Path,
    bus: RecordingBus,
    *,
    debounce_s: float = 0.1,
    max_wait_s: float = 0.5,
    excluded: set[str] | None = None,
) -> FileWatcher:
    """Create a FileWatcher with fast timers for testing."""
    scope = StubScope(excluded=excluded)
    settings = WatcherSettings(debounce_s=debounce_s, max_wait_s=max_wait_s)
    return FileWatcher(tmp_path, bus, scope, settings)  # type: ignore[arg-type]


def _make_monorepo_watcher(
    tmp_path: Path,
    bus: RecordingBus,
    subs: list[DetectedProject],
    *,
    root_name: str = "mono",
) -> FileWatcher:
    """Create a monorepo FileWatcher with detected sub-projects."""
    return FileWatcher(
        tmp_path,
        bus,  # type: ignore[arg-type]
        StubScope(),  # type: ignore[arg-type]
        WatcherSettings(debounce_s=0.05, max_wait_s=0.5),
        sub_projects=subs,
        root_name=root_name,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestChangeTypeMapping:
    """Verify watchfiles Change enum maps to correct change_type strings."""

    async def test_added_maps_to_created(self, tmp_path: Path) -> None:
        bus = RecordingBus()
        watcher = _make_watcher(tmp_path, bus)
        await watcher._on_change({(Change.added, str(tmp_path / "new.py"))})
        ev = watcher._pending["new.py"]
        assert ev.change_type == "created"

    async def test_modified_maps_to_modified(self, tmp_path: Path) -> None:
        bus = RecordingBus()
        watcher = _make_watcher(tmp_path, bus)
        await watcher._on_change({(Change.modified, str(tmp_path / "edit.py"))})
        ev = watcher._pending["edit.py"]
        assert ev.change_type == "modified"

    async def test_deleted_maps_to_deleted(self, tmp_path: Path) -> None:
        bus = RecordingBus()
        watcher = _make_watcher(tmp_path, bus)
        await watcher._on_change({(Change.deleted, str(tmp_path / "gone.py"))})
        ev = watcher._pending["gone.py"]
        assert ev.change_type == "deleted"


class TestExclusionFiltering:
    """Changes to excluded paths are ignored."""

    async def test_git_dir_excluded(self, tmp_path: Path) -> None:
        bus = RecordingBus()
        watcher = _make_watcher(tmp_path, bus, excluded={".git/"})
        await watcher._on_change({(Change.modified, str(tmp_path / ".git" / "index"))})
        assert watcher._pending == {}

    async def test_atlas_dir_excluded(self, tmp_path: Path) -> None:
        bus = RecordingBus()
        watcher = _make_watcher(tmp_path, bus, excluded={".atlas/"})
        await watcher._on_change({(Change.modified, str(tmp_path / ".atlas" / "state.json"))})
        assert watcher._pending == {}

    async def test_included_file_accepted(self, tmp_path: Path) -> None:
        bus = RecordingBus()
        watcher = _make_watcher(tmp_path, bus, excluded={".git/"})
        await watcher._on_change({(Change.modified, str(tmp_path / "src" / "main.py"))})
        assert "src/main.py" in watcher._pending

    async def test_path_outside_root_skipped(self, tmp_path: Path) -> None:
        bus = RecordingBus()
        watcher = _make_watcher(tmp_path, bus)
        await watcher._on_change({(Change.modified, "/some/other/path/file.py")})
        assert watcher._pending == {}


class TestTimerFiredFlush:
    """Timer-fired flushes must publish every pending event.

    Regression for the _flush self-cancel bug: _flush() cancelled the timer
    task currently executing it, so CancelledError fired at the first await
    inside bus.publish — after _pending was already cleared — and every
    timer-fired batch was silently dropped.
    """

    async def test_debounce_flush_publishes_all_events(self, tmp_path: Path) -> None:
        bus = RecordingBus()
        watcher = _make_watcher(tmp_path, bus, debounce_s=0.05, max_wait_s=5.0)

        await watcher._on_change(
            {
                (Change.modified, str(tmp_path / "a.py")),
                (Change.modified, str(tmp_path / "b.py")),
                (Change.modified, str(tmp_path / "c.py")),
            }
        )
        await asyncio.sleep(0.3)

        assert {ev.path for _, ev in bus.published} == {"a.py", "b.py", "c.py"}
        assert watcher._pending == {}

    async def test_max_wait_flush_publishes_all_events(self, tmp_path: Path) -> None:
        bus = RecordingBus()
        watcher = _make_watcher(tmp_path, bus, debounce_s=5.0, max_wait_s=0.1)

        await watcher._on_change({(Change.modified, str(tmp_path / "a.py"))})
        await watcher._on_change({(Change.modified, str(tmp_path / "b.py"))})
        await asyncio.sleep(0.4)

        assert {ev.path for _, ev in bus.published} == {"a.py", "b.py"}


class TestDebounceReset:
    """Rapid changes within the debounce window produce a single flush."""

    async def test_rapid_changes_single_flush(self, tmp_path: Path) -> None:
        bus = RecordingBus()
        watcher = _make_watcher(tmp_path, bus, debounce_s=0.15, max_wait_s=5.0)

        # Simulate rapid changes — each resets the debounce timer
        await watcher._on_change({(Change.modified, str(tmp_path / "a.py"))})
        await asyncio.sleep(0.05)
        await watcher._on_change({(Change.modified, str(tmp_path / "b.py"))})
        await asyncio.sleep(0.05)
        await watcher._on_change({(Change.modified, str(tmp_path / "c.py"))})

        # Wait for debounce to fire (0.15s after last change)
        await asyncio.sleep(0.25)

        # All three files should be flushed in a single batch
        assert len(bus.published) == 3
        paths = {ev.path for _, ev in bus.published}
        assert paths == {"a.py", "b.py", "c.py"}

    async def test_latest_change_type_wins(self, tmp_path: Path) -> None:
        bus = RecordingBus()
        watcher = _make_watcher(tmp_path, bus, debounce_s=0.1, max_wait_s=5.0)

        # Same file: created -> modified -> latest type wins
        await watcher._on_change({(Change.added, str(tmp_path / "x.py"))})
        await watcher._on_change({(Change.modified, str(tmp_path / "x.py"))})

        await asyncio.sleep(0.2)

        assert len(bus.published) == 1
        assert bus.published[0][1].change_type == "modified"


class TestMaxWaitCeiling:
    """Continuous changes beyond max-wait trigger a forced flush."""

    async def test_max_wait_forces_flush(self, tmp_path: Path) -> None:
        bus = RecordingBus()
        watcher = _make_watcher(tmp_path, bus, debounce_s=0.3, max_wait_s=0.2)

        # First change starts the max-wait timer
        await watcher._on_change({(Change.modified, str(tmp_path / "a.py"))})

        # Keep resetting debounce before it fires
        await asyncio.sleep(0.1)
        await watcher._on_change({(Change.modified, str(tmp_path / "b.py"))})

        # Max-wait (0.2s) should fire before debounce (0.3s)
        await asyncio.sleep(0.2)

        assert len(bus.published) >= 1
        paths = {ev.path for _, ev in bus.published}
        assert "a.py" in paths


class TestFlushSerialization:
    """Changes arriving during flush are queued for the next batch."""

    async def test_changes_during_flush_queued(self, tmp_path: Path) -> None:
        bus = RecordingBus()
        watcher = _make_watcher(tmp_path, bus, debounce_s=0.1, max_wait_s=5.0)

        # Add initial changes and trigger flush
        await watcher._on_change({(Change.modified, str(tmp_path / "first.py"))})
        await asyncio.sleep(0.15)

        # First flush should have completed
        assert len(bus.published) == 1
        assert bus.published[0][1].path == "first.py"

        # Now add more changes — should start a new batch
        await watcher._on_change({(Change.modified, str(tmp_path / "second.py"))})
        await asyncio.sleep(0.15)

        assert len(bus.published) == 2
        assert bus.published[1][1].path == "second.py"


class TestGracefulShutdown:
    """stop() cancels pending timers and allows clean exit."""

    async def test_stop_cancels_timers(self, tmp_path: Path) -> None:
        bus = RecordingBus()
        watcher = _make_watcher(tmp_path, bus, debounce_s=10.0, max_wait_s=30.0)

        # Add a change (starts timers) but stop before they fire
        await watcher._on_change({(Change.modified, str(tmp_path / "a.py"))})
        assert watcher._debounce_task is not None

        watcher.stop()
        assert watcher.stopped
        assert watcher._debounce_task is None
        assert watcher._max_wait_task is None

    async def test_stop_drains_pending(self, tmp_path: Path) -> None:
        """Calling stop during run() should drain remaining pending changes."""
        bus = RecordingBus()
        watcher = _make_watcher(tmp_path, bus, debounce_s=10.0, max_wait_s=30.0)

        # Manually add pending changes (simulating what _on_change does)
        watcher._pending["drain_me.py"] = FileChanged(path="drain_me.py", change_type="modified")

        # Flush should publish the pending change
        await watcher._flush()
        assert len(bus.published) == 1
        assert bus.published[0][1].path == "drain_me.py"


class TestFlushPublishFailure:
    """A publish failure must not drop the snapshotted batch.

    Regression for the silent-loss path: _flush cleared _pending before
    publishing, so a raised publish lost the whole batch and the exception
    died in the unobserved timer task.
    """

    async def test_failed_publish_requeues_remainder_and_retries(self, tmp_path: Path) -> None:
        bus = FlakyBus(failures=1)
        watcher = _make_watcher(tmp_path, bus, debounce_s=0.05, max_wait_s=5.0)
        await watcher._on_change(
            {
                (Change.modified, str(tmp_path / "a.py")),
                (Change.modified, str(tmp_path / "b.py")),
            }
        )

        # First flush: publish raises on the first event — the whole batch is re-queued
        await watcher._flush()
        assert bus.published == []
        assert set(watcher._pending) == {"a.py", "b.py"}
        assert watcher._debounce_task is not None  # retry flush is scheduled

        # Retry succeeds and drains the re-queued batch
        await watcher._flush()
        assert {ev.path for _, ev in bus.published} == {"a.py", "b.py"}
        assert watcher._pending == {}


class TestMonorepoPathIdentity:
    """Watcher seals the S4 path-identity contract into published events.

    FileChanged.path must be relative to the OWNING project's root (the
    sub-project root for monorepo sub-project files), project_name must be
    the graph identity ("{root}/{sub}"), and project_root must be the
    owning project root — matching what the full indexer stores on nodes.
    """

    async def test_sub_project_event_identity(self, tmp_path: Path) -> None:
        bus = RecordingBus()
        subs = [
            DetectedProject(
                name="core",
                path="packages/core",
                root=tmp_path / "packages" / "core",
                marker="pyproject.toml",
            )
        ]
        watcher = _make_monorepo_watcher(tmp_path, bus, subs)

        await watcher._on_change({(Change.modified, str(tmp_path / "packages" / "core" / "src" / "foo.py"))})
        await watcher._flush()

        assert len(bus.published) == 1
        ev = bus.published[0][1]
        assert ev.path == "src/foo.py"
        assert ev.project_name == "mono/core"
        assert ev.project_root == str(tmp_path / "packages" / "core")

    async def test_root_file_event_identity(self, tmp_path: Path) -> None:
        bus = RecordingBus()
        subs = [
            DetectedProject(
                name="core",
                path="packages/core",
                root=tmp_path / "packages" / "core",
                marker="pyproject.toml",
            )
        ]
        watcher = _make_monorepo_watcher(tmp_path, bus, subs)

        await watcher._on_change({(Change.modified, str(tmp_path / "tools" / "run.py"))})
        await watcher._flush()

        assert len(bus.published) == 1
        ev = bus.published[0][1]
        assert ev.path == "tools/run.py"
        assert ev.project_name == "mono"
        assert ev.project_root == str(tmp_path.resolve())

    async def test_colliding_sub_relative_paths_both_published(self, tmp_path: Path) -> None:
        bus = RecordingBus()
        subs = [
            DetectedProject(name="a", path="packages/a", root=tmp_path / "packages" / "a", marker="pyproject.toml"),
            DetectedProject(name="b", path="packages/b", root=tmp_path / "packages" / "b", marker="pyproject.toml"),
        ]
        watcher = _make_monorepo_watcher(tmp_path, bus, subs)

        await watcher._on_change(
            {
                (Change.modified, str(tmp_path / "packages" / "a" / "src" / "main.py")),
                (Change.modified, str(tmp_path / "packages" / "b" / "src" / "main.py")),
            }
        )
        await watcher._flush()

        assert len(bus.published) == 2
        events = [ev for _, ev in bus.published]
        assert {ev.path for ev in events} == {"src/main.py"}
        assert {ev.project_name for ev in events} == {"mono/a", "mono/b"}


class TestDirectoryRenameExpansion:
    """A directory rename/delete has no extension of its own, so the
    include spec always rejects the bare directory path — without
    expanding it against known/on-disk files, every entity it contained
    is silently orphaned in the graph.
    """

    async def test_directory_delete_expands_to_known_files(self, tmp_path: Path) -> None:
        bus = RecordingBus()
        settings = WatcherSettings(debounce_s=0.05, max_wait_s=5.0)
        watcher = FileWatcher(tmp_path, bus, ExtensionStubScope(), settings)  # type: ignore[arg-type]
        watcher._known_files = {"pkg/mod_a.py", "pkg/mod_b.py", "other.py"}

        await watcher._on_change({(Change.deleted, str(tmp_path / "pkg"))})
        await watcher._flush()

        assert {ev.path for _, ev in bus.published} == {"pkg/mod_a.py", "pkg/mod_b.py"}
        assert all(ev.change_type == "deleted" for _, ev in bus.published)
        assert watcher._known_files == {"other.py"}

    async def test_directory_delete_without_known_children_is_ignored(self, tmp_path: Path) -> None:
        bus = RecordingBus()
        settings = WatcherSettings(debounce_s=0.05, max_wait_s=5.0)
        watcher = FileWatcher(tmp_path, bus, ExtensionStubScope(), settings)  # type: ignore[arg-type]

        await watcher._on_change({(Change.deleted, str(tmp_path / "empty_dir"))})
        assert watcher._pending == {}

    async def test_directory_add_expands_to_contained_files(self, tmp_path: Path) -> None:
        bus = RecordingBus()
        new_dir = tmp_path / "pkg"
        new_dir.mkdir()
        (new_dir / "mod_a.py").write_text("a = 1\n")
        (new_dir / "mod_b.py").write_text("b = 1\n")
        (new_dir / "readme.txt").write_text("not code\n")

        settings = WatcherSettings(debounce_s=0.05, max_wait_s=5.0)
        watcher = FileWatcher(tmp_path, bus, ExtensionStubScope(), settings)  # type: ignore[arg-type]

        await watcher._on_change({(Change.added, str(new_dir))})
        await watcher._flush()

        assert {ev.path for _, ev in bus.published} == {"pkg/mod_a.py", "pkg/mod_b.py"}
        assert all(ev.change_type == "created" for _, ev in bus.published)
        assert watcher._known_files == {"pkg/mod_a.py", "pkg/mod_b.py"}

    async def test_directory_add_scan_does_not_block_event_loop(self, tmp_path: Path, monkeypatch) -> None:
        """The os.walk expansion must run off the event loop (asyncio.to_thread), like
        FileScope.scan in daemon.py -- otherwise a large directory add stalls the
        watcher's timers and event delivery for the duration of the walk."""
        import time

        import code_atlas.indexing.watcher as watcher_module

        new_dir = tmp_path / "big"
        new_dir.mkdir()
        (new_dir / "a.py").write_text("a = 1\n")

        real_walk = os.walk

        def slow_walk(path):
            time.sleep(0.3)  # simulate a large/slow directory scan
            yield from real_walk(path)

        monkeypatch.setattr(watcher_module.os, "walk", slow_walk)

        bus = RecordingBus()
        settings = WatcherSettings(debounce_s=0.05, max_wait_s=5.0)
        watcher = FileWatcher(tmp_path, bus, ExtensionStubScope(), settings)  # type: ignore[arg-type]

        ticks = 0

        async def ticker() -> None:
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        ticker_task = asyncio.create_task(ticker())
        try:
            await watcher._expand_directory_add(new_dir)
        finally:
            ticker_task.cancel()

        # If the walk ran directly on the event loop thread, the ticker would be
        # starved for the whole 0.3s sleep and accumulate close to 0 ticks.
        assert ticks >= 10, f"event loop appears blocked during os.walk (only {ticks} ticks)"

    async def test_normal_file_delete_still_untracks_known_files(self, tmp_path: Path) -> None:
        """Regression guard: ordinary per-file changes keep updating _known_files."""
        bus = RecordingBus()
        settings = WatcherSettings(debounce_s=0.05, max_wait_s=5.0)
        watcher = FileWatcher(tmp_path, bus, ExtensionStubScope(), settings)  # type: ignore[arg-type]
        watcher._known_files = {"solo.py"}

        await watcher._on_change({(Change.deleted, str(tmp_path / "solo.py"))})
        await watcher._flush()

        assert {ev.path for _, ev in bus.published} == {"solo.py"}
        assert watcher._known_files == set()
