"""Unit tests for the file watcher with hybrid debounce."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from watchfiles import Change

from code_atlas.settings import WatcherSettings
from code_atlas.watcher import FileWatcher

if TYPE_CHECKING:
    from pathlib import Path

    from code_atlas.events import FileChanged, Topic


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


class RecordingBus:
    """Fake EventBus that records published events."""

    def __init__(self) -> None:
        self.published: list[tuple[Topic, FileChanged]] = []

    async def publish(self, topic: Topic, event: FileChanged, *, maxlen: int = 10_000) -> bytes:
        self.published.append((topic, event))
        return b"fake-id"


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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestChangeTypeMapping:
    """Verify watchfiles Change enum maps to correct change_type strings."""

    async def test_added_maps_to_created(self, tmp_path: Path) -> None:
        bus = RecordingBus()
        watcher = _make_watcher(tmp_path, bus)
        watcher._on_change({(Change.added, str(tmp_path / "new.py"))})
        assert watcher._pending == {"new.py": ("created", "")}

    async def test_modified_maps_to_modified(self, tmp_path: Path) -> None:
        bus = RecordingBus()
        watcher = _make_watcher(tmp_path, bus)
        watcher._on_change({(Change.modified, str(tmp_path / "edit.py"))})
        assert watcher._pending == {"edit.py": ("modified", "")}

    async def test_deleted_maps_to_deleted(self, tmp_path: Path) -> None:
        bus = RecordingBus()
        watcher = _make_watcher(tmp_path, bus)
        watcher._on_change({(Change.deleted, str(tmp_path / "gone.py"))})
        assert watcher._pending == {"gone.py": ("deleted", "")}


class TestExclusionFiltering:
    """Changes to excluded paths are ignored."""

    async def test_git_dir_excluded(self, tmp_path: Path) -> None:
        bus = RecordingBus()
        watcher = _make_watcher(tmp_path, bus, excluded={".git/"})
        watcher._on_change({(Change.modified, str(tmp_path / ".git" / "index"))})
        assert watcher._pending == {}

    async def test_atlas_dir_excluded(self, tmp_path: Path) -> None:
        bus = RecordingBus()
        watcher = _make_watcher(tmp_path, bus, excluded={".atlas/"})
        watcher._on_change({(Change.modified, str(tmp_path / ".atlas" / "state.json"))})
        assert watcher._pending == {}

    async def test_included_file_accepted(self, tmp_path: Path) -> None:
        bus = RecordingBus()
        watcher = _make_watcher(tmp_path, bus, excluded={".git/"})
        watcher._on_change({(Change.modified, str(tmp_path / "src" / "main.py"))})
        assert "src/main.py" in watcher._pending

    async def test_path_outside_root_skipped(self, tmp_path: Path) -> None:
        bus = RecordingBus()
        watcher = _make_watcher(tmp_path, bus)
        watcher._on_change({(Change.modified, "/some/other/path/file.py")})
        assert watcher._pending == {}


class TestDebounceReset:
    """Rapid changes within the debounce window produce a single flush."""

    async def test_rapid_changes_single_flush(self, tmp_path: Path) -> None:
        bus = RecordingBus()
        watcher = _make_watcher(tmp_path, bus, debounce_s=0.15, max_wait_s=5.0)

        # Simulate rapid changes — each resets the debounce timer
        watcher._on_change({(Change.modified, str(tmp_path / "a.py"))})
        await asyncio.sleep(0.05)
        watcher._on_change({(Change.modified, str(tmp_path / "b.py"))})
        await asyncio.sleep(0.05)
        watcher._on_change({(Change.modified, str(tmp_path / "c.py"))})

        # Wait for debounce to fire (0.15s after last change)
        await asyncio.sleep(0.25)

        # All three files should be flushed in a single batch
        assert len(bus.published) == 3
        paths = {ev.path for _, ev in bus.published}
        assert paths == {"a.py", "b.py", "c.py"}

    async def test_latest_change_type_wins(self, tmp_path: Path) -> None:
        bus = RecordingBus()
        watcher = _make_watcher(tmp_path, bus, debounce_s=0.1, max_wait_s=5.0)

        # Same file: created → modified → latest type wins
        watcher._on_change({(Change.added, str(tmp_path / "x.py"))})
        watcher._on_change({(Change.modified, str(tmp_path / "x.py"))})

        await asyncio.sleep(0.2)

        assert len(bus.published) == 1
        assert bus.published[0][1].change_type == "modified"


class TestMaxWaitCeiling:
    """Continuous changes beyond max-wait trigger a forced flush."""

    async def test_max_wait_forces_flush(self, tmp_path: Path) -> None:
        bus = RecordingBus()
        watcher = _make_watcher(tmp_path, bus, debounce_s=0.3, max_wait_s=0.2)

        # First change starts the max-wait timer
        watcher._on_change({(Change.modified, str(tmp_path / "a.py"))})

        # Keep resetting debounce before it fires
        await asyncio.sleep(0.1)
        watcher._on_change({(Change.modified, str(tmp_path / "b.py"))})

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
        watcher._on_change({(Change.modified, str(tmp_path / "first.py"))})
        await asyncio.sleep(0.15)

        # First flush should have completed
        assert len(bus.published) == 1
        assert bus.published[0][1].path == "first.py"

        # Now add more changes — should start a new batch
        watcher._on_change({(Change.modified, str(tmp_path / "second.py"))})
        await asyncio.sleep(0.15)

        assert len(bus.published) == 2
        assert bus.published[1][1].path == "second.py"


class TestGracefulShutdown:
    """stop() cancels pending timers and allows clean exit."""

    async def test_stop_cancels_timers(self, tmp_path: Path) -> None:
        bus = RecordingBus()
        watcher = _make_watcher(tmp_path, bus, debounce_s=10.0, max_wait_s=30.0)

        # Add a change (starts timers) but stop before they fire
        watcher._on_change({(Change.modified, str(tmp_path / "a.py"))})
        assert watcher._debounce_task is not None

        watcher.stop()
        assert watcher._stop_event.is_set()
        assert watcher._debounce_task is None
        assert watcher._max_wait_task is None

    async def test_stop_drains_pending(self, tmp_path: Path) -> None:
        """Calling stop during run() should drain remaining pending changes."""
        bus = RecordingBus()
        watcher = _make_watcher(tmp_path, bus, debounce_s=10.0, max_wait_s=30.0)

        # Manually add pending changes (simulating what _on_change does)
        watcher._pending["drain_me.py"] = ("modified", "")

        # Flush should publish the pending change
        await watcher._flush()
        assert len(bus.published) == 1
        assert bus.published[0][1].path == "drain_me.py"
