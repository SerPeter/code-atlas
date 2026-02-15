"""Integration test: real filesystem watcher detects actual file changes."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from code_atlas.indexing.watcher import FileWatcher
from code_atlas.settings import WatcherSettings

if TYPE_CHECKING:
    from pathlib import Path

    from code_atlas.events import FileChanged, Topic

pytestmark = pytest.mark.integration


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


class TestEndToEnd:
    """Integration test: real filesystem watcher detects actual file changes."""

    async def test_end_to_end_watcher_detects_file_change(self, tmp_path: Path) -> None:
        """Start watcher, modify a file on disk, assert FileChanged event arrives."""
        # Create initial file
        py_file = tmp_path / "hello.py"
        py_file.write_text("x = 1\n", encoding="utf-8")

        bus = RecordingBus()
        watcher = _make_watcher(tmp_path, bus, debounce_s=0.2, max_wait_s=2.0)

        # Run watcher in background
        task = asyncio.create_task(watcher.run())

        # Allow watcher to warm up
        await asyncio.sleep(0.5)

        # Modify file on disk
        py_file.write_text("x = 2\n", encoding="utf-8")

        # Wait for debounce + flush
        await asyncio.sleep(1.5)

        # Stop watcher gracefully
        watcher.stop()
        await asyncio.wait_for(task, timeout=3.0)

        # Assert we got the change event
        assert len(bus.published) >= 1
        paths = {ev.path for _, ev in bus.published}
        assert "hello.py" in paths
        change_types = {ev.change_type for _, ev in bus.published if ev.path == "hello.py"}
        assert "modified" in change_types
