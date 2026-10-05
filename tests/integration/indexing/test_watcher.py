"""Integration test: real filesystem watcher detects actual file changes."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from code_atlas.indexing.orchestrator import FileScope
from code_atlas.indexing.watcher import FileWatcher
from code_atlas.settings import AtlasSettings, ScopeSettings, WatcherSettings

if TYPE_CHECKING:
    from collections.abc import Callable

    from watchfiles import Change

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
    return FileWatcher(tmp_path, bus, scope, settings)  # ty: ignore[invalid-argument-type]


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


# ---------------------------------------------------------------------------
# Watch roots (ATL-195)
# ---------------------------------------------------------------------------


def _published(bus: RecordingBus) -> set[str]:
    return {ev.path for _, ev in bus.published}


async def _until(cond: Callable[[], bool], what: str, *, timeout: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not cond():
        if loop.time() > deadline:
            pytest.fail(f"timed out waiting for {what}")
        await asyncio.sleep(0.05)


class TestWatchRoots:
    """Excluded top-level directories never become OS events at all.

    The assertions are on what reached ``_on_change``, not on what was published: the
    per-path filter already kept excluded files out of the published set, so checking
    that alone would pass against the old single recursive watch.
    """

    @staticmethod
    def _watcher(root: Path, bus: RecordingBus, **scope_kwargs: Any) -> tuple[FileWatcher, list[str]]:
        settings = AtlasSettings(project_root=root, scope=ScopeSettings(**scope_kwargs))
        scope = FileScope(root, settings)
        watcher = FileWatcher(
            root,
            bus,  # ty: ignore[invalid-argument-type]
            scope,
            WatcherSettings(debounce_s=0.1, max_wait_s=2.0),
            known_files=scope.scan(),
        )
        seen: list[str] = []
        real_on_change = watcher._on_change

        async def spy(changes: set[tuple[Change, str]]) -> None:
            seen.extend(Path(p).relative_to(root).as_posix() for _, p in changes)
            await real_on_change(changes)

        watcher._on_change = spy  # ty: ignore[invalid-assignment]
        return watcher, seen

    @staticmethod
    async def _start(watcher: FileWatcher) -> asyncio.Task[None]:
        task = asyncio.create_task(watcher.run())
        await _until(lambda: watcher._tree_task is not None or task.done(), "tree watch to arm")
        await asyncio.sleep(0.3)  # the root watch gives no arming signal of its own
        return task

    @staticmethod
    async def _stop(watcher: FileWatcher, task: asyncio.Task[None]) -> None:
        watcher.stop()
        await asyncio.wait_for(task, timeout=5.0)

    async def test_burst_under_excluded_dir_produces_no_events(self, tmp_path: Path) -> None:
        (tmp_path / "src").mkdir()
        (tmp_path / ".claude" / "worktrees").mkdir(parents=True)
        bus = RecordingBus()
        watcher, seen = self._watcher(tmp_path, bus)
        task = await self._start(watcher)

        burst = tmp_path / ".claude" / "worktrees" / "x" / ".venv" / "pkg"
        burst.mkdir(parents=True)
        for i in range(200):
            (burst / f"m{i}.py").write_text("x = 1\n", encoding="utf-8")
        (tmp_path / "src" / "real.py").write_text("y = 1\n", encoding="utf-8")
        await _until(lambda: "src/real.py" in _published(bus), "src/real.py to publish")
        await asyncio.sleep(0.5)  # let any straggling .claude batch arrive
        await self._stop(watcher, task)

        assert [p for p in seen if p.startswith(".claude/")] == []
        assert _published(bus) == {"src/real.py"}

    async def test_included_dir_reports_create_modify_delete(self, tmp_path: Path) -> None:
        (tmp_path / "src" / "pkg").mkdir(parents=True)
        bus = RecordingBus()
        watcher, _ = self._watcher(tmp_path, bus)
        task = await self._start(watcher)

        mod = tmp_path / "src" / "pkg" / "mod.py"
        types: list[str] = []
        for action, expected in (
            (lambda: mod.write_text("a = 1\n", encoding="utf-8"), "created"),
            (lambda: mod.write_text("a = 2\n", encoding="utf-8"), "modified"),
            (mod.unlink, "deleted"),
        ):
            bus.published.clear()
            action()
            await _until(lambda: "src/pkg/mod.py" in _published(bus), f"{expected} event")
            types.append(next(ev.change_type for _, ev in bus.published if ev.path == "src/pkg/mod.py"))
        await self._stop(watcher, task)

        assert types == ["created", "modified", "deleted"]

    async def test_top_level_file_reports(self, tmp_path: Path) -> None:
        (tmp_path / "src").mkdir()
        bus = RecordingBus()
        watcher, _ = self._watcher(tmp_path, bus)
        task = await self._start(watcher)

        (tmp_path / "top.py").write_text("x = 1\n", encoding="utf-8")
        await _until(lambda: "top.py" in _published(bus), "top.py to publish")
        await self._stop(watcher, task)

    async def test_new_eligible_dir_is_expanded_and_then_watched(self, tmp_path: Path) -> None:
        (tmp_path / "src").mkdir()
        bus = RecordingBus()
        watcher, _ = self._watcher(tmp_path, bus)
        task = await self._start(watcher)

        staged = tmp_path.parent / f"{tmp_path.name}-staged"
        staged.mkdir()
        (staged / "a.py").write_text("a = 1\n", encoding="utf-8")
        staged.rename(tmp_path / "newpkg")  # arrives populated: only the expansion can see a.py
        await _until(lambda: "newpkg" in watcher._tree_dirs, "re-arm onto newpkg")
        await _until(lambda: "newpkg/a.py" in _published(bus), "newpkg/a.py to publish")

        (tmp_path / "newpkg" / "b.py").write_text("b = 1\n", encoding="utf-8")
        await _until(lambda: "newpkg/b.py" in _published(bus), "newpkg/b.py to publish")
        await self._stop(watcher, task)

    async def test_new_excluded_dir_changes_nothing(self, tmp_path: Path) -> None:
        (tmp_path / "src").mkdir()
        bus = RecordingBus()
        watcher, seen = self._watcher(tmp_path, bus)
        task = await self._start(watcher)
        armed = watcher._tree_task

        venv = tmp_path / ".venv" / "lib"
        venv.mkdir(parents=True)
        for i in range(50):
            (venv / f"m{i}.py").write_text("x = 1\n", encoding="utf-8")
        await _until(lambda: ".venv" in seen, "the root watch to report .venv")
        await asyncio.sleep(0.5)
        replaced = watcher._tree_task is not armed
        await self._stop(watcher, task)

        assert not replaced
        assert [p for p in seen if p.startswith(".venv/")] == []
        assert _published(bus) == set()

    async def test_renamed_top_level_dir_moves_the_watch(self, tmp_path: Path) -> None:
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "mod.py").write_text("a = 1\n", encoding="utf-8")
        bus = RecordingBus()
        watcher, _ = self._watcher(tmp_path, bus)
        task = await self._start(watcher)

        (tmp_path / "src").rename(tmp_path / "lib")
        await _until(lambda: watcher._tree_dirs == ["lib"], "re-arm onto lib")
        await _until(lambda: {"src/mod.py", "lib/mod.py"} <= _published(bus), "delete + create")
        events = {ev.path: ev.change_type for _, ev in bus.published}

        bus.published.clear()
        (tmp_path / "lib" / "new.py").write_text("n = 1\n", encoding="utf-8")
        await _until(lambda: "lib/new.py" in _published(bus), "lib/new.py to publish")
        await self._stop(watcher, task)

        assert events["src/mod.py"] == "deleted"
        assert events["lib/mod.py"] == "created"

    async def test_scope_paths_narrow_the_watch(self, tmp_path: Path) -> None:
        for d in ("src", "docs"):
            (tmp_path / d).mkdir()
        bus = RecordingBus()
        watcher, seen = self._watcher(tmp_path, bus, paths=["src"])
        task = await self._start(watcher)
        watched = list(watcher._tree_dirs)

        (tmp_path / "docs" / "d.py").write_text("x = 1\n", encoding="utf-8")
        (tmp_path / "src" / "s.py").write_text("x = 1\n", encoding="utf-8")
        await _until(lambda: "src/s.py" in _published(bus), "src/s.py to publish")
        await asyncio.sleep(0.5)
        await self._stop(watcher, task)

        assert watched == ["src"]
        assert "docs/d.py" not in seen

    async def test_root_with_no_eligible_dirs(self, tmp_path: Path) -> None:
        (tmp_path / ".venv").mkdir()
        bus = RecordingBus()
        watcher, _ = self._watcher(tmp_path, bus)
        task = asyncio.create_task(watcher.run())
        await asyncio.sleep(0.5)

        assert watcher._tree_task is None
        (tmp_path / "top.py").write_text("x = 1\n", encoding="utf-8")
        await _until(lambda: "top.py" in _published(bus), "top.py to publish")
        await self._stop(watcher, task)

    async def test_rearm_loses_no_writes_to_watched_dirs(self, tmp_path: Path) -> None:
        (tmp_path / "src").mkdir()
        bus = RecordingBus()
        watcher, _ = self._watcher(tmp_path, bus)
        task = await self._start(watcher)
        written: set[str] = set()
        rearmed = False

        async def hammer() -> None:
            nonlocal rearmed
            for i in range(60):
                (tmp_path / "src" / f"hot_{i}.py").write_text("x = 1\n", encoding="utf-8")
                written.add(f"src/hot_{i}.py")
                rearmed = rearmed or "newpkg" in watcher._tree_dirs
                await asyncio.sleep(0.05)

        hammering = asyncio.create_task(hammer())
        await asyncio.sleep(0.5)
        (tmp_path / "newpkg").mkdir()
        await hammering
        await _until(lambda: written <= _published(bus), "every hot_*.py to publish", timeout=10.0)
        await self._stop(watcher, task)

        assert rearmed, "the re-arm did not happen while writes were in flight"
