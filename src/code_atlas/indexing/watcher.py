"""Filesystem watcher with hybrid debounce for real-time indexing.

Watches the project directory for file changes and publishes
:class:`~code_atlas.events.FileChanged` events to Valkey Streams.
Uses a hybrid debounce strategy: each change resets a short timer,
and the first change in a batch starts a max-wait ceiling.
Whichever fires first triggers a flush.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from watchfiles import Change, awatch

from code_atlas.events import EventBus, FileChanged, Topic
from code_atlas.indexing.orchestrator import classify_file_project

if TYPE_CHECKING:
    from code_atlas.indexing.orchestrator import DetectedProject, FileScope
    from code_atlas.settings import WatcherSettings

# Map watchfiles Change enum → FileChanged.change_type strings
_CHANGE_TYPE_MAP: dict[Change, str] = {
    Change.added: "created",
    Change.modified: "modified",
    Change.deleted: "deleted",
}


class FileWatcher:
    """Async filesystem watcher that produces FileChanged events.

    Uses ``watchfiles.awatch`` (Rust-backed) for efficient cross-platform
    filesystem monitoring, with a hybrid debounce strategy to coalesce
    rapid changes (e.g. branch switch, rebase, bulk save).

    Parameters
    ----------
    project_root:
        Absolute path to the project directory to watch.
    bus:
        EventBus for publishing FileChanged events.
    scope:
        FileScope filter — only changes passing ``is_included()`` are published.
    settings:
        WatcherSettings controlling debounce and max-wait timers.
    known_files:
        Watch-root-relative POSIX paths already known to be in scope (e.g.
        from an initial ``FileScope.scan()``). Seeds directory rename/delete
        detection — a bare directory path never matches the include spec, so
        without a known-files snapshot every file it contained would be
        silently orphaned instead of getting its own deletion event.
    """

    def __init__(
        self,
        project_root: str | Path,
        bus: EventBus,
        scope: FileScope,
        settings: WatcherSettings,
        *,
        sub_projects: list[DetectedProject] | None = None,
        root_name: str = "",
        known_files: list[str] | None = None,
    ) -> None:
        self._root = Path(project_root).resolve()
        self._bus = bus
        self._scope = scope
        self._debounce_s = settings.debounce_s
        self._max_wait_s = settings.max_wait_s
        self._sub_projects = sub_projects or []
        self._root_name = root_name
        self._known_files: set[str] = set(known_files or [])

        # Pending changes keyed by watch-root-relative POSIX path (latest event wins)
        self._pending: dict[str, FileChanged] = {}
        self._flush_lock = asyncio.Lock()
        self._stop_event = asyncio.Event()

        # Timer tasks
        self._debounce_task: asyncio.Task[None] | None = None
        self._max_wait_task: asyncio.Task[None] | None = None

    async def run(self) -> None:
        """Watch for filesystem changes until :meth:`stop` is called."""
        logger.info("Watching {} (debounce={}s, max_wait={}s)", self._root, self._debounce_s, self._max_wait_s)

        async for changes in awatch(
            self._root,
            debounce=200,
            step=100,
            stop_event=self._stop_event,
            watch_filter=None,
        ):
            if self._stop_event.is_set():
                break
            await self._on_change(changes)

        # Drain any remaining pending changes on shutdown
        if self._pending:
            await self._flush()

        logger.info("Watcher stopped")

    def stop(self) -> None:
        """Signal the watcher to stop and cancel pending timers."""
        self._stop_event.set()
        if self._debounce_task is not None:
            self._debounce_task.cancel()
            self._debounce_task = None
        if self._max_wait_task is not None:
            self._max_wait_task.cancel()
            self._max_wait_task = None

    @property
    def stopped(self) -> bool:
        """True once :meth:`stop` has been called."""
        return self._stop_event.is_set()

    async def _on_change(self, changes: set[tuple[Change, str]]) -> None:
        """Filter and accumulate changes, then reset debounce timer."""
        accepted = 0
        for change, abs_path in changes:
            try:
                rel_path = Path(abs_path).relative_to(self._root).as_posix()
            except ValueError:
                continue

            if not self._scope.is_included(rel_path):
                # A directory rename/delete has no extension of its own, so the
                # include spec always rejects the bare directory path — expand
                # it against known/on-disk files instead of dropping it, or every
                # entity it contained is silently orphaned in the graph.
                accepted += await self._accept_uncovered_path(change, abs_path, rel_path)
                continue

            change_type = _CHANGE_TYPE_MAP.get(change)
            if change_type is None:
                continue

            if change_type == "deleted":
                self._known_files.discard(rel_path)
            else:
                self._known_files.add(rel_path)

            self._queue_change(rel_path, change_type)
            accepted += 1

        if accepted == 0:
            return

        logger.debug("{} change(s) accepted, {} pending total", accepted, len(self._pending))

        # Start max-wait ceiling on first change in this batch
        if self._max_wait_task is None:
            self._start_max_wait()

        # Reset debounce timer
        self._reset_debounce()

    async def _accept_uncovered_path(self, change: Change, abs_path: str, rel_path: str) -> int:
        """Expand a directory rename/delete that the include spec rejected outright.

        Returns the number of individual file changes queued.
        """
        if change == Change.deleted:
            affected = self._expand_directory_delete(rel_path)
            change_type = "deleted"
        elif Path(abs_path).is_dir():
            affected = await self._expand_directory_add(Path(abs_path))
            change_type = _CHANGE_TYPE_MAP.get(change, "created")
        else:
            logger.trace("SKIP {}: excluded by scope", rel_path)
            return 0

        if not affected:
            logger.trace("SKIP {}: excluded by scope", rel_path)
            return 0

        for f_rel in affected:
            self._queue_change(f_rel, change_type)
        return len(affected)

    def _expand_directory_delete(self, rel_path: str) -> list[str]:
        """Return (and forget) previously known files under a deleted/renamed-away directory."""
        prefix = rel_path + "/"
        affected = [f for f in self._known_files if f == rel_path or f.startswith(prefix)]
        for f in affected:
            self._known_files.discard(f)
        return affected

    async def _expand_directory_add(self, abs_dir: Path) -> list[str]:
        """Discover in-scope files under a newly appeared/renamed-in directory.

        Runs the ``os.walk`` scan in a worker thread (like ``FileScope.scan`` in
        daemon.py) so a large directory add doesn't block the event loop and stall
        the watcher's debounce/flush timers and event delivery for other files.
        """
        discovered = await asyncio.to_thread(self._walk_dir_for_included_files, abs_dir)
        self._known_files.update(discovered)
        return discovered

    def _walk_dir_for_included_files(self, abs_dir: Path) -> list[str]:
        """Blocking os.walk scan -- must only be called via ``asyncio.to_thread``."""
        discovered: list[str] = []
        for dirpath, _dirnames, filenames in os.walk(abs_dir):
            cur_rel = Path(dirpath).relative_to(self._root).as_posix()
            for fname in filenames:
                f_rel = f"{cur_rel}/{fname}" if cur_rel else fname
                if self._scope.is_included(f_rel):
                    discovered.append(f_rel)
        return discovered

    def _queue_change(self, rel_path: str, change_type: str) -> None:
        """Resolve the owning project and enqueue a FileChanged for *rel_path*."""
        # Resolve the owning project: monorepo sub-project or the watch root itself
        sub = classify_file_project(rel_path, self._sub_projects) if self._sub_projects else None
        if sub is not None:
            project_name = f"{self._root_name}/{sub.name}" if self._root_name else sub.name
            event_path = rel_path[len(sub.path) + 1 :]  # re-relativize to the sub-project root
            project_root = str(sub.root)
        else:
            project_name = self._root_name
            event_path = rel_path
            project_root = str(self._root)

        self._pending[rel_path] = FileChanged(
            path=event_path,
            change_type=change_type,
            project_name=project_name,
            project_root=project_root,
        )

    def _reset_debounce(self) -> None:
        """Cancel and restart the debounce timer."""
        if self._debounce_task is not None:
            self._debounce_task.cancel()
        self._debounce_task = asyncio.get_running_loop().create_task(self._debounce_wait())

    async def _debounce_wait(self) -> None:
        """Wait for the debounce period, then flush."""
        await asyncio.sleep(self._debounce_s)
        await self._flush()

    def _start_max_wait(self) -> None:
        """Start the max-wait ceiling timer (once per batch)."""
        self._max_wait_task = asyncio.get_running_loop().create_task(self._max_wait_wait())

    async def _max_wait_wait(self) -> None:
        """Wait for the max-wait period, then flush."""
        await asyncio.sleep(self._max_wait_s)
        await self._flush()

    async def _flush(self) -> None:
        """Drain pending changes and publish FileChanged events.

        Uses a lock to prevent concurrent flushes. Changes arriving
        during flush are accumulated for the next batch.
        """
        async with self._flush_lock:
            # Cancel the timer that didn't trigger this flush — never the task
            # currently executing it (a self-cancel would abort the publish
            # loop at its first await, after _pending was already cleared).
            # Null both refs so a change arriving mid-publish starts fresh timers.
            current = asyncio.current_task()
            if self._debounce_task is not None:
                if self._debounce_task is not current:
                    self._debounce_task.cancel()
                self._debounce_task = None
            if self._max_wait_task is not None:
                if self._max_wait_task is not current:
                    self._max_wait_task.cancel()
                self._max_wait_task = None

            if not self._pending:
                return

            # Snapshot and clear
            batch = dict(self._pending)
            self._pending.clear()

        # Publish outside the lock so new changes can accumulate
        logger.info("Flushing {} file change(s)", len(batch))
        items = list(batch.items())
        for i, (_, event) in enumerate(items):
            try:
                await self._bus.publish(Topic.FILE_CHANGED, event)
            except Exception:
                # Never drop the snapshotted batch: re-queue the unpublished
                # remainder so the next flush retries it (S7 durability spirit).
                remainder = items[i:]
                logger.exception("Publish failed — re-queueing {} unpublished change(s) for retry", len(remainder))
                for key, ev in remainder:
                    # A newer event that arrived mid-publish wins over the requeued one
                    self._pending.setdefault(key, ev)
                if not self._stop_event.is_set():
                    self._reset_debounce()  # schedule the retry flush
                return
