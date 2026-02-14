"""Filesystem watcher with hybrid debounce for real-time indexing.

Watches the project directory for file changes and publishes
:class:`~code_atlas.events.FileChanged` events to Valkey Streams.
Uses a hybrid debounce strategy: each change resets a short timer,
and the first change in a batch starts a max-wait ceiling.
Whichever fires first triggers a flush.
"""

from __future__ import annotations

import asyncio
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
    """

    def __init__(
        self,
        project_root: str | Path,
        bus: EventBus,
        scope: FileScope,
        settings: WatcherSettings,
        *,
        sub_projects: list[DetectedProject] | None = None,
    ) -> None:
        self._root = Path(project_root).resolve()
        self._bus = bus
        self._scope = scope
        self._debounce_s = settings.debounce_s
        self._max_wait_s = settings.max_wait_s
        self._sub_projects = sub_projects or []

        # Pending changes: {rel_path: (change_type, project_name)}  (latest wins)
        self._pending: dict[str, tuple[str, str]] = {}
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
            self._on_change(changes)

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

    def _on_change(self, changes: set[tuple[Change, str]]) -> None:
        """Filter and accumulate changes, then reset debounce timer."""
        accepted = 0
        for change, abs_path in changes:
            try:
                rel_path = Path(abs_path).relative_to(self._root).as_posix()
            except ValueError:
                continue

            if not self._scope.is_included(rel_path):
                logger.trace("SKIP {}: excluded by scope", rel_path)
                continue

            change_type = _CHANGE_TYPE_MAP.get(change)
            if change_type is None:
                continue

            # Classify sub-project for monorepo mode
            project_name = ""
            if self._sub_projects:
                project_name = classify_file_project(rel_path, self._sub_projects)

            self._pending[rel_path] = (change_type, project_name)
            accepted += 1

        if accepted == 0:
            return

        logger.debug("{} change(s) accepted, {} pending total", accepted, len(self._pending))

        # Start max-wait ceiling on first change in this batch
        if self._max_wait_task is None:
            self._start_max_wait()

        # Reset debounce timer
        self._reset_debounce()

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
            # Cancel timers (whoever didn't trigger the flush)
            if self._debounce_task is not None:
                self._debounce_task.cancel()
                self._debounce_task = None
            if self._max_wait_task is not None:
                self._max_wait_task.cancel()
                self._max_wait_task = None

            if not self._pending:
                return

            # Snapshot and clear
            batch = dict(self._pending)
            self._pending.clear()

        # Publish outside the lock so new changes can accumulate
        logger.info("Flushing {} file change(s)", len(batch))
        for rel_path, (change_type, project_name) in batch.items():
            event = FileChanged(path=rel_path, change_type=change_type, project_name=project_name)
            await self._bus.publish(Topic.FILE_CHANGED, event)
