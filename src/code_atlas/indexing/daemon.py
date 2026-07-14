"""Daemon manager — reusable watcher + pipeline lifecycle.

Encapsulates the EventBus, FileWatcher, EmbedClient, EmbedCache,
and AST/Embed consumers.  Used by both the CLI (``atlas watch``,
``atlas daemon start``) and the MCP server for auto-indexing.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from code_atlas.events import EventBus
from code_atlas.indexing.consumers import ASTConsumer, EmbedConsumer
from code_atlas.indexing.orchestrator import (
    FileScope,
    detect_sub_projects,
    gc_vanished_worktree_projects,
    index_monorepo,
    index_project,
    publish_project_changes,
)
from code_atlas.indexing.watcher import FileWatcher
from code_atlas.search.embeddings import EmbedCache, EmbedClient
from code_atlas.settings import derive_project_name

if TYPE_CHECKING:
    from code_atlas.graph.client import GraphClient
    from code_atlas.settings import AtlasSettings, ExtraVaultSettings


@dataclass
class DaemonManager:
    """Manages watcher + pipeline lifecycle.  Reusable across CLI and MCP."""

    _bus: EventBus | None = field(default=None, repr=False)
    _watcher: FileWatcher | None = field(default=None, repr=False)
    _vault_watchers: list[FileWatcher] = field(default_factory=list, repr=False)
    _consumers: list[ASTConsumer | EmbedConsumer] = field(default_factory=list, repr=False)
    _tasks: list[asyncio.Task[None]] = field(default_factory=list, repr=False)
    _cache: EmbedCache | None = field(default=None, repr=False)
    _embed: EmbedClient | None = field(default=None, repr=False)
    _crash_counts: dict[str, int] = field(default_factory=dict, repr=False)
    _last_crash: dict[str, str] = field(default_factory=dict, repr=False)

    def status(self) -> dict[str, Any]:
        """Task liveness + crash state, consumed by the ``pipeline`` health check."""
        return {
            "tasks_running": sum(1 for t in self._tasks if not t.done()),
            "tasks_total": len(self._tasks),
            "crash_counts": dict(self._crash_counts),
            "last_crash": dict(self._last_crash),
        }

    async def start(
        self,
        settings: AtlasSettings,
        graph: GraphClient,
        *,
        include_watcher: bool = True,
        catchup: bool = True,
    ) -> bool:
        """Try to start watcher + pipeline.

        Returns ``False`` if Valkey is unreachable (graceful degradation).
        The caller-owned *graph* is shared — **not** closed by this manager.

        Parameters
        ----------
        settings:
            Full atlas settings (redis, embeddings, watcher, scope, …).
        graph:
            An already-connected :class:`GraphClient`.
        include_watcher:
            If ``False``, only start the tier consumers (no filesystem watcher).
        catchup:
            If ``True``, run one delta index pass before consuming so edits
            made while the daemon was down are indexed. Failures are logged
            and non-fatal.
        """
        bus = EventBus(settings.redis, project_name=derive_project_name(settings.project_root))
        try:
            await bus.ping()
        except Exception:
            logger.warning("Valkey unavailable — running without auto-indexing")
            await bus.close()
            return False

        self._bus = bus

        try:
            removed = await gc_vanished_worktree_projects(graph)
            if removed:
                logger.info("GC: removed {} vanished worktree project(s): {}", len(removed), ", ".join(removed))
        except Exception:
            logger.exception("Worktree GC sweep failed — continuing startup")

        embed: EmbedClient | None = None
        cache: EmbedCache | None = None
        if settings.embeddings.enabled:
            embed = EmbedClient(settings.embeddings)
            self._embed = embed
            if settings.embeddings.cache_ttl_days > 0:
                cache = EmbedCache(settings.redis, settings.embeddings)
            self._cache = cache

        consumers: list[ASTConsumer | EmbedConsumer] = [
            ASTConsumer(bus, graph, settings, cooldown_s=settings.watcher.cooldown_s),
        ]
        if embed is not None:
            consumers.append(EmbedConsumer(bus, graph, embed, cache=cache))
        self._consumers = consumers

        if include_watcher:
            scope = FileScope(settings.project_root, settings)
            # FileScope only discovers nested .gitignore files as a side effect
            # of scan() (recorded while walking) — without it, the watcher
            # would filter live changes without ever loading them, indexing
            # files the full/delta indexer excludes. The returned file list
            # also seeds known-files tracking for directory rename/delete
            # detection (a bare directory path never matches the include spec).
            known_files = await asyncio.to_thread(scope.scan)
            subs = detect_sub_projects(settings.project_root, settings.monorepo)
            root_name = derive_project_name(settings.project_root)
            self._watcher = FileWatcher(
                settings.project_root,
                bus,
                scope,
                settings.watcher,
                sub_projects=subs or None,
                root_name=root_name,
                known_files=known_files,
            )

        # Spawn the watcher first so no change is missed while catch-up runs;
        # its events wait in the stream until the consumers start.
        if self._watcher is not None:
            self._tasks.append(asyncio.get_running_loop().create_task(self._run_watcher()))

        # Catch-up must finish BEFORE the daemon's consumers start: its inline
        # pipeline uses the same consumer names, so the two must never coexist
        # in this process.
        if catchup:
            await self._catchup(settings, graph, bus)

        for consumer in self._consumers:
            self._tasks.append(asyncio.get_running_loop().create_task(self._run_consumer(consumer)))

        # Extra vaults (global vault, harness memory dir) live outside project_root,
        # so each gets its own FileWatcher instance rather than riding the main
        # project's one (FileWatcher itself is single-root — see watcher.py). This
        # is independent of include_watcher (which only gates the main project's
        # watcher) — vaults have always indexed regardless of that flag.
        for vault in settings.knowledge.extra_vaults:
            await self._start_vault(vault, settings, graph, bus, catchup=catchup)

        return True

    async def _start_vault(
        self,
        vault: ExtraVaultSettings,
        settings: AtlasSettings,
        graph: GraphClient,
        bus: EventBus,
        *,
        catchup: bool,
    ) -> None:
        """Scan, optionally catch up, and spawn a live FileWatcher for one extra vault.

        Publishing reuses the persistent consumers already started in start()
        (they accept any project_name/project_root per event, same as monorepo
        sub-projects), so no per-vault consumer pair is created here.
        """
        vault_root = Path(vault.path).expanduser().resolve()
        if not vault_root.is_dir():
            logger.warning("Extra vault '{}' not found at {} — skipping", vault.project_name, vault_root)
            return

        scope = FileScope(vault_root, settings)
        known_files = await asyncio.to_thread(scope.scan)

        if catchup:
            await self._catchup_vault(vault.project_name, vault_root, known_files, settings, graph)

        vault_watcher = FileWatcher(
            vault_root,
            bus,
            scope,
            settings.watcher,
            root_name=vault.project_name,
            known_files=known_files,
        )
        self._vault_watchers.append(vault_watcher)
        self._tasks.append(
            asyncio.get_running_loop().create_task(self._run_vault_watcher(vault.project_name, vault_watcher))
        )

    async def _catchup(self, settings: AtlasSettings, graph: GraphClient, bus: EventBus) -> None:
        """One delta index pass so changes made while the daemon was down get indexed."""
        try:
            if detect_sub_projects(settings.project_root, settings.monorepo):
                await index_monorepo(settings, graph, bus)
            else:
                await index_project(settings, graph, bus)
        except Exception:
            logger.exception("Startup catch-up index failed — continuing with live events only")

    async def wait(self) -> None:
        """Block until all background tasks finish (or are cancelled)."""
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    async def stop(self) -> None:
        """Graceful shutdown: stop watcher, consumers, close connections."""
        if self._watcher is not None:
            self._watcher.stop()

        for vault_watcher in self._vault_watchers:
            vault_watcher.stop()

        for consumer in self._consumers:
            consumer.stop()

        # Let tasks observe the stop flags first — the watcher drains its
        # pending changes and consumers finish their current batch — then
        # cancel whatever is still running.
        if self._tasks:
            _done, still_pending = await asyncio.wait(self._tasks, timeout=10.0)
            for task in still_pending:
                task.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        if self._cache is not None:
            await self._cache.close()

        if self._bus is not None:
            await self._bus.close()

        logger.debug("DaemonManager stopped")

    async def _run_watcher(self) -> None:
        """Run the file watcher under supervision: crash → log + backoff restart."""
        watcher = self._watcher
        if watcher is None:  # pragma: no cover — spawned only when a watcher was built
            return
        backoff = 1.0
        while not watcher.stopped:
            started = asyncio.get_running_loop().time()
            try:
                await watcher.run()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._crash_counts["watcher"] = self._crash_counts.get("watcher", 0) + 1
                self._last_crash["watcher"] = repr(exc)
                logger.exception("File watcher crashed — restarting in {:.0f}s", backoff)
                if asyncio.get_running_loop().time() - started > 60.0:
                    backoff = 1.0  # healthy for a while before this crash — reset
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
            else:
                return  # clean exit via stop()

    async def _run_consumer(self, consumer: ASTConsumer | EmbedConsumer) -> None:
        """Run a consumer under supervision: crash → log + backoff restart.

        ``run()`` re-runs ``ensure_group()`` at its top, so a Valkey restart
        that lost the consumer group (NOGROUP) heals on the first restart.
        """
        backoff = 1.0
        while not consumer.stopped:
            started = asyncio.get_running_loop().time()
            try:
                await consumer.run()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._crash_counts[consumer.consumer_name] = self._crash_counts.get(consumer.consumer_name, 0) + 1
                self._last_crash[consumer.consumer_name] = repr(exc)
                logger.exception("Consumer {} crashed — restarting in {:.0f}s", consumer.consumer_name, backoff)
                if asyncio.get_running_loop().time() - started > 60.0:
                    backoff = 1.0  # healthy for a while before this crash — reset
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
            else:
                return  # clean exit via stop()

    async def _catchup_vault(
        self,
        project_name: str,
        vault_root: Path,
        files: list[str],
        settings: AtlasSettings,
        graph: GraphClient,
    ) -> None:
        """One-time scan+publish for an extra vault, mirroring the main project's ``_catchup``.

        Extra vaults are never git-tracked as far as this daemon is concerned (even
        if the directory happens to be a git repo, no git_hash is stored for them),
        so this always runs in "full" delta mode; the AST consumer's per-file
        content-hash gate is what keeps unchanged files cheap. Unlike ``_catchup``,
        this has no consumer-startup ordering constraint — ``publish_project_changes``
        only publishes events for the already-running persistent consumers to pick
        up, it never starts its own.
        """
        bus = self._bus
        if bus is None:  # pragma: no cover — only called after start() sets self._bus
            return
        try:
            result = await publish_project_changes(settings, graph, bus, project_name, vault_root, files)
            entity_count = await graph.count_entities(project_name)
            await graph.update_project_metadata(
                project_name,
                last_indexed_at=time.time(),
                file_count=result.files_scanned,
                entity_count=entity_count,
                index_mode=result.mode,
            )
            if result.files_published:
                logger.info("Vault '{}': {} file(s) published ({})", project_name, result.files_published, result.mode)
        except Exception:
            logger.exception("Startup catch-up failed for vault '{}' — continuing with live events only", project_name)

    async def _run_vault_watcher(self, label: str, watcher: FileWatcher) -> None:
        """Run one extra vault's FileWatcher under supervision: crash → log + backoff restart."""
        crash_key = f"vault:{label}"
        backoff = 1.0
        while not watcher.stopped:
            started = asyncio.get_running_loop().time()
            try:
                await watcher.run()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._crash_counts[crash_key] = self._crash_counts.get(crash_key, 0) + 1
                self._last_crash[crash_key] = repr(exc)
                logger.exception("Vault watcher '{}' crashed — restarting in {:.0f}s", label, backoff)
                if asyncio.get_running_loop().time() - started > 60.0:
                    backoff = 1.0  # healthy for a while before this crash — reset
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
            else:
                return  # clean exit via stop()
