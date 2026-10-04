"""Daemon manager — reusable watcher + pipeline lifecycle.

Encapsulates the EventBus, FileWatcher, EmbedClient,
and AST/Embed consumers.  Used by both the CLI (``atlas watch``,
``atlas daemon start``) and the MCP server for auto-indexing.
"""

from __future__ import annotations

import asyncio
import contextlib
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from code_atlas.events import INDEXER_LEASE_TTL_MS, EventBus, Topic, new_lease_owner
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
from code_atlas.search.embeddings import EmbedClient, EmbedPolicy
from code_atlas.settings import derive_project_name
from code_atlas.telemetry import is_enabled as telemetry_enabled
from code_atlas.telemetry import set_backlog

if TYPE_CHECKING:
    from code_atlas.graph.client import GraphClient
    from code_atlas.search.ratelimit import Limiter
    from code_atlas.settings import AtlasSettings, ExtraVaultSettings


# Slow enough to be free, fast enough to see a stall inside a minute.
_BACKLOG_SAMPLE_S = 15.0

# Restart backoff for the three supervised loops (watcher, consumer, vault watcher):
# first pause after a crash, doubling to the cap, reset once a run has been healthy for
# a while. Named rather than left as a local `backoff = 1.0` in each loop so tests can
# shrink it -- as locals they were unpatchable, and the three crash-restart tests each
# paid the full second.
_RESTART_BACKOFF_S = 1.0
_RESTART_BACKOFF_MAX_S = 60.0
# How long a run must survive before its next crash is treated as unrelated.
_RESTART_HEALTHY_S = 60.0

# Exactly one process per checkout indexes it: whoever holds the indexer lease. Every other
# session stands by, query-only, and re-tries the lease on this interval, jittered so
# sessions an agent client spawned together do not retry in lockstep. A holder that is
# killed stops renewing, so its lease expires within INDEXER_LEASE_TTL_MS and the next
# standby poll after that takes over.
_STANDBY_POLL_S = 15.0
_STANDBY_POLL_JITTER = 0.5

# How often the holder renews its lease and checks whether a one-shot `atlas index` has
# asked it to yield. Well inside the TTL, and short because whoever asked is waiting.
_HOLDER_TICK_S = 5.0


@dataclass
class DaemonManager:
    """Manages watcher + pipeline lifecycle.  Reusable across CLI and MCP."""

    _bus: EventBus | None = field(default=None, repr=False)
    _watcher: FileWatcher | None = field(default=None, repr=False)
    _vault_watchers: list[FileWatcher] = field(default_factory=list, repr=False)
    _consumers: list[ASTConsumer | EmbedConsumer] = field(default_factory=list, repr=False)
    _tasks: list[asyncio.Task[None]] = field(default_factory=list, repr=False)
    _embed: EmbedClient | None = field(default=None, repr=False)
    _crash_counts: dict[str, int] = field(default_factory=dict, repr=False)
    _last_crash: dict[str, str] = field(default_factory=dict, repr=False)
    #: Why this manager is not indexing, if it deliberately isn't. Without it the
    #: pipeline health check reports "0 task(s) running" as OK, which reads identically
    #: to a pipeline that died on startup.
    disabled_reason: str = ""
    # What start() was given, kept so a standby session can build the same stack when it
    # takes the lease over.
    _settings: AtlasSettings | None = field(default=None, repr=False)
    _graph: GraphClient | None = field(default=None, repr=False)
    _limiter: Limiter | None = field(default=None, repr=False)
    _include_watcher: bool = field(default=True, repr=False)
    _catchup_enabled: bool = field(default=True, repr=False)
    #: This process's identity on the indexer lease.
    _owner: str | None = field(default=None, repr=False)
    #: The caller took the lease and renews it (`atlas index --watch`), so this manager
    #: neither elects, yields nor releases.
    _lease_is_callers: bool = field(default=False, repr=False)
    _active: bool = field(default=False, repr=False)
    _elector: asyncio.Task[None] | None = field(default=None, repr=False)
    _activation: asyncio.Task[None] | None = field(default=None, repr=False)

    @property
    def standby(self) -> bool:
        """True while another process holds this checkout's indexer lease and this one waits."""
        return self._elector is not None and not self._active

    @property
    def bus(self) -> EventBus | None:
        """The event bus this manager is actually running on (``None`` before ``start()``).

        Declared type stays ``EventBus`` per the same structurally-compatible-fallback
        convention as ``start()``'s ``bus`` local — it may hold a ``SqliteEventBus`` at
        runtime. Lets the health-check honesty fix report the real active queue backend
        instead of probing a disconnected one.
        """
        return self._bus

    def status(self) -> dict[str, Any]:
        """Task liveness + crash state, consumed by the ``pipeline`` health check."""
        return {
            "tasks_running": sum(1 for t in self._tasks if not t.done()),
            "tasks_total": len(self._tasks),
            "crash_counts": dict(self._crash_counts),
            "last_crash": dict(self._last_crash),
            "disabled_reason": self.disabled_reason,
            "standby": self.standby,
        }

    async def start(
        self,
        settings: AtlasSettings,
        graph: GraphClient,
        bus: EventBus,
        *,
        limiter: Limiter,
        include_watcher: bool = True,
        catchup: bool = True,
        first_index_ready: asyncio.Event | None = None,
        lease_owner: str | None = None,
    ) -> bool:
        """Start watcher + pipeline if this process wins the checkout's indexer lease.

        Returns ``False`` if the queue backend is unreachable (graceful degradation).
        Both *graph* and *bus* are caller-owned and shared — **not** closed by this
        manager. It used to build its own bus while being handed a graph, which left
        ownership split down the middle of one object: the caller closed half of what
        this used and the manager closed the other, and neither could see the whole.

        Every MCP session starts with indexing on, and nobody can pass ``--no-index`` to
        each one, so which process indexes a checkout is decided here: the one that holds
        the indexer lease. The winner keeps it for as long as it runs, renewing it, and
        steps down when a one-shot ``atlas index`` asks it to yield. A loser runs nothing
        -- no scan, no watcher, no consumers, no catch-up -- and re-tries the lease every
        ``_STANDBY_POLL_S``, taking over when the holder exits or dies. Before this, each
        session ran its own watcher over the same tree, and three of them sat at 99% CPU
        re-walking the same excluded ``.venv``.

        Parameters
        ----------
        settings:
            Full atlas settings (redis, embeddings, watcher, scope, …).
        graph:
            An already-connected :class:`GraphClient`.
        limiter:
            The process's embedding rate limiter (``Backends.limiter``). Caller-owned like
            the bus. Required: no pacing is ``unpaced()``, asked for by name.
        include_watcher:
            If ``False``, only start the tier consumers (no filesystem watcher).
        catchup:
            If ``True``, run one delta index pass before consuming so edits
            made while no process was indexing are indexed — at startup and again on
            every takeover. Failures are logged and non-fatal.
        first_index_ready:
            Set once startup catch-up has run (success or swallowed failure)
            or is skipped, and at once when this process stands by — lets a caller (MCP's
            first-index readiness gate) block tool calls against a genuinely fresh
            backend without hanging forever. ``None`` (default) means no one is waiting.
        lease_owner:
            The indexer lease the caller already holds and renews, which is what
            ``atlas index --watch`` does. The stack then starts under it unconditionally:
            no election, no yielding, and the lease is the caller's to release. Consumers
            are told it too, or they would stand down for their own caller's lease.
        """
        try:
            await bus.ping()
        except Exception:
            # Not closed here: the caller opened it and will close it. Reporting the
            # degradation is this manager's job; disposing of someone else's connection
            # is not.
            logger.warning("Queue backend unavailable — running without auto-indexing")
            return False

        self._bus = bus
        self._settings, self._graph, self._limiter = settings, graph, limiter
        self._include_watcher, self._catchup_enabled = include_watcher, catchup

        if lease_owner is not None:
            self._owner, self._lease_is_callers, self._active = lease_owner, True, True
            await self._activate(first_index_ready)
            return True

        self._owner = new_lease_owner()
        won = await self._try_acquire()
        self._active = won
        self._elector = asyncio.get_running_loop().create_task(self._elect())
        if not won:
            await self._enter_standby()
            if first_index_ready is not None:
                first_index_ready.set()
            return True

        activation = self._spawn_activation(first_index_ready)
        try:
            await asyncio.wait({activation})
        except asyncio.CancelledError:
            activation.cancel()
            raise
        finally:
            if first_index_ready is not None:
                first_index_ready.set()
        if not activation.cancelled() and activation.exception() is not None:
            # Same contract as before the election: a stack that fails to start fails
            # start(), and the lease is not left held by a process that indexes nothing.
            await self.stop()
            activation.result()
        return True

    async def _activate(self, first_index_ready: asyncio.Event | None) -> None:
        """Build and start the watcher + pipeline, under a lease this process holds."""
        settings, graph, bus, limiter = self._settings, self._graph, self._bus, self._limiter
        assert settings is not None
        assert graph is not None
        assert bus is not None
        assert limiter is not None

        try:
            removed = await gc_vanished_worktree_projects(graph)
            if removed:
                logger.info("GC: removed {} vanished worktree project(s): {}", len(removed), ", ".join(removed))
        except Exception:
            logger.exception("Worktree GC sweep failed — continuing startup")

        embed: EmbedClient | None = None
        if settings.embeddings.enabled:
            embed = EmbedClient(settings.embeddings, limiter=limiter)
            self._embed = embed

        consumers: list[ASTConsumer | EmbedConsumer] = [
            ASTConsumer(
                bus,
                graph,
                settings,
                cooldown_s=settings.watcher.cooldown_s,
                defer_to_lease=True,
                lease_owner=self._owner,
            ),
        ]
        if embed is not None:
            consumers.append(
                EmbedConsumer(
                    bus,
                    graph,
                    embed,
                    defer_to_lease=True,
                    lease_owner=self._owner,
                    embedding_policy=EmbedPolicy.from_settings(settings.embeddings),
                )
            )
        self._consumers = consumers

        if self._include_watcher:
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
        if self._catchup_enabled:
            await self._catchup(settings, graph, bus, limiter, first_index_ready)
        elif first_index_ready is not None:
            first_index_ready.set()

        for consumer in self._consumers:
            self._tasks.append(asyncio.get_running_loop().create_task(self._run_consumer(consumer)))

        self._start_backlog_sampler()

        # Extra vaults (global vault, harness memory dir) live outside project_root,
        # so each gets its own FileWatcher instance rather than riding the main
        # project's one (FileWatcher itself is single-root — see watcher.py). This
        # is independent of include_watcher (which only gates the main project's
        # watcher) — vaults have always indexed regardless of that flag.
        for vault in settings.knowledge.extra_vaults:
            try:
                await self._start_vault(vault, settings, graph, bus, catchup=self._catchup_enabled)
            except Exception:
                logger.exception("Failed to start extra vault '{}' — continuing without it", vault.project_name)

    def _spawn_activation(self, first_index_ready: asyncio.Event | None) -> asyncio.Task[None]:
        """Start the stack as a task, so the elector can cancel it on a yield mid catch-up."""
        self._active = True
        self.disabled_reason = ""
        self._activation = asyncio.get_running_loop().create_task(self._activate(first_index_ready))
        return self._activation

    async def _try_acquire(self) -> bool:
        """Take the lease, unless a one-shot index is waiting for exactly that."""
        bus, owner = self._bus, self._owner
        assert bus is not None
        assert owner is not None
        if await bus.read_indexer_yield():
            return False
        return await bus.acquire_indexer_lease(owner, INDEXER_LEASE_TTL_MS)

    async def _elect(self) -> None:
        """Keep the lease while indexing; take it over while standing by. Runs until stop()."""
        while True:
            try:
                if self._active:
                    await asyncio.sleep(_HOLDER_TICK_S)
                    reason = await self._reason_to_step_down()
                    if reason:
                        logger.info("Stepping down as this checkout's indexer: {}", reason)
                        await self._step_down()
                else:
                    jitter = random.uniform(1 - _STANDBY_POLL_JITTER, 1 + _STANDBY_POLL_JITTER)
                    await asyncio.sleep(_STANDBY_POLL_S * jitter)
                    if await self._try_acquire():
                        assert self._settings is not None
                        logger.info("Took over the indexer lease — now indexing {}", self._settings.project_root)
                        self._spawn_activation(None)
                    else:
                        await self._enter_standby(announce=False)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Indexer lease check failed — retrying")

    async def _reason_to_step_down(self) -> str:
        """Renew the lease and say why this process should stop indexing, or ``""``."""
        bus, owner = self._bus, self._owner
        assert bus is not None
        assert owner is not None
        activation = self._activation
        if activation is not None and activation.done() and not activation.cancelled():
            exc = activation.exception()
            if exc is not None:
                logger.opt(exception=exc).error("Indexer takeover failed to start the watcher and pipeline")
                return "its watcher and pipeline failed to start"
        requester = await bus.read_indexer_yield()
        if requester:
            return f"{requester} asked it to yield"
        if await bus.renew_indexer_lease(owner, INDEXER_LEASE_TTL_MS):
            return ""
        if await bus.acquire_indexer_lease(owner, INDEXER_LEASE_TTL_MS):
            logger.warning("The indexer lease had vanished (queue backend restarted?) — re-acquired it")
            return ""
        holder = await bus.read_indexer_lease()
        return f"the lease passed to {holder or 'another process'}"

    async def _step_down(self) -> None:
        """Stop the stack, then release the lease — in that order, so no two indexers overlap."""
        self._active = False
        activation, self._activation = self._activation, None
        if activation is not None and not activation.done():
            activation.cancel()
            await asyncio.gather(activation, return_exceptions=True)
        await self._stop_stack()
        if self._bus is not None and self._owner is not None:
            await self._bus.release_indexer_lease(self._owner)
        await self._enter_standby()

    async def _enter_standby(self, *, announce: bool = True) -> None:
        """Record who indexes this checkout instead, for the health check and the log."""
        assert self._bus is not None
        assert self._settings is not None
        try:
            holder = await self._bus.read_indexer_lease()
        except Exception:
            holder = None
        self.disabled_reason = f"standby — {holder or 'another process'} holds the indexer lease"
        if announce:
            logger.info(
                "Standing by: {} indexes {} — this session is query-only until that lease frees",
                holder or "another process",
                self._settings.project_root,
            )

    async def _start_vault(
        self,
        vault: ExtraVaultSettings,
        settings: AtlasSettings,
        graph: GraphClient,
        bus: EventBus,
        *,
        catchup: bool,
    ) -> None:
        """Scan, spawn a live FileWatcher, then optionally catch up for one extra vault.

        Publishing reuses the persistent consumers already started in start()
        (they accept any project_name/project_root per event, same as monorepo
        sub-projects), so no per-vault consumer pair is created here.

        The watcher is spawned before catch-up runs — same ordering (and
        rationale) as the main project above: no change is missed while
        catch-up runs; its events wait in the stream until the consumers
        start.
        """
        vault_root = Path(vault.path).expanduser().resolve()
        if not vault_root.is_dir():
            logger.warning("Extra vault '{}' not found at {} — skipping", vault.project_name, vault_root)
            return

        scope = FileScope(vault_root, settings)
        known_files = await asyncio.to_thread(scope.scan)

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

        if catchup:
            await self._catchup_vault(vault.project_name, vault_root, known_files, settings, graph)

    async def _catchup(
        self,
        settings: AtlasSettings,
        graph: GraphClient,
        bus: EventBus,
        limiter: Limiter,
        first_index_ready: asyncio.Event | None = None,
    ) -> None:
        """One delta index pass so changes made while nothing was indexing get indexed.

        Runs under the indexer lease this process already holds for the whole session,
        so it takes no lease of its own. Without one, every MCP session's catch-up raced
        every other session's (and any concurrent CLI run's) directly against Memgraph --
        exactly the "two processes writing the same nodes" scenario the lease exists to
        prevent (see ``hold_indexer_lease``'s docstring), which surfaced as Memgraph MVCC
        conflicts and a 60s write timeout that killed an indexing run outright.

        *first_index_ready* (if given) is set in the ``finally`` block regardless of
        outcome — both success and the swallowed-failure path below must unblock a
        caller waiting on it (MCP's first-index readiness gate), never hang forever.
        """
        try:
            if detect_sub_projects(settings.project_root, settings.monorepo):
                await index_monorepo(
                    settings, graph, bus, drain_timeout_s=settings.index.drain_timeout_s, limiter=limiter
                )
            else:
                await index_project(
                    settings, graph, bus, drain_timeout_s=settings.index.drain_timeout_s, limiter=limiter
                )
        except Exception:
            logger.exception("Startup catch-up index failed — continuing with live events only")
        finally:
            if first_index_ready is not None:
                first_index_ready.set()

    async def wait(self) -> None:
        """Block until stop(): through standby and takeovers, or until the stack's tasks end."""
        if self._elector is not None:
            await asyncio.gather(self._elector, return_exceptions=True)
        elif self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    def _start_backlog_sampler(self) -> None:
        """Spawn the backlog sampler, but only when something is collecting.

        It is an XINFO round-trip on a timer; with no exporter attached it would be
        pure overhead for the life of the daemon.
        """
        if telemetry_enabled():
            self._tasks.append(asyncio.get_running_loop().create_task(self._sample_backlog()))

    async def _sample_backlog(self) -> None:
        """Feed the backlog gauge on a timer.

        Backlog is the one pipeline number that falls as well as rises, so it has to be
        a gauge, and observable-gauge callbacks are synchronous while the depth comes
        from an async XINFO. Hence a sampler that pushes into a cache the callback
        reads, rather than a callback that queries.
        """
        while True:
            await asyncio.sleep(_BACKLOG_SAMPLE_S)
            try:
                counts = await self.pending_event_counts()
            except Exception:
                logger.debug("Backlog sample failed — will retry")
                continue
            for topic, pending in (counts or {}).items():
                set_backlog(topic, pending)

    async def pending_event_counts(self) -> dict[str, int] | None:
        """Current backlog size per topic (file_changed, embed_dirty), or ``None`` if not running.

        A one-shot read of the same pending+lag figures ``_wait_for_drain`` polls during
        catchup (``pending + lag``, or just ``pending`` when ``lag`` is unknown) — lets a
        caller surface "how much indexing work is left" without duplicating that polling loop.
        """
        if self._bus is None:
            return None
        queries: list[tuple[Topic, str]] = [(Topic.FILE_CHANGED, "ast")]
        if self._embed is not None:
            queries.append((Topic.EMBED_DIRTY, "embed"))
        infos = await self._bus.stream_group_info_multi(queries)
        counts: dict[str, int] = {}
        for (topic, _group), info in zip(queries, infos, strict=True):
            pending, lag = info["pending"], info["lag"]
            counts[topic.value] = pending if lag is None else pending + lag
        return counts

    async def stop(self) -> None:
        """Graceful shutdown: stop the elector and the stack, then release the lease."""
        elector, self._elector = self._elector, None
        if elector is not None:
            elector.cancel()
            await asyncio.gather(elector, return_exceptions=True)
        activation, self._activation = self._activation, None
        if activation is not None and not activation.done():
            activation.cancel()
            await asyncio.gather(activation, return_exceptions=True)

        await self._stop_stack()

        if self._active and not self._lease_is_callers and self._bus is not None and self._owner is not None:
            # Released at once rather than left to expire, so a standby session takes
            # over on its next poll instead of a minute later. The bus may already be
            # gone on a shutdown, and the TTL covers that.
            with contextlib.suppress(Exception):
                await self._bus.release_indexer_lease(self._owner)
        self._active = False

        # The bus is deliberately not closed: it is the caller's, and closing it here
        # once meant a restart_daemon() left the MCP server holding a dead connection.
        logger.debug("DaemonManager stopped")

    async def _stop_stack(self) -> None:
        """Stop watcher, vault watchers and consumers, and forget them, so a takeover rebuilds."""
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
        self._watcher = None
        self._vault_watchers = []
        self._consumers = []

        # Cleared so a restart builds a fresh embed client. Nothing to close: it holds no
        # connection, and the limiter it paced through is the caller's, like the bus.
        self._embed = None

    async def _run_watcher(self) -> None:
        """Run the file watcher under supervision: crash → log + backoff restart."""
        watcher = self._watcher
        if watcher is None:  # pragma: no cover — spawned only when a watcher was built
            return
        backoff = _RESTART_BACKOFF_S
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
                if asyncio.get_running_loop().time() - started > _RESTART_HEALTHY_S:
                    backoff = _RESTART_BACKOFF_S  # healthy for a while before this crash — reset
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, _RESTART_BACKOFF_MAX_S)
            else:
                return  # clean exit via stop()

    async def _run_consumer(self, consumer: ASTConsumer | EmbedConsumer) -> None:
        """Run a consumer under supervision: crash → log + backoff restart.

        ``run()`` re-runs ``ensure_group()`` at its top, so a lost consumer group heals on the
        first restart: every bus raises ``ConsumerGroupMissingError`` for it (Valkey after a
        restart that lost it, any bus after ``atlas project rm``).
        """
        backoff = _RESTART_BACKOFF_S
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
                if asyncio.get_running_loop().time() - started > _RESTART_HEALTHY_S:
                    backoff = _RESTART_BACKOFF_S  # healthy for a while before this crash — reset
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, _RESTART_BACKOFF_MAX_S)
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
        backoff = _RESTART_BACKOFF_S
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
                if asyncio.get_running_loop().time() - started > _RESTART_HEALTHY_S:
                    backoff = _RESTART_BACKOFF_S  # healthy for a while before this crash — reset
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, _RESTART_BACKOFF_MAX_S)
            else:
                return  # clean exit via stop()
