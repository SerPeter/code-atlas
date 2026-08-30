"""Two-stage consumer pipeline for event-driven indexing.

    FileChanged → AST stage (hash gate + AST parse + diff)
                → significance gate → EmbedDirty → Embed stage (embeddings)

Each stage uses batch-pull with configurable time/count policy and
deduplicates within its batch window.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import socket
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from loguru import logger

from code_atlas.events import (
    EmbedDirty,
    EntityRef,
    Event,
    EventBus,
    FileChanged,
    Significance,
    Topic,
    decode_event,
)
from code_atlas.graph.client import _CONFIG_REF_REL_TYPES, EmbedChunkWrite
from code_atlas.parsing.ast import ParsedEntity, ParsedFile, ParsedRelationship, parse_file
from code_atlas.parsing.detectors import DetectorResult, get_enabled_detectors, run_detectors
from code_atlas.schema import NodeLabel, RelType
from code_atlas.search.embeddings import _CODE_ENTITY_LABELS, build_embed_text, hash_text
from code_atlas.settings import derive_project_name
from code_atlas.telemetry import get_metrics, get_tracer, timed_phase

if TYPE_CHECKING:
    from collections.abc import Callable

    from code_atlas.chunking import SplitResult
    from code_atlas.graph.client import GraphClient
    from code_atlas.search.embeddings import EmbedClient
    from code_atlas.settings import AtlasSettings

_tracer = get_tracer(__name__)

_COLLAPSE_BLANK_RE = re.compile(rb"\n{3,}")


def _retry_key(rel: ParsedRelationship) -> tuple[str, str, str, str, str]:
    """Identity of a call/import/type site, for deduplicating ``ASTConsumer._retry_rels``.

    Everything ``_resolve_one_call`` reads: re-parsing an unchanged file must
    produce the same key so the buffer stays the size of the codebase rather
    than growing once per re-index.
    """
    return (
        str(rel.rel_type),
        rel.from_qualified_name,
        rel.to_name,
        str(rel.properties.get("receiver") or ""),
        str(rel.properties.get("receiver_type") or ""),
    )


def _compute_file_hash(source: bytes, *, strip_whitespace: bool = True) -> str:
    """Compute a short SHA-256 hash of file contents.

    When *strip_whitespace* is True: strip trailing whitespace per line,
    collapse consecutive blank lines, then hash.  This makes the gate
    ignore formatting-only changes (e.g. ``ruff format``) while preserving
    leading indentation for indentation-sensitive languages.
    """
    if strip_whitespace:
        lines = [line.rstrip() for line in source.splitlines()]
        normalized = b"\n".join(lines)
        normalized = _COLLAPSE_BLANK_RE.sub(b"\n\n", normalized)
        return hashlib.sha256(normalized).hexdigest()[:16]
    return hashlib.sha256(source).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Batch policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BatchPolicy:
    """Controls when a consumer flushes its accumulated batch."""

    time_window_s: float  # Max seconds to accumulate before flush
    max_batch_size: int  # Max items before flush (whichever hits first)
    block_ms: int | None = None  # Override for XREADGROUP block; None = derive from time_window_s


# ---------------------------------------------------------------------------
# Abstract tier consumer
# ---------------------------------------------------------------------------

# Batches a message may fail before it is parked (ACKed + dropped) on the next PEL reclaim.
_MAX_BATCH_FAILURES = 5

# Concurrent embedding-write transactions per embed consumer. See EmbedConsumer
# __init__ for the measurements and for why this is bounded rather than unbounded.
_EMBED_WRITE_CONCURRENCY = 2

# Adaptive resolution cadence (reindex only) -- see _flush_deferred_resolution.
#
# A flush costs O(project size): build_resolution_lookup alone measured 405ms on a
# 592-module project and 812ms on one of 581 with more classes, and it grows with the
# project while the number of flushes grows with it too. A fixed every-5-batches
# cadence therefore spends a growing share of a reindex on resolution.
#
# So the next flush waits at least this multiple of the last one's duration, which
# caps resolution at roughly 1/(1+ratio) of wall time -- 20% at 4. Small projects are
# unaffected: a 50ms flush yields a 200ms gap, far below the batch cadence that
# triggers it anyway, so this only engages once flushes get expensive.
_RESOLVE_DUTY_RATIO = 4.0
_RESOLVE_MAX_GAP_S = 300.0

# Safety valve, not a tuned number: deferring a flush accumulates unresolved
# relationships in memory, so a flush happens regardless once the buffer reaches
# this many. Set generously because the real ceiling should come from a measured
# reindex rather than a guess; the count is logged at each flush to make that
# measurement possible.
_RESOLVE_PENDING_CEILING = 500_000


def _next_resolve_gap(duration_s: float) -> float:
    """Minimum spacing before the next resolution flush, given what the last cost."""
    return min(_RESOLVE_MAX_GAP_S, max(0.0, duration_s) * _RESOLVE_DUTY_RATIO)


# How long an entry must sit untouched before another consumer may adopt it. Long enough
# that a live consumer's own in-flight batch never qualifies, short enough that a crash
# does not strand work for a whole session.
_ABANDONED_MIN_IDLE_MS = 120_000

# Which pipeline stage each topic belongs to, for the `stage` metric label. The topic
# name ("file-changed") describes the event; the stage name ("ast") describes the work,
# and it is the work you compare when asking which stage the time went into.
_STAGE_BY_TOPIC = {Topic.FILE_CHANGED: "ast", Topic.EMBED_DIRTY: "embed"}

# The AST consumer's default batch window. Also sets the blocking read: block_ms is
# max(100, time_window_s * 1000 // 2), so 3.0s here means every XREADGROUP blocks 1.5s.
# Named so the integration suite can shrink it. Do NOT shrink it to 0 -- `is_reindex`
# below tests `time_window_s == 0` and zero silently switches the resolution cadence.
_AST_WINDOW_S = 3.0

# How often a paused consumer re-checks whether the foreign lease has been released.
_LEASE_POLL_S = 1.0

# How often to sweep for work abandoned by a dead process, once this consumer's own PEL
# is drained. Without a periodic sweep the adopt path only ever runs during a consumer's
# initial drain, so a crashed indexer's messages sit stranded until something restarts —
# observed live: 96 file and 1037 embed messages held by an exited process while a
# healthy consumer polled beside them and never looked.
_RECLAIM_SWEEP_S = 30.0

# How long a consumer registration must go untouched before it is treated as belonging to
# a process that is gone. This is not a poll interval: the longest legitimate silence for a
# LIVE consumer is a long CLI index that it is standing down for (see _defer_to_foreign_lease),
# so the threshold is in hours. Pruning early would deregister a consumer that is merely
# waiting its turn.
_STALE_CONSUMER_IDLE_MS = 3_600_000


def _process_tag() -> str:
    """Short identity for this process, unique across hosts and PIDs.

    Consumer names used to be the constants "ast-0" and "embed", so every process that
    built a consumer claimed the same identity in the same group. Measured consequence
    on live Valkey: `XREADGROUP ... 0` returns the OTHER process's in-flight messages,
    and either process's XACK of the other's message succeeds and removes it from the
    PEL — silently deleting the peer's crash-recovery net. Redis identifies a consumer
    solely by this string, so making it unique is what separates the two PELs.
    """
    return f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}"


def _stream_id_key(msg_id: bytes) -> tuple[int, int]:
    """Numeric sort key for a Redis Stream id (``b"<ms>-<seq>"``)."""
    ms, _, seq = msg_id.partition(b"-")
    return int(ms), int(seq or 0)


class TierConsumer(ABC):
    """Base class for tiered pipeline consumers.

    Implements the batch-pull loop: XREADGROUP → accumulate → dedup →
    flush when policy triggers → ACK. Subclasses implement
    ``process_batch`` for tier-specific work.
    """

    def __init__(
        self,
        bus: EventBus,
        input_topic: Topic,
        group: str,
        consumer_name: str,
        policy: BatchPolicy,
        *,
        project_filter: set[str] | None = None,
        defer_to_lease: bool = False,
        lease_owner: str | None = None,
        abandoned_min_idle_ms: int = _ABANDONED_MIN_IDLE_MS,
        stale_consumer_idle_ms: int = _STALE_CONSUMER_IDLE_MS,
    ) -> None:
        self.bus = bus
        self.input_topic = input_topic
        self.group = group
        self.consumer_name = consumer_name
        self.policy = policy
        self._project_filter = project_filter
        self._stop = False
        self._progress_at: float = 0.0
        self._pel_dirty = False
        self._fail_counts: dict[bytes, int] = {}  # msg_id → failed-batch count (poison cap)
        # Long-lived consumers (daemon, MCP server) stand down while a CLI index holds the
        # lease; a CLI index's own inline consumers hold it and must not wait on it.
        self.defer_to_lease = defer_to_lease
        # Whose lease is *ours*. Until a persistent indexer existed nothing ever set
        # this: the daemon released its lease as soon as catch-up finished, so every
        # holder it could see afterwards genuinely was foreign and None was right by
        # accident. `atlas index --watch` holds its lease for the whole session, so
        # without this its own consumers would stand down against it and the pipeline
        # would idle forever with a full backlog -- the exact symptom the lease exists
        # to prevent, produced by the guard against it.
        self._lease_owner: str | None = lease_owner
        self._lease_waiting = False
        # Injectable so crash recovery is testable: a test cannot wait out the production
        # threshold, and an untestable recovery path is how the old shared consumer name
        # survived this long.
        self._abandoned_min_idle_ms = abandoned_min_idle_ms
        self._stale_consumer_idle_ms = stale_consumer_idle_ms

    @abstractmethod
    async def process_batch(self, events: list[Event], batch_id: str) -> set[str] | None:
        """Process a deduplicated batch. Subclasses implement tier logic.

        Returns dedup keys of events that were DEFERRED and must stay
        un-ACKed in the PEL; None/empty when fully handled.
        """

    def dedup_key(self, event: Event) -> str:
        """Return a dedup key for an event. Override for custom logic.

        FileChanged keys include project_name — monorepo sub-projects routinely
        share relative paths, and equal keys ACK-supersede each other in _dedup_put.
        """
        if isinstance(event, FileChanged):
            return f"{event.project_name}:{event.path}"
        return str(id(event))

    def _matches_project(self, event: Event) -> bool:
        """Check if an event belongs to the filtered project(s)."""
        if self._project_filter is None:
            return True
        pn = ""
        if isinstance(event, FileChanged):
            pn = event.project_name
        elif isinstance(event, EmbedDirty):
            # EmbedDirty doesn't carry project_name directly — always accept
            return True
        return pn in self._project_filter

    async def _dedup_put(
        self,
        pending: dict[str, tuple[bytes, Event]],
        key: str,
        msg_id: bytes,
        event: Event,
    ) -> None:
        """Insert into *pending*, keeping the NEWEST msg_id per dedup key.

        The superseded (older) msg_id is ACKed; the retained one stays in the
        PEL. A byte-equal msg_id is a PEL re-read — no ACK. Keep-newest makes
        the PEL reclaim idempotent: re-feeding an older un-ACKed message never
        displaces (or double-ACKs against) a newer one already held in *pending*.
        """
        old = pending.get(key)
        if old is None:
            pending[key] = (msg_id, event)
            return
        if old[0] == msg_id:
            return
        if _stream_id_key(msg_id) < _stream_id_key(old[0]):
            await self._ack(msg_id)
            return
        await self._ack(old[0])
        pending[key] = (msg_id, event)

    async def _ack(self, *msg_ids: bytes) -> None:
        """ACK messages and drop their poison-tracking state.

        Every ACK path must go through here: an ACKed message can never be
        re-delivered, so keeping its ``_fail_counts`` entry (supersession,
        project-filter, undecodable, empty-fields and park paths never reach
        ``_ack_processed``) would leak memory unboundedly.
        """
        await self.bus.ack(self.input_topic, self.group, *msg_ids)
        for mid in msg_ids:
            self._fail_counts.pop(mid, None)

    def stop(self) -> None:
        """Signal the consumer to stop after the current iteration."""
        self._stop = True

    def note_progress(self) -> None:
        """Record that real work just completed.

        Teardown reads this to tell a consumer that is slow from one that is stuck.
        The final flush is unbounded work — it scales with the project — so any fixed
        grace period is wrong at some size, and being wrong means the whole-project
        sweeps at the end of that flush are cancelled and silently skipped.
        """
        self._progress_at = asyncio.get_event_loop().time()

    @property
    def progress_at(self) -> float:
        """Loop-clock timestamp of the last :meth:`note_progress`, 0.0 if never."""
        return self._progress_at

    @property
    def stopped(self) -> bool:
        """True once ``stop()`` has been called (used by daemon supervision)."""
        return self._stop

    async def _pre_run(self) -> None:  # noqa: B027
        """Hook called before the main loop starts. Override for setup."""

    async def _post_run(self) -> None:  # noqa: B027
        """Hook called after the main loop exits. Override for teardown."""

    async def _wait_for_slot(self) -> bool:
        """Hook called at the top of each iteration before reading messages.

        Return ``True`` to proceed, ``False`` to break the loop.
        Default always proceeds.  Override for backpressure (e.g. semaphore).
        """
        return True

    async def _defer_to_foreign_lease(self) -> None:
        """Stand down while another process holds the indexer lease.

        Placed at the top of the loop, BEFORE any read, so a paused consumer is simply
        not asking for work — no in-flight batch is abandoned and no ACK path is touched.
        ADR-0009 exists because a review found six silent-drop bugs in those paths, so the
        yield is deliberately coarse: finish the current batch, then stop reading.

        Only long-lived consumers defer. A CLI index holds the lease itself and must not
        wait on it.
        """
        if not self.defer_to_lease:
            return
        while not self._stop:
            try:
                holder = await self.bus.read_indexer_lease()
            except Exception:
                return
            if not holder or holder == self._lease_owner:
                return
            if not self._lease_waiting:
                self._lease_waiting = True
                logger.info("{} pausing — another indexer holds the lease ({})", self.consumer_name, holder)
            await asyncio.sleep(_LEASE_POLL_S)
        self._lease_waiting = False

    async def _prune_consumer_registrations(self) -> None:
        """Deregister consumers left behind by processes that are no longer running.

        Redis registers a consumer on first read and never unregisters it, so now that
        names carry a process identity the group grows by one entry per index run —
        measured at 10 after a single session on this repo, unbounded over a project's life.

        Only entries holding nothing are removed. ``XGROUP DELCONSUMER`` destroys a PEL
        rather than reassigning it, so pruning a consumer that still owns work would delete
        exactly the messages the reclaim sweep exists to rescue. That ordering is the whole
        design: the sweep adopts first, which is what empties a dead PEL and makes its
        registration prunable on a later pass.
        """
        try:
            registrations = await self.bus.consumer_registrations(self.input_topic, self.group)
        except Exception:
            logger.opt(exception=True).debug("{} could not list consumer registrations", self.consumer_name)
            return

        for name, pending, idle_ms in registrations:
            if name == self.consumer_name or pending or idle_ms < self._stale_consumer_idle_ms:
                continue
            destroyed = await self.bus.drop_consumer(self.input_topic, self.group, name)
            if destroyed:
                # Raced: work landed between the listing and the delete, and is now gone.
                logger.error(
                    "{} destroyed {} pending message(s) deregistering {} — it took work after {}s of silence",
                    self.consumer_name,
                    destroyed,
                    name,
                    idle_ms // 1000,
                )
            else:
                logger.info("{} deregistered stale consumer {} (idle {}s)", self.consumer_name, name, idle_ms // 1000)

    async def _deregister_self(self) -> None:
        """Give up this process's registration on a clean exit.

        The idle-based prune above is the backstop for crashes; this is the common case,
        and doing it here means a well-behaved process leaves nothing for an hour-long
        timer to notice. Skipped when work is still unacked — that PEL is a crash-recovery
        net for the reclaim sweep to adopt, and deleting it would discard the messages.
        """
        try:
            for name, pending, _idle_ms in await self.bus.consumer_registrations(self.input_topic, self.group):
                if name != self.consumer_name:
                    continue
                if pending:
                    logger.warning(
                        "{} exiting with {} unacked message(s); leaving its registration for another "
                        "consumer to adopt from",
                        self.consumer_name,
                        pending,
                    )
                    return
                await self.bus.drop_consumer(self.input_topic, self.group, self.consumer_name)
                return
        except Exception:
            logger.opt(exception=True).debug("{} could not deregister itself", self.consumer_name)

    async def _ack_processed(self, events: list[Event], msg_ids: list[bytes], deferred: set[str]) -> None:
        """ACK msg_ids whose events were fully handled; deferred ones stay in the PEL."""
        ack_ids = [mid for mid, ev in zip(msg_ids, events, strict=True) if self.dedup_key(ev) not in deferred]
        if ack_ids:
            await self._ack(*ack_ids)
        if deferred:
            self._pel_dirty = True  # deferred messages stay in PEL; reclaim re-delivers them

    def _note_batch_failure(self, msg_ids: list[bytes]) -> None:
        for mid in msg_ids:
            self._fail_counts[mid] = self._fail_counts.get(mid, 0) + 1
        self._pel_dirty = True

    def _record_batch(self, *, events: int, outcome: str, started: float) -> None:
        """Stage total plus batch/event counters, for one finished batch.

        Called from **both** dispatch paths. `_worker` (the embed consumer's async
        dispatch) and `_dispatch_batch` (everything else, including the CLI's inline
        index pipeline) are separate code paths, and instrumenting only the first meant
        `atlas index` -- much the most common way anyone indexes -- reported no stage
        totals and no batch counters at all. Found by running an index and querying for
        the series, not by reading the code.

        `phase="total"` is the whole batch, so `sum by (stage)` answers "which stage
        costs the most" directly; summing the individual phases would miss whatever is
        not inside one of them and quietly under-report.
        """
        topic = self.input_topic.value
        stage = _STAGE_BY_TOPIC.get(self.input_topic, topic)
        metrics = get_metrics()
        metrics.stage_seconds.record(time.perf_counter() - started, {"stage": stage, "phase": "total"})
        metrics.batches_processed.add(1, {"topic": topic, "outcome": outcome})
        metrics.events_consumed.add(events, {"topic": topic})

    async def _dispatch_batch(
        self,
        events: list[Event],
        msg_ids: list[bytes],
        batch_id: str,
    ) -> None:
        """Process and ACK a batch. Override for async dispatch (e.g. worker tasks).

        Default: process inline, ACK non-deferred on success, leave in PEL on failure.
        """
        outcome = "ok"
        started = time.perf_counter()
        try:
            with logger.contextualize(consumer=self.consumer_name):
                deferred = await self.process_batch(events, batch_id) or set()
            await self._ack_processed(events, msg_ids, deferred)
            self.note_progress()
        except Exception:
            outcome = "failed"
            logger.exception("{} batch {} failed, will retry", self.consumer_name, batch_id)
            self._note_batch_failure(msg_ids)
        finally:
            self._record_batch(events=len(events), outcome=outcome, started=started)

    async def run(self) -> None:  # noqa: PLR0912, PLR0915
        """Main consumer loop — runs until ``stop()`` is called."""
        await self.bus.ensure_group(self.input_topic, self.group)
        logger.debug("{} started (group={}, topic={})", self.consumer_name, self.group, self.input_topic.value)
        await self._pre_run()

        pending: dict[str, tuple[bytes, Event]] = {}  # dedup_key → (msg_id, event)
        window_start: float | None = None
        pel_drained = False  # True once all pending (unacked) messages have been reclaimed
        last_reclaim_at = 0.0
        self._pel_dirty = False
        block_ms = (
            self.policy.block_ms
            if self.policy.block_ms is not None
            else max(100, int(self.policy.time_window_s * 1000 // 2))
        )

        try:
            while not self._stop:
                if not await self._wait_for_slot():
                    break
                await self._defer_to_foreign_lease()

                # Reclaim unacknowledged messages from PEL (failed batches), then adopt
                # anything a dead process abandoned. The second half exists because
                # consumer names now carry a process identity: read_pending only ever
                # returns THIS consumer's history, so a killed process's entries would
                # otherwise be orphaned under a name nothing will use again.
                now = asyncio.get_event_loop().time()
                due_for_sweep = now - last_reclaim_at >= _RECLAIM_SWEEP_S
                if not pel_drained or self._pel_dirty or due_for_sweep:
                    self._pel_dirty = False
                    if due_for_sweep:
                        await self._prune_consumer_registrations()
                    last_reclaim_at = now
                    reclaimed = await self.bus.read_pending(
                        self.input_topic,
                        self.group,
                        self.consumer_name,
                        count=self.policy.max_batch_size,
                    )
                    if not reclaimed:
                        reclaimed = await self.bus.reclaim_abandoned(
                            self.input_topic,
                            self.group,
                            self.consumer_name,
                            min_idle_ms=self._abandoned_min_idle_ms,
                            count=self.policy.max_batch_size,
                        )
                    if reclaimed:
                        for msg_id, fields in reclaimed:
                            if not fields:
                                await self._ack(msg_id)
                                continue
                            try:
                                event = decode_event(self.input_topic, fields)
                                key = self.dedup_key(event)
                            except KeyError, TypeError, ValueError:
                                logger.exception("{} failed to decode pending message, skipping", self.consumer_name)
                                await self._ack(msg_id)
                                continue
                            if not self._matches_project(event):
                                await self._ack(msg_id)
                                continue
                            if self._fail_counts.get(msg_id, 0) >= _MAX_BATCH_FAILURES:
                                logger.error(
                                    "{} parking poison message {} (key={}) after {} failed batches — "
                                    "change is dropped until the file is re-indexed",
                                    self.consumer_name,
                                    msg_id,
                                    key,
                                    _MAX_BATCH_FAILURES,
                                )
                                await self._ack(msg_id)
                                continue
                            await self._dedup_put(pending, key, msg_id, event)
                            if window_start is None:
                                window_start = asyncio.get_event_loop().time()
                    else:
                        pel_drained = True

                # Pull new messages
                messages = await self.bus.read_batch(
                    self.input_topic,
                    self.group,
                    self.consumer_name,
                    count=self.policy.max_batch_size,
                    block_ms=block_ms,
                )

                for msg_id, fields in messages:
                    try:
                        event = decode_event(self.input_topic, fields)
                        key = self.dedup_key(event)
                    except KeyError, TypeError, ValueError:
                        logger.exception("{} failed to decode message, skipping", self.consumer_name)
                        await self._ack(msg_id)
                        continue
                    if not self._matches_project(event):
                        await self._ack(msg_id)
                        continue
                    await self._dedup_put(pending, key, msg_id, event)
                    if window_start is None:
                        window_start = asyncio.get_event_loop().time()

                # Decide whether to flush
                if not pending:
                    continue

                elapsed = asyncio.get_event_loop().time() - (window_start or 0)
                if len(pending) < self.policy.max_batch_size and elapsed < self.policy.time_window_s:
                    continue

                # Flush
                msg_ids = [mid for mid, _ in pending.values()]
                events = [ev for _, ev in pending.values()]
                batch_id = uuid.uuid4().hex[:12]

                logger.debug("{} flushing batch {} ({} events)", self.consumer_name, batch_id, len(events))
                await self._dispatch_batch(events, msg_ids, batch_id)

                pending.clear()
                window_start = None
        finally:
            await self._post_run()
            await self._deregister_self()

        logger.debug("{} stopped", self.consumer_name)


# ---------------------------------------------------------------------------
# AST stage: parse + graph write (medium cost)
# ---------------------------------------------------------------------------

# Significance levels for the AST → Embed gate
#
# | Condition                                  | Level    | Action       |
# |--------------------------------------------|----------|--------------|
# | Docstring-only changed                     | MODERATE | Gate through |
# | Signature/body/name/tags/visibility change | HIGH     | Gate through |
# | Entity added/deleted                       | HIGH     | Always gate  |
#
# Whitespace-only file changes never reach classification — the file hash
# gate strips whitespace (_compute_file_hash). Every added|modified entity
# becomes an embed candidate; the embed_hash gate (read_embed_hashes) is
# what suppresses re-embedding when the embed text is unchanged.


_SIG_ORDER: dict[Significance, int] = {
    Significance.NONE: 0,
    Significance.TRIVIAL: 1,
    Significance.MODERATE: 2,
    Significance.HIGH: 3,
}


# Labels a citation can resolve to. A batch that adds or changes one of these
# is the signal that previously unresolvable citations may now have a target —
# see ASTConsumer._citation_retry_projects.
_DOCUMENT_LABELS: frozenset[NodeLabel] = frozenset({NodeLabel.DOC_FILE, NodeLabel.DOC_SECTION, NodeLabel.NOTE})


@dataclass
class ASTStats:
    """Accumulated delta statistics for AST stage processing."""

    files_processed: int = 0
    files_skipped: int = 0
    files_deferred: int = 0
    files_deleted: int = 0
    entities_added: int = 0
    entities_modified: int = 0
    entities_deleted: int = 0
    entities_unchanged: int = 0


@dataclass(frozen=True)
class _ParsedFileData:
    """Parse results for a single file, ready for batched graph write.

    ``parsed_file`` retains the raw tree-sitter parse so graph-querying
    detectors can run in a SECOND pass, after this batch's entities are
    upserted (see process_batch) — running them during parsing (before any
    entity in the batch exists in the graph) silently drops TESTS/OVERRIDES/
    INJECTED_INTO edges for same-batch subject/reference pairs.
    """

    file_path: str
    parsed_file: ParsedFile
    entities: list[ParsedEntity]
    non_import_rels: list[ParsedRelationship]
    import_rels: list[ParsedRelationship]
    call_rels: list[ParsedRelationship]
    type_rels: list[ParsedRelationship]
    inherit_rels: list[ParsedRelationship]
    ref_rels: list[ParsedRelationship]
    member_rels: list[ParsedRelationship]
    anchor_rels: list[ParsedRelationship]
    # Heuristic DOCUMENTS (symbol_mention / file_ref / explicit). Deferred for
    # cost rather than for visibility: resolving one inline meant a whole-graph
    # scan per doc file, and pooling them into the flush also lets a reference to
    # a file indexed later in the same run resolve instead of being dropped.
    doc_rels: list[ParsedRelationship]
    # READS_ENV / REFERENCES_FILE. Deferred like imports because the target
    # EnvVar/ResourceFile node does not exist until resolution MERGEs it.
    config_rels: list[ParsedRelationship]
    # entity uid → the raw ADR/RFC strings extract_rationale found in its
    # comments. Not a ParsedRelationship like the rest: citations are captured
    # as an entity *property*, and only post-batch resolution
    # (GraphClient.resolve_citations) knows which document node they mean.
    citations: dict[str, list[str]]


_SENTINEL_DELETED = _ParsedFileData(
    file_path="",
    parsed_file=ParsedFile(file_path="", language="", entities=[], relationships=[]),
    entities=[],
    non_import_rels=[],
    import_rels=[],
    call_rels=[],
    type_rels=[],
    inherit_rels=[],
    ref_rels=[],
    member_rels=[],
    anchor_rels=[],
    doc_rels=[],
    config_rels=[],
    citations={},
)


class ASTConsumer(TierConsumer):
    """AST stage: Parse AST via tree-sitter, write entities to graph, publish EmbedDirty."""

    def __init__(
        self,
        bus: EventBus,
        graph: GraphClient,
        settings: AtlasSettings,
        *,
        project_root: Path | None = None,
        project_filter: set[str] | None = None,
        policy: BatchPolicy | None = None,
        cooldown_s: float = 0.0,
        defer_to_lease: bool = False,
        lease_owner: str | None = None,
        abandoned_min_idle_ms: int = _ABANDONED_MIN_IDLE_MS,
    ) -> None:
        super().__init__(
            bus=bus,
            input_topic=Topic.FILE_CHANGED,
            group="ast",
            consumer_name=f"ast-{_process_tag()}",
            policy=policy or BatchPolicy(time_window_s=_AST_WINDOW_S, max_batch_size=30),
            project_filter=project_filter,
            defer_to_lease=defer_to_lease,
            lease_owner=lease_owner,
            abandoned_min_idle_ms=abandoned_min_idle_ms,
        )
        self.graph = graph
        self.settings = settings
        self._project_root = project_root or Path(settings.project_root)
        self.stats = ASTStats()
        self._detectors = get_enabled_detectors(settings.detectors.enabled)

        # Per-file cooldown state (daemon mode). Cooldown-deferred events stay
        # un-ACKed in the PEL and are redelivered by the reclaim loop.
        self._cooldown_s = cooldown_s
        self._cooldowns: dict[str, float] = {}  # "project_name:path" → expiry (monotonic)

        # Deferred resolution state — accumulate rels across batches, flush periodically.
        # In reindex mode (time_window_s=0, block_ms=50) use larger intervals to skip
        # redundant resolution; daemon mode (default policy) resolves every batch.
        is_reindex = self.policy.time_window_s == 0
        self._resolve_batch_interval: int = 5 if is_reindex else 1
        self._resolve_time_interval_s: float = 30.0 if is_reindex else 5.0
        self._batches_since_resolve: int = 0
        self._last_resolve_time: float = 0.0
        # Adaptivity is reindex-only. In watch mode `final` arrives at shutdown and a
        # stretched gap would delay a single edited file's edges by that gap; there the
        # flushes are small and frequent, which is the point of watch mode.
        self._resolve_adaptive: bool = is_reindex
        self._resolve_min_gap_s: float = 0.0
        self._pending_import_rels: list[ParsedRelationship] = []
        self._pending_call_rels: list[ParsedRelationship] = []
        self._pending_type_rels: list[ParsedRelationship] = []
        self._pending_inherit_rels: list[ParsedRelationship] = []
        self._pending_ref_rels: list[ParsedRelationship] = []
        self._pending_member_rels: list[ParsedRelationship] = []
        self._pending_anchor_rels: list[ParsedRelationship] = []
        # project_name -> {file_path: rels}. Keyed by FILE, not a flat list: each
        # parse deletes the file's outbound edges and then re-buffers its rels, so
        # a file parsed twice before one flush would otherwise contribute its rels
        # twice and CREATE every DOCUMENTS edge twice over. Keying by path makes the
        # second parse overwrite the first, which is what the delete already assumed.
        self._pending_doc_rels: dict[str, dict[str, list[ParsedRelationship]]] = {}
        # Notes re-parsed this flush. Drives the supersession stamp pass, which is
        # scoped to these rather than sweeping the project: a project-wide pass
        # inside the batch loop is erased by the next batch (ADR-0026).
        self._pending_note_uids: set[str] = set()
        self._pending_config_rels: list[ParsedRelationship] = []
        self._pending_citations: dict[str, dict[str, list[str]]] = {}  # project_name -> {uid: citations}
        # Every file this flush's batches actually re-parsed, per project — the
        # revoke scope handed to resolve_citations. Deliberately NOT derived
        # from _pending_citations: a file whose last "see ADR-14" comment was
        # just deleted yields no citations at all, and that is exactly the file
        # whose stale citation edge has to be cleared.
        self._pending_citation_files: dict[str, set[str]] = {}  # project_name -> {file_path}
        self._pending_project_names: set[str] = set()
        # Survives every flush: _pending_project_names is cleared each time, so a
        # final-flush-only sweep would iterate an empty set and silently do nothing.
        self._projects_seen: set[str] = set()
        # IMPORTS/CALLS/USES_TYPE rels that resolution could not settle for good.
        # It reads the graph as it stands at that flush, and set_batch_file_hashes
        # then makes the hash gate skip the caller for good — so a callee upserted
        # by a LATER batch loses its edge permanently, and "no inbound edge"
        # degrades from "unreachable" into "nothing was resolvable the moment that
        # file was last indexed". Measured on this repo: consumers.py had ZERO
        # edges of any type to events.py.
        #
        # Two buffers because replaying them costs different amounts — see
        # ReplayableRels. Both are keyed rather than appended so re-parsing a call
        # site replaces its entry instead of adding one; most entries never
        # resolve (builtins, external libraries) and an unkeyed list would grow
        # without bound in a long-running daemon.
        # Reindex mode only, on both counts: it is the mode whose ordering causes
        # the staleness (a bulk run resolves most files before the modules they
        # call into), and the only one that reliably reaches a final flush to
        # spend the buffer on. A daemon resolves against an already-complete
        # graph, so retaining this there would be megabytes held for nothing.
        self._retry_rels: dict[str, dict[tuple[str, str, str, str, str], ParsedRelationship]] = {}
        self._stale_candidate_rels: dict[str, dict[tuple[str, str, str, str, str], ParsedRelationship]] | None = (
            {} if is_reindex else None
        )
        # Every project that has ever contributed a citation in this consumer's
        # lifetime — NOT cleared per flush. The end-of-run retry sweep needs it
        # because a citation's target document is usually indexed in a LATER
        # batch than the citing code file (src/ sorts before wiki/), so the
        # per-batch pass legitimately cannot resolve it yet.
        self._citation_projects: set[str] = set()
        # Projects whose current batch added or changed a DocFile/DocSection/
        # Note. Cleared per flush, like the rel buffers. This is what makes
        # citations resolve in a long-running daemon: the end-of-run sweep only
        # fires when the consumer shuts down, so without a live trigger an ADR
        # written today would stay unlinked until the next restart. Indexing
        # the document IS the event that can newly satisfy a pending citation,
        # so the sweep rides on it instead of polling.
        self._citation_retry_projects: set[str] = set()

        # File hashes withheld from the graph until their batch's deferred
        # IMPORTS/CALLS/USES_TYPE/member-DEFINES rels are actually resolved —
        # writing the hash any earlier would make a crash before that point
        # permanently unrecoverable (hash gate would skip the file forever).
        self._pending_file_hashes: dict[str, dict[str, str]] = {}  # project_name -> {file_path: hash}

    async def run(self) -> None:
        try:
            await super().run()
        finally:
            # Final resolution flush for any remaining deferred rels, plus the
            # end-of-run citation retry sweep (which must run even when every
            # buffer is already empty — see _flush_deferred_resolution).
            await self._flush_deferred_resolution(final=True)

    def _pending_rel_count(self) -> int:
        """Relationships buffered since the last flush — what a deferred flush costs in memory."""
        return (
            len(self._pending_import_rels)
            + len(self._pending_call_rels)
            + len(self._pending_type_rels)
            + len(self._pending_inherit_rels)
            + len(self._pending_ref_rels)
            + len(self._pending_member_rels)
            + len(self._pending_anchor_rels)
            + len(self._pending_note_uids)
            + len(self._pending_config_rels)
            + sum(len(rels) for by_file in self._pending_doc_rels.values() for rels in by_file.values())
        )

    async def _flush_deferred_resolution(self, *, final: bool = False) -> None:  # noqa: PLR0912, PLR0915
        """Run resolution for all accumulated rels across batches.

        Citations are re-attempted whenever this flush's batches touched a
        document node (``_citation_retry_projects``) — indexing the ADR is what
        makes an earlier, unresolvable citation resolvable — and once more when
        *final* marks the last flush of a run, as a backstop for a document
        that was already in the graph before this run started.

        The per-project citation call also carries ``_pending_citation_files``,
        the revoke scope, so this flush's parsed files get delete-then-recreate
        rather than merge-only. The two retry sweeps deliberately pass no scope:
        they cover the whole project and reparse nothing, so a delete there
        would wipe citations for files nobody touched.

        IMPORTS/CALLS/USES_TYPE additionally carry ``_retry_rels`` forward — see
        that attribute for why a batch-local resolution alone loses edges for
        good. ``_stale_candidate_rels`` rides along only on the *final* flush,
        because every earlier replay of it is superseded by this one and each
        costs a full rewrite of the edges it owns. Projects with a backlog are
        visited even when this flush parsed nothing for them, which is what lets
        the final flush close out a full-index run.
        """
        flush_started = asyncio.get_event_loop().time()
        pending_before = self._pending_rel_count()

        if self._pending_anchor_rels:
            # Anchors may target code in any project (uid/project-prefixed/
            # absolute path forms) — resolved once, cross-project, rather
            # than per-project like CALLS/IMPORTS/USES_TYPE below.
            await self.graph.resolve_anchors(self._pending_anchor_rels)
            self.note_progress()

        if self._pending_note_uids:
            # After the note edges for this flush exist, not before: the pass reads
            # SUPERSEDES/CONTRADICTS to decide what to stamp.
            stamped = await self.graph.stamp_note_relations(sorted(self._pending_note_uids))
            logger.debug("Stamped supersession/contradiction on {} note(s)", stamped)
            self.note_progress()

        # `_stale_candidate_rels is not None` is the reindex flag (set to {} only when
        # is_reindex, see __init__), which the `final and ...` guard below already keys
        # on. During a reindex the *retry* backlog is deferred to the final flush too.
        #
        # _retry_rels[p] is dominated by rels that never resolve -- builtins,
        # third-party attribute calls, anything the receiver-type gate rejects -- so
        # once a project has been touched it stays in the backlog for the whole run,
        # and every later flush paid build_resolution_lookup(p) plus a full
        # _resolve_one_call pass over p's entire retry buffer even when this flush
        # parsed nothing for p. In a monorepo that is one such pass per sub-project per
        # flush, and the sub-projects are exactly what a monorepo has many of.
        #
        # Only *untouched* projects are affected: a project this flush parsed for is in
        # _pending_project_names already, and its retry buffer is merged inside the loop
        # regardless. An untouched project's backlog cannot become resolvable through
        # writes this consumer did not make, so replaying it mid-run is a no-op.
        #
        # Deliberately NOT deferred in watch mode. There `final` only arrives at
        # shutdown, which can be hours; another session's consumer writing to an
        # untouched project would otherwise leave its backlog unresolved until then.
        # Watch-mode flushes carry few projects and small buffers, so per-flush replay
        # costs little there — the amplification is a full-index problem.
        reindexing = self._stale_candidate_rels is not None
        backlog: set[str] = set()
        if final or not reindexing:
            backlog |= {p for p, r in self._retry_rels.items() if r}
        if final and self._stale_candidate_rels is not None:
            backlog |= {p for p, r in self._stale_candidate_rels.items() if r}
        for project_name in self._pending_project_names | backlog:
            proj_imports = [
                r for r in self._pending_import_rels if r.from_qualified_name.startswith(project_name + ":")
            ]
            proj_calls = [r for r in self._pending_call_rels if r.from_qualified_name.startswith(project_name + ":")]
            proj_types = [r for r in self._pending_type_rels if r.from_qualified_name.startswith(project_name + ":")]
            proj_members = [
                r for r in self._pending_member_rels if r.from_qualified_name.startswith(project_name + ":")
            ]
            proj_config = [r for r in self._pending_config_rels if r.from_qualified_name.startswith(project_name + ":")]

            # Strictly BEFORE resolve_imports, which MERGEs ExternalSymbol/
            # ExternalPackage stubs. Those carry a `name`, so the symbol branch would
            # match them — and docs name libraries constantly. Resolving inline, this
            # ran during the file's own upsert and never saw the same flush's stubs;
            # running it here keeps that candidate set rather than quietly widening it
            # under cover of a performance change. Already keyed by project, so no
            # from_qualified_name filter. Pooling every buffered file into one call is
            # the point: the file-ref branch scans the project once per call, and it
            # used to run once per doc file.
            proj_doc_rels = [r for rels in self._pending_doc_rels.get(project_name, {}).values() for r in rels]
            if proj_doc_rels:
                await self.graph.resolve_doc_links(project_name, proj_doc_rels)
                self.note_progress()

            retry = self._retry_rels.setdefault(project_name, {})
            all_stale = self._stale_candidate_rels
            stale = None if all_stale is None else all_stale.setdefault(project_name, {})
            replayed = list(retry.values()) + (list(stale.values()) if final and stale else [])
            proj_imports += [r for r in replayed if r.rel_type == RelType.IMPORTS]
            proj_calls += [r for r in replayed if r.rel_type == RelType.CALLS]
            proj_types += [r for r in replayed if r.rel_type == RelType.USES_TYPE]
            unresolved: list[ParsedRelationship] = []
            stale_candidates: list[ParsedRelationship] = []

            if proj_imports:
                replay = await self.graph.resolve_imports(project_name, proj_imports)
                self.note_progress()
                unresolved += replay.unresolved
                stale_candidates += replay.stale_candidates

            if proj_config:
                await self.graph.resolve_config_refs(project_name, proj_config)
                self.note_progress()

            # Strictly after resolve_imports: a base is usually external, and the
            # ExternalSymbol it points at does not exist until imports are resolved.
            proj_inherits = [
                r for r in self._pending_inherit_rels if r.from_qualified_name.startswith(project_name + ":")
            ]
            if proj_inherits:
                await self.graph.resolve_inherits(project_name, proj_inherits)
                self.note_progress()

            proj_refs = [r for r in self._pending_ref_rels if r.from_qualified_name.startswith(project_name + ":")]
            if proj_refs:
                await self.graph.resolve_value_references(project_name, proj_refs)
                self.note_progress()

            if proj_calls or proj_types or proj_members:
                # Built after resolve_imports above, so a retried import that only
                # just became resolvable is already in this lookup's import_map —
                # strategy 1 of _resolve_one_call reads exactly that.
                shared_lookup, td_map = await self.graph.build_resolution_lookup(project_name)
                if proj_calls:
                    replay = await self.graph.resolve_calls(
                        project_name,
                        proj_calls,
                        lookup=shared_lookup,
                        name_to_typedefs=td_map,
                        test_patterns=self.settings.search.test_patterns,
                    )
                    unresolved += replay.unresolved
                    stale_candidates += replay.stale_candidates
                if proj_types:
                    replay = await self.graph.resolve_type_refs(
                        project_name, proj_types, lookup=shared_lookup, name_to_typedefs=td_map
                    )
                    unresolved += replay.unresolved
                    stale_candidates += replay.stale_candidates
                if proj_members:
                    await self.graph.resolve_member_defines(
                        project_name, proj_members, lookup=shared_lookup, name_to_typedefs=td_map
                    )
                self.note_progress()

            # Rebuild rather than update: the whole backlog was just replayed, so
            # a rel that resolved this time is absent from `unresolved` and has to
            # drop out. Updating in place would keep it forever.
            retry.clear()
            for rel in unresolved:
                retry[_retry_key(rel)] = rel
            if stale is not None:
                if final:
                    # Just replayed against the complete graph: spent, not carried.
                    stale.clear()
                else:
                    # Accumulated across flushes — never cleared here, or a rel
                    # first seen two flushes ago would drop out before the replay.
                    for rel in stale_candidates:
                        stale[_retry_key(rel)] = rel

            proj_citations = self._pending_citations.pop(project_name, None)
            citation_files = self._pending_citation_files.pop(project_name, None)
            retry_citations = project_name in self._citation_retry_projects
            if proj_citations or citation_files or retry_citations:
                await self.graph.resolve_citations(
                    project_name,
                    proj_citations or {},
                    file_paths=citation_files,
                    retry_unresolved=retry_citations,
                )

            # Only now — after this project's deferred rels are actually
            # resolved — persist the file hashes withheld in process_batch.
            # A crash before this point leaves the stored hash unset, so the
            # hash gate reprocesses the file (and regenerates the rels) on
            # the next run instead of silently skipping it forever.
            pending_hashes = self._pending_file_hashes.pop(project_name, None)
            if pending_hashes:
                await self.graph.set_batch_file_hashes(project_name, pending_hashes)

        if final:
            for project_name in self._citation_projects:
                await self.graph.resolve_citations(project_name, {}, retry_unresolved=True)
            self._citation_projects.clear()

        # Reference-counted GC, last: every project's resolve_config_refs above
        # has already re-created this flush's references, so anything still at
        # zero incoming edges really has lost its last callsite. Running it
        # any earlier would delete nodes that are about to be re-linked.
        #
        # Gated on the flush having processed a project rather than on
        # proj_config being non-empty: a file whose LAST os.getenv() call was
        # just deleted produces no config rels at all, and that is precisely
        # the case that orphans a node. Cost is two label-index scans over the
        # two smallest labels in the graph — bounded by them, not by the graph.
        if final:
            # Whole-project sweep, and ONLY once every batch has been written. Run per
            # batch, the edges it writes run FROM a concrete method in some OTHER file, so
            # the next batch to re-process that file deletes them again in its
            # delete-then-recreate phase. Measured both ways: per batch, IMPLEMENTS
            # collapsed 261 -> 42 and GraphBackend went back to zero implementers.
            for project_name in self._projects_seen:
                await self.graph.resolve_protocol_conformance(project_name)
                self.note_progress()
            self._projects_seen.clear()

        if self._pending_project_names:
            await self.graph.gc_orphaned_reference_nodes()
            # An overflow chunk has no edge to its parent (the link is a property, which
            # is what makes finding one cheap), so a DETACH DELETE of the parent cannot
            # take it with it. Left behind it stays in the vector index, answering
            # searches with text no node in the graph still contains.
            orphaned = await self.graph.gc_orphaned_embed_chunks()
            if orphaned:
                logger.debug("Swept {} orphaned embed chunk(s)", orphaned)

        self._pending_import_rels.clear()
        self._pending_call_rels.clear()
        self._pending_type_rels.clear()
        self._pending_inherit_rels.clear()
        self._pending_ref_rels.clear()
        self._pending_member_rels.clear()
        self._pending_anchor_rels.clear()
        self._pending_doc_rels.clear()
        self._pending_note_uids.clear()
        self._pending_config_rels.clear()
        self._pending_citations.clear()
        self._pending_citation_files.clear()
        self._citation_retry_projects.clear()
        self._pending_project_names.clear()
        self._batches_since_resolve = 0
        self._last_resolve_time = asyncio.get_event_loop().time()

        # Pace the next flush by what this one cost. Reported unconditionally at debug
        # level because the pending count is the number _RESOLVE_PENDING_CEILING should
        # eventually be derived from.
        duration = self._last_resolve_time - flush_started
        if self._resolve_adaptive:
            self._resolve_min_gap_s = _next_resolve_gap(duration)
        logger.debug(
            "Resolution flush took {:.2f}s over {} pending rel(s); next flush deferred at least {:.1f}s",
            duration,
            pending_before,
            self._resolve_min_gap_s,
        )

    async def _parse_file(
        self,
        project_name: str,
        file_path: str,
        *,
        project_root: Path | None = None,
        source: bytes | None = None,
    ) -> _ParsedFileData | None:
        """Parse a single file (pure tree-sitter parse; no graph queries, no graph writes).

        Returns ``None`` for unreadable/unsupported files, ``_SENTINEL_DELETED``
        for deleted files, or a ``_ParsedFileData`` with parsed results.

        Graph-querying detectors are NOT run here — they run in process_batch,
        AFTER this batch's entities are upserted, so same-batch cross-file
        targets (TESTS/OVERRIDES/INJECTED_INTO) are resolvable instead of
        silently missing.

        If *source* is provided, it is used directly (avoids re-reading from disk
        when the hash gate has already read the file).
        """
        root = project_root if project_root is not None else self._project_root
        if source is None:
            full_path = root / file_path
            if not full_path.is_file():
                return _SENTINEL_DELETED
            try:
                source = full_path.read_bytes()
            except OSError:
                logger.warning("AST: cannot read {}", file_path)
                return None

        parsed = parse_file(
            file_path,
            source,
            project_name,
            max_source_chars=self.settings.index.max_source_chars,
            max_doc_section_chars=self.settings.index.max_doc_section_chars,
            max_parse_bytes=self.settings.index.max_parse_bytes,
            rationale=self.settings.rationale,
        )
        if parsed is None:
            logger.debug("AST: unsupported language for {}", file_path)
            return None

        _deferred = {
            RelType.IMPORTS,
            RelType.CALLS,
            RelType.USES_TYPE,
            RelType.INHERITS,
            RelType.REFERENCES,
            RelType.REGISTERED_BY,
            # Both DOCUMENTS lanes are post-batch: anchors via resolve_anchors,
            # heuristic symbol/file refs via resolve_doc_links.
            RelType.DOCUMENTS,
        } | _CONFIG_REF_REL_TYPES

        def _is_member(r: ParsedRelationship) -> bool:
            # Member DEFINES whose parent type may live in another file —
            # resolved post-batch via GraphClient.resolve_member_defines.
            return r.rel_type == RelType.DEFINES and "parent_type_name" in r.properties

        def _is_anchor(r: ParsedRelationship) -> bool:
            # Explicit anchors: frontmatter DOCUMENTS edges — resolved by
            # GraphClient.resolve_anchors, whose path/uid/symbol lookup is
            # cross-project. The heuristic refs go to resolve_doc_links, which is
            # per-project. DOCUMENTS as a whole is deferred (see _deferred above),
            # so this only picks which of the two lanes a rel belongs to.
            return r.rel_type == RelType.DOCUMENTS and r.properties.get("link_type") == "anchor"

        def _keep_config_ref(r: ParsedRelationship) -> bool:
            # A directory path in a string literal looks exactly like a file path to the
            # parser, which is a pure no-I/O function and cannot tell them apart. This
            # wrapper already owns the project root and already stats the file, so the
            # check belongs here. A path that does not exist is kept: an unresolved
            # reference to a data file is the case this node type is for.
            if r.rel_type not in _CONFIG_REF_REL_TYPES:
                return False
            if r.rel_type != RelType.REFERENCES_FILE:
                return True
            try:
                return not (root / r.to_name).is_dir()
            except OSError:
                return True

        return _ParsedFileData(
            file_path=file_path,
            parsed_file=parsed,
            entities=parsed.entities,
            non_import_rels=[r for r in parsed.relationships if r.rel_type not in _deferred and not _is_member(r)],
            import_rels=[r for r in parsed.relationships if r.rel_type == RelType.IMPORTS],
            call_rels=[r for r in parsed.relationships if r.rel_type == RelType.CALLS],
            # Signature-derived USES_TYPE resolves through the Callable lookup; one whose
            # source is a Value cannot, because a Value is not in that lookup. Split so each
            # reaches the resolver that can actually see its source.
            type_rels=[
                r for r in parsed.relationships if r.rel_type == RelType.USES_TYPE and r.properties.get("on") != "value"
            ],
            inherit_rels=[r for r in parsed.relationships if r.rel_type == RelType.INHERITS],
            ref_rels=[
                r
                for r in parsed.relationships
                if r.rel_type in (RelType.REFERENCES, RelType.REGISTERED_BY)
                or (r.rel_type == RelType.USES_TYPE and r.properties.get("on") == "value")
            ],
            anchor_rels=[r for r in parsed.relationships if _is_anchor(r)],
            doc_rels=[r for r in parsed.relationships if r.rel_type == RelType.DOCUMENTS and not _is_anchor(r)],
            member_rels=[r for r in parsed.relationships if _is_member(r)],
            config_rels=[r for r in parsed.relationships if _keep_config_ref(r)],
            citations={e.qualified_name: list(e.citations) for e in parsed.entities if e.citations},
        )

    async def process_batch(self, events: list[Event], batch_id: str) -> set[str]:  # noqa: PLR0912, PLR0915
        deferred_keys: set[str] = set()
        with _tracer.start_as_current_span("ast.process_batch", attributes={"batch_id": batch_id}) as span:
            # Per-file cooldown filter: defer events for recently-processed files.
            # Deferred events stay un-ACKed in the PEL and are redelivered every
            # batch window until the cooldown expires, so files_deferred counts
            # retry passes too.
            if self._cooldown_s > 0:
                now = asyncio.get_event_loop().time()
                # Clean expired cooldowns
                self._cooldowns = {k: exp for k, exp in self._cooldowns.items() if exp > now}
                processable: list[Event] = []
                deferred_count = 0
                for ev in events:
                    if isinstance(ev, FileChanged):
                        key = self.dedup_key(ev)
                        if key in self._cooldowns:
                            deferred_keys.add(key)
                            deferred_count += 1
                            continue
                    processable.append(ev)
                if deferred_count:
                    self.stats.files_deferred += deferred_count
                    logger.debug("AST batch {}: {} event(s) deferred by cooldown", batch_id, deferred_count)
                events = processable
                if not events:
                    return deferred_keys

            # Group paths by (project_name, project_root) — monorepo batches can mix sub-projects
            groups: dict[tuple[str, str], list[str]] = {}
            for e in events:
                if isinstance(e, FileChanged):
                    key = (e.project_name, e.project_root)
                    groups.setdefault(key, []).append(e.path)

            # Deduplicate paths within each group
            groups = {key: list(dict.fromkeys(paths)) for key, paths in groups.items()}

            total_paths = sum(len(p) for p in groups.values())
            logger.debug("AST batch {}: {} unique path(s) in {} group(s)", batch_id, total_paths, len(groups))

            embed_candidates: dict[str, tuple[EntityRef, str]] = {}  # uid → (ref, text_hash)
            changed_uids: set[str] = set()  # accumulated across every group in this batch
            skipped_before = self.stats.files_skipped
            total_changed = 0
            batch_max_sig = Significance.NONE

            for (event_project_name, event_project_root), unique_paths in groups.items():
                project_name = event_project_name or derive_project_name(Path(self.settings.project_root))
                effective_root = Path(event_project_root) if event_project_root else None
                root = effective_root if effective_root is not None else self._project_root

                # 0. File hash gate — read files, compute hashes, skip unchanged
                use_hash_gate = self.settings.index.file_hash_gate
                strip_ws = self.settings.index.strip_whitespace
                file_sources: dict[str, bytes] = {}  # file_path → source bytes (pre-read)
                deleted_files: list[str] = []

                # Separate deleted files (always process) and read live files
                live_paths: list[str] = []
                unreadable_paths: list[str] = []
                for fp in unique_paths:
                    full_path = root / fp
                    if not full_path.is_file():
                        deleted_files.append(fp)
                    else:
                        try:
                            file_sources[fp] = full_path.read_bytes()
                            live_paths.append(fp)
                        except OSError:
                            # Transient (editor/AV/indexer lock, sharing violation
                            # mid-save) — defer instead of dropping so the PEL
                            # retries it, rather than losing the change silently.
                            logger.warning("AST: cannot read {}, deferring for retry", fp)
                            unreadable_paths.append(fp)

                if unreadable_paths:
                    for fp in unreadable_paths:
                        deferred_keys.add(f"{event_project_name}:{fp}")
                    self.stats.files_deferred += len(unreadable_paths)

                # Apply hash gate to live files
                if use_hash_gate and live_paths:
                    new_hashes = {
                        fp: _compute_file_hash(file_sources[fp], strip_whitespace=strip_ws) for fp in live_paths
                    }
                    stored_hashes = await self.graph.get_batch_file_hashes(project_name, live_paths)

                    gate_passed: list[str] = []
                    for fp in live_paths:
                        stored = stored_hashes.get(fp)
                        if stored is not None and stored == new_hashes[fp]:
                            self.stats.files_skipped += 1
                        else:
                            gate_passed.append(fp)

                    hash_skipped = len(live_paths) - len(gate_passed)
                    if hash_skipped:
                        logger.debug(
                            "AST batch {}: hash gate skipped {}/{} file(s)",
                            batch_id,
                            hash_skipped,
                            len(live_paths),
                        )
                    live_paths = gate_passed
                else:
                    new_hashes = {}

                # 1. Parse loop (async, per-file) — no graph writes
                parsed_files: dict[str, _ParsedFileData] = {}

                with timed_phase("ast", "parse", files=len(live_paths)):
                    for file_idx, file_path in enumerate(live_paths, 1):
                        if file_idx % 50 == 0:
                            logger.debug("AST batch {}: parsed {}/{} files", batch_id, file_idx, len(live_paths))
                        pfd = await self._parse_file(
                            project_name,
                            file_path,
                            project_root=effective_root,
                            source=file_sources.get(file_path),
                        )
                        if pfd is _SENTINEL_DELETED:
                            deleted_files.append(file_path)
                        elif pfd is not None:
                            parsed_files[file_path] = pfd

                # 2. Handle deleted files
                for fp in deleted_files:
                    logger.debug("AST: file deleted, removing entities for {}", fp)
                    deleted = await self.graph.delete_file_entities(project_name, fp)
                    self.stats.files_deleted += 1
                    self.stats.entities_deleted += len(deleted)
                    if deleted:
                        batch_max_sig = Significance.HIGH

                # 3. Batched upsert (2 managed transactions) — entities + parser-only
                #    rels. Graph-querying detectors run AFTER this write (step 3.5)
                #    so this batch's own entities are visible for same-batch
                #    cross-file matches (TESTS/OVERRIDES/INJECTED_INTO would
                #    otherwise silently miss subjects added in the same batch).
                if parsed_files:
                    file_data = {fp: (pfd.entities, pfd.non_import_rels) for fp, pfd in parsed_files.items()}
                    with timed_phase("ast", "upsert", files=len(file_data)):
                        results = await self.graph.upsert_batch_entities(project_name, file_data)

                    # 3.5. Graph-querying detectors, now that this batch's entities exist.
                    det_results: dict[str, DetectorResult] = {}
                    if self._detectors:
                        with timed_phase("ast", "detectors", detectors=len(self._detectors)):
                            for fp, pfd in parsed_files.items():
                                det_result = await run_detectors(
                                    self._detectors, pfd.parsed_file, project_name, self.graph
                                )
                                if det_result.relationships or det_result.enrichments:
                                    det_results[fp] = det_result

                    # 4. Batched enrichments
                    all_enrichments = [e for det in det_results.values() for e in det.enrichments]
                    if all_enrichments:
                        with timed_phase("ast", "enrich", count=len(all_enrichments)):
                            await self.graph.apply_property_enrichments(all_enrichments)

                    # 4b. Re-write relationships for files with new detector-emitted
                    #     rels — merged with the original parser rels, since TX2
                    #     deletes then recreates each file's rel set (a partial
                    #     rewrite would drop the parser rels just written in step 3).
                    # A re-exported name has no uid yet — its target lives in a submodule
                    # this one imports, which only exists after resolve_imports. Split it
                    # out of the uid-routed write and resolve it with the rest.
                    for det in det_results.values():
                        self._pending_ref_rels.extend(r for r in det.relationships if r.properties.get("by_name"))
                    det_rel_files = {
                        fp: [r for r in det.relationships if not r.properties.get("by_name")]
                        for fp, det in det_results.items()
                        if any(not r.properties.get("by_name") for r in det.relationships)
                    }
                    if det_rel_files:
                        second_file_data = {
                            fp: (parsed_files[fp].entities, parsed_files[fp].non_import_rels + rels)
                            for fp, rels in det_rel_files.items()
                        }
                        # rels_only: the entities are byte-identical to the ones step 3
                        # just wrote, so the entity transaction would classify to nothing
                        # and issue a read plus an empty begin/commit. The merge below is
                        # load-bearing and unchanged -- TX2 deletes each file's whole rel
                        # set before recreating it, so dropping non_import_rels here would
                        # discard the parser rels written in step 3.
                        await self.graph.upsert_batch_entities(project_name, second_file_data, rels_only=True)

                    # 5. Accumulate stats + entity refs from per-file results
                    for fp, pfd in parsed_files.items():
                        result = results.get(fp)
                        if result is None:
                            continue

                        self.stats.files_processed += 1
                        self.stats.entities_added += len(result.added)
                        self.stats.entities_modified += len(result.modified)
                        self.stats.entities_deleted += len(result.deleted)
                        self.stats.entities_unchanged += len(result.unchanged)

                        changed_qns = set(result.added) | set(result.modified)
                        if not changed_qns:
                            self.stats.files_skipped += 1
                            continue

                        total_changed += len(changed_qns)

                        # Compute file-level significance
                        if result.added or result.deleted:
                            file_sig = Significance.HIGH
                        elif result.modified_significance:
                            file_sig = max(
                                (Significance(v) for v in result.modified_significance.values()),
                                key=lambda s: _SIG_ORDER[s],
                            )
                        else:
                            file_sig = Significance.NONE

                        if _SIG_ORDER[file_sig] > _SIG_ORDER[batch_max_sig]:
                            batch_max_sig = file_sig

                        entity_map = {
                            (e.qualified_name.split(":", 1)[1] if ":" in e.qualified_name else e.qualified_name): e
                            for e in pfd.entities
                        }
                        for qn in changed_qns:
                            entity = entity_map.get(qn)
                            if entity is not None:
                                changed_uids.add(entity.qualified_name)
                                if entity.label in _DOCUMENT_LABELS:
                                    self._citation_retry_projects.add(project_name)
                                ref = EntityRef(
                                    qualified_name=entity.qualified_name,
                                    node_type=entity.label.value,
                                    file_path=entity.file_path,
                                )
                                # Build embed text from parsed entity data (same fields as graph)
                                qn_bare = (
                                    entity.qualified_name.split(":", 1)[1]
                                    if ":" in entity.qualified_name
                                    else entity.qualified_name
                                )
                                props = {
                                    "_label": entity.label.value,
                                    "qualified_name": qn_bare,
                                    "name": entity.name,
                                    "kind": entity.kind,
                                    "signature": entity.signature or "",
                                    "docstring": entity.docstring or "",
                                    "source": entity.source or "",
                                    "tags": entity.tags,
                                }
                                text = build_embed_text(props)
                                if text:
                                    text_hash = hash_text(text)
                                    embed_candidates[entity.qualified_name] = (ref, text_hash)

                # 6. Withhold EVERY processed file's hash until the deferred flush.
                #    A hash written before _flush_deferred_resolution completes lets a
                #    crash in between drop that file's deferred work permanently — the
                #    gate skips the file forever afterwards.
                #
                #    This used to write immediately for files with "nothing deferred",
                #    gated on entity_changed_files. That signal is derived from the
                #    step-3 upsert delta, and step 3 has ALREADY advanced the entity's
                #    stored content_hash. So after an interrupted run the delta compares
                #    against a partially-applied state: the file reports "unchanged",
                #    takes the immediate path, and a second crash strands its citation
                #    edge exactly as before (ATL-090). The old comment claimed an
                #    unchanged entity has "provably identical citations" — true only if
                #    the previous run finished, which is precisely what a crash denies.
                #
                #    Withholding unconditionally is strictly MORE conservative: recovery
                #    re-parses more, never less. It also costs nothing extra — the flush
                #    already issues one set_batch_file_hashes per project, so this
                #    removes a second write rather than adding one.
                if new_hashes:
                    self._pending_file_hashes.setdefault(project_name, {}).update(new_hashes)

                # 7. Accumulate rels for deferred resolution
                group_import_rels = [r for pfd in parsed_files.values() for r in pfd.import_rels]
                group_call_rels = [r for pfd in parsed_files.values() for r in pfd.call_rels]
                group_type_rels = [r for pfd in parsed_files.values() for r in pfd.type_rels]
                group_inherit_rels = [r for pfd in parsed_files.values() for r in pfd.inherit_rels]
                group_ref_rels = [r for pfd in parsed_files.values() for r in pfd.ref_rels]
                group_member_rels = [r for pfd in parsed_files.values() for r in pfd.member_rels]
                group_anchor_rels = [r for pfd in parsed_files.values() for r in pfd.anchor_rels]
                group_config_rels = [r for pfd in parsed_files.values() for r in pfd.config_rels]

                self._pending_import_rels.extend(group_import_rels)
                self._pending_call_rels.extend(group_call_rels)
                self._pending_type_rels.extend(group_type_rels)
                self._pending_inherit_rels.extend(group_inherit_rels)
                self._pending_ref_rels.extend(group_ref_rels)
                self._pending_member_rels.extend(group_member_rels)
                self._pending_anchor_rels.extend(group_anchor_rels)
                # Every re-parsed file is recorded, doc rels or not: an empty list
                # overwrites a previous parse's entries, which is what makes a file
                # whose last reference was just deleted stop re-creating the edge.
                self._pending_doc_rels.setdefault(project_name, {}).update(
                    {fp: pfd.doc_rels for fp, pfd in parsed_files.items()}
                )
                self._pending_note_uids.update(
                    f"{project_name}:{e.qualified_name}"
                    for pfd in parsed_files.values()
                    for e in pfd.entities
                    if e.label is NodeLabel.NOTE
                )
                self._pending_config_rels.extend(group_config_rels)
                self._pending_project_names.add(project_name)
                self._projects_seen.add(project_name)

                group_citations = {uid: raws for pfd in parsed_files.values() for uid, raws in pfd.citations.items()}
                # The scope covers every re-parsed file, citations or not: it is
                # what lets resolve_citations revoke an edge whose comment was
                # deleted. Files the hash gate skipped are absent by
                # construction, so their citations are left alone.
                if parsed_files:
                    self._pending_citation_files.setdefault(project_name, set()).update(parsed_files)
                if group_citations:
                    self._pending_citations.setdefault(project_name, {}).update(group_citations)
                    self._citation_projects.add(project_name)

                # 8. Set per-file cooldown for processed files
                if self._cooldown_s > 0:
                    expiry = asyncio.get_event_loop().time() + self._cooldown_s
                    for fp in list(parsed_files) + deleted_files:
                        self._cooldowns[f"{event_project_name}:{fp}"] = expiry

            self._batches_since_resolve += 1
            now = asyncio.get_event_loop().time()
            elapsed = now - self._last_resolve_time
            due = (
                self._batches_since_resolve >= self._resolve_batch_interval or elapsed >= self._resolve_time_interval_s
            )
            # The adaptive gap only ever *delays* a due flush; it can never bring one
            # forward, so the existing cadence stays an upper bound on staleness. The
            # buffer ceiling overrides it so a long gap cannot grow memory without limit.
            if (due and elapsed >= self._resolve_min_gap_s) or self._pending_rel_count() >= _RESOLVE_PENDING_CEILING:
                with timed_phase("ast", "resolve", rels=self._pending_rel_count()):
                    await self._flush_deferred_resolution()

            # Anchor invalidation runs every batch, unlike the deferred-resolution
            # cadence above — a docstring-only edit that never clears the embed
            # significance gate should still flag its anchoring notes stale.
            # Best-effort by design: staleness is freshness metadata, and a timeout
            # here once killed a full index at close-out, leaving the project marked
            # "not indexed" over data that had landed completely.
            if changed_uids:
                try:
                    await self.graph.invalidate_stale_anchors(changed_uids)
                except Exception as exc:
                    logger.warning("Anchor staleness pass failed (continuing): {}", exc)

            span.set_attribute("files_count", total_paths)
            span.set_attribute("entities_changed", total_changed)

            logger.debug(
                "AST batch {}: {} files, {} skipped, {} entities changed",
                batch_id,
                total_paths,
                self.stats.files_skipped - skipped_before,
                total_changed,
            )

            if (
                self.settings.embeddings.enabled
                and embed_candidates
                and _SIG_ORDER[batch_max_sig] >= _SIG_ORDER[Significance.MODERATE]
            ):
                # Batch-read stored embed_hashes to filter graph hits
                cand_uids = list(embed_candidates.keys())
                cand_labels = [embed_candidates[uid][0].node_type for uid in cand_uids]
                stored = await self.graph.read_embed_hashes(cand_uids, labels=cand_labels)

                to_publish: list[EntityRef] = []
                for uid, (ref, text_hash) in embed_candidates.items():
                    stored_info = stored.get(uid)
                    if stored_info is not None:
                        stored_hash, has_embedding = stored_info
                        if stored_hash == text_hash and has_embedding:
                            continue  # graph hit — embedding still valid
                    to_publish.append(ref)

                if to_publish:
                    await self.bus.publish_many(
                        Topic.EMBED_DIRTY,
                        [EmbedDirty(entity=ref, significance=batch_max_sig) for ref in to_publish],
                    )

        return deferred_keys


# ---------------------------------------------------------------------------
# Embed stage: Embeddings (expensive, heavily batched)
# ---------------------------------------------------------------------------


EMBED_CHUNK_UID_SEPARATOR = "#chunk"
"""Joins a node's uid to a chunk index to make the chunk's own uid.

Not a character a qualified_name produces, so a chunk uid can never collide with an
entity's, and the parent is recoverable from the chunk uid by inspection alone.
"""


SNIPPET_CHARS = 240
"""How much of a chunk's text travels with a hit that matched it.

Enough to recognise what matched -- a test title, a function signature, the head of a
config block -- without storing the text twice. A chunk otherwise keeps only a vector
and a hash, so if this is not stored the information does not exist anywhere.
"""


class _ChunkFacts(NamedTuple):
    """What a chunk can tell a result about itself once its parent stands in for it."""

    snippet: str
    line_start: int | None
    line_end: int | None


def _chunk_line_span(
    chunk: str,
    offset: int,
    source: str,
    source_offset: int,
    entity_line_start: int | None,
) -> tuple[int | None, int | None]:
    """File lines this chunk covers, or ``(None, None)`` when that is not derivable.

    Only the ``source`` portion of an embed text has a known offset from the entity's
    own ``line_start``; the breadcrumb header and the docstring do not, because
    ``build_embed_text`` reorders and re-wraps them. A chunk landing there gets nulls.

    Null rather than a guess on purpose: a wrong line sends an agent to the wrong place,
    which is worse than sending it to the node and letting it look.
    """
    if entity_line_start is None or not source or source_offset < 0 or offset < source_offset:
        return None, None
    relative = offset - source_offset
    if relative > len(source):
        return None, None
    start = entity_line_start + source[:relative].count("\n")
    return start, start + chunk.count("\n")


@dataclass(frozen=True)
class _ChunkPlan:
    """How one embed batch's texts map onto the vectors that get written.

    A node whose embed text fits the model's input cap -- all but a fraction of a
    percent of them -- contributes exactly one unit and no chunk entry, so the plan is
    a pass-through for the normal batch.
    """

    units: list[tuple[str, str, str]]
    """``(target_uid, text, hash of that text)`` — what actually gets embedded."""

    store_hash: dict[str, str]
    """``target_uid -> the hash to persist on it``.

    Differs from the unit's own hash for exactly one case: the parent of a split node
    stores the hash of its *whole* text, not of chunk 1, because that is the value
    ``process_batch``'s freshness check compares against. Storing chunk 1's hash there
    would make every subsequent batch believe the node had changed and re-embed it
    forever.
    """

    chunk_of: dict[str, tuple[str, int]]
    """``chunk uid -> (parent uid, 1-based chunk index)`` for the overflow chunks."""

    chunk_facts: dict[str, _ChunkFacts]
    """``chunk uid -> what that chunk can tell a search result about itself``."""


def _plan_embed_chunks(
    to_process: list[tuple[str, str, str]],
    uid_to_label: dict[str, str],
    split: Callable[[str], SplitResult],
    props_by_uid: dict[str, dict[str, Any]] | None = None,
) -> _ChunkPlan:
    """Expand any text over the model's input cap into one unit per chunk.

    Chunk 1 is written to the node itself, so a node that fits creates no extra graph
    state at all and a node that does not keeps answering searches at its own uid.

    The warning is the point of the label check, not the split: a Callable or TypeDef
    that needs several chunks is usually one that is too large to be a single unit of
    anything, which is worth saying out loud. Documents are not -- an oversized
    DocSection has already been split into separate nodes by the parser, and a Note is
    deliberately never split, so neither is evidence of a defect.
    """
    units: list[tuple[str, str, str]] = []
    store_hash: dict[str, str] = {}
    chunk_of: dict[str, tuple[str, int]] = {}
    chunk_facts: dict[str, _ChunkFacts] = {}
    props_by_uid = props_by_uid or {}

    for uid, text, full_hash in to_process:
        split_result = split(text)
        chunks, hard_split = split_result.chunks, split_result.hard_split
        if split_result.dropped:
            # The cap is a cost ceiling, not a licence to lose text silently. Saying so
            # is the difference between "this node is long" and "this node is longer
            # than we are willing to index, and here is what that cost".
            logger.warning(
                "Embed text for {} exceeds {} chunks; ~{} tokens past the cap are not indexed",
                uid,
                len(chunks),
                split_result.dropped,
            )
        if len(chunks) <= 1:
            units.append((uid, text, full_hash))
            store_hash[uid] = full_hash
            continue

        label = uid_to_label.get(uid, "")
        log = logger.warning if label in _CODE_ENTITY_LABELS else logger.debug
        log(
            "Embed text for {} ({}) needs {} chunks to fit the model's input cap{}",
            uid,
            label or "unknown label",
            len(chunks),
            " — no natural border to cut at" if hard_split else "",
        )

        props = props_by_uid.get(uid, {})
        source = props.get("source") or ""
        # rfind, not find: build_embed_text appends the source last, and a short source
        # can also appear inside the docstring above it.
        source_offset = text.rfind(source) if source else -1
        entity_line_start = props.get("line_start")

        cursor = 0
        for index, chunk in enumerate(chunks, start=1):
            chunk_hash = hash_text(chunk)
            found = text.find(chunk, cursor)
            offset = found if found >= 0 else cursor
            cursor = offset + len(chunk)
            if index == 1:
                units.append((uid, chunk, chunk_hash))
                store_hash[uid] = full_hash
            else:
                chunk_uid = f"{uid}{EMBED_CHUNK_UID_SEPARATOR}{index}"
                units.append((chunk_uid, chunk, chunk_hash))
                store_hash[chunk_uid] = chunk_hash
                chunk_of[chunk_uid] = (uid, index)
                line_start, line_end = _chunk_line_span(chunk, offset, source, source_offset, entity_line_start)
                chunk_facts[chunk_uid] = _ChunkFacts(
                    snippet=chunk[:SNIPPET_CHARS], line_start=line_start, line_end=line_end
                )

    return _ChunkPlan(units=units, store_hash=store_hash, chunk_of=chunk_of, chunk_facts=chunk_facts)


def _partition_written_vectors(
    resolved: list[tuple[str, list[float], str]],
    plan: _ChunkPlan,
) -> tuple[list[tuple[str, list[float], str]], list[EmbedChunkWrite]]:
    """Sort embedded units into the two things they are written as.

    A unit's uid says which: anything the plan recorded as a chunk becomes an
    EmbedChunk node, everything else is a vector on the node itself. The hash written
    is the plan's, not the unit's — see ``_ChunkPlan.store_hash``.
    """
    parent_items: list[tuple[str, list[float], str]] = []
    chunk_items: list[EmbedChunkWrite] = []
    for target_uid, vector, _unit_hash in resolved:
        stored = plan.store_hash[target_uid]
        parent = plan.chunk_of.get(target_uid)
        if parent is None:
            parent_items.append((target_uid, vector, stored))
            continue
        parent_uid, chunk_index = parent
        facts = plan.chunk_facts.get(target_uid)
        chunk_items.append(
            EmbedChunkWrite(
                uid=target_uid,
                parent_uid=parent_uid,
                project_name=parent_uid.split(":", 1)[0],
                chunk_index=chunk_index,
                vector=vector,
                embed_hash=stored,
                snippet=facts.snippet if facts else "",
                line_start=facts.line_start if facts else None,
                line_end=facts.line_end if facts else None,
            )
        )
    return parent_items, chunk_items


class EmbedConsumer(TierConsumer):
    """Embed stage: Re-embed entities via TEI. Deduplicates by qualified name.

    Implements a three-level lookup to minimize expensive embedding API calls:
      1. **Graph hit** — node already has ``embed_hash`` matching current text (free).
      2. **Graph dedup hit** — some node, in any project, already has a vector for
         this exact text under this model (1 round-trip, ADR-0036).
      3. **API call** — embed via TEI / cloud provider (expensive).
    """

    def __init__(
        self,
        bus: EventBus,
        graph: GraphClient,
        embed: EmbedClient,
        *,
        project_filter: set[str] | None = None,
        policy: BatchPolicy | None = None,
        max_concurrency: int | None = None,
        defer_to_lease: bool = False,
        lease_owner: str | None = None,
        abandoned_min_idle_ms: int = _ABANDONED_MIN_IDLE_MS,
    ) -> None:
        _max_conc = max_concurrency or embed.max_concurrency
        super().__init__(
            bus=bus,
            input_topic=Topic.EMBED_DIRTY,
            group="embed",
            consumer_name=f"embed-{_process_tag()}",
            policy=policy
            or BatchPolicy(
                time_window_s=10.0,
                max_batch_size=embed.batch_size * _max_conc,
            ),
            project_filter=project_filter,
            defer_to_lease=defer_to_lease,
            lease_owner=lease_owner,
            abandoned_min_idle_ms=abandoned_min_idle_ms,
        )
        self.graph = graph
        self.embed = embed
        # Read once, here, not per batch: reached lazily at write time a missing
        # attribute surfaces as a swallowed per-batch error and the vectors simply
        # never appear. Read at construction it is an immediate, obvious failure.
        self._embed_model: str = embed.configured_model
        self._max_concurrency = _max_conc
        self._sem = asyncio.Semaphore(self._max_concurrency)
        self._inflight: set[asyncio.Task[None]] = set()
        # Bounded, not serialised. The lock this replaces protected nothing that
        # _inflight_uids does not already protect: concurrent embed workers hold
        # provably disjoint uid sets (the membership test and the .update(claimed)
        # are consecutive statements with no await between them, so the claim is
        # atomic under asyncio), and every written uid comes from that claimed set.
        #
        # Measured against Memgraph 3.12 with live vector indices, disjoint uids,
        # 250 entities per worker: serialised 702ms vs concurrent 592ms at 2 workers
        # (1.19x), 1579 vs 1211 at 4 (1.30x), 2866 vs 2379 at 8 (1.20x). The ratio is
        # flat, which says Memgraph already serialises most of this itself and the
        # app-level lock was adding ~20-30% on top of that. So 2 buys essentially the
        # whole win with a quarter of the exposure of 8.
        #
        # Exposure is the reason it is not simply removed. ADR-0024 records
        # memgraph#4473, an unfixed Storage GC segfault during vector index GC with
        # concurrent inserts. A stress run here -- 8 workers x 250 entities x 25
        # rounds, 50k concurrent vector-index writes -- did not reproduce it
        # (RestartCount 0, every row written), but that is one machine and the bug is
        # intermittent, so this stays conservative rather than treating one clean run
        # as proof of absence. Drop to 1 if a Storage GC crash ever appears.
        self._write_gate = asyncio.Semaphore(_EMBED_WRITE_CONCURRENCY)

        # Uids currently being read+embedded+written by an in-flight worker.
        # A second concurrently-dispatched batch for the SAME uid is deferred
        # (not processed) until the first worker releases it — otherwise a
        # slow worker holding a stale read can finish writing AFTER a faster,
        # later-dispatched worker already wrote a newer vector/hash for the
        # same entity, silently clobbering it with stale data (lost update).
        self._inflight_uids: set[str] = set()

    def dedup_key(self, event: Event) -> str:
        if isinstance(event, EmbedDirty):
            return event.entity.qualified_name
        return super().dedup_key(event)

    async def _pre_run(self) -> None:
        logger.debug("{} concurrency={}", self.consumer_name, self._max_concurrency)

    async def _post_run(self) -> None:
        if self._inflight:
            logger.debug("{} draining {} in-flight worker(s)", self.consumer_name, len(self._inflight))
            await asyncio.gather(*self._inflight, return_exceptions=True)

    async def _dispatch_batch(
        self,
        events: list[Event],
        msg_ids: list[bytes],
        batch_id: str,
    ) -> None:
        """Acquire a worker slot, then dispatch as a background task."""
        await self._sem.acquire()
        if self._stop:
            self._sem.release()
            return
        task = asyncio.create_task(self._worker(events, msg_ids, batch_id))
        self._inflight.add(task)
        task.add_done_callback(self._inflight.discard)

    async def _worker(
        self,
        events: list[Event],
        msg_ids: list[bytes],
        batch_id: str,
    ) -> None:
        """Execute process_batch for a single batch, then release the semaphore."""
        outcome = "ok"
        started = time.perf_counter()
        try:
            logger.debug("{} dispatching batch {} ({} events)", self.consumer_name, batch_id, len(events))
            with logger.contextualize(consumer=self.consumer_name):
                deferred = await self.process_batch(events, batch_id) or set()
            await self._ack_processed(events, msg_ids, deferred)
        except Exception:
            outcome = "failed"
            logger.exception("{} batch {} failed, will retry via PEL", self.consumer_name, batch_id)
            self._note_batch_failure(msg_ids)
        finally:
            self._record_batch(events=len(events), outcome=outcome, started=started)
            self._sem.release()

    async def _resolve_from_graph(
        self, to_process: list[tuple[str, str, str]]
    ) -> tuple[list[tuple[str, list[float], str]], list[tuple[str, str, str]], int]:
        """Copy vectors the graph already holds for these texts.

        Returns ``(resolved, need_embed, hits)``. The graph is the dedup layer: the
        vectors are already there, durably, keyed by exactly this hash, and a Valkey
        copy of them was the thing that filled the shared instance and failed the
        pipeline's stream writes (ADR-0036).
        """
        if not to_process:
            return [], [], 0
        found = await self.graph.find_embeddings_by_hash([h for _, _, h in to_process], self._embed_model)
        if not found:
            return [], list(to_process), 0
        resolved: list[tuple[str, list[float], str]] = []
        need_embed: list[tuple[str, str, str]] = []
        for uid, text, text_hash in to_process:
            vec = found.get(text_hash)
            if vec is None:
                need_embed.append((uid, text, text_hash))
            else:
                resolved.append((uid, vec, text_hash))
        return resolved, need_embed, len(resolved)

    async def _embed_and_store(self, need_embed: list[tuple[str, str, str]]) -> list[tuple[str, list[float], str]]:
        """Embed texts via the API. Returns (uid, vector, hash) tuples.

        Identical texts inside one batch are embedded **once**. Two entities can share
        a hash — a moved file, a copied helper, a re-exported symbol — and paying the
        provider twice in the same request for the same string is pure waste. This is
        the one hit class the old Valkey cache could never serve either, because a
        ``--full`` reindex cleared it before the run started.
        """
        if not need_embed:
            return []
        first_of: dict[str, str] = {}
        unique: list[tuple[str, str, str]] = []
        for uid, text, text_hash in need_embed:
            if text_hash not in first_of:
                first_of[text_hash] = text
                unique.append((uid, text, text_hash))
        vectors = await self.embed.embed_batch([text for _, text, _ in unique])
        by_hash = {th: vec for (_u, _t, th), vec in zip(unique, vectors, strict=True)}
        return [(uid, by_hash[th], th) for uid, _text, th in need_embed]

    async def process_batch(self, events: list[Event], batch_id: str) -> set[str] | None:  # noqa: PLR0915
        # Collect and deduplicate entities across all events in the batch
        seen: dict[str, EntityRef] = {}
        for e in events:
            if isinstance(e, EmbedDirty):
                seen[e.entity.qualified_name] = e.entity

        # Defer any uid already claimed by another in-flight worker (see
        # __init__ docstring for _inflight_uids) — the dedup key for EmbedDirty
        # IS the qualified_name, so this set is returned as-is for the PEL to
        # retain and redeliver once the earlier worker releases the claim.
        deferred: set[str] = {uid for uid in seen if uid in self._inflight_uids}
        entities = [ref for uid, ref in seen.items() if uid not in deferred]
        claimed = [uid for uid in seen if uid not in deferred]
        self._inflight_uids.update(claimed)

        try:
            with _tracer.start_as_current_span("embed.process_batch", attributes={"batch_id": batch_id}) as span:
                logger.debug("Embed batch {}: {} unique entity(ies)", batch_id, len(entities))

                if not entities:
                    return deferred or None

                t0 = asyncio.get_event_loop().time()

                # 1. Read entity properties from graph (includes embed_hash + embedding)
                #    Pass labels so queries use per-label indices instead of full scans.
                qns = [e.qualified_name for e in entities]
                entity_labels = [e.node_type for e in entities]
                with timed_phase("embed", "read_entities", entities=len(qns)):
                    entity_props = await self.graph.read_entity_texts(qns, labels=entity_labels)

                # 2. Build embed texts — graph-check for unchanged content
                to_process: list[tuple[str, str, str]] = []  # (uid, text, text_hash)
                uid_to_label: dict[str, str] = {}
                graph_hits = 0
                for props in entity_props:
                    text = build_embed_text(props)
                    if not text:
                        continue
                    uid = props["uid"]
                    text_hash = hash_text(text)
                    if lbl := props.get("_label"):
                        uid_to_label[uid] = lbl
                    if props.get("embed_hash") == text_hash and props.get("has_embedding"):
                        graph_hits += 1
                    else:
                        to_process.append((uid, text, text_hash))

                total = graph_hits + len(to_process)
                if not to_process:
                    elapsed = asyncio.get_event_loop().time() - t0
                    logger.debug(
                        "Embed batch {}: {} entities, {} unchanged, 0 deduped, 0 embedded ({:.1f}s)",
                        batch_id,
                        total,
                        graph_hits,
                        elapsed,
                    )
                    return deferred or None

                # 2b. Expand anything over the model's input cap into one unit per chunk.
                plan = _plan_embed_chunks(
                    to_process,
                    uid_to_label,
                    self.embed.split_text,
                    {p["uid"]: p for p in entity_props if p.get("uid")},
                )

                # 3. Graph dedup check → 4. API call for what is genuinely new
                with timed_phase("embed", "dedup", candidates=len(plan.units)):
                    dedup_resolved, need_embed, dedup_hits = await self._resolve_from_graph(plan.units)
                # The expensive one, and the only phase whose cost is someone else's
                # network: split out so a slow provider cannot be mistaken for a slow graph.
                with timed_phase("embed", "provider", entities=len(need_embed)):
                    api_vectors = await self._embed_and_store(need_embed)

                # 5. Write all new/changed vectors + embed_hashes to graph (single UNWIND)
                #    Bounded to _EMBED_WRITE_CONCURRENCY rather than fully serialised --
                #    see the _write_gate comment in __init__ for the measurements.
                all_resolved = dedup_resolved + api_vectors
                parent_items, chunk_items = _partition_written_vectors(all_resolved, plan)

                if all_resolved:
                    with timed_phase("embed", "write_lock_wait"):
                        await self._write_gate.acquire()
                    try:
                        with timed_phase("embed", "write", vectors=len(all_resolved)):
                            # Every node here is being re-embedded, so its old overflow
                            # chunks describe text it no longer contains. Dropping them
                            # unconditionally costs one indexed statement that matches
                            # nothing in the normal case, and is the only thing standing
                            # between a node that shrank and vectors of deleted content
                            # still answering searches.
                            await self.graph.delete_embed_chunks([uid for uid, _t, _h in to_process])
                            write_labels = [uid_to_label[uid] for uid, _, _ in parent_items] if uid_to_label else None
                            # Stamp the model: a vector only means anything inside the space
                            # its model defines, and one database holds several (ATL-135).
                            await self.graph.write_embeddings_and_hashes(
                                parent_items,
                                labels=write_labels,
                                model=self._embed_model,
                            )
                            await self.graph.write_embed_chunks(chunk_items, model=self._embed_model)
                    finally:
                        self._write_gate.release()

                elapsed = asyncio.get_event_loop().time() - t0
                span.set_attribute("entities_count", total)
                span.set_attribute("graph_hits", graph_hits)
                span.set_attribute("dedup_hits", dedup_hits)
                span.set_attribute("api_embedded", len(api_vectors))

                get_metrics().embedding_latency.record(elapsed)
                # Where each vector came from. "2,346 API calls, 0 cache hits" was the
                # measurement that justified ADR-0036; nothing was reporting it, so it
                # had to be reconstructed from logs after the fact.
                embeddings = get_metrics().embeddings_total
                embeddings.add(graph_hits, {"source": "unchanged"})
                embeddings.add(dedup_hits, {"source": "dedup"})
                embeddings.add(len(api_vectors), {"source": "api"})

                logger.debug(
                    "Embed batch {}: {} entities, {} unchanged, {} deduped, {} embedded ({:.1f}s)",
                    batch_id,
                    total,
                    graph_hits,
                    dedup_hits,
                    len(api_vectors),
                    elapsed,
                )

                return deferred or None
        finally:
            self._inflight_uids.difference_update(claimed)
