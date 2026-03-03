"""Two-stage consumer pipeline for event-driven indexing.

    FileChanged → AST stage (hash gate + AST parse + diff)
                → significance gate → EmbedDirty → Embed stage (embeddings)

Each stage uses batch-pull with configurable time/count policy and
deduplicates within its batch window.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

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
from code_atlas.parsing.ast import ParsedEntity, ParsedRelationship, parse_file
from code_atlas.parsing.detectors import PropertyEnrichment, get_enabled_detectors, run_detectors
from code_atlas.schema import RelType
from code_atlas.search.embeddings import EmbedCache, build_embed_text
from code_atlas.settings import derive_project_name
from code_atlas.telemetry import get_metrics, get_tracer

if TYPE_CHECKING:
    from code_atlas.graph.client import GraphClient
    from code_atlas.search.embeddings import EmbedClient
    from code_atlas.settings import AtlasSettings

_tracer = get_tracer(__name__)

_COLLAPSE_BLANK_RE = re.compile(rb"\n{3,}")


def _compute_file_hash(source: bytes, *, strip_whitespace: bool = True) -> str:
    """Compute a short SHA-256 hash of file contents.

    When *strip_whitespace* is True: strip leading/trailing whitespace per
    line, collapse consecutive blank lines, then hash.  This makes the gate
    ignore formatting-only changes (e.g. ``ruff format``).
    """
    if strip_whitespace:
        lines = [line.strip() for line in source.splitlines()]
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
    ) -> None:
        self.bus = bus
        self.input_topic = input_topic
        self.group = group
        self.consumer_name = consumer_name
        self.policy = policy
        self._project_filter = project_filter
        self._stop = False

    @abstractmethod
    async def process_batch(self, events: list[Event], batch_id: str) -> None:
        """Process a deduplicated batch. Subclasses implement tier logic."""

    def dedup_key(self, event: Event) -> str:
        """Return a dedup key for an event. Override for custom logic."""
        if isinstance(event, FileChanged):
            return event.path
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
        """Insert into *pending*, ACKing the superseded msg_id if the key already exists."""
        old = pending.get(key)
        if old is not None:
            await self.bus.ack(self.input_topic, self.group, old[0])
        pending[key] = (msg_id, event)

    def stop(self) -> None:
        """Signal the consumer to stop after the current iteration."""
        self._stop = True

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

    async def _dispatch_batch(
        self,
        events: list[Event],
        msg_ids: list[bytes],
        batch_id: str,
    ) -> None:
        """Process and ACK a batch. Override for async dispatch (e.g. worker tasks).

        Default: process inline, ACK on success, leave in PEL on failure.
        """
        try:
            with logger.contextualize(consumer=self.consumer_name):
                await self.process_batch(events, batch_id)
            await self.bus.ack(self.input_topic, self.group, *msg_ids)
        except Exception:
            logger.exception("{} batch {} failed, will retry", self.consumer_name, batch_id)
            self._pel_dirty = True

    async def run(self) -> None:  # noqa: PLR0912, PLR0915
        """Main consumer loop — runs until ``stop()`` is called."""
        await self.bus.ensure_group(self.input_topic, self.group)
        logger.debug("{} started (group={}, topic={})", self.consumer_name, self.group, self.input_topic.value)
        await self._pre_run()

        pending: dict[str, tuple[bytes, Event]] = {}  # dedup_key → (msg_id, event)
        window_start: float | None = None
        pel_drained = False  # True once all pending (unacked) messages have been reclaimed
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

                # Reclaim unacknowledged messages from PEL (failed batches).
                if not pel_drained or self._pel_dirty:
                    self._pel_dirty = False
                    reclaimed = await self.bus.read_pending(
                        self.input_topic,
                        self.group,
                        self.consumer_name,
                        count=self.policy.max_batch_size,
                    )
                    if reclaimed:
                        for msg_id, fields in reclaimed:
                            if not fields:
                                await self.bus.ack(self.input_topic, self.group, msg_id)
                                continue
                            try:
                                event = decode_event(self.input_topic, fields)
                                key = self.dedup_key(event)
                            except KeyError, TypeError, ValueError:
                                logger.exception("{} failed to decode pending message, skipping", self.consumer_name)
                                await self.bus.ack(self.input_topic, self.group, msg_id)
                                continue
                            if not self._matches_project(event):
                                await self.bus.ack(self.input_topic, self.group, msg_id)
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
                        await self.bus.ack(self.input_topic, self.group, msg_id)
                        continue
                    if not self._matches_project(event):
                        await self.bus.ack(self.input_topic, self.group, msg_id)
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

        logger.debug("{} stopped", self.consumer_name)


# ---------------------------------------------------------------------------
# AST stage: parse + graph write (medium cost)
# ---------------------------------------------------------------------------

# Significance levels for the AST → Embed gate
#
# | Condition                        | Level    | Action        |
# |----------------------------------|----------|---------------|
# | Whitespace/formatting only       | NONE     | Stop          |
# | Non-docstring comment            | TRIVIAL  | Stop          |
# | Docstring changed                | MODERATE | Gate through  |
# | Body changed < 20% AST diff     | MODERATE | Gate through  |
# | Body changed >= 20%             | HIGH     | Gate through  |
# | Signature changed                | HIGH     | Always gate   |
# | Entity added/deleted             | HIGH     | Always gate   |


_SIG_ORDER: dict[Significance, int] = {
    Significance.NONE: 0,
    Significance.TRIVIAL: 1,
    Significance.MODERATE: 2,
    Significance.HIGH: 3,
}


@dataclass
class ASTStats:
    """Accumulated delta statistics for AST stage processing."""

    files_processed: int = 0
    files_skipped: int = 0
    files_deleted: int = 0
    entities_added: int = 0
    entities_modified: int = 0
    entities_deleted: int = 0
    entities_unchanged: int = 0


@dataclass(frozen=True)
class _ParsedFileData:
    """Parse + detect results for a single file, ready for batched graph write."""

    file_path: str
    entities: list[ParsedEntity]
    non_import_rels: list[ParsedRelationship]
    import_rels: list[ParsedRelationship]
    call_rels: list[ParsedRelationship]
    type_rels: list[ParsedRelationship]
    enrichments: list[PropertyEnrichment]


_SENTINEL_DELETED = _ParsedFileData(
    file_path="", entities=[], non_import_rels=[], import_rels=[], call_rels=[], type_rels=[], enrichments=[]
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
    ) -> None:
        super().__init__(
            bus=bus,
            input_topic=Topic.FILE_CHANGED,
            group="ast",
            consumer_name="ast-0",
            policy=policy or BatchPolicy(time_window_s=3.0, max_batch_size=30),
            project_filter=project_filter,
        )
        self.graph = graph
        self.settings = settings
        self._project_root = project_root or Path(settings.project_root)
        self.stats = ASTStats()
        self._detectors = get_enabled_detectors(settings.detectors.enabled)

        # Per-file cooldown state (daemon mode)
        self._cooldown_s = cooldown_s
        self._cooldowns: dict[str, float] = {}  # file_path → expiry (monotonic)
        self._deferred: dict[str, FileChanged] = {}  # file_path → latest deferred event
        self._deferred_drain_task: asyncio.Task[None] | None = None

        # Deferred resolution state — accumulate rels across batches, flush periodically.
        # In reindex mode (time_window_s=0, block_ms=50) use larger intervals to skip
        # redundant resolution; daemon mode (default policy) resolves every batch.
        is_reindex = self.policy.time_window_s == 0
        self._resolve_batch_interval: int = 5 if is_reindex else 1
        self._resolve_time_interval_s: float = 30.0 if is_reindex else 5.0
        self._batches_since_resolve: int = 0
        self._last_resolve_time: float = 0.0
        self._pending_import_rels: list[ParsedRelationship] = []
        self._pending_call_rels: list[ParsedRelationship] = []
        self._pending_type_rels: list[ParsedRelationship] = []
        self._pending_project_names: set[str] = set()

    def dedup_key(self, event: Event) -> str:
        if isinstance(event, FileChanged):
            return event.path
        return super().dedup_key(event)

    async def _pre_run(self) -> None:
        if self._cooldown_s > 0:
            self._deferred_drain_task = asyncio.create_task(self._drain_deferred_loop())

    async def _post_run(self) -> None:
        if self._deferred_drain_task is not None:
            self._deferred_drain_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._deferred_drain_task
            self._deferred_drain_task = None

    async def _drain_deferred_loop(self) -> None:
        """Re-publish deferred events whose cooldown has expired.

        Runs as a background task so deferred events are eventually
        processed even if no new events arrive to trigger ``process_batch``.
        """
        while not self._stop:
            await asyncio.sleep(2.0)
            now = asyncio.get_event_loop().time()
            expired = [(fp, ev) for fp, ev in self._deferred.items() if now >= self._cooldowns.get(fp, 0)]
            for fp, _ in expired:
                del self._deferred[fp]
            if expired:
                await self.bus.publish_many(Topic.FILE_CHANGED, [ev for _, ev in expired])

    async def run(self) -> None:
        try:
            await super().run()
        finally:
            # Final resolution flush for any remaining deferred rels
            if self._pending_import_rels or self._pending_call_rels or self._pending_type_rels:
                await self._flush_deferred_resolution()

    async def _flush_deferred_resolution(self) -> None:
        """Run resolution for all accumulated rels across batches."""
        for project_name in self._pending_project_names:
            proj_imports = [
                r for r in self._pending_import_rels if r.from_qualified_name.startswith(project_name + ":")
            ]
            proj_calls = [r for r in self._pending_call_rels if r.from_qualified_name.startswith(project_name + ":")]
            proj_types = [r for r in self._pending_type_rels if r.from_qualified_name.startswith(project_name + ":")]

            if proj_imports:
                await self.graph.resolve_imports(project_name, proj_imports)

            if proj_calls or proj_types:
                shared_lookup, td_map = await self.graph.build_resolution_lookup(project_name)
                if proj_calls:
                    await self.graph.resolve_calls(project_name, proj_calls, lookup=shared_lookup)
                if proj_types:
                    await self.graph.resolve_type_refs(
                        project_name, proj_types, lookup=shared_lookup, name_to_typedefs=td_map
                    )

        self._pending_import_rels.clear()
        self._pending_call_rels.clear()
        self._pending_type_rels.clear()
        self._pending_project_names.clear()
        self._batches_since_resolve = 0
        self._last_resolve_time = asyncio.get_event_loop().time()

    async def _parse_file(
        self,
        project_name: str,
        file_path: str,
        *,
        project_root: Path | None = None,
        source: bytes | None = None,
    ) -> _ParsedFileData | None:
        """Parse and detect a single file without graph writes.

        Returns ``None`` for unreadable/unsupported files, ``_SENTINEL_DELETED``
        for deleted files, or a ``_ParsedFileData`` with parsed results.

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

        parsed = parse_file(file_path, source, project_name, max_source_chars=self.settings.index.max_source_chars)
        if parsed is None:
            logger.debug("AST: unsupported language for {}", file_path)
            return None

        det_result = await run_detectors(self._detectors, parsed, project_name, self.graph)
        all_rels = parsed.relationships + det_result.relationships

        _deferred = {RelType.IMPORTS, RelType.CALLS, RelType.USES_TYPE}
        return _ParsedFileData(
            file_path=file_path,
            entities=parsed.entities,
            non_import_rels=[r for r in all_rels if r.rel_type not in _deferred],
            import_rels=[r for r in all_rels if r.rel_type == RelType.IMPORTS],
            call_rels=[r for r in all_rels if r.rel_type == RelType.CALLS],
            type_rels=[r for r in all_rels if r.rel_type == RelType.USES_TYPE],
            enrichments=det_result.enrichments,
        )

    async def process_batch(self, events: list[Event], batch_id: str) -> None:  # noqa: PLR0912, PLR0915
        with _tracer.start_as_current_span("ast.process_batch", attributes={"batch_id": batch_id}) as span:
            # Per-file cooldown filter: defer events for recently-processed files
            if self._cooldown_s > 0:
                now = asyncio.get_event_loop().time()
                # Clean expired cooldowns
                self._cooldowns = {fp: exp for fp, exp in self._cooldowns.items() if exp > now}
                processable: list[Event] = []
                deferred_count = 0
                for ev in events:
                    if isinstance(ev, FileChanged) and ev.path in self._cooldowns:
                        self._deferred[ev.path] = ev  # latest overwrites
                        deferred_count += 1
                    else:
                        processable.append(ev)
                if deferred_count:
                    logger.debug("AST batch {}: {} event(s) deferred by cooldown", batch_id, deferred_count)
                events = processable
                if not events:
                    return

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
                for fp in unique_paths:
                    full_path = root / fp
                    if not full_path.is_file():
                        deleted_files.append(fp)
                    else:
                        try:
                            file_sources[fp] = full_path.read_bytes()
                            live_paths.append(fp)
                        except OSError:
                            logger.warning("AST: cannot read {}", fp)

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

                # 3. Batched upsert (2 managed transactions)
                if parsed_files:
                    file_data = {fp: (pfd.entities, pfd.non_import_rels) for fp, pfd in parsed_files.items()}
                    results = await self.graph.upsert_batch_entities(project_name, file_data)

                    # 4. Batched enrichments
                    all_enrichments = [e for pfd in parsed_files.values() for e in pfd.enrichments]
                    if all_enrichments:
                        await self.graph.apply_property_enrichments(all_enrichments)

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
                                    "kind": entity.kind,
                                    "signature": entity.signature or "",
                                    "docstring": entity.docstring or "",
                                    "source": entity.source or "",
                                }
                                text = build_embed_text(props)
                                if text:
                                    text_hash = EmbedCache.hash_text(text)
                                    embed_candidates[entity.qualified_name] = (ref, text_hash)

                # 6. Write back file hashes for processed files
                if new_hashes:
                    hashes_to_write = {fp: new_hashes[fp] for fp in parsed_files if fp in new_hashes}
                    if hashes_to_write:
                        await self.graph.set_batch_file_hashes(project_name, hashes_to_write)

                # 7. Accumulate rels for deferred resolution
                group_import_rels = [r for pfd in parsed_files.values() for r in pfd.import_rels]
                group_call_rels = [r for pfd in parsed_files.values() for r in pfd.call_rels]
                group_type_rels = [r for pfd in parsed_files.values() for r in pfd.type_rels]

                self._pending_import_rels.extend(group_import_rels)
                self._pending_call_rels.extend(group_call_rels)
                self._pending_type_rels.extend(group_type_rels)
                self._pending_project_names.add(project_name)

                # 8. Set per-file cooldown for processed files
                if self._cooldown_s > 0:
                    expiry = asyncio.get_event_loop().time() + self._cooldown_s
                    for fp in list(parsed_files) + deleted_files:
                        self._cooldowns[fp] = expiry

            self._batches_since_resolve += 1
            now = asyncio.get_event_loop().time()
            if (
                self._batches_since_resolve >= self._resolve_batch_interval
                or (now - self._last_resolve_time) >= self._resolve_time_interval_s
            ):
                await self._flush_deferred_resolution()

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


# ---------------------------------------------------------------------------
# Embed stage: Embeddings (expensive, heavily batched)
# ---------------------------------------------------------------------------


class EmbedConsumer(TierConsumer):
    """Embed stage: Re-embed entities via TEI. Deduplicates by qualified name.

    Implements a three-level lookup to minimize expensive embedding API calls:
      1. **Graph hit** — node already has ``embed_hash`` matching current text (free).
      2. **Valkey cache hit** — vector stored in Redis from a previous run (1 round-trip).
      3. **API call** — embed via TEI / cloud provider (expensive).
    """

    def __init__(
        self,
        bus: EventBus,
        graph: GraphClient,
        embed: EmbedClient,
        *,
        cache: EmbedCache | None = None,
        project_filter: set[str] | None = None,
        policy: BatchPolicy | None = None,
        max_concurrency: int | None = None,
    ) -> None:
        _max_conc = max_concurrency or embed.max_concurrency
        super().__init__(
            bus=bus,
            input_topic=Topic.EMBED_DIRTY,
            group="embed",
            consumer_name="embed",
            policy=policy
            or BatchPolicy(
                time_window_s=10.0,
                max_batch_size=embed.batch_size * _max_conc,
            ),
            project_filter=project_filter,
        )
        self.graph = graph
        self.embed = embed
        self.cache = cache
        self._max_concurrency = _max_conc
        self._sem = asyncio.Semaphore(self._max_concurrency)
        self._inflight: set[asyncio.Task[None]] = set()
        self._pel_dirty = False
        self._write_lock = asyncio.Lock()

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
        try:
            logger.debug("{} dispatching batch {} ({} events)", self.consumer_name, batch_id, len(events))
            with logger.contextualize(consumer=self.consumer_name):
                await self.process_batch(events, batch_id)
            await self.bus.ack(self.input_topic, self.group, *msg_ids)
        except Exception:
            logger.exception("{} batch {} failed, will retry via PEL", self.consumer_name, batch_id)
            self._pel_dirty = True
        finally:
            self._sem.release()

    async def _resolve_cache(
        self, to_process: list[tuple[str, str, str]]
    ) -> tuple[list[tuple[str, list[float], str]], list[tuple[str, str, str]], int]:
        """Resolve vectors from Valkey cache, returning (cache_resolved, need_embed, cache_hits)."""
        if self.cache is not None and to_process:
            remaining_hashes = [h for _, _, h in to_process]
            cached = await self.cache.get_many(remaining_hashes)
            cache_resolved: list[tuple[str, list[float], str]] = []
            need_embed: list[tuple[str, str, str]] = []
            hits = 0
            for uid, text, text_hash in to_process:
                if text_hash in cached:
                    cache_resolved.append((uid, cached[text_hash], text_hash))
                    hits += 1
                else:
                    need_embed.append((uid, text, text_hash))
            return cache_resolved, need_embed, hits
        return [], list(to_process), 0

    async def _embed_and_store(self, need_embed: list[tuple[str, str, str]]) -> list[tuple[str, list[float], str]]:
        """Embed texts via API and store results in cache. Returns (uid, vector, hash) tuples."""
        if not need_embed:
            return []
        texts = [text for _, text, _ in need_embed]
        vectors = await self.embed.embed_batch(texts)
        result = [(uid, vec, th) for (uid, _t, th), vec in zip(need_embed, vectors, strict=True)]
        if self.cache is not None:
            await self.cache.put_many([(th, vec) for _, vec, th in result])
        return result

    async def process_batch(self, events: list[Event], batch_id: str) -> None:
        with _tracer.start_as_current_span("embed.process_batch", attributes={"batch_id": batch_id}) as span:
            # Collect and deduplicate entities across all events in the batch
            seen: dict[str, EntityRef] = {}
            for e in events:
                if isinstance(e, EmbedDirty):
                    seen[e.entity.qualified_name] = e.entity

            entities = list(seen.values())
            logger.debug("Embed batch {}: {} unique entity(ies)", batch_id, len(entities))

            if not entities:
                return

            t0 = asyncio.get_event_loop().time()

            # 1. Read entity properties from graph (includes embed_hash + embedding)
            #    Pass labels so queries use per-label indices instead of full scans.
            qns = [e.qualified_name for e in entities]
            entity_labels = [e.node_type for e in entities]
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
                text_hash = EmbedCache.hash_text(text)
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
                    "Embed batch {}: {} entities, {} graph hits, 0 cache hits, 0 embedded ({:.1f}s)",
                    batch_id,
                    total,
                    graph_hits,
                    elapsed,
                )
                return

            # 3. Valkey cache check → 4. API call for misses
            cache_resolved, need_embed, cache_hits = await self._resolve_cache(to_process)
            api_vectors = await self._embed_and_store(need_embed)

            # 5. Write all new/changed vectors + embed_hashes to graph (single UNWIND)
            #    Serialized via _write_lock to avoid Memgraph write-lock contention.
            all_resolved = cache_resolved + api_vectors
            if all_resolved:
                with _tracer.start_as_current_span("embed.write_lock_wait"):
                    await self._write_lock.acquire()
                try:
                    with _tracer.start_as_current_span("embed.write_embeddings"):
                        write_labels = [uid_to_label[uid] for uid, _, _ in all_resolved] if uid_to_label else None
                        await self.graph.write_embeddings_and_hashes(all_resolved, labels=write_labels)
                finally:
                    self._write_lock.release()

            elapsed = asyncio.get_event_loop().time() - t0
            span.set_attribute("entities_count", total)
            span.set_attribute("graph_hits", graph_hits)
            span.set_attribute("cache_hits", cache_hits)
            span.set_attribute("api_embedded", len(api_vectors))

            get_metrics().embedding_latency.record(elapsed)

            logger.debug(
                "Embed batch {}: {} entities, {} graph hits, {} cache hits, {} embedded ({:.1f}s)",
                batch_id,
                total,
                graph_hits,
                cache_hits,
                len(api_vectors),
                elapsed,
            )
