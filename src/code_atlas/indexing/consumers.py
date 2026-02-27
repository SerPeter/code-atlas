"""Tiered consumer pipeline for event-driven indexing.

Three tiers form a linear pipeline, each pulling at its own pace:

    FileChanged → Tier 1 (graph metadata) → ASTDirty → Tier 2 (AST parse)
                → significance gate → EmbedDirty → Tier 3 (embeddings)

Each tier uses batch-pull with configurable time/count policy and
deduplicates within its batch window.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from code_atlas.events import (
    ASTDirty,
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
        if isinstance(event, (FileChanged, ASTDirty)):
            pn = event.project_name
        elif isinstance(event, EmbedDirty):
            # EmbedDirty doesn't carry project_name directly — always accept
            return True
        return pn in self._project_filter

    def stop(self) -> None:
        """Signal the consumer to stop after the current iteration."""
        self._stop = True

    async def run(self) -> None:  # noqa: PLR0912, PLR0915
        """Main consumer loop — runs until ``stop()`` is called."""
        await self.bus.ensure_group(self.input_topic, self.group)
        logger.debug("{} started (group={}, topic={})", self.consumer_name, self.group, self.input_topic.value)

        pending: dict[str, tuple[bytes, Event]] = {}  # dedup_key → (msg_id, event)
        window_start: float | None = None
        pel_drained = False  # True once all pending (unacked) messages have been reclaimed

        while not self._stop:
            # First, reclaim any unacknowledged messages from the PEL (failed batches).
            # Once the PEL is empty, switch to reading new messages only.
            if not pel_drained:
                reclaimed = await self.bus.read_pending(
                    self.input_topic,
                    self.group,
                    self.consumer_name,
                    count=self.policy.max_batch_size,
                )
                if reclaimed:
                    for msg_id, fields in reclaimed:
                        if not fields:
                            # Tombstone (deleted message) — just ACK it
                            await self.bus.ack(self.input_topic, self.group, msg_id)
                            continue
                        try:
                            event = decode_event(self.input_topic, fields)
                            key = self.dedup_key(event)
                        except KeyError, TypeError, ValueError, json.JSONDecodeError:
                            logger.exception("{} failed to decode pending message, skipping", self.consumer_name)
                            await self.bus.ack(self.input_topic, self.group, msg_id)
                            continue
                        if not self._matches_project(event):
                            await self.bus.ack(self.input_topic, self.group, msg_id)
                            continue
                        pending[key] = (msg_id, event)
                        if window_start is None:
                            window_start = asyncio.get_event_loop().time()
                else:
                    pel_drained = True

            # Pull new messages (short block so we can check flush + stop)
            block_ms = (
                self.policy.block_ms
                if self.policy.block_ms is not None
                else max(100, int(self.policy.time_window_s * 1000 // 2))
            )
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
                except KeyError, TypeError, ValueError, json.JSONDecodeError:
                    logger.exception("{} failed to decode message, skipping", self.consumer_name)
                    await self.bus.ack(self.input_topic, self.group, msg_id)
                    continue
                if not self._matches_project(event):
                    await self.bus.ack(self.input_topic, self.group, msg_id)
                    continue
                pending[key] = (msg_id, event)  # latest wins
                if window_start is None:
                    window_start = asyncio.get_event_loop().time()

            # Decide whether to flush
            if not pending:
                continue

            elapsed = asyncio.get_event_loop().time() - (window_start or 0)
            should_flush = len(pending) >= self.policy.max_batch_size or elapsed >= self.policy.time_window_s

            if not should_flush:
                continue

            # Flush: extract events and message IDs
            msg_ids = [mid for mid, _ in pending.values()]
            events = [ev for _, ev in pending.values()]
            batch_id = uuid.uuid4().hex[:12]

            logger.debug("{} flushing batch {} ({} events)", self.consumer_name, batch_id, len(events))

            try:
                with logger.contextualize(consumer=self.consumer_name):
                    await self.process_batch(events, batch_id)
                await self.bus.ack(self.input_topic, self.group, *msg_ids)
            except Exception:
                # Failed batch: do NOT ack — messages stay in PEL for reclaim
                logger.exception("{} batch {} failed, will retry", self.consumer_name, batch_id)
                pel_drained = False  # re-check PEL on next iteration

            pending.clear()
            window_start = None

        logger.debug("{} stopped", self.consumer_name)


# ---------------------------------------------------------------------------
# Tier 1: Graph metadata (cheap, fast)
# ---------------------------------------------------------------------------


class Tier1GraphConsumer(TierConsumer):
    """Tier 1: Update file-level graph metadata, always publish ASTDirty downstream."""

    def __init__(
        self, bus: EventBus, graph: GraphClient, settings: AtlasSettings, *, project_filter: set[str] | None = None
    ) -> None:
        super().__init__(
            bus=bus,
            input_topic=Topic.FILE_CHANGED,
            group="tier1-graph",
            consumer_name="tier1-graph-0",
            policy=BatchPolicy(time_window_s=0.5, max_batch_size=50),
            project_filter=project_filter,
        )
        self.graph = graph
        self.settings = settings

    def dedup_key(self, event: Event) -> str:
        if isinstance(event, FileChanged):
            return event.path
        return super().dedup_key(event)

    async def process_batch(self, events: list[Event], batch_id: str) -> None:
        with _tracer.start_as_current_span("tier1.process_batch", attributes={"batch_id": batch_id}):
            # Group files by (project_name, project_root) — monorepo batches can mix sub-projects
            groups: dict[tuple[str, str], list[str]] = {}
            for e in events:
                if isinstance(e, FileChanged):
                    key = (e.project_name, e.project_root)
                    groups.setdefault(key, []).append(e.path)

            total = sum(len(p) for p in groups.values())
            logger.debug("Tier1 batch {}: {} file(s) in {} group(s)", batch_id, total, len(groups))

            # TODO: Update Memgraph file nodes (timestamps, staleness flags)

            # Publish one ASTDirty per file — Tier 2 decides significance
            for (project_name, project_root), paths in groups.items():
                await self.bus.publish_many(
                    Topic.AST_DIRTY,
                    [ASTDirty(path=p, project_name=project_name, project_root=project_root) for p in paths],
                )


# ---------------------------------------------------------------------------
# Tier 2: AST parse + graph write (medium cost)
# ---------------------------------------------------------------------------

# Significance levels for the Tier 2 → Tier 3 gate
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
class Tier2Stats:
    """Accumulated delta statistics for Tier 2 processing."""

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


class Tier2ASTConsumer(TierConsumer):
    """Tier 2: Parse AST via tree-sitter, write entities to graph, publish EmbedDirty."""

    def __init__(
        self,
        bus: EventBus,
        graph: GraphClient,
        settings: AtlasSettings,
        *,
        project_root: Path | None = None,
        project_filter: set[str] | None = None,
        policy: BatchPolicy | None = None,
    ) -> None:
        super().__init__(
            bus=bus,
            input_topic=Topic.AST_DIRTY,
            group="tier2-ast",
            consumer_name="tier2-ast-0",
            policy=policy or BatchPolicy(time_window_s=3.0, max_batch_size=30),
            project_filter=project_filter,
        )
        self.graph = graph
        self.settings = settings
        self._project_root = project_root or Path(settings.project_root)
        self.stats = Tier2Stats()
        self._detectors = get_enabled_detectors(settings.detectors.enabled)

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
        if isinstance(event, ASTDirty):
            return event.path
        return super().dedup_key(event)

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
    ) -> _ParsedFileData | None:
        """Parse and detect a single file without graph writes.

        Returns ``None`` for unreadable/unsupported files, ``_SENTINEL_DELETED``
        for deleted files, or a ``_ParsedFileData`` with parsed results.
        """
        root = project_root if project_root is not None else self._project_root
        full_path = root / file_path
        if not full_path.is_file():
            return _SENTINEL_DELETED

        try:
            source = full_path.read_bytes()
        except OSError:
            logger.warning("Tier2: cannot read {}", file_path)
            return None

        parsed = parse_file(file_path, source, project_name, max_source_chars=self.settings.index.max_source_chars)
        if parsed is None:
            logger.debug("Tier2: unsupported language for {}", file_path)
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
        with _tracer.start_as_current_span("tier2.process_batch", attributes={"batch_id": batch_id}) as span:
            # Group paths by (project_name, project_root) — monorepo batches can mix sub-projects
            groups: dict[tuple[str, str], list[str]] = {}
            for e in events:
                if isinstance(e, ASTDirty):
                    key = (e.project_name, e.project_root)
                    groups.setdefault(key, []).append(e.path)

            # Deduplicate paths within each group
            groups = {key: list(dict.fromkeys(paths)) for key, paths in groups.items()}

            total_paths = sum(len(p) for p in groups.values())
            logger.debug("Tier2 batch {}: {} unique path(s) in {} group(s)", batch_id, total_paths, len(groups))

            changed_entity_refs: list[EntityRef] = []
            skipped_before = self.stats.files_skipped
            total_changed = 0
            batch_max_sig = Significance.NONE

            for (event_project_name, event_project_root), unique_paths in groups.items():
                project_name = event_project_name or derive_project_name(Path(self.settings.project_root))
                effective_root = Path(event_project_root) if event_project_root else None

                # 1. Parse loop (async, per-file) — no graph writes
                parsed_files: dict[str, _ParsedFileData] = {}
                deleted_files: list[str] = []

                for file_idx, file_path in enumerate(unique_paths, 1):
                    if file_idx % 50 == 0:
                        logger.debug("Tier2 batch {}: parsed {}/{} files", batch_id, file_idx, len(unique_paths))
                    pfd = await self._parse_file(project_name, file_path, project_root=effective_root)
                    if pfd is _SENTINEL_DELETED:
                        deleted_files.append(file_path)
                    elif pfd is not None:
                        parsed_files[file_path] = pfd

                # 2. Handle deleted files
                for fp in deleted_files:
                    logger.debug("Tier2: file deleted, removing entities for {}", fp)
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
                                changed_entity_refs.append(
                                    EntityRef(
                                        qualified_name=entity.qualified_name,
                                        node_type=entity.label.value,
                                        file_path=entity.file_path,
                                    )
                                )

                # 6. Accumulate rels for deferred resolution
                group_import_rels = [r for pfd in parsed_files.values() for r in pfd.import_rels]
                group_call_rels = [r for pfd in parsed_files.values() for r in pfd.call_rels]
                group_type_rels = [r for pfd in parsed_files.values() for r in pfd.type_rels]

                self._pending_import_rels.extend(group_import_rels)
                self._pending_call_rels.extend(group_call_rels)
                self._pending_type_rels.extend(group_type_rels)
                self._pending_project_names.add(project_name)

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
                "Tier2 batch {}: {} files, {} skipped, {} entities changed",
                batch_id,
                total_paths,
                self.stats.files_skipped - skipped_before,
                total_changed,
            )

            if (
                self.settings.embeddings.enabled
                and changed_entity_refs
                and _SIG_ORDER[batch_max_sig] >= _SIG_ORDER[Significance.MODERATE]
            ):
                await self.bus.publish_many(
                    Topic.EMBED_DIRTY,
                    [EmbedDirty(entity=ref, significance=batch_max_sig) for ref in changed_entity_refs],
                )


# ---------------------------------------------------------------------------
# Tier 3: Embeddings (expensive, heavily batched)
# ---------------------------------------------------------------------------


class Tier3EmbedConsumer(TierConsumer):
    """Tier 3: Re-embed entities via TEI. Deduplicates by qualified name.

    Implements a three-tier lookup to minimize expensive embedding API calls:
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
    ) -> None:
        super().__init__(
            bus=bus,
            input_topic=Topic.EMBED_DIRTY,
            group="tier3-embed",
            consumer_name="tier3-embed-0",
            policy=policy or BatchPolicy(time_window_s=15.0, max_batch_size=100),
            project_filter=project_filter,
        )
        self.graph = graph
        self.embed = embed
        self.cache = cache

    def dedup_key(self, event: Event) -> str:
        if isinstance(event, EmbedDirty):
            return event.entity.qualified_name
        return super().dedup_key(event)

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

    async def process_batch(self, events: list[Event], batch_id: str) -> None:  # noqa: PLR0912
        with _tracer.start_as_current_span("tier3.process_batch", attributes={"batch_id": batch_id}) as span:
            # Collect and deduplicate entities across all events in the batch
            seen: dict[str, EntityRef] = {}
            for e in events:
                if isinstance(e, EmbedDirty):
                    seen[e.entity.qualified_name] = e.entity

            entities = list(seen.values())
            logger.debug("Tier3 batch {}: {} unique entity(ies)", batch_id, len(entities))

            if not entities:
                return

            t0 = asyncio.get_event_loop().time()

            # 1. Read entity properties from graph (includes embed_hash + embedding)
            #    Pass labels so queries use per-label indices instead of full scans.
            qns = [e.qualified_name for e in entities]
            entity_labels = [e.node_type for e in entities]
            entity_props = await self.graph.read_entity_texts(qns, labels=entity_labels)

            # Build uid→label map for downstream writes
            uid_label: dict[str, str] = {}
            for props_row in entity_props:
                lbl = props_row.get("_label")
                if lbl:
                    uid_label[props_row["uid"]] = lbl

            # 2. Build embed texts — graph-check for unchanged content
            to_process: list[tuple[str, str, str]] = []  # (uid, text, text_hash)
            graph_hits = 0
            for props in entity_props:
                text = build_embed_text(props)
                if not text:
                    continue
                uid = props["uid"]
                text_hash = EmbedCache.hash_text(text)
                if props.get("embed_hash") == text_hash and props.get("embedding") is not None:
                    graph_hits += 1
                else:
                    to_process.append((uid, text, text_hash))

            total = graph_hits + len(to_process)
            if not to_process:
                elapsed = asyncio.get_event_loop().time() - t0
                logger.debug(
                    "Tier3 batch {}: {} entities, {} graph hits, 0 cache hits, 0 embedded ({:.1f}s)",
                    batch_id,
                    total,
                    graph_hits,
                    elapsed,
                )
                return

            # 3. Valkey cache check → 4. API call for misses
            cache_resolved, need_embed, cache_hits = await self._resolve_cache(to_process)
            api_vectors = await self._embed_and_store(need_embed)

            # 5. Write all new/changed vectors + embed_hashes to graph
            all_resolved = cache_resolved + api_vectors
            if all_resolved:
                # Split into labeled (index-backed) and unlabeled (fallback) writes
                labeled = [(qn, vec, th) for qn, vec, th in all_resolved if qn in uid_label]
                unlabeled = [(qn, vec, th) for qn, vec, th in all_resolved if qn not in uid_label]

                if labeled:
                    lbl_list = [uid_label[qn] for qn, _, _ in labeled]
                    await self.graph.write_embeddings(
                        [(qn, vec) for qn, vec, _ in labeled],
                        labels=lbl_list,
                    )
                    await self.graph.write_embed_hashes(
                        [(qn, th) for qn, _, th in labeled],
                        labels=lbl_list,
                    )
                if unlabeled:
                    await self.graph.write_embeddings([(qn, vec) for qn, vec, _ in unlabeled])
                    await self.graph.write_embed_hashes([(qn, th) for qn, _, th in unlabeled])

            elapsed = asyncio.get_event_loop().time() - t0
            span.set_attribute("entities_count", total)
            span.set_attribute("graph_hits", graph_hits)
            span.set_attribute("cache_hits", cache_hits)
            span.set_attribute("api_embedded", len(api_vectors))

            get_metrics().embedding_latency.record(elapsed)

            logger.debug(
                "Tier3 batch {}: {} entities, {} graph hits, {} cache hits, {} embedded ({:.1f}s)",
                batch_id,
                total,
                graph_hits,
                cache_hits,
                len(api_vectors),
                elapsed,
            )
