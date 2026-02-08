"""Tiered consumer pipeline for event-driven indexing.

Three tiers form a linear pipeline, each pulling at its own pace:

    FileChanged → Tier 1 (graph metadata) → ASTDirty → Tier 2 (AST parse)
                → significance gate → EmbedDirty → Tier 3 (embeddings)

Each tier uses batch-pull with configurable time/count policy and
deduplicates within its batch window.
"""

from __future__ import annotations

import asyncio
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from code_atlas.embeddings import EmbedCache, build_embed_text
from code_atlas.events import (
    ASTDirty,
    EmbedDirty,
    EntityRef,
    Event,
    EventBus,
    FileChanged,
    Topic,
    decode_event,
)
from code_atlas.parser import parse_file

if TYPE_CHECKING:
    from code_atlas.embeddings import EmbedClient
    from code_atlas.graph import GraphClient
    from code_atlas.settings import AtlasSettings

# ---------------------------------------------------------------------------
# Batch policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BatchPolicy:
    """Controls when a consumer flushes its accumulated batch."""

    time_window_s: float  # Max seconds to accumulate before flush
    max_batch_size: int  # Max items before flush (whichever hits first)


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
    ) -> None:
        self.bus = bus
        self.input_topic = input_topic
        self.group = group
        self.consumer_name = consumer_name
        self.policy = policy
        self._stop = False

    @abstractmethod
    async def process_batch(self, events: list[Event], batch_id: str) -> None:
        """Process a deduplicated batch. Subclasses implement tier logic."""

    def dedup_key(self, event: Event) -> str:
        """Return a dedup key for an event. Override for custom logic."""
        if isinstance(event, FileChanged):
            return event.path
        return str(id(event))

    def stop(self) -> None:
        """Signal the consumer to stop after the current iteration."""
        self._stop = True

    async def run(self) -> None:
        """Main consumer loop — runs until ``stop()`` is called."""
        await self.bus.ensure_group(self.input_topic, self.group)
        logger.info("{} started (group={}, topic={})", self.consumer_name, self.group, self.input_topic.value)

        pending: dict[str, tuple[bytes, Event]] = {}  # dedup_key → (msg_id, event)
        window_start: float | None = None

        while not self._stop:
            # Pull messages (short block so we can check flush + stop)
            block_ms = max(100, int(self.policy.time_window_s * 1000 // 2))
            messages = await self.bus.read_batch(
                self.input_topic,
                self.group,
                self.consumer_name,
                count=self.policy.max_batch_size,
                block_ms=block_ms,
            )

            for msg_id, fields in messages:
                event = decode_event(self.input_topic, fields)
                key = self.dedup_key(event)
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
                await self.process_batch(events, batch_id)
                await self.bus.ack(self.input_topic, self.group, *msg_ids)
            except Exception:
                # Failed batch: do NOT ack — Redis will redeliver via PEL
                logger.exception("{} batch {} failed, will retry", self.consumer_name, batch_id)

            pending.clear()
            window_start = None

        logger.info("{} stopped", self.consumer_name)


# ---------------------------------------------------------------------------
# Tier 1: Graph metadata (cheap, fast)
# ---------------------------------------------------------------------------


class Tier1GraphConsumer(TierConsumer):
    """Tier 1: Update file-level graph metadata, always publish ASTDirty downstream."""

    def __init__(self, bus: EventBus, graph: GraphClient, settings: AtlasSettings) -> None:
        super().__init__(
            bus=bus,
            input_topic=Topic.FILE_CHANGED,
            group="tier1-graph",
            consumer_name="tier1-graph-0",
            policy=BatchPolicy(time_window_s=0.5, max_batch_size=50),
        )
        self.graph = graph
        self.settings = settings

    def dedup_key(self, event: Event) -> str:
        if isinstance(event, FileChanged):
            return event.path
        return super().dedup_key(event)

    async def process_batch(self, events: list[Event], batch_id: str) -> None:
        paths = [e.path for e in events if isinstance(e, FileChanged)]
        logger.info("Tier1 batch {}: {} file(s)", batch_id, len(paths))

        # TODO: Update Memgraph file nodes (timestamps, staleness flags)

        # Always publish downstream — Tier 2 decides significance
        await self.bus.publish(Topic.AST_DIRTY, ASTDirty(paths=paths, batch_id=batch_id))


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


class Tier2ASTConsumer(TierConsumer):
    """Tier 2: Parse AST via tree-sitter, write entities to graph, publish EmbedDirty."""

    def __init__(self, bus: EventBus, graph: GraphClient, settings: AtlasSettings) -> None:
        super().__init__(
            bus=bus,
            input_topic=Topic.AST_DIRTY,
            group="tier2-ast",
            consumer_name="tier2-ast-0",
            policy=BatchPolicy(time_window_s=3.0, max_batch_size=20),
        )
        self.graph = graph
        self.settings = settings
        self.stats = Tier2Stats()

    async def process_batch(self, events: list[Event], batch_id: str) -> None:
        all_paths: list[str] = []
        for e in events:
            if isinstance(e, ASTDirty):
                all_paths.extend(e.paths)
        # Deduplicate paths within this batch
        unique_paths = list(dict.fromkeys(all_paths))

        logger.info("Tier2 batch {}: {} unique path(s)", batch_id, len(unique_paths))

        project_name = self.settings.project_root.name
        changed_entity_refs: list[EntityRef] = []
        files_skipped = 0
        total_changed = 0

        for file_path in unique_paths:
            # Read file from disk — delete file's entities if it was removed
            full_path = Path(self.settings.project_root) / file_path
            if not full_path.is_file():
                logger.debug("Tier2: file deleted, removing entities for {}", file_path)
                deleted = await self.graph.delete_file_entities(project_name, file_path)
                self.stats.files_deleted += 1
                self.stats.entities_deleted += len(deleted)
                continue

            try:
                source = full_path.read_bytes()
            except OSError:
                logger.warning("Tier2: cannot read {}", file_path)
                continue

            # Parse with tree-sitter
            parsed = parse_file(file_path, source, project_name)
            if parsed is None:
                logger.debug("Tier2: unsupported language for {}", file_path)
                continue

            # Write to graph (delta-aware)
            result = await self.graph.upsert_file_entities(
                project_name=project_name,
                file_path=file_path,
                entities=parsed.entities,
                relationships=parsed.relationships,
            )

            # Accumulate stats
            self.stats.files_processed += 1
            self.stats.entities_added += len(result.added)
            self.stats.entities_modified += len(result.modified)
            self.stats.entities_deleted += len(result.deleted)
            self.stats.entities_unchanged += len(result.unchanged)

            # Only collect refs for added + modified entities (not unchanged)
            changed_qns = set(result.added) | set(result.modified)
            if not changed_qns:
                files_skipped += 1
                self.stats.files_skipped += 1
                continue

            total_changed += len(changed_qns)
            entity_map = {
                (e.qualified_name.split(":", 1)[1] if ":" in e.qualified_name else e.qualified_name): e
                for e in parsed.entities
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

        logger.info(
            "Tier2 batch {}: {} files, {} skipped, {} entities changed",
            batch_id,
            len(unique_paths),
            files_skipped,
            total_changed,
        )

        if changed_entity_refs:
            await self.bus.publish(
                Topic.EMBED_DIRTY,
                EmbedDirty(entities=changed_entity_refs, significance="HIGH", batch_id=batch_id),
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
        self, bus: EventBus, graph: GraphClient, embed: EmbedClient, *, cache: EmbedCache | None = None
    ) -> None:
        super().__init__(
            bus=bus,
            input_topic=Topic.EMBED_DIRTY,
            group="tier3-embed",
            consumer_name="tier3-embed-0",
            policy=BatchPolicy(time_window_s=15.0, max_batch_size=100),
        )
        self.graph = graph
        self.embed = embed
        self.cache = cache

    def dedup_key(self, event: Event) -> str:
        # For EmbedDirty we can't dedup at the event level easily —
        # dedup happens inside process_batch across all entities
        return str(id(event))

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
            for qn, text, text_hash in to_process:
                if text_hash in cached:
                    cache_resolved.append((qn, cached[text_hash], text_hash))
                    hits += 1
                else:
                    need_embed.append((qn, text, text_hash))
            return cache_resolved, need_embed, hits
        return [], list(to_process), 0

    async def _embed_and_store(self, need_embed: list[tuple[str, str, str]]) -> list[tuple[str, list[float], str]]:
        """Embed texts via API and store results in cache. Returns (qn, vector, hash) tuples."""
        if not need_embed:
            return []
        texts = [text for _, text, _ in need_embed]
        vectors = await self.embed.embed_batch(texts)
        result = [(qn, vec, th) for (qn, _t, th), vec in zip(need_embed, vectors, strict=True)]
        if self.cache is not None:
            await self.cache.put_many([(th, vec) for _, vec, th in result])
        return result

    async def process_batch(self, events: list[Event], batch_id: str) -> None:
        # Collect and deduplicate entities across all events in the batch
        seen: dict[str, EntityRef] = {}
        for e in events:
            if isinstance(e, EmbedDirty):
                for entity in e.entities:
                    seen[entity.qualified_name] = entity

        entities = list(seen.values())
        logger.info("Tier3 batch {}: {} unique entity(ies)", batch_id, len(entities))

        if not entities:
            return

        t0 = asyncio.get_event_loop().time()

        # 1. Read entity properties from graph (includes embed_hash + embedding)
        qns = [e.qualified_name for e in entities]
        entity_props = await self.graph.read_entity_texts(qns)

        # 2. Build embed texts — graph-check for unchanged content
        to_process: list[tuple[str, str, str]] = []  # (qn, text, text_hash)
        graph_hits = 0
        for props in entity_props:
            text = build_embed_text(props)
            if not text:
                continue
            qn = props["qualified_name"]
            text_hash = EmbedCache.hash_text(text)
            if props.get("embed_hash") == text_hash and props.get("embedding") is not None:
                graph_hits += 1
            else:
                to_process.append((qn, text, text_hash))

        total = graph_hits + len(to_process)
        if not to_process:
            elapsed = asyncio.get_event_loop().time() - t0
            logger.info(
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
            await self.graph.write_embeddings([(qn, vec) for qn, vec, _ in all_resolved])
            await self.graph.write_embed_hashes([(qn, th) for qn, _, th in all_resolved])

        elapsed = asyncio.get_event_loop().time() - t0
        logger.info(
            "Tier3 batch {}: {} entities, {} graph hits, {} cache hits, {} embedded ({:.1f}s)",
            batch_id,
            total,
            graph_hits,
            cache_hits,
            len(api_vectors),
            elapsed,
        )
