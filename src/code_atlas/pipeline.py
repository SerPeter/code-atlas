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

    async def process_batch(self, events: list[Event], batch_id: str) -> None:
        all_paths: list[str] = []
        for e in events:
            if isinstance(e, ASTDirty):
                all_paths.extend(e.paths)
        # Deduplicate paths within this batch
        unique_paths = list(dict.fromkeys(all_paths))

        logger.info("Tier2 batch {}: {} unique path(s)", batch_id, len(unique_paths))

        project_name = self.settings.project_root.name
        all_entity_refs: list[EntityRef] = []

        for file_path in unique_paths:
            # Read file from disk (skip if deleted/missing)
            full_path = Path(self.settings.project_root) / file_path
            if not full_path.is_file():
                logger.debug("Tier2: skipping missing file {}", file_path)
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

            # Write to graph
            await self.graph.upsert_file_entities(
                project_name=project_name,
                file_path=file_path,
                entities=parsed.entities,
                relationships=parsed.relationships,
            )

            # Collect entity refs for downstream
            all_entity_refs.extend(
                EntityRef(
                    qualified_name=entity.qualified_name,
                    node_type=entity.label.value,
                    file_path=entity.file_path,
                )
                for entity in parsed.entities
            )

        # No significance gate yet (always pass through) — epic 05-delta
        if all_entity_refs:
            await self.bus.publish(
                Topic.EMBED_DIRTY,
                EmbedDirty(entities=all_entity_refs, significance="HIGH", batch_id=batch_id),
            )


# ---------------------------------------------------------------------------
# Tier 3: Embeddings (expensive, heavily batched)
# ---------------------------------------------------------------------------


class Tier3EmbedConsumer(TierConsumer):
    """Tier 3: Re-embed entities via TEI. Deduplicates by qualified name."""

    def __init__(self, bus: EventBus) -> None:
        super().__init__(
            bus=bus,
            input_topic=Topic.EMBED_DIRTY,
            group="tier3-embed",
            consumer_name="tier3-embed-0",
            policy=BatchPolicy(time_window_s=15.0, max_batch_size=100),
        )

    def dedup_key(self, event: Event) -> str:
        # For EmbedDirty we can't dedup at the event level easily —
        # dedup happens inside process_batch across all entities
        return str(id(event))

    async def process_batch(self, events: list[Event], batch_id: str) -> None:
        # Collect and deduplicate entities across all events in the batch
        seen: dict[str, EntityRef] = {}
        for e in events:
            if isinstance(e, EmbedDirty):
                for entity in e.entities:
                    seen[entity.qualified_name] = entity

        entities = list(seen.values())
        logger.info("Tier3 batch {}: {} unique entity(ies)", batch_id, len(entities))

        # TODO: Call TEI for embeddings, write vectors to Memgraph
