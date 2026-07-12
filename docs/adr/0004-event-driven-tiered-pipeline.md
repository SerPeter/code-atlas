# ADR-0004: Event-Driven Tiered Pipeline

## Status

Accepted — amended by [ADR-0009](./0009-event-pipeline-durability-contract.md) (significance
gating and durability guarantees superseded; tiered pipeline shape unchanged)

## Date

2026-02-07

## Context

Code Atlas has two distinct event sources — file system changes (continuous) and MCP requests (on-demand) — that feed
into an indexing pipeline where operations have vastly different costs:

- **Cheap**: Graph node metadata updates (sub-millisecond Cypher writes)
- **Medium**: AST re-parsing (tree-sitter, fast) + entity diffing + graph edge updates
- **Expensive**: Embedding generation via TEI (network call, batch processing)

A flat debounce strategy (as attempted in code-graph-rag PR #213) treats all work equally. This leads to either:

- Re-embedding on every minor change (wasteful — TEI calls are the bottleneck)
- Delaying cheap updates unnecessarily (stale graph metadata while waiting for debounce)

We need an architecture that processes cheap work immediately and expensive work only when semantically justified.

## Decision

We adopt an **event-driven tiered pipeline** using **Redis/Valkey Streams** as the event bus backbone, following a "dumb
pipes, smart endpoints" principle: Redis routes messages between decoupled stages, and each consumer implements its own
batching, dedup, and gating logic.

### Event Bus: Redis Streams

Redis Streams provide the pub/sub backbone with consumer groups:

- **Native batch-pull** (`XREADGROUP COUNT N BLOCK ms`) — consumers control their own pace
- **Multi-process ready** — no single-process rewrite later when scaling
- **Dual-use** — same Valkey instance serves as embedding cache (task 02-search-04)
- **Lightweight** — 30MB Docker image, 5-10MB idle RAM
- **Reliable** — failed batches stay in the pending entries list for redelivery

### Event Types

Typed frozen dataclasses with JSON serialization for Redis transport:

- `FileChanged(path, change_type, timestamp)` — published by file watcher
- `EmbedDirty(entities: list[EntityRef], significance, batch_id)` — published by AST stage

### Two-Stage Pipeline

```
                     atlas:file-changed                                atlas:embed-dirty
                          stream                                           stream
                            │                                                │
                     ┌──────▼───────┐                                ┌──────▼───────┐
  File Watcher ────► │  AST Stage   │ ─────── significance gate ───► │ Embed Stage  │
                     │ hash gate +  │  only if semantically changed  │  Embeddings  │
                     │ parse + diff │                                │ (15s batch)  │
                     │  (3s batch)  │                                └──────────────┘
                     └──────────────┘
```

Each stage pulls at its own pace via `XREADGROUP`, deduplicates within its batch window, and publishes downstream only
if warranted.

### Per-Consumer Batch Policy

| Stage | Window | Max Batch | Dedup Key             |
| ----- | ------ | --------- | --------------------- |
| AST   | 3.0s   | 30        | File path             |
| Embed | 15.0s  | 100       | Entity qualified name |

Hybrid batching: flush when count OR time threshold hit, whichever first. Same file changed 5× in window = 1 work item.

### Event Data Flow

```
FileChanged                                         EmbedDirty
┌─────────────┐                                     ┌──────────────────────────┐
│ path: str   │                                     │ entity: EntityRef        │
│ change_type │ ─── AST stage ── sig gate ────────► │ significance: str        │
│ timestamp   │                                     └──────────────────────────┘
└─────────────┘                                      EntityRef:
                                                       qualified_name, node_type,
                                                       file_path
```

### Significance Gating (AST → Embed)

The AST stage evaluates whether a change is semantically significant enough to warrant re-embedding:

| Condition                   | Level    | Action              |
| --------------------------- | -------- | ------------------- |
| Whitespace/formatting only  | NONE     | Stop                |
| Non-docstring comment       | TRIVIAL  | Stop                |
| Docstring changed           | MODERATE | Gate through        |
| Body changed < 20% AST diff | MODERATE | Gate through        |
| Body changed >= 20%         | HIGH     | Gate through        |
| Signature changed           | HIGH     | Always gate through |
| Entity added/deleted        | HIGH     | Always gate through |

### Error Handling

Failed batches are NOT acknowledged — Redis re-delivers via the pending entries list (PEL). Each consumer handles its
own retries through this mechanism, avoiding the need for a separate dead-letter queue.

## Consequences

### Positive

- Cheap operations (staleness flags, graph metadata) are near-instant — MCP queries reflect changes within ~1s
- Expensive operations (embeddings) only run when semantically justified — significant cost reduction
- Decoupled stages can be developed, tested, and scaled independently
- Batching per stage matches the cost profile of each operation
- Multi-process from day one — no rewrite needed when scaling
- Dual-use of Valkey for event bus + embedding cache
- Natural extension point: new stages or event types can be added without restructuring

### Negative

- More architectural complexity than a simple "reindex everything on change"
- Significance threshold heuristics need tuning and may produce false negatives (skipping re-embeds that should have
  happened)
- Debugging event flow across stages is harder than a linear pipeline
- Additional infrastructure dependency (Valkey), though lightweight

### Risks

- Threshold tuning: too aggressive = stale embeddings, too conservative = excessive TEI calls. Need observability on
  gate decisions.
- Event ordering: if the AST stage processes file A before file B, but B depends on A's entities, the diff may be
  incorrect. Batch boundaries must align with dependency boundaries.
- Complexity creep: the event bus must stay simple. If we find ourselves adding routing rules, dead-letter queues, or
  retry logic, we've gone too far.

## Alternatives Considered

### Alternative 1: Flat Debounce (code-graph-rag PR #213 approach)

Single debounce timer + max-wait ceiling. All work (graph updates, AST parsing, embedding) happens in one batch.

**Why rejected:**

- Treats all work equally — either everything is delayed (bad for cheap updates) or everything is eager (wasteful for
  expensive updates)
- No way to skip embedding for trivial changes
- code-graph-rag's implementation had unresolved bugs around threading and timer cancellation

### Alternative 2: In-Process asyncio.Queue

Pure asyncio queues per topic, task per subscriber. Zero external dependencies.

**Why rejected:**

- Locks to single-process — multi-process scaling requires a full rewrite
- No consumer groups, no persistent pending entries, no batch-pull primitives
- Simpler initially but creates technical debt when daemon + MCP need to communicate

### Alternative 3: SQLite Job Queue

Use SQLite WAL as a durable work queue between tiers.

**Why rejected:**

- Persistence adds complexity without benefit (process re-evaluates on startup)
- Polling-based consumption adds latency vs event-driven push
- Doesn't naturally support the pub/sub pattern needed for tier fan-out

### Alternative 4: External Message Broker (RabbitMQ)

Full-featured message broker for event routing.

**Why rejected:**

- Heavy infrastructure for a local developer tool (100MB+ Docker image)
- Feature-rich but most features (persistence, routing rules, exchanges) are unnecessary
- Valkey provides the needed subset at a fraction of the weight

## References

- [code-graph-rag PR #213 — Debounce implementation](https://github.com/vitali87/code-graph-rag/pull/213)
- [code-graph-rag #286 — Infinite reindex loop](https://github.com/vitali87/code-graph-rag/issues/286)
